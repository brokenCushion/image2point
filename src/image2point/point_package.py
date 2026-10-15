"""Portable point-cache contract and CPU artist operations (NumPy + stdlib only)."""

import json
import os
import tempfile
from pathlib import Path

import numpy as np

VERSION = 1


def srgb_to_linear(rgb):
    rgb = np.asarray(rgb, dtype=np.float32)
    return np.where(rgb <= 0.04045, rgb / 12.92, ((rgb + 0.055) / 1.055) ** 2.4).astype(np.float32)


def atomic(path, writer):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(dir=path.parent, suffix=".tmp")
    try:
        with os.fdopen(fd, "wb") as stream:
            writer(stream)
        os.replace(temporary, path)
    finally:
        Path(temporary).unlink(missing_ok=True)


def json_write(path, value):
    atomic(path, lambda f: f.write(json.dumps(value, indent=2, allow_nan=False).encode()))


def child(root, name):
    result = (Path(root) / name).resolve()
    if not result.is_relative_to(Path(root).resolve()):
        raise ValueError("Cache paths must stay inside the package directory")
    return result


def load_frame(manifest_path, frame):
    path = Path(manifest_path)
    manifest = json.loads(path.read_text())
    # Packages written before the image2depth -> image2point rename are still readable.
    kinds = ("image2point.point_package", "image2depth.point_package")
    if manifest.get("version") != VERSION or manifest.get("kind") not in kinds:
        raise ValueError("Unsupported Image2Point point package")
    item = next((x for x in manifest["frames"] if x["frame"] == frame), None)
    if item is None:
        raise ValueError(f"Frame {frame} is not in this cache")
    with np.load(child(path.parent, item["file"]), allow_pickle=False) as archive:
        arrays = {key: archive[key] for key in archive.files}
    if manifest.get("color_space") not in (None, "linear_srgb"):
        raise ValueError("Unsupported point colour space")
    if manifest.get("color_space") == "linear_srgb" and "linear_rgb" not in arrays:
        raise ValueError("Linear point colour missing from cache")
    return manifest, item, arrays


def frame_points(path, item):
    """World points and labels of one frame, read from its NPZ."""
    with np.load(child(path.parent, item["file"]), allow_pickle=False) as archive:
        points = archive["P"]
        labels = archive["object_id"]
    return points, labels


def project(points, item, shape):
    """Pixel (v, u) and camera depth of world points seen by a frame's camera; only those in view."""
    pose = np.asarray(item["c2w"], dtype=float)
    k = np.asarray(item["K"], dtype=float)
    cam = (points - pose[:3, 3]) @ pose[:3, :3]
    cam = cam[cam[:, 2] > 0]
    u = np.floor(k[0, 0] * cam[:, 0] / cam[:, 2] + k[0, 2]).astype(int)
    v = np.floor(k[1, 1] * cam[:, 1] / cam[:, 2] + k[1, 2]).astype(int)
    inside = (u >= 0) & (u < shape[1]) & (v >= 0) & (v < shape[0])
    return v[inside], u[inside], cam[inside, 2]


def camera_depth(points, item):
    pose = np.asarray(item["c2w"], dtype=float)
    return ((points - pose[:3, 3]) @ pose[:3, :3])[..., 2]


def measure_shot(manifest_path, smoothing=12.0, neighbours=3):
    """Read every frame once: per-frame scale factors and each actor's depth per frame.

    MoGe guesses the metric scale of every frame on its own, so a whole frame can come out a
    few percent nearer or farther than its neighbours. The static background shows it: each
    frame's background points are moved into the cameras of the next `neighbours` frames, and
    the median ratio of the depths is the scale change between the two. All the changes are
    solved together (least squares), smoothed over `smoothing` frames, and each frame is
    scaled onto the smooth curve: fast flicker goes, slow drift and the overall scale stay.
    Actors are not used for the measurement, since they move.

    Returns (scales, actor_depths): scales[frame] is the factor for the whole frame;
    actor_depths[actor][frame] is that actor's median camera depth before any scaling.
    """
    path = Path(manifest_path)
    items = sorted(json.loads(path.read_text())["frames"], key=lambda x: x["frame"])
    recent = []  # (index, every fourth background point) of the last `neighbours` frames
    pairs = []  # (earlier index, later index, log of the depth ratio)
    actor_depths = {}
    for index, item in enumerate(items):
        points, labels = frame_points(path, item)
        depth = camera_depth(points, item)
        valid = np.isfinite(points).all(-1) & (depth > 0)
        background = valid & (labels == 0)
        for actor in np.unique(labels[valid & (labels > 0)]):
            selected = valid & (labels == actor)
            if selected.sum() >= 200:
                actor_depths.setdefault(int(actor), {})[item["frame"]] = float(np.median(depth[selected]))
        for earlier, seen_points in recent:
            v, u, seen = project(seen_points, item, labels.shape)
            matched = background[v, u]
            if matched.sum() >= 100:
                change = np.median(np.log(depth[v[matched], u[matched]] / seen[matched]))
                pairs.append((earlier, index, float(change)))
        recent = (recent + [(index, points[::4, ::4][background[::4, ::4]])])[-neighbours:]
    count = len(items)
    # Solve log-levels x: x[later] - x[earlier] = measured change, x[0] = 0, plus a weak
    # "no change" link between neighbours so frames without a measurement stay connected.
    rows = [(0, 0, 1.0, 0.0)]
    rows += [(earlier, later, 1.0, change) for earlier, later, change in pairs]
    rows += [(index - 1, index, 0.01, 0.0) for index in range(1, count)]
    system = np.zeros((len(rows), count))
    target = np.zeros(len(rows))
    for row, (earlier, later, weight, change) in enumerate(rows):
        system[row, later] += weight
        if row:
            system[row, earlier] -= weight
        target[row] = weight * change
    levels = np.exp(np.linalg.lstsq(system, target, rcond=None)[0])
    offsets = np.arange(count)
    weights = np.exp(-0.5 * ((offsets[:, None] - offsets[None, :]) / smoothing) ** 2)
    smooth = weights @ levels / weights.sum(1)
    scales = {item["frame"]: float(goal / level) for item, goal, level in zip(items, smooth, levels)}
    return scales, actor_depths


def depth_scales(manifest_path, smoothing=12.0, neighbours=3):
    """Per-frame factor, keyed by frame, that removes MoGe's frame-to-frame scale flicker."""
    return measure_shot(manifest_path, smoothing, neighbours)[0]


def actor_factors(depths, scales, smoothing):
    """Per-frame factor for one actor that smooths its own depth wobble over `smoothing` frames.

    depths is that actor's median depth per frame (from measure_shot), taken after the scene
    scale fix. The factor moves the actor onto the smooth curve, so a steady walk is kept while
    frame-to-frame wobble goes. 0 turns it off (empty result).
    """
    if smoothing <= 0 or not depths:
        return {}
    frames = np.array(sorted(depths))
    values = np.array([depths[frame] * scales.get(frame, 1.0) for frame in frames])
    weights = np.exp(-0.5 * ((frames[:, None] - frames[None, :]) / smoothing) ** 2)
    smooth = weights @ values / weights.sum(1)
    return {int(frame): float(goal / value) for frame, goal, value in zip(frames, smooth, values)}


def background_factors(manifest_path, frame, radius, scales):
    """Per-pixel factor for one frame that averages its background with `radius` frames each side.

    The set does not move, so a background pixel can be compared with the same spot in
    neighbouring frames, matched through the tracked camera. Neighbour depths more than 2%
    away are left out (an actor passing in front, or an edge). Actor pixels keep factor 1.
    """
    path = Path(manifest_path)
    items = sorted(json.loads(path.read_text())["frames"], key=lambda x: x["frame"])
    index = next(i for i, item in enumerate(items) if item["frame"] == frame)
    points, labels = frame_points(path, items[index])
    depth = camera_depth(points, items[index]) * scales.get(frame, 1.0)
    background = np.isfinite(points).all(-1) & (depth > 0) & (labels == 0)
    total = np.where(background, depth, 0.0)
    count = background.astype(np.float64)
    for item in items[max(0, index - radius) : index + radius + 1]:
        if item["frame"] == frame:
            continue
        other, other_labels = frame_points(path, item)
        pose = np.asarray(item["c2w"], dtype=float)
        usable = np.isfinite(other).all(-1) & (other_labels == 0)
        # The neighbour's own scale fix, applied along its camera rays.
        corrected = (other[usable] - pose[:3, 3]) * scales.get(item["frame"], 1.0) + pose[:3, 3]
        v, u, seen = project(corrected, items[index], labels.shape)
        own = np.where(background[v, u], depth[v, u], np.inf)
        close = np.abs(seen / own - 1) < 0.02
        np.add.at(total, (v[close], u[close]), seen[close])
        np.add.at(count, (v[close], u[close]), 1)
    factor = np.ones(labels.shape, np.float32)
    factor[background] = (total[background] / count[background] / depth[background]).astype(np.float32)
    return factor


def camera_points(depth, k):
    y, x = np.indices(depth.shape, dtype=np.float32)
    rays = np.stack((x + 0.5, y + 0.5, np.ones_like(x)), -1) @ np.linalg.inv(k).T
    with np.errstate(invalid="ignore"):  # Invalid model depth stays NaN.
        return (rays * depth[..., None]).astype(np.float32)


def tag_masks(masks, ids, shape):
    h, w = shape
    if len(masks) != len(ids) or len(set(map(int, ids))) != len(ids):
        raise ValueError("Supply one unique integer ID per actor mask")
    labels = np.zeros((h, w), np.int32)
    alpha = np.zeros((h, w), np.float32)
    for mask, object_id in zip(masks, ids):
        if int(object_id) <= 0 or int(object_id) != object_id:
            raise ValueError("Actor IDs must be positive; zero is reserved for background")
        if mask.shape != (h, w) or not np.isfinite(mask).all() or np.any((mask < 0) | (mask > 1)):
            raise ValueError("Mask shape/values do not match the point grid")
        # A pixel belongs to an actor only at 0.5 coverage or more (the same rule as edge repair), so
        # faint values such as video-compression fringes around an RGB roto stay background.
        # Highest coverage wins; ties go to the earlier supplied actor.
        replace = (mask >= 0.5) & (mask > alpha)
        labels[replace] = object_id
        alpha[replace] = np.clip(mask[replace], 0, 1)
    return labels, alpha


def morph(mask, radius, dilate):
    result = mask.astype(bool)
    for axis in (0, 1):
        pad = [(0, 0), (0, 0)]
        pad[axis] = (radius, radius)
        padded = np.pad(result, pad, constant_values=False)
        result = np.full_like(result, not dilate)
        combine = np.logical_or if dilate else np.logical_and
        for i in range(2 * radius + 1):
            section = [slice(None), slice(None)]
            section[axis] = slice(i, i + result.shape[axis])
            combine(result, padded[tuple(section)], out=result)
    return result


def extend_depth(depth, mask, inset, extension, tolerance=0.0):
    """Nearest reliable actor Z, with a bounded NumPy fallback for stock Houdini.

    Reliable ("seed") pixels lie at least `inset` pixels inside the mask with a valid depth
    and, when tolerance > 0, within `tolerance` metres of the actor's median seed depth
    (MoGe blends depths where a thin actor meets a far background). Every other pixel in
    the mask, plus `extension` pixels around it, takes the nearest seed's depth. Pixels
    with no seed within `reach` steps (stray mask specks that Edge Repair removes entirely)
    are dropped rather than given a far-away depth; the limit is the same with or without
    SciPy, so local and farm bakes agree.
    """
    inside = mask > 0.5
    support = morph(inside, extension, True)
    seeds = morph(inside, inset, False) & np.isfinite(depth) & (depth > 0)
    if not support.any():
        return depth.copy(), support
    if tolerance > 0 and seeds.any():
        seeds &= np.abs(depth - np.median(depth[seeds])) <= tolerance
    if not seeds.any():
        raise ValueError("No reliable actor interior; reduce Edge Repair or correct the mask")
    reach = 2 * (inset + extension) + 8
    try:
        from scipy.ndimage import distance_transform_cdt, distance_transform_edt

        nearest = distance_transform_edt(~seeds, return_distances=False, return_indices=True)
        filled = depth[tuple(nearest)]
        support &= distance_transform_cdt(~seeds, metric="chessboard") <= reach
    except ImportError:
        # Bounded local propagation; install SciPy for faster large-frame bakes.
        h, w = depth.shape
        yy, xx = np.indices((h, w))
        source = np.where(seeds, yy * w + xx, -1)
        distance = np.where(seeds, 0.0, np.inf)
        for _ in range(reach):
            changed = False
            for dy, dx in ((-1, 0), (1, 0), (0, -1), (0, 1), (-1, -1), (-1, 1), (1, -1), (1, 1)):
                shifted = np.full_like(source, -1)
                ys, xs = slice(max(0, dy), min(h, h + dy)), slice(max(0, dx), min(w, w + dx))
                shifted[ys, xs] = source[slice(max(0, -dy), min(h, h - dy)), slice(max(0, -dx), min(w, w - dx))]
                d = (shifted // w - yy) ** 2 + (shifted % w - xx) ** 2
                better = (shifted >= 0) & (d < distance) & support
                if better.any():
                    source[better] = shifted[better]
                    distance[better] = d[better]
                    changed = True
            if not changed:
                break
        support &= source >= 0
        filled = depth.ravel()[np.maximum(source, 0)]
    out = depth.copy()
    out[support & ~seeds] = filled[support & ~seeds]
    return out, support


def process(arrays, item, settings):
    """Return flat render attributes; input is the immutable full-frame point grid."""
    p = arrays["P"]
    h, w = p.shape[:2]
    colour = arrays.get("linear_rgb")
    if colour is None:
        colour = srgb_to_linear(arrays["rgb"].astype(np.float32) / 255)
    if colour.shape != p.shape or not np.isfinite(colour).all():
        raise ValueError("Invalid linear point colour")
    ids = arrays["actor_ids"].astype(int)
    masks = arrays["masks"]
    labels = arrays["object_id"]
    if (
        p.shape != (h, w, 3)
        or masks.shape != (len(ids), h, w)
        or labels.shape != (h, w)
        or arrays["rgb"].shape != (h, w, 3)
    ):
        raise ValueError("Malformed point grid or mask dimensions")
    if len(set(ids.tolist())) != len(ids) or np.any(ids <= 0):
        raise ValueError("Cache actor IDs must be unique and positive")
    pose = np.asarray(item["c2w"], dtype=float)
    k = np.asarray(item["K"], dtype=float)
    if (
        pose.shape != (4, 4)
        or k.shape != (3, 3)
        or not np.isfinite(pose).all()
        or not np.isfinite(k).all()
        or k[0, 0] <= 0
        or k[1, 1] <= 0
    ):
        raise ValueError("Invalid camera matrices")
    if not np.allclose(pose[:3, :3].T @ pose[:3, :3], np.eye(3), atol=1e-5):
        raise ValueError("Cache camera must have a rigid transform")
    cam = (p - pose[:3, 3]) @ pose[:3, :3]
    # Depth stabilisation (depth_scales): scaling camera-space points moves them along their
    # camera rays, so the plate alignment is kept.
    cam = cam * float(settings.get("depth_scale", 1.0))
    if "background_factor" in settings:
        # Background smoothing (background_factors): per pixel, along the camera rays.
        cam = cam * settings["background_factor"][..., None]
    # Per-Actor Overrides choose the actors: once any row exists only actors with a row are
    # included, and actors switched off (Keep unticked) are left out entirely.
    only = settings.get("only")
    dropped = set(settings.get("drop", []))
    selected = [actor for actor in ids.tolist() if (only is None or actor in only) and actor not in dropped]
    stride = int(settings.get("stride", 1))
    if stride < 1:
        raise ValueError("Point spacing must be positive")
    names = ["P", "Cd", "object_id", "pscale", "uv", "source_pixel"]
    roto = arrays.get("roto_grid")
    if roto is not None:
        names.append("roto")  # The artist roto colour at each point's pixel: the matte for shaders.
        if roto.ndim == 2:
            roto = np.repeat(roto[..., None], 3, axis=-1)  # Older packages stored one greyscale matte.
    outputs = {name: [] for name in names}
    yy, xx = np.indices((h, w))
    sample = (xx % stride == 0) & (yy % stride == 0)
    if settings.get("remove_background", False) and not len(ids):
        raise ValueError("Remove Background needs actor masks; no actors are tagged in this package")
    parts = ([0] if not settings.get("remove_background", False) else []) + list(selected)
    for actor in parts:
        # The main depth offset moves the whole cloud; a per-actor offset then applies on top.
        # Edge and coverage overrides replace the main values.
        override = settings.get("overrides", {}).get(str(actor), {}) if actor else {}
        cfg = {**settings, **override}
        offset = float(settings.get("offset", 0)) + float(override.get("offset", 0))
        coverage = float(cfg.get("coverage", 1.05))
        inset, extension = int(cfg.get("inset", 0)), int(cfg.get("extension", 0))
        tolerance = float(cfg.get("tolerance", 0))
        if (
            not np.isfinite([offset, coverage, tolerance]).all()
            or coverage <= 0
            or tolerance < 0
            or not 0 <= inset <= 64
            or not 0 <= extension <= 64
        ):
            raise ValueError("Invalid depth, edge or coverage settings")
        if actor == 0:
            keep = labels == 0
            local = cam
        else:
            # The actor's mask (SAM) decides which pixels belong to it and drives edge repair.
            mask = masks[list(ids).index(actor)].astype(np.float32)
            if inset or extension or tolerance:
                depth, keep = extend_depth(cam[..., 2], mask, inset, extension, tolerance)
                local = camera_points(depth, k)
            else:
                keep = labels == actor
                local = cam
        keep &= sample & np.isfinite(local).all(-1) & (local[..., 2] > 0)
        local = local[keep]
        # Per-actor depth smoothing (actor_factors), along the camera rays.
        local = local * float(settings.get("actor_scales", {}).get(actor, 1.0))
        distance = np.linalg.norm(local, axis=1)
        factor = np.maximum(0.001, distance + offset) / np.maximum(distance, 1e-12)
        local = local * factor[:, None]
        positions = local @ pose[:3, :3].T + pose[:3, 3]
        # Circumscribed pixel footprint provides continuous front-facing coverage.
        radius = 0.5 * local[:, 2] * np.sqrt((stride / k[0, 0]) ** 2 + (stride / k[1, 1]) ** 2) * coverage
        outputs["P"].append(positions.astype("f4"))
        outputs["Cd"].append(colour[keep].astype("f4"))
        outputs["object_id"].append(np.full(keep.sum(), actor, np.int32))
        outputs["pscale"].append(radius.astype("f4"))
        outputs["uv"].append(
            np.column_stack(((xx[keep] + 0.5) / w, 1 - (yy[keep] + 0.5) / h, np.zeros(keep.sum()))).astype("f4")
        )
        outputs["source_pixel"].append((yy[keep] * w + xx[keep]).astype(np.int32))
        if "roto" in outputs:
            outputs["roto"].append(roto[keep].astype(np.float32))
    return {
        name: np.concatenate(chunks)
        if chunks
        else np.empty(
            (0, 3) if name in ("P", "Cd", "uv", "roto") else (0,),
            dtype=np.int32 if name in ("object_id", "source_pixel") else np.float32,
        )
        for name, chunks in outputs.items()
    }
