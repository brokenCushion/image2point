"""Embedded after point_package.py in the HDA PythonModule. No model dependencies."""

import threading
import uuid
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from itertools import islice

import hou

# Bake Locally prepares this many upcoming frames in background threads (about 2 GB each
# for a 4K frame). Measured on a 4K shot: 2 and 4 threads are equally fast (Python's lock
# and Houdini's main-thread geometry steps are the limit), 8 is slower. Farm bakes go through
# a standard Geometry ROP on the asset's internal OUT_RENDER instead.
LOCAL_BAKE_WORKERS = 2

# Depth stabilisation reads every frame of a package once; keep the result per manifest for the
# session. A lock because Bake Locally asks from background threads.
SCALES = {}
SCALES_LOCK = threading.Lock()


def settings(node, bake=False):
    result = {name: node.evalParm(name) for name in ("offset", "coverage", "inset", "extension", "remove_background")}
    result["stride"] = 1 if bake else node.evalParm("preview_stride")
    result["stabilize"] = bool(node.evalParm("stabilize"))
    result["bg_smoothing"] = node.evalParm("bg_smoothing")
    result["overrides"] = {}
    result["drop"] = []
    for i in range(1, node.evalParm("actor_controls") + 1):
        actor = str(node.evalParm("actor_id" + str(i)))
        if actor in result["overrides"]:
            raise ValueError("Two Per-Actor Override rows use actor ID " + actor)
        if not node.evalParm("keep" + str(i)):
            result["drop"].append(int(actor))
        result["overrides"][actor] = {
            name: node.evalParm(name + str(i))
            for name in ("offset", "coverage", "inset", "extension", "tolerance", "smoothing")
        }
    if result["overrides"]:
        result["only"] = [int(actor) for actor in result["overrides"]]
    return result


def rows_changed(node):
    """Per-Actor Overrides callback: a newly added row (default ID 1) gets the next unused actor ID."""
    used = set()
    for i in range(1, node.evalParm("actor_controls") + 1):
        parm = node.parm("actor_id" + str(i))
        if parm.eval() in used:
            parm.set(max(used) + 1)
        used.add(parm.eval())


def check_colours(node, values):
    """Viewport-only Cd for Roto Check / ID Check; bakes always keep the plate colour."""
    if node.evalParm("roto_check"):
        if "roto" not in values:
            raise ValueError("Roto Check: this package has no roto")
        values["Cd"] = values["roto"]  # The roto's own colours, as in the roto images.
    elif node.evalParm("id_check"):
        ids = values["object_id"]
        # Golden-ratio hue steps keep neighbouring IDs far apart for any number of actors.
        hue = (ids * 0.618034) % 1.0
        sector = hue * 6
        rgb = np.clip(np.abs((sector[:, None] + [0, 4, 2]) % 6 - 3) - 1, 0, 1)
        rgb[ids == 0] = 0.3  # Background: grey
        values["Cd"] = rgb.astype(np.float32)
    return values


def create_points(target, positions):
    """Add len(positions) points to target and set P from raw bytes.

    createPoints() only accepts a Python list of positions, which takes ~15 s for a 4K
    frame (8.8M points). Merging a one-point geometry in doubling blocks creates the same
    number of points in well under a second; P is then written in one call.
    """
    count = len(positions)
    block = hou.Geometry()
    block.createPoint()
    while count:
        if count & 1:
            target.merge(block)
        count >>= 1
        if count:
            doubled = hou.Geometry()
            doubled.merge(block)
            doubled.merge(block)
            block = doubled
    if len(positions):
        target.setPointFloatAttribValuesFromString("P", np.ascontiguousarray(positions, dtype="f4"))


def cached_measure(manifest_path):
    """(scales, actor depths) for a package, kept per manifest for the session.

    ComfyUI stores them in the manifest, so normally nothing is measured here; every farm task
    is a fresh Houdini and would otherwise re-read the whole shot. Older packages without the
    stored block are measured once (measure_shot reads every frame).
    """
    path = Path(manifest_path)
    key = (str(path.resolve()), path.stat().st_mtime_ns)
    with SCALES_LOCK:
        if key not in SCALES:
            stored = json.loads(path.read_text()).get("stabilisation")
            if stored:
                scales = {int(frame): scale for frame, scale in stored["scales"].items()}
                depths = {
                    int(actor): {int(frame): depth for frame, depth in values.items()}
                    for actor, values in stored["actor_depths"].items()
                }
                SCALES[key] = (scales, depths)
            else:
                SCALES[key] = measure_shot(path)
        return SCALES[key]


def stabilised(manifest_path, frame, cfg):
    """cfg plus this frame's stabilisation: scene scale, background and per-actor smoothing."""
    smoothing = {int(actor): o["smoothing"] for actor, o in cfg["overrides"].items() if o.get("smoothing")}
    if not (cfg.get("stabilize") or cfg.get("bg_smoothing") or smoothing):
        return cfg
    scales, depths = cached_measure(manifest_path)
    if not cfg.get("stabilize"):
        scales = {}
    extra = {"depth_scale": scales.get(frame, 1.0)}
    if cfg.get("bg_smoothing"):
        extra["background_factor"] = background_factors(manifest_path, frame, cfg["bg_smoothing"], scales)
    extra["actor_scales"] = {}
    for actor, amount in smoothing.items():
        factor = actor_factors(depths.get(actor, {}), scales, amount).get(frame)
        if factor is not None:
            extra["actor_scales"][actor] = factor
    return {**cfg, **extra}


def frame_values(manifest_path, frame, cfg):
    """Pure NumPy work for one frame (load + process); safe to run in a background thread."""
    manifest, item, arrays = load_frame(manifest_path, frame)
    cfg = stabilised(manifest_path, frame, cfg)
    return manifest, process(arrays, item, cfg)


def write_geometry(target, manifest, values, manifest_path):
    """Houdini geometry for one frame's values; must run on Houdini's main thread."""
    if abs(hou.fps() - manifest["fps"]) > 1e-5:
        raise ValueError("Scene FPS differs from the cache. Check the studio shot setup and cache FPS.")
    target.clear()
    create_points(target, values["P"])
    for name, data in values.items():
        if name == "P":
            continue
        default = (0.0, 0.0, 0.0) if data.ndim == 2 else (0 if data.dtype.kind in "iu" else 0.0)
        target.addAttrib(hou.attribType.Point, name, default)
        if data.dtype.kind in "iu":
            target.setPointIntAttribValues(name, data.ravel().tolist())
        else:
            target.setPointFloatAttribValuesFromString(name, np.ascontiguousarray(data, dtype="f4"))
    for name, value in [
        ("i2p_up_axis", manifest["up_axis"]),
        ("i2p_manifest", manifest_path),
        ("i2p_color_space", "linear_srgb"),
    ]:
        target.addAttrib(hou.attribType.Global, name, "")
        target.setGlobalAttribValue(name, value)


def geometry(node, target, bake=False):
    frame = hou.frame()  # Evaluating time makes the Python SOP time-dependent.
    if abs(frame - round(frame)) > 1e-5:
        raise ValueError("This cache contains integer frames; subframe interpolation is not supported")
    manifest_path = node.evalParm("manifest")
    manifest, values = frame_values(manifest_path, int(round(frame)), settings(node, bake))
    if not bake:
        values = check_colours(node, values)
    write_geometry(target, manifest, values, manifest_path)
    return manifest


def compute(job):
    frame, manifest_path, cfg = job
    return frame_values(manifest_path, frame, cfg)


def computed(jobs, workers):
    """Yield (job, frame_values result) in frame order, computing up to `workers` frames ahead."""
    if workers <= 1:
        for job in jobs:
            yield job, compute(job)
        return
    jobs = iter(jobs)
    with ThreadPoolExecutor(max_workers=workers) as pool:
        pending = deque((job, pool.submit(compute, job)) for job in islice(jobs, workers))
        while pending:
            job, future = pending.popleft()
            result = future.result()
            following = next(jobs, None)
            if following is not None:
                pending.append((following, pool.submit(compute, following)))
            yield job, result


def manifest_changed(node):
    """Initialize the bake range without changing studio scene timing."""
    path = node.evalParm("manifest").strip()
    if path and Path(path).is_file():
        manifest = json.loads(Path(path).read_text())
        frames = [item["frame"] for item in manifest["frames"]]
        node.parm("start").set(min(frames))
        node.parm("end").set(max(frames))


def bake_locally(node):
    """Bake Locally button: what a Geometry ROP on OUT_RENDER writes, with frames prepared in parallel."""
    if hou.isUIAvailable():
        with hou.InterruptableOperation("Baking point clouds", open_interrupt_dialog=True) as progress:
            progress.updateProgress(0.0)
            return bake(node, workers=LOCAL_BAKE_WORKERS, progress=progress)
    return bake(node, workers=LOCAL_BAKE_WORKERS)


def bake(node, workers=1, progress=None):
    """Local Bake: write the OUT_RENDER points (full resolution, no viewport checks) to Output File.

    One file per frame from Start to End; existing files are overwritten, like a Geometry ROP.
    On error or cancel the files written by this bake are removed. Returns the written paths.
    """
    start, end = node.evalParm("start"), node.evalParm("end")
    if end < start:
        raise ValueError("End frame must be at or after Start frame")
    paths = [Path(node.parm("output_file").evalAtFrame(frame)).expanduser() for frame in range(start, end + 1)]
    if not str(paths[0]) or str(paths[0]) == ".":
        raise ValueError("Choose an Output File, e.g. $HIP/image2point/points.$F4.bgeo.sc")
    if len(set(paths)) != len(paths):
        raise ValueError("Output File needs a frame number such as $F4, or every frame overwrites the same file")
    previous = hou.frame()
    written = []
    temporary = None
    try:
        # Parameters can be animated: read every frame's settings here, on the main thread.
        jobs = []
        for frame in range(start, end + 1):
            hou.setFrame(frame)
            jobs.append((frame, node.evalParm("manifest"), settings(node, True)))
        for path, ((frame, manifest_path, cfg), (manifest, values)) in zip(paths, computed(jobs, workers)):
            output = hou.Geometry()
            write_geometry(output, manifest, values, manifest_path)
            path.parent.mkdir(parents=True, exist_ok=True)
            temporary = path.with_name(uuid.uuid4().hex + "." + path.name)
            output.saveToFile(str(temporary))
            os.replace(temporary, path)
            written.append(path)
            if progress is not None:
                progress.updateProgress(len(written) / len(jobs))
    except BaseException:
        for path in written:
            path.unlink(missing_ok=True)
        if temporary is not None:
            temporary.unlink(missing_ok=True)
        raise
    finally:
        hou.setFrame(previous)
    return [str(path) for path in written]
