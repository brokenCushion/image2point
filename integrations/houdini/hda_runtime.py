"""Embedded after point_package.py in the HDA PythonModule. No model dependencies."""

import threading
import uuid
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from itertools import islice

import hou

# Bake Locally prepares this many upcoming frames in background threads (about 2 GB each
# for a 4K frame). Measured on a 4K shot: 2 and 4 threads are equally fast (Python's lock
# and Houdini's main-thread geometry steps are the limit), 8 is slower. The farm job bakes
# one frame at a time.
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
    """measure_shot reads every frame once; keep its (scales, actor depths) per manifest."""
    path = Path(manifest_path)
    key = (str(path.resolve()), path.stat().st_mtime_ns)
    with SCALES_LOCK:
        if key not in SCALES:
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
    """Bake Locally button: same output as the farm bake, with frames prepared in parallel."""
    if hou.isUIAvailable():
        with hou.InterruptableOperation("Baking point clouds", open_interrupt_dialog=True) as progress:
            progress.updateProgress(0.0)
            return bake(node, workers=LOCAL_BAKE_WORKERS, progress=progress)
    return bake(node, workers=LOCAL_BAKE_WORKERS)


def bake(node, destination=None, workers=1, progress=None):
    root = Path(destination or node.evalParm("publish_directory")).expanduser()
    if not str(root) or str(root) == ".":
        raise ValueError("Choose a shared Publish Directory")
    root = root.resolve()
    start, end = node.evalParm("start"), node.evalParm("end")
    if end < start:
        raise ValueError("End frame must be at or after Start frame")
    filenames = ["points.%04d.bgeo.sc" % frame for frame in range(start, end + 1)]
    for filename in ["manifest.json", *filenames]:
        if (root / filename).exists():
            raise FileExistsError("Preserving existing cache: " + str(root / filename) + ". Choose another folder.")
    root.mkdir(parents=True, exist_ok=True)
    previous = hou.frame()
    entries = []
    temporary = None
    try:
        # Parameters can be animated: read every frame's settings here, on the main thread.
        jobs = []
        for frame in range(start, end + 1):
            hou.setFrame(frame)
            jobs.append((frame, node.evalParm("manifest"), settings(node, True)))
        for (frame, manifest_path, cfg), (manifest, values) in computed(jobs, workers):
            output = hou.Geometry()
            write_geometry(output, manifest, values, manifest_path)
            filename = "points.%04d.bgeo.sc" % frame
            temporary = root / (uuid.uuid4().hex + "." + filename)
            output.saveToFile(str(temporary))
            os.replace(temporary, root / filename)
            entries.append({"frame": frame, "file": filename, "settings": cfg})
            if progress is not None:
                progress.updateProgress(len(entries) / len(jobs))
        json_write(
            root / "manifest.json",
            {
                "kind": "image2point.render_cache",
                "color_space": "linear_srgb",
                "version": 1,
                "houdini_license": str(hou.licenseCategory()),
                "source": node.evalParm("manifest"),
                "fps": manifest["fps"],
                "up_axis": manifest["up_axis"],
                "meters_per_unit": 1.0,
                "frames": entries,
            },
        )
    except Exception:
        for entry in entries:
            (root / entry["file"]).unlink(missing_ok=True)
        if temporary is not None:
            temporary.unlink(missing_ok=True)
        raise
    finally:
        hou.setFrame(previous)
    return str(root / "manifest.json")


def prepare_farm(node):
    """Freeze this HDA (including animated controls) in a new HIP for studio submission."""
    if hou.licenseCategory() == hou.licenseCategoryType.Apprentice:
        raise ValueError(
            "Apprentice is for isolated local tests. Use Bake Locally; "
            "prepare studio farm jobs with a commercial license."
        )
    directory = node.evalParm("publish_directory").strip()
    if not directory:
        raise ValueError("Choose a shared Publish Directory first")
    root = Path(directory).resolve() / ("job_" + uuid.uuid4().hex)
    root.mkdir(parents=True, exist_ok=False)
    # Store the exact asset definition with the job; no installed HDA needed on workers.
    definition = node.type().definition()
    definition.save(str(root / "image2point.hda"))
    original = hou.hipFile.path()
    original_manifest = node.parm("manifest").unexpandedString()
    absolute_manifest = str(Path(node.evalParm("manifest")).resolve())
    suffix = Path(original).suffix if Path(original).suffix in (".hip", ".hiplc", ".hipnc") else ".hip"
    scene = root / ("bake" + suffix)
    try:
        node.parm("manifest").set(absolute_manifest)
        hou.hipFile.save(str(scene), save_to_recent_files=False)
    finally:
        hou.hipFile.setName(original)
        node.parm("manifest").set(original_manifest)
    json_write(
        root / "job.json",
        {
            "scene": scene.name,
            "asset": "image2point.hda",
            "node": node.path(),
            "publish_directory": str(Path(directory).resolve()),
        },
    )
    if hou.isUIAvailable():
        hou.ui.displayMessage(
            "Farm job saved:\n"
            + str(root / "job.json")
            + "\nSubmit with the supplied farm_bake.py using your studio Houdini runner."
            + "\nNo job has been submitted automatically."
        )
    return str(root / "job.json")
