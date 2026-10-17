"""ComfyUI custom nodes: Load MoGe-3 Model and Generate Point Cloud.

Load MoGe-3 Model picks a local checkpoint file (nothing is ever downloaded). Generate
Point Cloud turns a batch of plate images, a tracked USD camera and an optional actor
mask into per-frame point clouds (PLY files) and a manifest.json that the Houdini HDA
reads. The heavy lifting is done by MoGe-3 (see src/image2point/moge3.py).
"""

import gc
import hashlib
import json
import os
import shutil
import uuid
from pathlib import Path

import folder_paths
import numpy as np

from image2point.point_package import atomic, json_write, measure_shot, srgb_to_linear, tag_masks

# Name of the model folder ComfyUI searches for checkpoints: models/image2point, plus any
# "image2point:" entries a studio adds to extra_model_paths.yaml (e.g. a server share).
MODEL_FOLDER = "image2point"
# Output resolution choices shown in ComfyUI, mapped to how much each side is shrunk
# (1 = full size, 2 = half size, and so on).
RESOLUTION_DIVISORS = {"Max - Full": 1, "Medium - Half": 2, "Low - Quarter": 4, "Draft - Eighth": 8}
# Register our model folder with ComfyUI at import time. MoGe ships PyTorch .pt checkpoints only.
folder_paths.add_model_folder_path(MODEL_FOLDER, str(Path(folder_paths.models_dir) / MODEL_FOLDER))
folder_paths.folder_names_and_paths[MODEL_FOLDER][1].update({".pt"})


def bundled(relative):
    """Release ZIPs vendor files beside this node; a repository checkout keeps them at the root or in src/."""
    here = Path(__file__).resolve().parent
    repository = here.parents[1]
    for base in (here, repository, repository / "src"):
        if (base / relative).exists():
            return base / relative
    return here / relative


def actor_masks(track_data, index, count, height, width):
    """Native SAM3_TRACK_DATA: bit-packed [frame, object, y, packed-x]."""
    import torch

    if track_data is None:
        return np.empty((0, height, width), np.float32), np.empty(0, np.int32)
    if track_data.get("n_frames") != count:
        raise ValueError("Tracking and plate batches must have identical frame counts/order")
    if tuple(track_data.get("orig_size", ())) != (height, width):
        raise ValueError("Tracking must come from these original plate dimensions")
    packed = track_data.get("packed_masks")
    if packed is None or packed.shape[1] == 0:
        return np.empty((0, height, width), np.float32), np.empty(0, np.int32)
    if packed.ndim != 4 or packed.shape[0] != count or packed.dtype != torch.uint8:
        raise ValueError("Unsupported SAM3 tracking layout; expected bit-packed uint8 masks")
    bits = np.unpackbits(packed[index].detach().cpu().numpy(), axis=-1, bitorder="little")
    values = torch.from_numpy(bits.astype(np.float32)).unsqueeze(1)
    masks = torch.nn.functional.interpolate(values, size=(height, width), mode="nearest")[:, 0].numpy()
    return masks, np.arange(1, len(masks) + 1, dtype=np.int32)


def moge_source():
    """Pinned MoGe source: vendored in release ZIPs, cache/MoGe in a repository checkout."""
    source = bundled("vendor/MoGe")
    if not source.exists():
        source = bundled("cache/MoGe")
    if not source.exists():
        raise ValueError("MoGe source missing; reinstall the node from the release ZIP")
    return source


class MoGe3Model:
    """What Load MoGe-3 Model hands to Generate Point Cloud: a checked checkpoint, not loaded weights.

    The weights are only loaded onto the GPU when Generate Point Cloud has frames that are
    not already cached, and freed again when it finishes.
    """

    def __init__(self, weights, source, use_fp16, identity):
        self.weights = weights
        self.source = source
        self.use_fp16 = use_fp16
        self.identity = identity


class LoadMoGe3:
    """Pick a local MoGe-3 checkpoint. Separate from generation so studios can swap how models are found."""

    CATEGORY = "Image2Point"
    FUNCTION = "load"
    RETURN_TYPES = ("I2P_MOGE3",)
    RETURN_NAMES = ("moge3",)

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "checkpoint": (
                    folder_paths.get_filename_list(MODEL_FOLDER),
                    {
                        "tooltip": (
                            "MoGe-3 .pt files in ComfyUI's models/image2point folder or any 'image2point' "
                            "path in extra_model_paths.yaml. Nothing is downloaded."
                        )
                    },
                ),
                "precision": (["FP16", "FP32"],),
            }
        }

    @classmethod
    def IS_CHANGED(cls, checkpoint, precision):
        """Re-check only when the file itself changes, so its fingerprint is not recomputed every run."""
        path = folder_paths.get_full_path(MODEL_FOLDER, checkpoint)
        if not path:
            return float("nan")
        stat = Path(path).stat()
        return f"{path}|{stat.st_size}|{stat.st_mtime_ns}|{precision}"

    def load(self, checkpoint, precision):
        import torch

        from image2point.io import digest

        weights = folder_paths.get_full_path(MODEL_FOLDER, checkpoint)
        if not weights:
            searched = ", ".join(folder_paths.get_folder_paths(MODEL_FOLDER))
            raise ValueError(f"Checkpoint '{checkpoint}' not found. Searched: {searched}")
        # Memory-mapped read: checks the file's structure without loading 1.5 GB of weights.
        try:
            contents = torch.load(weights, map_location="cpu", weights_only=True, mmap=True)
        except Exception as error:
            raise ValueError(f"'{checkpoint}' is not a readable PyTorch checkpoint: {error}") from error
        is_moge3 = isinstance(contents, dict) and {"model_config", "model"} <= contents.keys()
        if not is_moge3 or "refiner" not in contents["model_config"]:
            raise ValueError(f"'{checkpoint}' is not a MoGe-3 checkpoint (MoGe-1/2 or another model)")
        del contents
        source = moge_source()
        source_files = sorted(p for p in source.rglob("*") if p.suffix in (".py", ".yaml"))
        identity = {
            "model": "MoGe-3",
            "weights": digest(weights),
            "precision": precision,
            "engine": digest(bundled("image2point/moge3.py")),
            "source": hashlib.sha256("".join(digest(p) for p in source_files).encode()).hexdigest(),
        }
        return (MoGe3Model(weights, source, precision == "FP16", identity),)


def publish(staging, target, key, versioned):
    """Move a finished package from its temporary folder to `target` (or a version of it); return the folder.

    With versioned=True (a user-chosen output_directory) the folder is target_v001, then
    target_v002, target_v003...: an existing folder is reused only when it holds exactly
    the same results (same content key) and is never overwritten, so Houdini scenes and
    farm jobs pointing at a package never see it change. Automatic folders are named by
    the content key, so an existing one always holds the same results.
    """
    (staging / ".package_key").write_text(key)
    candidates = [target]
    if versioned:
        candidates = [target.with_name(f"{target.name}_v{v:03d}") for v in range(1, 1000)]
    for final in candidates:
        if (final / "manifest.json").is_file():
            existing = final / ".package_key"
            if not versioned or (existing.is_file() and existing.read_text() == key):
                shutil.rmtree(staging)
                return final
            continue
        try:
            os.replace(staging, final)
            return final
        except OSError:
            if not (final / "manifest.json").is_file():
                raise
    raise ValueError(f"No free version left for output directory '{target}'")


def file_names(stem):
    """Frame, PLY and manifest names: a package folder's layout, or flat files beside a custom path."""
    if stem is None:
        return "frame.{:04d}.npz", "points.{:04d}.ply", "manifest.json"
    return stem + ".{:04d}.npz", stem + ".{:04d}.ply", stem + ".manifest.json"


def publish_flat(staging, folder, manifest_name, key):
    """Move finished flat files beside a custom output path, the manifest last; return the folder.

    The caller owns names and versions, so an existing manifest is reused only when it holds
    exactly these results and is otherwise never overwritten. Frame files without a manifest
    are leftovers of an interrupted run and are replaced. If a move fails, the files already
    moved are removed again, so a manifest on disk always means a complete package.
    """
    manifest = folder / manifest_name
    if manifest.is_file():
        if json.loads(manifest.read_text()).get("package_key") == key:
            return folder
        raise ValueError(f"{manifest} already holds different results; connect a new output path")
    folder.mkdir(parents=True, exist_ok=True)
    moved = []
    try:
        for name in sorted(path.name for path in staging.iterdir() if path.name != manifest_name):
            os.replace(staging / name, folder / name)
            moved.append(folder / name)
        os.replace(staging / manifest_name, manifest)
    except BaseException:
        for path in moved:
            path.unlink(missing_ok=True)
        raise
    return folder


class GeneratePointCloud:
    """The ComfyUI node itself. ComfyUI reads the class attributes below to build the node."""

    CATEGORY = "Image2Point"  # Folder in the ComfyUI add-node menu.
    FUNCTION = "generate"  # Name of the method ComfyUI calls when the node runs.
    RETURN_TYPES = ("FILE_3D_PLY", "STRING")  # Point files and the manifest path.
    RETURN_NAMES = ("point_cloud", "houdini_cache")
    OUTPUT_IS_LIST = (True, False)  # One PLY per frame, plus one manifest path.
    OUTPUT_NODE = True  # Run this node even when nothing is connected downstream.

    @classmethod
    def INPUT_TYPES(cls):
        """Describe the widgets and sockets ComfyUI shows on the node.

        "required" inputs must be filled in. "optional" inputs can be left unconnected.
        """
        return {
            "required": {
                "moge3": ("I2P_MOGE3", {"tooltip": "Connect a Load MoGe-3 Model node"}),
                "plates": ("IMAGE",),
                "camera_usd": ("STRING", {"default": ""}),
                "output_resolution": (list(RESOLUTION_DIVISORS), {"default": "Medium - Half"}),
                "start_frame": ("INT", {"default": 1001}),
                "end_frame": ("INT", {"default": 1024}),
                "input_first_frame": (
                    "INT",
                    {
                        "default": 1001,
                        "tooltip": (
                            "Frame number of the FIRST image in the input batch, even when rendering a shorter range"
                        ),
                    },
                ),
                "inference_quality": ("INT", {"default": 9, "min": 0, "max": 9}),
                "refinement_steps": (
                    "INT",
                    {"default": 3, "min": 0, "max": 16, "tooltip": "More is not always better"},
                ),
                "up_axis": (["Y", "Z"], {"default": "Y"}),
                "camera_prim": (
                    "STRING",
                    {"default": "", "tooltip": "Leave blank when USD contains exactly one camera"},
                ),
                "output_directory": (
                    "STRING",
                    {
                        "default": "",
                        "tooltip": "Full path of the package folder, e.g. /shots/BA001/pole (a plain name goes in "
                        "ComfyUI's output folder). Saved as _v001; different results never overwrite: "
                        "they go to _v002, _v003... "
                        "Blank uses an automatic folder in ComfyUI output/image2point.",
                    },
                ),
            },
            "optional": {
                "output_path": (
                    "STRING",
                    {
                        "forceInput": True,
                        "tooltip": "Connect a custom output path, without a file extension. When connected, "
                        "Image2Point does not name or version the output, and Output Directory is ignored. "
                        "Files: <path>.####.npz, <path>.####.ply, <path>.manifest.json.",
                    },
                ),
                "actor_tracks": ("SAM3_TRACK_DATA",),
                "roto": (
                    "IMAGE",
                    {
                        "tooltip": "RGB roto, stored on every point. Without actor_tracks its channels are the "
                        "actors: red = 1, green = 2, blue = 3 (a greyscale roto is actor 1)."
                    },
                ),
            },
        }

    @classmethod
    def IS_CHANGED(cls, **kwargs):
        """Tell ComfyUI this node always needs to run again.

        ComfyUI skips a node when its inputs have not changed. Returning NaN (which never
        equals itself) forces a re-run, so changes to the camera file on disk are noticed.
        Re-running is cheap because depth and packages are cached by content.
        """
        # Re-enter to validate external camera changes; raw inference and packages are content-cached.
        return float("nan")

    def generate(
        self,
        moge3,
        plates,
        camera_usd,
        output_resolution,
        start_frame,
        end_frame,
        input_first_frame=1001,
        inference_quality=9,
        refinement_steps=3,
        up_axis="Y",
        camera_prim="",
        output_directory="",
        actor_tracks=None,
        roto=None,
        output_path=None,
    ):
        """Run MoGe-3 on each requested frame and write the point-cloud package.

        Steps:
          1. Check the inputs (shapes, frame range, settings).
          2. Read the USD camera for frame rate and per-frame camera poses.
          3. Build the fingerprint ("identity") of this run from the loaded model and settings.
          4. For each frame: get depth (from cache or MoGe-3), apply the actor mask,
             move the points into world space, and save a .npz and a .ply.
          5. Measure depth stabilisation over all frames, write the manifest, then move the
             finished folder into place (or the flat files beside a connected output_path).
          6. Clean up temporary files and GPU memory.

        Returns the list of PLY files (one per frame) and the path to manifest.json.
        """
        import comfy.model_management
        import comfy.utils
        import torch
        from comfy_api.latest import Types
        from pxr import Usd

        from image2point.camera import camera_frames
        from image2point.io import digest, write_ply

        if plates.ndim != 4 or plates.shape[-1] != 3:
            raise ValueError("Plates must be an RGB IMAGE batch [frames,height,width,3]")
        n, h, w, _ = plates.shape
        if roto is not None:
            if roto.ndim != 4 or tuple(roto.shape[:3]) != (n, h, w):
                raise ValueError("Roto must be an IMAGE batch the same size and length as the plates")
        # Without SAM tracks the roto's colour channels are the actors: red 1, green 2, blue 3.
        # A greyscale roto (channels equal, allowing for video compression) is one actor. A channel
        # that never reaches 0.5 in the shot is compression noise, not an actor.
        roto_channels = []
        if roto is not None and actor_tracks is None:
            greyscale = all((roto[..., 0] - roto[..., c]).abs().max() <= 0.05 for c in (1, 2))
            channels = [0] if greyscale else [0, 1, 2]
            roto_channels = [c for c in channels if roto[..., c].max() > 0.5]
        if not input_first_frame <= start_frame <= end_frame < input_first_frame + n:
            raise ValueError("Requested frame range must be inside the input plate batch")
        if not 0 <= inference_quality <= 9 or not 0 <= refinement_steps <= 16:
            raise ValueError("Invalid MoGe-3 quality/refinement setting")
        if up_axis not in ("Y", "Z"):
            raise ValueError("Unsupported up axis")

        camera_path = Path(camera_usd).resolve()
        stage = Usd.Stage.Open(str(camera_path))
        if stage is None or not stage.HasAuthoredMetadata("framesPerSecond"):
            raise ValueError("Camera USD must contain explicit framesPerSecond metadata")
        fps = stage.GetFramesPerSecond()
        # VFX frame numbers refer to USD time codes; reject implicit rate conversions here.
        if abs(stage.GetTimeCodesPerSecond() - fps) > 1e-6:
            raise ValueError(
                "USD timeCodesPerSecond must match FPS for frame-number mapping; normalize the camera export"
            )
        divisor = RESOLUTION_DIVISORS[output_resolution]
        iw, ih = max(1, round(w / divisor)), max(1, round(h / divisor))
        job = {
            "camera": {"path": str(camera_path), "time_code_start": start_frame},
            "plate": {"resolution": [w, h], "calibration_resolution": [w, h], "inference_resolution": [iw, ih]},
            "frames": {"start": start_frame, "end": end_frame, "fps": fps},
        }
        if camera_prim:
            job["camera"]["prim_path"] = camera_prim
        report = camera_frames(job, Path("."))

        stem = None
        if output_path is not None:
            # Connected: the caller owns naming and versions; never fall back to a default place.
            if not output_path.strip():
                raise ValueError("output_path is connected but empty; connect a path or disconnect it")
            flat = Path(output_path.strip()).expanduser()
            if not flat.is_absolute():
                flat = Path(folder_paths.get_output_directory()) / flat
            flat = flat.resolve()
            target, parent, stem = None, flat.parent, flat.name
        elif output_directory.strip():
            target = Path(output_directory.strip()).expanduser()
            if not target.is_absolute():
                target = Path(folder_paths.get_output_directory()) / target
            target = target.resolve()
            parent = target.parent
        else:
            target, parent = None, Path(folder_paths.get_output_directory()) / "image2point"
        frame_name, ply_name, manifest_name = file_names(stem)
        # MoGe depth cache: only this node reads it (re-runs skip inference). ComfyUI empties its
        # temp folder at startup and shutdown, so the cache lasts one ComfyUI session.
        depth_cache = Path(folder_paths.get_temp_directory()) / "image2point-depth"
        if not isinstance(moge3, MoGe3Model):
            raise ValueError("Connect the moge3 input to a Load MoGe-3 Model node")
        identity = {
            **moge3.identity,
            "quality": inference_quality,
            "steps": refinement_steps,
            "resolution": [iw, ih],
            "adapter": digest(__file__),
        }
        manifest = {
            "kind": "image2point.point_package",
            "version": 1,
            "up_axis": up_axis,
            "meters_per_unit": 1.0,
            "fps": fps,
            "plate_resolution": [w, h],
            "grid_resolution": [iw, ih],
            "pixel_convention": "image-edge origin; centers x+0.5,y+0.5",
            "model": identity,
            "roto": roto is not None,
            "color_space": "linear_srgb",
            "mask_kind": "binary SAM segmentation"
            if actor_tracks is not None
            else ("supplied roto" if roto is not None else "none"),
            "actor_id_mapping": "SAM track slot + 1; background = 0"
            if actor_tracks is not None
            else "roto channel: red = 1, green = 2, blue = 3 (greyscale roto = 1); background = 0",
            "frames": [],
        }
        axis = np.eye(4)
        if up_axis == "Y":
            axis[:3, :3] = [[1, 0, 0], [0, 0, 1], [0, -1, 0]]

        staging = parent / (".staging-" + uuid.uuid4().hex)
        package_key = hashlib.sha256()
        ply_names = []
        engine = None
        progress = comfy.utils.ProgressBar(end_frame - start_frame + 1)
        try:
            for frame in report["frames"]:
                comfy.model_management.throw_exception_if_processing_interrupted()
                index = frame["frame"] - input_first_frame
                image = plates[index].detach().cpu()
                if not torch.isfinite(image).all() or image.min() < 0 or image.max() > 1:
                    raise ValueError(
                        "Supply display-ready sRGB plates in [0,1]; convert linear/log in the input workflow"
                    )
                rgb_tensor = torch.nn.functional.interpolate(
                    image.permute(2, 0, 1).unsqueeze(0),
                    size=(ih, iw),
                    mode="bicubic",
                    align_corners=False,
                    antialias=True,
                )
                # Convert the resized image back to 0-255 colour values for storage.
                srgb = rgb_tensor[0].permute(1, 2, 0).clamp(0, 1).numpy()
                rgb = (srgb * 255).astype(np.uint8)
                linear_rgb = srgb_to_linear(srgb)
                # K is the camera's intrinsics matrix: focal length and image centre at the grid size.
                k = np.asarray(frame["K_inference"])
                raw_key = hashlib.sha256(
                    rgb.tobytes() + json.dumps({**identity, "K": k.tolist()}, sort_keys=True).encode()
                ).hexdigest()
                raw_file = depth_cache / f"{raw_key}.npz"
                # Cache hit: reuse the depth saved by an earlier run with the same image and settings.
                # Cache miss: run MoGe-3 (slow, GPU) and save the result for next time.
                if raw_file.exists():
                    with np.load(raw_file, allow_pickle=False) as data:
                        local = data["points"]
                else:
                    if engine is None:
                        comfy.model_management.unload_all_models()
                        from image2point.moge3 import MoGe3Engine

                        engine = MoGe3Engine(
                            moge3.weights, moge3.source, inference_quality, refinement_steps, moge3.use_fp16
                        )
                    local, valid, _ = engine.predict(rgb, k)
                    local = np.where((valid > 0.5)[..., None], local, np.nan)
                    atomic(raw_file, lambda f: np.savez_compressed(f, points=local))
                if local.shape != (ih, iw, 3):
                    raise ValueError("Raw cache shape mismatch")

                # SAM tracks provide actor IDs and the masks that drive edge repair; a roto connected
                # alongside them only provides the RGB roto attribute.
                masks, ids = actor_masks(actor_tracks, index, n, h, w)
                roto_frame = None
                if roto is not None:
                    roto_frame = roto[index, ..., :3].detach().cpu().numpy().astype(np.float32)
                    if actor_tracks is None:
                        masks = np.ascontiguousarray(roto_frame[..., roto_channels].transpose(2, 0, 1))
                        ids = np.array([c + 1 for c in roto_channels], np.int32)
                if len(ids):
                    grid_masks = torch.nn.functional.interpolate(
                        torch.from_numpy(masks).unsqueeze(1), size=(ih, iw), mode="bilinear", align_corners=False
                    )[:, 0].numpy()
                else:
                    grid_masks = np.empty((0, ih, iw), np.float32)
                labels, alpha = tag_masks(grid_masks, ids, (ih, iw))
                # Move every point from camera space into world space using the camera pose
                # (rotation in the top-left 3x3 block, position in the last column).
                pose = axis @ np.asarray(frame["c2w"])
                world = (local @ pose[:3, :3].T + pose[:3, 3]).astype(np.float32)
                record = {
                    "frame": frame["frame"],
                    "file": frame_name.format(frame["frame"]),
                    "c2w": pose.tolist(),
                    "K": k.tolist(),
                    "K_plate": frame["K_plate"],
                    "raw_cache_key": raw_key,
                }
                # Add this frame to the running fingerprint of the whole package.
                extra = {}
                if roto_frame is not None:
                    roto_grid = (
                        torch.nn.functional.interpolate(
                            torch.from_numpy(roto_frame).permute(2, 0, 1)[None],
                            size=(ih, iw),
                            mode="bilinear",
                            align_corners=False,
                        )[0]
                        .permute(1, 2, 0)
                        .numpy()
                    )
                    extra["roto_grid"] = roto_grid.astype(np.float16)
                    package_key.update(roto_frame.tobytes())
                package_key.update(json.dumps(record, sort_keys=True).encode() + masks.tobytes() + ids.tobytes())
                package_key.update(linear_rgb.tobytes())
                # Save the full point grid and masks for Houdini edits (.npz is numpy's binary format).
                atomic(
                    staging / record["file"],
                    lambda f: np.savez_compressed(
                        f,
                        P=world,
                        rgb=rgb,
                        linear_rgb=linear_rgb,
                        object_id=labels,
                        alpha=alpha,
                        masks=grid_masks.astype(np.float16),
                        original_masks=masks.astype(np.float16),
                        actor_ids=ids,
                        **extra,
                    ),
                )
                manifest["frames"].append(record)

                # Keep only points with a real depth value; invalid ones are skipped in the PLY.
                valid = np.isfinite(world).all(-1) & (local[..., 2] > 0)
                ply_names.append(ply_name.format(frame["frame"]))
                attributes = {
                    "object_id": labels[valid],
                    "source_pixel": np.arange(ih * iw, dtype=np.int32).reshape(ih, iw)[valid],
                }
                # Standard PLY RGB stays display-encoded for native previews; float colour is explicit.
                for channel, suffix in enumerate("rgb"):
                    attributes["linear_" + suffix] = linear_rgb[..., channel][valid]
                if extra:
                    # The artist roto colour at each point's pixel (all points), for shaders.
                    roto_points = extra["roto_grid"].astype(np.float32)[valid]
                    for channel, suffix in enumerate("rgb"):
                        attributes["roto_" + suffix] = roto_points[:, channel]
                write_ply(
                    staging / ply_names[-1],
                    world[valid],
                    rgb[valid],
                    np.ones(valid.sum(), np.float32),
                    coordinate_system=f"{up_axis}-up meters; frame {frame['frame']}",
                    attributes=attributes,
                )
                progress.update(1)
            # Depth stabilisation is measured once here, over every frame, so Houdini (and each
            # farm task) reads it from the manifest instead of re-reading the whole shot.
            json_write(staging / manifest_name, manifest)
            scales, actor_depths = measure_shot(staging / manifest_name)
            manifest["stabilisation"] = {
                "smoothing": 12.0,
                "neighbours": 3,
                "scales": {str(f): s for f, s in scales.items()},
                "actor_depths": {str(a): {str(f): d for f, d in v.items()} for a, v in actor_depths.items()},
            }
            json_write(staging / manifest_name, manifest)
            package_key.update(
                json.dumps({key: value for key, value in manifest.items() if key != "frames"}, sort_keys=True).encode()
            )
            # The package folder is named by a short hash of everything inside it.
            key = package_key.hexdigest()
            if stem is None:
                publication = publish(staging, target or parent / key[:16], key, versioned=target is not None)
            else:
                manifest["package_key"] = key
                json_write(staging / manifest_name, manifest)
                publication = publish_flat(staging, parent, manifest_name, key)
        finally:
            # Runs even if an error happened: remove leftovers and free GPU memory.
            shutil.rmtree(staging, ignore_errors=True)
            del engine
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        # Wrap each PLY path so ComfyUI can show it in its 3D preview and save nodes.
        point_files = [Types.File3D(str(publication / name), "ply") for name in ply_names]
        return (point_files, str(publication / manifest_name))
