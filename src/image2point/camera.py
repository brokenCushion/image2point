"""Resolve a tracked USD camera to per-frame OpenCV intrinsics and Z-up metre poses."""

import math
from pathlib import Path

import numpy as np
from pxr import Usd, UsdGeom


def require(condition, message):
    if not condition:
        raise ValueError(message)


def positive(value, name):
    require(type(value) in (int, float) and math.isfinite(value) and value > 0, f"{name} must be positive and finite")
    return value


def camera_frames(job, root):
    """Per-frame K and camera-to-world poses.

    job holds camera {path, prim_path?, time_code_start?}, plate resolutions/crop and frames {start, end, fps, step?}.
    """
    cfg, plate, frames = job["camera"], job["plate"], job["frames"]
    camera_path = (Path(root) / cfg["path"]).resolve()
    require(camera_path.is_file(), f"USD camera file missing: {camera_path}")
    stage = Usd.Stage.Open(str(camera_path))
    require(stage is not None, f"Cannot open USD stage: {camera_path}")
    require(not stage.GetCompositionErrors(), f"USD composition errors: {stage.GetCompositionErrors()}")
    cameras = [p for p in stage.Traverse() if p.IsA(UsdGeom.Camera)]
    if cfg.get("prim_path"):
        prim = stage.GetPrimAtPath(cfg["prim_path"])
        require(bool(prim) and prim.IsA(UsdGeom.Camera), "camera.prim_path does not identify a USD Camera")
    else:
        require(len(cameras) == 1, f"Select camera.prim_path explicitly; found {[str(p.GetPath()) for p in cameras]}")
        prim = cameras[0]
    require(
        stage.HasAuthoredMetadata("metersPerUnit") and stage.HasAuthoredMetadata("upAxis"),
        "USD must explicitly author metersPerUnit and upAxis",
    )
    units = positive(UsdGeom.GetStageMetersPerUnit(stage), "USD metersPerUnit")
    up = str(UsdGeom.GetStageUpAxis(stage))
    require(up in ("Y", "Z"), "USD upAxis must be Y or Z")
    world = np.eye(4)
    if up == "Y":
        world[:3, :3] = [[1, 0, 0], [0, 0, -1], [0, 1, 0]]
    tcps = positive(stage.GetTimeCodesPerSecond(), "USD timeCodesPerSecond")
    start = cfg.get("time_code_start", frames["start"])
    require(type(start) in (int, float) and math.isfinite(start), "time_code_start must be finite")
    cam = UsdGeom.Camera(prim)
    cw, ch = plate["calibration_resolution"]
    x, y, w, h = plate.get("crop", [0, 0, cw, ch])
    pw, ph = plate["resolution"]
    iw, ih = plate["inference_resolution"]
    calibration_to_plate = np.array([[pw / w, 0, -x * pw / w], [0, ph / h, -y * ph / h], [0, 0, 1]])
    # Pixel coordinates use image edges as origin: pixel centers are (x+.5,y+.5).
    plate_to_inference = np.diag([iw / pw, ih / ph, 1.0])
    result = []
    for frame in range(frames["start"], frames["end"] + 1, frames.get("step", 1)):
        time = start + (frame - frames["start"]) * tcps / frames["fps"]
        if stage.HasAuthoredTimeCodeRange():
            require(
                stage.GetStartTimeCode() <= time <= stage.GetEndTimeCode(),
                f"Frame {frame} maps outside USD time range: {time}",
            )
        require(cam.GetProjectionAttr().Get(time) == "perspective", "Only perspective cameras are supported")
        a = positive(cam.GetHorizontalApertureAttr().Get(time), "horizontal aperture")
        b = positive(cam.GetVerticalApertureAttr().Get(time), "vertical aperture")
        f = positive(cam.GetFocalLengthAttr().Get(time), "focal length")
        require(
            math.isclose(a / b, cw / ch, rel_tol=0.001),
            f"USD aperture aspect differs from calibration resolution at frame {frame}",
        )
        ox, oy = cam.GetHorizontalApertureOffsetAttr().Get(time), cam.GetVerticalApertureOffsetAttr().Get(time)
        require(math.isfinite(ox) and math.isfinite(oy), "Camera aperture offsets must be finite")
        k = np.array([[f * cw / a, 0, cw * (0.5 - ox / a)], [0, f * ch / b, ch * (0.5 + oy / b)], [0, 0, 1]])
        matrix = np.array(UsdGeom.XformCache(time).GetLocalToWorldTransform(prim)).T
        require(np.isfinite(matrix).all() and np.allclose(matrix[3], [0, 0, 0, 1]), "Invalid camera transform")
        r = matrix[:3, :3]
        require(
            np.allclose(r.T @ r, np.eye(3), atol=1e-5) and np.isclose(np.linalg.det(r), 1, atol=1e-5),
            "Camera transform contains scale, shear or reflection; bake to a rigid camera first",
        )
        matrix[:3, 3] *= units
        blender_c2w = world @ matrix
        cv_c2w = blender_c2w @ np.diag([1, -1, -1, 1])
        result.append(
            {
                "frame": frame,
                "time_code": time,
                "c2w": cv_c2w.tolist(),
                "blender_c2w": blender_c2w.tolist(),
                "K_plate": (calibration_to_plate @ k).tolist(),
                "K_inference": (plate_to_inference @ calibration_to_plate @ k).tolist(),
            }
        )
    return {
        "prim_path": str(prim.GetPath()),
        "meters_per_unit": units,
        "source_up_axis": up,
        "time_codes_per_second": tcps,
        "world_convention": "Z-up meters",
        "camera_convention": "OpenCV X-right Y-down Z-forward",
        "pixel_convention": "image-edge origin; pixel centers at x+0.5,y+0.5",
        "plate_to_inference": plate_to_inference.tolist(),
        "frames": result,
    }
