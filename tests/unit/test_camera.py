import copy
import tempfile
import unittest
from pathlib import Path

import numpy as np
from pxr import Gf, Usd, UsdGeom

from image2point.camera import camera_frames


def tracked_camera(path):
    """Y-up centimetre rig with a parent offset, lens shift and per-frame translation."""
    stage = Usd.Stage.CreateNew(str(path))
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.y)
    UsdGeom.SetStageMetersPerUnit(stage, 0.01)
    stage.SetTimeCodesPerSecond(24)
    stage.SetStartTimeCode(1001)
    stage.SetEndTimeCode(1003)
    rig = UsdGeom.Xform.Define(stage, "/Rig")
    rig.AddTranslateOp().Set(Gf.Vec3d(100, 200, 300))
    camera = UsdGeom.Camera.Define(stage, "/Rig/Camera")
    camera.CreateHorizontalApertureAttr(36)
    camera.CreateVerticalApertureAttr(24)
    camera.CreateHorizontalApertureOffsetAttr(1.2)
    camera.CreateVerticalApertureOffsetAttr(-0.8)
    camera.CreateFocalLengthAttr(50)
    translate = camera.AddTranslateOp()
    for frame in range(1001, 1004):
        translate.Set(Gf.Vec3d((frame - 1001) * 10, 0, 0), frame)
    stage.GetRootLayer().Save()


class CameraTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        tracked_camera(self.root / "camera.usda")
        self.job = {
            "camera": {"path": "camera.usda", "prim_path": "/Rig/Camera"},
            "plate": {
                "resolution": [420, 280],
                "calibration_resolution": [420, 280],
                "inference_resolution": [210, 140],
            },
            "frames": {"start": 1001, "end": 1003, "fps": 24},
        }
        self.report = camera_frames(self.job, self.root)

    def tearDown(self):
        self.tmp.cleanup()

    def test_usd_frustum_agrees_with_intrinsics_and_parent_transform(self):
        stage = Usd.Stage.Open(str(self.root / "camera.usda"))
        camera = UsdGeom.Camera(stage.GetPrimAtPath("/Rig/Camera"))
        for item in self.report["frames"]:
            projection = np.array(camera.GetCamera(item["time_code"]).frustum.ComputeProjectionMatrix())
            k = np.array(item["K_plate"])
            for p in ([0, 0, 5], [0.2, 0.3, 4], [-0.7, -0.2, 3]):
                p = np.array(p)
                ndc = np.array([p[0], -p[1], -p[2], 1]) @ projection
                ndc /= ndc[3]
                usd_uv = np.array([(ndc[0] + 1) * 420 / 2, (1 - ndc[1]) * 280 / 2])
                uv = k @ p
                np.testing.assert_allclose(uv[:2] / uv[2], usd_uv, atol=1e-5)
            # Parent translation is cm -> metres, and Y-up rotates to Z-up.
            np.testing.assert_allclose(np.array(item["c2w"])[:3, 3], [1 + 0.1 * (item["frame"] - 1001), -3, 2])
            self.assertAlmostEqual(np.linalg.det(np.array(item["c2w"])[:3, :3]), 1)

    def test_resize_intrinsics_and_explicit_crop(self):
        job = copy.deepcopy(self.job)
        job["plate"]["calibration_resolution"] = [840, 560]
        job["plate"]["crop"] = [100, 50, 630, 420]
        k = np.array(camera_frames(job, self.root)["frames"][0]["K_plate"])
        reference = np.array(self.report["frames"][0]["K_plate"])
        transform = np.array([[2 / 3, 0, -100 * 2 / 3], [0, 2 / 3, -50 * 2 / 3], [0, 0, 1]])
        np.testing.assert_allclose(k, transform @ np.diag([2, 2, 1]) @ reference)
        np.testing.assert_allclose(
            camera_frames(job, self.root)["frames"][0]["K_inference"], np.diag([0.5, 0.5, 1]) @ k
        )

    def test_reflection_rejected(self):
        stage = Usd.Stage.Open(str(self.root / "camera.usda"))
        UsdGeom.Xformable(stage.GetPrimAtPath("/Rig")).AddScaleOp().Set(Gf.Vec3f(-1, 1, 1))
        stage.GetRootLayer().Save()
        with self.assertRaisesRegex(ValueError, "reflection"):
            camera_frames(self.job, self.root)

    def test_time_mapping(self):
        self.job["frames"]["fps"] = 48
        report = camera_frames(self.job, self.root)
        self.assertEqual([f["time_code"] for f in report["frames"]], [1001, 1001.5, 1002])


if __name__ == "__main__":
    unittest.main()
