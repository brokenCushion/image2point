"""Synthetic three-frame sample package and example ComfyUI workflows (no AI, no production data)."""

import sys
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from image2point.point_package import camera_points, json_write, tag_masks  # noqa: E402


def sample(root):
    """Two actors at 3 m and 6 m in front of a 15 m wall, moving one pixel per frame."""
    from pxr import Usd, UsdGeom

    root.mkdir(parents=True)
    w, h = 128, 96
    k = np.array([[128.0, 0, 64], [0, 128.0, 48], [0, 0, 1]])
    pose = np.diag([1.0, -1.0, -1.0, 1.0])
    manifest = dict(
        kind="image2point.point_package",
        version=1,
        up_axis="Y",
        meters_per_unit=1.0,
        fps=24,
        plate_resolution=[w, h],
        grid_resolution=[w, h],
        frames=[],
        model={"name": "synthetic, no AI"},
        mask_kind="synthetic roto",
    )
    for frame in range(1001, 1004):
        masks = np.zeros((2, h, w), np.float32)
        dx = frame - 1001
        masks[0, 20:80, 20 + dx : 50 + dx] = 1
        masks[1, 28:84, 75 - dx : 105 - dx] = 1
        depth = np.full((h, w), 15.0, np.float32)
        depth[masks[0] > 0] = 3.0
        depth[masks[1] > 0] = 6.0
        labels, alpha = tag_masks(masks, [1, 2], (h, w))
        rgb = np.full((h, w, 3), 64, np.uint8)
        rgb[masks[0] > 0] = [65, 190, 120]
        rgb[masks[1] > 0] = [210, 130, 70]
        filename = f"frame.{frame:04d}.npz"
        np.savez_compressed(
            root / filename,
            P=camera_points(depth, k) @ pose[:3, :3].T,
            rgb=rgb,
            masks=masks,
            original_masks=masks,
            actor_ids=np.array([1, 2], np.int32),
            object_id=labels,
            alpha=alpha,
            # Artist colour matte: actor 1 red, actor 2 green, a little soft.
            roto_grid=(np.stack([masks[0], masks[1], np.zeros_like(masks[0])], -1) * 0.9).astype(np.float16),
        )
        manifest["frames"].append(dict(frame=frame, file=filename, c2w=pose.tolist(), K=k.tolist(), K_plate=k.tolist()))
        if frame == 1001:
            Image.fromarray(rgb).save(root / "plate.png")
    json_write(root / "manifest.json", manifest)
    stage = Usd.Stage.CreateNew(str(root / "camera.usda"))
    stage.SetStartTimeCode(1001)
    stage.SetEndTimeCode(1003)
    stage.SetFramesPerSecond(24)
    stage.SetTimeCodesPerSecond(24)
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.y)
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)
    cam = UsdGeom.Camera.Define(stage, "/Camera")
    cam.CreateHorizontalApertureAttr(36.0)
    cam.CreateVerticalApertureAttr(27.0)
    cam.CreateFocalLengthAttr(36.0)
    stage.GetRootLayer().Save()
    return root / "manifest.json"


def workflow(root):
    """One-frame example in both ComfyUI formats.

    Placeholders to replace: CAMERA_USD_PATH (absolute path to sample/camera.usda) and
    MOGE3_CHECKPOINT (pick your MoGe-3 .pt file in the Load MoGe-3 Model node).
    """
    root.mkdir()
    loader_inputs = dict(checkpoint="MOGE3_CHECKPOINT", precision="FP16")
    generate_inputs = dict(
        moge3=["3", 0],
        plates=["1", 0],
        camera_usd="CAMERA_USD_PATH",
        output_resolution="Medium - Half",
        start_frame=1001,
        end_frame=1001,
        input_first_frame=1001,
        inference_quality=9,
        refinement_steps=3,
        up_axis="Y",
        camera_prim="",
        output_directory="",
    )
    api = {
        "1": {"class_type": "LoadImage", "inputs": {"image": "plate.png"}},
        "3": {"class_type": "Image2PointLoadMoGe3", "inputs": loader_inputs},
        "2": {"class_type": "Image2PointGeneratePointCloud", "inputs": generate_inputs},
        "4": {
            "class_type": "SavePointCloud",
            "inputs": dict(
                model_3d=["2", 0], filename_prefix="image2point/points", viewport_state={}, width=1024, height=1024
            ),
        },
    }
    json_write(root / "comfyui-api.json", api)

    # UI format: links are [link id, from node, from slot, to node, to slot, type].
    load_image = dict(
        id=1,
        type="LoadImage",
        pos=[80, 100],
        size=[300, 300],
        flags={},
        order=0,
        mode=0,
        inputs=[],
        outputs=[dict(name="IMAGE", type="IMAGE", links=[1], slot_index=0), dict(name="MASK", type="MASK", links=None)],
        properties={"Node name for S&R": "LoadImage"},
        widgets_values=["plate.png", "image"],
    )
    load_model = dict(
        id=3,
        type="Image2PointLoadMoGe3",
        pos=[80, 460],
        size=[300, 110],
        flags={},
        order=1,
        mode=0,
        inputs=[],
        outputs=[dict(name="moge3", type="I2P_MOGE3", links=[3], slot_index=0)],
        properties={"Node name for S&R": "Image2PointLoadMoGe3"},
        widgets_values=["MOGE3_CHECKPOINT", "FP16"],
    )
    generate = dict(
        id=2,
        type="Image2PointGeneratePointCloud",
        pos=[480, 100],
        size=[420, 500],
        flags={},
        order=2,
        mode=0,
        inputs=[
            dict(name="moge3", type="I2P_MOGE3", link=3),
            dict(name="plates", type="IMAGE", link=1),
            dict(name="actor_tracks", type="SAM3_TRACK_DATA", link=None),
            dict(name="roto", type="IMAGE", link=None),
        ],
        outputs=[
            dict(name="point_cloud", type="FILE_3D_PLY", links=[2], slot_index=0),
            dict(name="houdini_cache", type="STRING", links=None, slot_index=1),
        ],
        properties={"Node name for S&R": "Image2PointGeneratePointCloud"},
        widgets_values=["CAMERA_USD_PATH", "Medium - Half", 1001, 1001, 1001, 9, 3, "Y", "", ""],
    )
    save = dict(
        id=4,
        type="SavePointCloud",
        pos=[980, 100],
        size=[400, 500],
        flags={},
        order=3,
        mode=0,
        inputs=[
            dict(name="model_3d", type="FILE_3D_POINT_CLOUD_ANY,FILE_3D_PLY", link=2),
            dict(name="model_3d_info", type="LOAD_3D_MODEL_INFO", link=None),
            dict(name="camera_info", type="LOAD_3D_CAMERA", link=None),
        ],
        outputs=[],
        properties={"Node name for S&R": "SavePointCloud"},
        widgets_values=["image2point/points", {}, 1024, 1024],
    )
    json_write(
        root / "comfyui-workflow.json",
        dict(
            last_node_id=4,
            last_link_id=3,
            nodes=[load_image, load_model, generate, save],
            links=[[1, 1, 0, 2, 1, "IMAGE"], [2, 2, 0, 4, 0, "FILE_3D_PLY"], [3, 3, 0, 2, 0, "I2P_MOGE3"]],
            groups=[],
            config={},
            extra={},
            version=0.4,
        ),
    )


if __name__ == "__main__":
    print(sample(Path(sys.argv[1])))
