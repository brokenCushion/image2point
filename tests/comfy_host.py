"""Run the real ComfyUI node against stand-in ComfyUI host modules (folder_paths, comfy, comfy_api)."""

import importlib.util
import socket
import sys
import tempfile
import types
from contextlib import contextmanager, nullcontext
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch
from pxr import Usd, UsdGeom

REPOSITORY = Path(__file__).resolve().parents[1]
NODE = REPOSITORY / "integrations/comfyui"
CHECKPOINT = REPOSITORY / "cache/MoGe/checkpoints/moge3/model.pt"


def camera(path, width, height):
    stage = Usd.Stage.CreateNew(str(path))
    stage.SetFramesPerSecond(24)
    stage.SetTimeCodesPerSecond(24)
    stage.SetStartTimeCode(1001)
    stage.SetEndTimeCode(1002)
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.y)
    UsdGeom.SetStageMetersPerUnit(stage, 1)
    cam = UsdGeom.Camera.Define(stage, "/Camera")
    cam.CreateHorizontalApertureAttr(36)
    cam.CreateVerticalApertureAttr(36 * height / width)
    cam.CreateFocalLengthAttr(35)
    stage.GetRootLayer().Save()


def host_modules(root, checkpoint):
    folders = types.ModuleType("folder_paths")
    folders.models_dir = str(root)
    folders.folder_names_and_paths = {}
    folders.add_model_folder_path = lambda name, path: folders.folder_names_and_paths.setdefault(name, ([path], set()))
    folders.get_filename_list = lambda name: ["local.pt"]
    folders.get_full_path = lambda name, file: str(checkpoint) if Path(checkpoint).is_file() else None
    folders.get_folder_paths = lambda name: [str(root / "models" / name)]
    folders.get_output_directory = lambda: str(root)
    folders.get_temp_directory = lambda: str(root / "temp")
    comfy = types.ModuleType("comfy")
    comfy.__path__ = []
    management = types.ModuleType("comfy.model_management")
    management.throw_exception_if_processing_interrupted = lambda: None
    management.unload_all_models = lambda: None
    utils = types.ModuleType("comfy.utils")
    utils.ProgressBar = lambda count: types.SimpleNamespace(update=lambda n: None)
    comfy.model_management, comfy.utils = management, utils
    api = types.ModuleType("comfy_api.latest")
    api.Types = types.SimpleNamespace(
        File3D=lambda path, fmt: types.SimpleNamespace(get_source=lambda: path, format=fmt)
    )
    return {
        "folder_paths": folders,
        "comfy": comfy,
        "comfy.model_management": management,
        "comfy.utils": utils,
        "comfy_api": types.ModuleType("comfy_api"),
        "comfy_api.latest": api,
    }


def load_node():
    """Import integrations/comfyui under its installed name, as ComfyUI does."""
    spec = importlib.util.spec_from_file_location(
        "comfyui_image2point", NODE / "__init__.py", submodule_search_locations=[str(NODE)]
    )
    package = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = package
    spec.loader.exec_module(package)
    return sys.modules["comfyui_image2point.nodes"]


@contextmanager
def offline():
    """Block and record every network lookup or connection; yields the list of attempts."""
    attempts = []

    def refuse(*args, **kwargs):
        attempts.append(args)
        raise OSError("Network access attempted while offline")

    with patch.object(socket, "getaddrinfo", refuse), patch.object(socket.socket, "connect", refuse):
        yield attempts


def stand_in_moge(root):
    """CPU runs (and CI) have no real MoGe: a stand-in source tree and a tiny checkpoint with MoGe-3's layout."""
    stub = root / "moge-stub"
    (stub / "MoGe/moge").mkdir(parents=True)
    (stub / "MoGe/moge/__init__.py").write_text("")
    checkpoint = stub / "model.pt"
    torch.save({"model_config": {"refiner": {}}, "model": {}}, checkpoint)
    return checkpoint, stub / "MoGe"


@contextmanager
def comfy_node(root, checkpoint, source=None):
    """The node module loaded inside stand-in ComfyUI host modules; source overrides the MoGe source folder."""
    with patch.dict(sys.modules, host_modules(root, checkpoint)):
        node = load_node()
        if source is not None:
            real_bundled = node.bundled
            node.bundled = lambda relative: source if relative == "vendor/MoGe" else real_bundled(relative)
        yield node


W, H = 128, 96


def sample_args(root, model):
    """Two-frame 128x96 shot with two SAM-tracked actors; renders frame 1002 at half resolution."""
    camera(root / "camera.usda", W, H)
    plates = torch.full((2, H, W, 3), 0.4)
    plates[:, 25:75, 35:85, :] = 0.8
    masks = np.zeros((2, 2, H, W), np.uint8)
    masks[:, 0, 20:70, 20:50] = 1
    masks[:, 1, 30:80, 75:105] = 1
    tracks = dict(
        n_frames=2,
        orig_size=(H, W),
        packed_masks=torch.from_numpy(np.packbits(masks, axis=-1, bitorder="little")),
    )
    return dict(
        moge3=model,
        plates=plates,
        camera_usd=str(root / "camera.usda"),
        output_resolution="Medium - Half",
        start_frame=1002,
        end_frame=1002,
        input_first_frame=1001,
        inference_quality=0,
        refinement_steps=1,
        actor_tracks=tracks,
    )


def exercise(gpu=False):
    from image2point.point_package import camera_points, load_frame, process

    with tempfile.TemporaryDirectory() as temp:
        root = Path(temp)
        w, h = W, H
        checkpoint, source = (CHECKPOINT, None) if gpu else stand_in_moge(root)
        with comfy_node(root, checkpoint, source) as node:
            (model,) = node.LoadMoGe3().load("local.pt", "FP16")
            args = sample_args(root, model)

            class FakeEngine:
                calls = 0

                def __init__(self, *args):
                    pass

                def predict(self, rgb, k):
                    FakeEngine.calls += 1
                    return camera_points(np.full(rgb.shape[:2], 4), k), np.ones(rgb.shape[:2]), {}

            engine = nullcontext() if gpu else patch("image2point.moge3.MoGe3Engine", FakeEngine)
            with engine, offline() as attempts:
                files, manifest_path = generate_package(node, **args)
                assert attempts == [], "The nodes must never reach the network"
                assert len(files) == 1 and files[0].format == "ply"
                blob = Path(files[0].get_source()).read_bytes()
                assert b"property int object_id\n" in blob and b"Alpha" not in blob.split(b"end_header")[0]
                manifest, item, arrays = load_frame(manifest_path, 1002)
                assert manifest["up_axis"] == "Y" and manifest["fps"] == 24
                assert set(np.unique(arrays["object_id"])) == {0, 1, 2}
                assert arrays["P"].shape == (h // 2, w // 2, 3)
                assert np.isfinite(arrays["P"]).any()
                if not gpu:
                    np.testing.assert_allclose(arrays["P"][..., 2], -4)
                result = process(arrays, item, dict(remove_background=True, inset=0))
                assert set(np.unique(result["object_id"])) == {1, 2}

                _, again = generate_package(node, **args)
                assert again == manifest_path, "Identical inputs must reuse the published package"
                _, untagged = generate_package(node, **{**args, "actor_tracks": None})
                assert untagged != manifest_path
                packages = Path(manifest_path).parents[1]
                assert [p.name for p in packages.iterdir() if p.name.startswith(".staging")] == []
                assert len(list(packages.iterdir())) == 2
                if not gpu:
                    assert FakeEngine.calls == 1, "Mask changes must reuse depth"
            return manifest["grid_resolution"]


def generate_package(node, **args):
    """Run the node and return its PLY files and manifest path."""
    return node.GeneratePointCloud().generate(**args)
