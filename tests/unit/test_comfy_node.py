"""ComfyUI nodes on CPU: model loading checks, frame mapping, actor tagging, package reuse (MoGe-3 stubbed)."""

import os

import pytest
import torch
from comfy_host import comfy_node, exercise, generate_package, offline, sample_args, stand_in_moge


def test_generate_node_exposes_ply_and_manifest_path_outputs(tmp_path):
    checkpoint, source = stand_in_moge(tmp_path)
    with comfy_node(tmp_path, checkpoint, source) as node:
        assert node.GeneratePointCloud.RETURN_TYPES == ("FILE_3D_PLY", "STRING")
        assert node.GeneratePointCloud.RETURN_NAMES == ("point_cloud", "houdini_cache")


def test_frame_mapping_tagging_and_reuse():
    assert exercise() == [64, 48]


def test_loader_reports_missing_checkpoint_and_searched_folders(tmp_path):
    with comfy_node(tmp_path, tmp_path / "absent.pt") as node:
        with pytest.raises(ValueError, match="not found. Searched: .*image2point"):
            node.LoadMoGe3().load("local.pt", "FP16")


@pytest.mark.parametrize(
    "contents, message",
    [
        (b"not a torch file", "not a readable PyTorch checkpoint"),
        ({"model_config": {"encoder": {}}, "model": {}}, "not a MoGe-3 checkpoint"),
        ({"state_dict": {}}, "not a MoGe-3 checkpoint"),
    ],
)
def test_loader_rejects_files_that_are_not_moge3(tmp_path, contents, message):
    _, source = stand_in_moge(tmp_path)
    checkpoint = tmp_path / "other.pt"
    if isinstance(contents, bytes):
        checkpoint.write_bytes(contents)
    else:
        torch.save(contents, checkpoint)
    with comfy_node(tmp_path, checkpoint, source) as node:
        with pytest.raises(ValueError, match=message):
            node.LoadMoGe3().load("local.pt", "FP16")


class CodeOnLoad:
    """Pickles to a call that creates a folder: stands in for code hidden in a hostile checkpoint."""

    def __init__(self, marker):
        self.marker = marker

    def __reduce__(self):
        return os.makedirs, (str(self.marker),)


def test_loader_never_runs_code_inside_a_checkpoint(tmp_path):
    marker = tmp_path / "code-ran"
    _, source = stand_in_moge(tmp_path)
    checkpoint = tmp_path / "hostile.pt"
    torch.save({"model_config": {"refiner": {}}, "model": CodeOnLoad(marker)}, checkpoint)
    with comfy_node(tmp_path, checkpoint, source) as node:
        with pytest.raises(ValueError, match="not a readable PyTorch checkpoint"):
            node.LoadMoGe3().load("local.pt", "FP16")
    assert not marker.exists()


def test_offline_guard_catches_network_access():
    import urllib.request

    with offline() as attempts:
        with pytest.raises(OSError):
            urllib.request.urlopen("https://huggingface.co", timeout=5)
    assert attempts


def test_sam_ids_with_roto_alpha(tmp_path):
    """SAM tracks give actor IDs and edge masks; the RGB roto is its own colour attribute."""
    from pathlib import Path
    from unittest.mock import patch

    import numpy as np

    from image2point.point_package import camera_points, load_frame, process

    class FakeEngine:
        def __init__(self, *args):
            pass

        def predict(self, rgb, k):
            assert (rgb == 127).all(), "MoGe must still receive display-encoded sRGB"
            return camera_points(np.full(rgb.shape[:2], 4), k), np.ones(rgb.shape[:2]), {}

    checkpoint, source = stand_in_moge(tmp_path)
    with comfy_node(tmp_path, checkpoint, source) as node, patch("image2point.moge3.MoGe3Engine", FakeEngine):
        (model,) = node.LoadMoGe3().load("local.pt", "FP16")
        args = sample_args(tmp_path, model)
        args["plates"] = torch.full_like(args["plates"], 0.5)
        roto = torch.zeros(2, 96, 128, 3)  # An RGB colour matte, as Load Image / Load Video give it.
        roto[:, 22:68, 22:48, 0] = 0.75  # Soft red matte, a little tighter than SAM actor 1.
        roto[:, 30:80, 80:100, 1] = 1  # Green matte over actor 2.
        _, manifest_path = generate_package(node, **args, roto=roto)
        manifest, item, arrays = load_frame(manifest_path, 1002)
        assert manifest["roto"] is True
        assert manifest["color_space"] == "linear_srgb"
        np.testing.assert_allclose(arrays["linear_rgb"], 0.21404114, atol=1e-6)
        assert set(np.unique(arrays["object_id"])) == {0, 1, 2}  # IDs still from SAM
        assert not list(Path(manifest_path).parent.glob("*.png"))  # No roto image files
        everything = process(arrays, item, dict(inset=0))
        assert (everything["object_id"] == 0).any()  # Background stays
        assert everything["roto"].shape == (len(everything["P"]), 3)
        assert (everything["roto"][everything["object_id"] == 0] == 0).all(1).any()  # Black over background
        inside_red = everything["source_pixel"] == 20 * 64 + 15  # Grid pixel (15, 20) is inside the red matte
        np.testing.assert_allclose(everything["roto"][inside_red], [[0.75, 0, 0]], atol=1e-3)
        inside_green = everything["source_pixel"] == 25 * 64 + 45  # Grid pixel (45, 25) is inside the green matte
        np.testing.assert_allclose(everything["roto"][inside_green], [[0, 1, 0]], atol=1e-3)
        ply = (Path(manifest_path).parent / "points.1002.ply").read_bytes()
        header = ply.split(b"end_header")[0].decode("ascii").splitlines()
        assert {"property float roto_r", "property float roto_g", "property float roto_b"} <= set(header)
        assert {"property float linear_r", "property float linear_g", "property float linear_b"} <= set(header)
        assert not any("Alpha" in line for line in header)
        # A different roto must give a different package.
        _, other = generate_package(node, **args, roto=roto * 0.5)
        assert other != manifest_path
        # Roto without SAM tracks: each roto colour channel is an actor (red 1, green 2, blue 3).
        _, alone = generate_package(node, **{**args, "actor_tracks": None}, roto=roto)
        manifest, item, arrays = load_frame(alone, 1002)
        assert arrays["actor_ids"].tolist() == [1, 2]  # Blue is empty in every frame: no actor 3
        assert arrays["object_id"][20, 15] == 1 and arrays["object_id"][25, 45] == 2
        assert "red = 1" in manifest["actor_id_mapping"]
        assert "roto" in process(arrays, item, dict(inset=0))
        # A greyscale roto (all channels equal) is a single actor 1.
        grey = roto[..., :1].expand(-1, -1, -1, 3).clone()
        _, single = generate_package(node, **{**args, "actor_tracks": None}, roto=grey)
        _, _, arrays = load_frame(single, 1002)
        assert arrays["actor_ids"].tolist() == [1]
        assert set(np.unique(arrays["object_id"])) == {0, 1}


def test_output_directory_is_the_package_folder_and_versions(tmp_path):
    from pathlib import Path
    from unittest.mock import patch

    import numpy as np

    from image2point.point_package import camera_points

    class FakeEngine:
        calls = 0

        def __init__(self, *args):
            pass

        def predict(self, rgb, k):
            FakeEngine.calls += 1
            return camera_points(np.full(rgb.shape[:2], 4), k), np.ones(rgb.shape[:2]), {}

    checkpoint, source = stand_in_moge(tmp_path)
    shot = tmp_path / "shots" / "BA001_pole"
    with comfy_node(tmp_path, checkpoint, source) as node, patch("image2point.moge3.MoGe3Engine", FakeEngine):
        (model,) = node.LoadMoGe3().load("local.pt", "FP16")
        args = {**sample_args(tmp_path, model), "output_directory": str(shot)}

        def run(**overrides):
            return Path(generate_package(node, **{**args, **overrides})[1]).parent

        first = shot.with_name("BA001_pole_v001")
        assert run() == first  # Packages are always versioned, starting at _v001
        assert sorted(p.name for p in shot.parent.iterdir()) == ["BA001_pole_v001"]  # No staging or cache beside it
        assert {p.name for p in first.iterdir()} >= {"manifest.json", "frame.1002.npz", "points.1002.ply"}
        assert run() == first  # Identical results reuse the folder
        assert run(actor_tracks=None) == shot.with_name("BA001_pole_v002")  # Different results never overwrite
        assert FakeEngine.calls == 1  # The mask change reused cached depth
        assert list((tmp_path / "temp" / "image2point-depth").glob("*.npz"))  # Depth cache in ComfyUI temp
        assert run(output_directory="relative_name") == tmp_path / "relative_name_v001"  # Plain name: ComfyUI output
        automatic = run(output_directory="")
        assert automatic.parent == tmp_path / "image2point" and len(automatic.name) == 16


def test_custom_output_path_writes_flat_files_without_versions(tmp_path):
    """A connected output_path is used as given: no _v001, Output Directory ignored."""
    import json
    from pathlib import Path
    from unittest.mock import patch

    import numpy as np

    from image2point.point_package import camera_points, load_frame, measure_shot

    class FakeEngine:
        def __init__(self, *args):
            pass

        def predict(self, rgb, k):
            return camera_points(np.full(rgb.shape[:2], 4), k), np.ones(rgb.shape[:2]), {}

    checkpoint, source = stand_in_moge(tmp_path)
    shots = tmp_path / "shots"
    path = shots / "BA001_pole_v003"
    with comfy_node(tmp_path, checkpoint, source) as node, patch("image2point.moge3.MoGe3Engine", FakeEngine):
        (model,) = node.LoadMoGe3().load("local.pt", "FP16")
        args = {
            **sample_args(tmp_path, model),
            "start_frame": 1001,
            "output_directory": str(tmp_path / "ignored"),
            "output_path": str(path),
        }
        files, manifest_path = generate_package(node, **args)
        assert Path(manifest_path) == shots / "BA001_pole_v003.manifest.json"
        assert sorted(p.name for p in shots.iterdir()) == [
            "BA001_pole_v003.1001.npz",
            "BA001_pole_v003.1001.ply",
            "BA001_pole_v003.1002.npz",
            "BA001_pole_v003.1002.ply",
            "BA001_pole_v003.manifest.json",
        ]
        assert [Path(f.get_source()).name for f in files] == ["BA001_pole_v003.1001.ply", "BA001_pole_v003.1002.ply"]
        assert not (tmp_path / "ignored").exists() and not list(tmp_path.glob("ignored_v*"))
        manifest, _, arrays = load_frame(manifest_path, 1002)
        assert arrays["P"].shape[:2] == (48, 64)
        # Stabilisation is measured once here, so Houdini (and every farm task) just reads it.
        scales, depths = measure_shot(manifest_path)
        stored = manifest["stabilisation"]
        assert {int(f): s for f, s in stored["scales"].items()} == scales
        assert {int(a): {int(f): d for f, d in v.items()} for a, v in stored["actor_depths"].items()} == depths

        # Same path, same results: reused. Same path, different results: refused, first files intact.
        before = {p.name: p.read_bytes() for p in shots.iterdir()}
        assert generate_package(node, **args)[1] == manifest_path
        with pytest.raises(ValueError, match="already holds different results"):
            generate_package(node, **{**args, "actor_tracks": None})
        assert {p.name: p.read_bytes() for p in shots.iterdir()} == before

        with pytest.raises(ValueError, match="output_path is connected but empty"):
            generate_package(node, **{**args, "output_path": "  "})
        _, relative = generate_package(node, **{**args, "output_path": "rel/take_a"})
        assert Path(relative) == tmp_path / "rel" / "take_a.manifest.json"

        # A failure partway leaves nothing behind (no frames, no staging, no temporary files).
        failing = tmp_path / "failing" / "take_b"
        with patch("image2point.io.write_ply", side_effect=[None, RuntimeError("disk full")]):
            with pytest.raises(RuntimeError, match="disk full"):
                generate_package(node, **{**args, "output_path": str(failing)})
        assert not failing.parent.exists() or list(failing.parent.iterdir()) == []

        # Crash leftovers (frame files with no manifest) are overwritten.
        leftover = tmp_path / "left" / "take_c.1001.npz"
        leftover.parent.mkdir()
        leftover.write_bytes(b"stale")
        _, recovered = generate_package(node, **{**args, "output_path": str(tmp_path / "left" / "take_c")})
        assert leftover.read_bytes() != b"stale"
        assert json.loads(Path(recovered).read_text())["frames"][0]["file"] == "take_c.1001.npz"
