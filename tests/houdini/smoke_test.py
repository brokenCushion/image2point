"""Run with hython on a clean worker. Builds, cooks and bakes the supplied test shot.

Release ZIPs place this beside build_hda.py; in a repository checkout pytest runs it
(tests/houdini/test_hda.py) with a freshly generated sample manifest.
"""

import json
import sys
import tempfile
from pathlib import Path
from unittest.mock import patch

import hou
import numpy as np

here = Path(__file__).resolve().parent
if not (here / "build_hda.py").exists():
    sys.path.insert(0, str(here.parents[1] / "integrations/houdini"))
from build_hda import build  # noqa: E402


def main():
    with tempfile.TemporaryDirectory() as directory:
        asset = build(Path(directory) / "test.hda")
        hou.hda.installFile(str(asset))
        container = hou.node("/obj").createNode("geo", "image2point_test", run_init_scripts=False)
        try:
            node = container.createNode("image2point::proxy::1.0")
            node.allSubChildren(sync_delayed_definition=True)
            sample = (
                Path(sys.argv[1])
                if len(sys.argv) > 1
                else Path(__file__).resolve().parent.parent / "sample/manifest.json"
            )
            node.parm("manifest").set(str(sample))
            hou.setFps(48)
            hou.playbar.setFrameRange(900, 1100)
            hou.playbar.setPlaybackRange(950, 1050)
            hou.setFrame(999)
            scene_range = hou.playbar.frameRange()
            playback_range = hou.playbar.playbackRange()
            template = node.type().definition().parmTemplateGroup().find("manifest")
            assert "manifest_changed" in template.scriptCallback()
            node.hdaModule().manifest_changed(node)  # parm.set() does not fire UI callbacks
            assert hou.fps() == 48 and hou.frame() == 999
            assert hou.playbar.frameRange() == scene_range
            assert hou.playbar.playbackRange() == playback_range
            assert node.evalParm("start") == 1001 and node.evalParm("end") == 1003
            assert node.parm("shot_range") is None
            try:
                node.hdaModule().write_geometry(hou.Geometry(), {"fps": 24}, {}, str(sample))
            except ValueError as error:
                assert "FPS differs" in str(error)
            else:
                raise AssertionError("Mismatched scene FPS must be rejected")
            assert hou.fps() == 48
            hou.setFps(24)
            hou.setFrame(1001)
            group = node.type().definition().parmTemplateGroup()
            assert [t.label() for t in group.entries()] == [
                "Point Cloud Cache",
                "Global",
                "Actors",
                "Display",
                "Bake",
            ]
            for label in ("Global", "Actors", "Display", "Bake"):
                assert group.findFolder(label).folderType() == hou.folderType.Tabs
            assert group.find("advanced_depth") is None
            assert group.find("tolerance") is None
            assert "remove_background" in [t.name() for t in group.findFolder("Global").parmTemplates()]
            stabilisation = group.find("stabilisation_section")
            assert "stabilisation_section" in [t.name() for t in group.findFolder("Global").parmTemplates()]
            assert [t.name() for t in stabilisation.parmTemplates()] == ["stabilize", "bg_smoothing"]
            assert node.evalParm("bg_smoothing") == 0
            assert node.evalParm("stabilize") == 1  # Depth stabilisation is on by default
            assert [t.name() for t in group.find("mask_section").parmTemplates()] == ["roto_check", "id_check"]
            assert [t.name() for t in group.findFolder("Actors").parmTemplates()] == ["actor_controls"]
            assert [t.name() for t in group.findFolder("Bake").parmTemplates()][:3] == [
                "publish_directory",
                "start",
                "end",
            ]
            # Edge Repair and Extension start at 1 (also per actor); Depth Tolerance starts off, since it
            # flattens actors that span a long depth.
            assert node.evalParm("inset") == 1 and node.evalParm("extension") == 1
            node.parm("actor_controls").set(1)
            assert node.evalParm("tolerance1") == 0
            assert node.evalParm("inset1") == 1 and node.evalParm("extension1") == 1
            node.parm("actor_controls").set(0)
            node.parm("remove_background").set(True)
            node.parm("extension").set(2)
            node.parm("preview_stride").set(1)
            geometry = node.geometry()
            assert geometry is not None, node.errors()
            assert len(geometry.points()) > 0
            assert set(geometry.pointIntAttribValues("object_id")) == {1, 2}
            # Bulk point creation must keep every point where the core maths put it, in order.
            module = node.hdaModule()
            _, item, arrays = module.load_frame(str(sample), int(hou.frame()))
            scale = module.depth_scales(str(sample))[int(hou.frame())]
            expected = module.process(arrays, item, {**module.settings(node), "depth_scale": scale})["P"]
            cooked = np.frombuffer(geometry.pointFloatAttribValuesAsString("P"), np.float32).reshape(-1, 3)
            assert np.array_equal(cooked, expected.astype(np.float32)), "cooked P differs from process()"
            # Stabilize off: the raw MoGe depth, unscaled.
            node.parm("stabilize").set(False)
            raw = np.frombuffer(node.geometry().pointFloatAttribValuesAsString("P"), np.float32).reshape(-1, 3)
            unscaled = module.process(arrays, item, module.settings(node))["P"]
            assert np.array_equal(raw, unscaled.astype(np.float32)), "Stabilize off must not scale"
            node.parm("stabilize").set(True)
            # Background Smoothing and per-actor Depth Smoothing reach the cooked points.
            node.parm("bg_smoothing").set(1)
            node.parm("actor_controls").set(2)
            module.rows_changed(node)
            assert node.evalParm("smoothing1") == 0  # Off by default
            node.parm("smoothing2").set(2)
            frame = int(hou.frame())
            scales, depths = module.measure_shot(str(sample))
            cfg = {
                **module.settings(node),
                "depth_scale": scales[frame],
                "background_factor": module.background_factors(str(sample), frame, 1, scales),
                "actor_scales": {2: module.actor_factors(depths[2], scales, 2)[frame]},
            }
            smoothed = np.frombuffer(node.geometry().pointFloatAttribValuesAsString("P"), np.float32).reshape(-1, 3)
            assert not node.errors(), node.errors()
            assert np.array_equal(smoothed, module.process(arrays, item, cfg)["P"].astype(np.float32))
            node.parm("bg_smoothing").set(0)
            node.parm("actor_controls").set(0)
            # Check colours are viewport only. Roto Check shows the roto's own colours.
            node.parm("roto_check").set(True)
            roto_view = node.geometry()
            shown = np.array(roto_view.pointFloatAttribValues("Cd"))
            assert np.array_equal(shown, np.array(roto_view.pointFloatAttribValues("roto")))
            assert np.isclose(shown.reshape(-1, 3), np.float16(0.9) * np.array([1, 0, 0])).all(1).any()
            node.parm("roto_check").set(False)
            node.parm("id_check").set(True)
            checked = node.geometry()
            ids = np.array(checked.pointIntAttribValues("object_id"))
            cd = np.array(checked.pointFloatAttribValues("Cd")).reshape(-1, 3)
            assert len({tuple(cd[ids == actor][0]) for actor in (1, 2)}) == 2, "each actor gets its own colour"
            assert all(len({tuple(c) for c in cd[ids == actor]}) == 1 for actor in (1, 2))
            # Keep unticked drops that actor.
            node.parm("actor_controls").set(1)
            node.parm("actor_id1").set(2)
            listed = set(node.geometry().pointIntAttribValues("object_id"))
            assert listed == {2}, (listed, module.settings(node), node.errors())  # Only listed actors
            node.parm("keep1").set(False)
            assert len(node.geometry().points()) == 0  # The only listed actor is switched off
            node.parm("actor_controls").set(0)
            # Adding override rows gives each the next unused actor ID, so adding rows never clashes.
            template = node.type().definition().parmTemplateGroup().find("actor_controls")
            assert "rows_changed" in template.scriptCallback()
            node.parm("actor_controls").set(2)
            module.rows_changed(node)  # parm.set() does not fire UI callbacks
            assert [node.evalParm(f"actor_id{i}") for i in (1, 2)] == [1, 2]
            assert node.geometry() is not None and not node.errors(), node.errors()
            node.parm("actor_controls").set(0)
            for count in (0, 1, 5, 1000):  # Edge cases of the doubling merge.
                small = hou.Geometry()
                module.create_points(small, np.arange(count * 3, dtype=np.float32).reshape(-1, 3))
                assert small.intrinsicValue("pointcount") == count
            node.parm("publish_directory").set(directory)
            published = Path(node.hdaModule().bake(node))
            assert published == Path(directory) / "manifest.json"
            original = published.read_bytes()
            try:
                node.hdaModule().bake(node)
            except FileExistsError:
                pass
            else:
                raise AssertionError("Bake must not overwrite existing cache files")
            assert published.read_bytes() == original
            frames = json.loads(published.read_text())["frames"]
            assert len(frames) == 3
            assert [entry["file"] for entry in frames] == [f"points.{f}.bgeo.sc" for f in (1001, 1002, 1003)]
            loaded = hou.Geometry()
            loaded.loadFromFile(str(published.parent / frames[0]["file"]))
            assert loaded.attribValue("i2p_color_space") == "linear_srgb"
            assert json.loads(published.read_text())["color_space"] == "linear_srgb"
            # id_check is still on, but a bake keeps the plate colours.
            module_values = module.frame_values(str(sample), 1001, module.settings(node, True))[1]
            assert np.allclose(np.array(loaded.pointFloatAttribValues("Cd")), module_values["Cd"].ravel())
            assert loaded.findPointAttrib("pscale") and loaded.findPointAttrib("Alpha") is None
            # The artist roto is baked as the roto point attribute; no image files travel with the bake.
            assert not list(published.parent.glob("*.png"))
            baked_roto = np.array(loaded.pointFloatAttribValues("roto")).reshape(-1, 3)
            assert np.isclose(baked_roto, np.float16(0.9) * np.array([1, 0, 0])).all(1).any()  # RGB roto attribute
            # Bake Locally (threaded) must write exactly what the sequential farm bake writes.
            node.parm("publish_directory").set(str(Path(directory) / "threaded"))
            with (
                patch.object(hou, "isUIAvailable", return_value=True),
                patch.object(hou, "InterruptableOperation") as bar,
            ):
                threaded = Path(node.hdaModule().bake_locally(node))
                assert bar.call_args.kwargs["open_interrupt_dialog"] is True
                progress = bar.return_value.__enter__.return_value.updateProgress.call_args_list
                assert progress[0].args == (0.0,)
                assert progress[-1].args == (1.0,)
                assert len(progress) >= 4
            assert json.loads(threaded.read_text())["frames"] == frames
            for entry in frames:
                sequential_geo, threaded_geo = hou.Geometry(), hou.Geometry()
                sequential_geo.loadFromFile(str(published.parent / entry["file"]))
                threaded_geo.loadFromFile(str(threaded.parent / entry["file"]))
                for attrib in sequential_geo.pointAttribs():
                    name = attrib.name()
                    if attrib.dataType() == hou.attribData.Int:
                        same = sequential_geo.pointIntAttribValuesAsString(
                            name
                        ) == threaded_geo.pointIntAttribValuesAsString(name)
                    else:
                        same = sequential_geo.pointFloatAttribValuesAsString(
                            name
                        ) == threaded_geo.pointFloatAttribValuesAsString(name)
                    assert same, f"threaded bake differs: frame {entry['frame']} {name}"
            cancelled = Path(directory) / "cancelled"
            node.parm("publish_directory").set(str(cancelled))
            previous = hou.frame()
            with (
                patch.object(hou, "isUIAvailable", return_value=True),
                patch.object(hou, "InterruptableOperation") as bar,
            ):
                bar.return_value.__enter__.return_value.updateProgress.side_effect = [None, hou.OperationInterrupted()]
                try:
                    module.bake_locally(node)
                except hou.OperationInterrupted:
                    pass
                else:
                    raise AssertionError("Bake did not cancel")
            assert hou.frame() == previous
            assert not list(cancelled.iterdir()), "Cancelled bake left partial output"
            assert Path(module.bake(node)).is_file(), "Cancelled output folder cannot be reused"
            print("PASS: HDA build, point cook, actor extraction, BGEO bake, progress, cancellation, threaded parity")
        finally:
            container.destroy()


if __name__ == "__main__":
    main()
