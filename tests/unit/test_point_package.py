import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

from image2point.point_package import (
    actor_factors,
    background_factors,
    camera_points,
    depth_scales,
    extend_depth,
    json_write,
    load_frame,
    measure_shot,
    process,
    tag_masks,
)


def scene(actor_ids):
    """Wall at 20 m with optional actors at 3 m and 6 m, seen by a rotated, offset camera."""
    h, w = 16, 24
    k = np.array([[24.0, 0, 12], [0, 24, 8], [0, 0, 1]])
    depth = np.full((h, w), 20.0, np.float32)
    masks = np.zeros((len(actor_ids), h, w), np.float32)
    for i, (rows, cols, z) in enumerate([((3, 13), (2, 10), 3), ((3, 13), (14, 22), 6)][: len(actor_ids)]):
        masks[i, rows[0] : rows[1], cols[0] : cols[1]] = 1
        depth[masks[i] > 0] = z
    labels, alpha = tag_masks(masks, actor_ids, (h, w))
    pose = np.eye(4)
    pose[:3, 3] = [1, 2, 3]
    pose[:3, :3] = [[1, 0, 0], [0, 0, 1], [0, -1, 0]]
    arrays = dict(
        P=camera_points(depth, k) @ pose[:3, :3].T + pose[:3, 3],
        rgb=np.full((h, w, 3), 128, np.uint8),
        masks=masks,
        actor_ids=np.array(actor_ids, np.int32),
        object_id=labels,
        alpha=alpha,
    )
    return arrays, dict(frame=1001, file="frame.npz", c2w=pose.tolist(), K=k.tolist()), pose


def ray_distance(result, pose, actor):
    """Distance from the camera, which is what Depth Scale/Offset act on (along each pixel's ray)."""
    return np.linalg.norm(result["P"][result["object_id"] == actor] - pose[:3, 3], axis=1)


class PointPackageTest(unittest.TestCase):
    def test_depth_offset_moves_whole_cloud_and_actor_offsets_stack(self):
        # No actor masks: the main offset must still move every point.
        arrays, item, pose = scene([])
        original = ray_distance(process(arrays, item, dict(inset=0)), pose, 0)
        moved = process(arrays, item, dict(offset=0.5, inset=0))
        np.testing.assert_allclose(ray_distance(moved, pose, 0), original + 0.5, rtol=1e-5)
        # With actors: background and actors all follow the main offset...
        arrays, item, pose = scene([7, 9])
        base = process(arrays, item, dict(inset=0))
        both = process(arrays, item, dict(offset=-0.5, inset=0))
        for actor in (0, 7, 9):
            np.testing.assert_allclose(
                ray_distance(both, pose, actor), ray_distance(base, pose, actor) - 0.5, rtol=1e-5
            )
        # ...and a per-actor offset adds on top, to that actor only.
        stacked = process(arrays, item, dict(offset=-0.5, inset=0, overrides={"7": {"offset": 2}}))
        np.testing.assert_allclose(ray_distance(stacked, pose, 7), ray_distance(base, pose, 7) + 1.5, rtol=1e-5)
        for actor in (0, 9):
            np.testing.assert_allclose(ray_distance(stacked, pose, actor), ray_distance(both, pose, actor), rtol=1e-5)

    def test_camera_edges_ids_and_publication(self):
        h, w = 16, 24
        k = np.array([[24.0, 0, 12], [0, 24, 8], [0, 0, 1]])
        depth = np.full((h, w), 20.0, np.float32)
        masks = np.zeros((2, h, w), np.float32)
        masks[0, 3:13, 2:10] = 1
        masks[1, 3:13, 14:22] = 1
        depth[masks[0] > 0] = 3
        depth[masks[1] > 0] = 6
        depth[3, 2:10] = 17  # Deliberately bad predicted actor boundary.
        labels, alpha = tag_masks(masks, [7, 9], (h, w))
        pose = np.eye(4)
        pose[:3, 3] = [1, 2, 3]
        pose[:3, :3] = [[1, 0, 0], [0, 0, 1], [0, -1, 0]]
        p = camera_points(depth, k) @ pose[:3, :3].T + pose[:3, 3]
        arrays = dict(
            P=p,
            rgb=np.full((h, w, 3), 128, np.uint8),
            masks=masks,
            actor_ids=np.array([7, 9]),
            object_id=labels,
            alpha=alpha,
        )
        item = dict(frame=1001, file="frame.npz", c2w=pose.tolist(), K=k.tolist())
        cfg = dict(inset=1, extension=1, remove_background=True, stride=1)
        result = process(arrays, item, cfg)
        cam = (result["P"] - pose[:3, 3]) @ pose[:3, :3]
        np.testing.assert_allclose(cam[result["object_id"] == 7, 2], 3, atol=1e-6)
        np.testing.assert_allclose(cam[result["object_id"] == 9, 2], 6, atol=1e-6)
        pixels = cam @ k.T
        pixels = pixels[:, :2] / pixels[:, 2:]
        source = result["source_pixel"]
        np.testing.assert_allclose(pixels, np.column_stack((source % w + 0.5, source // w + 0.5)), atol=2e-6)
        self.assertTrue(np.all(result["pscale"] > 0))
        # Ray adjustment retains projection and responds independently per actor.
        adjusted = process(arrays, item, {**cfg, "overrides": {"7": {"offset": 1}}})
        np.testing.assert_allclose(ray_distance(adjusted, pose, 7), ray_distance(result, pose, 7) + 1, rtol=1e-6)
        np.testing.assert_allclose(ray_distance(adjusted, pose, 9), ray_distance(result, pose, 9), rtol=1e-6)
        # NumPy-only Houdini fallback agrees on this local repair.
        with patch.dict("sys.modules", {"scipy.ndimage": None}):
            filled, support = extend_depth(depth, masks[0], 1, 1)
        np.testing.assert_allclose(filled[support], 3)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            np.savez(root / "frame.npz", **arrays)
            json_write(root / "manifest.json", dict(kind="image2point.point_package", version=1, frames=[item]))
            _, _, loaded = load_frame(root / "manifest.json", 1001)
            np.testing.assert_array_equal(loaded["P"], p)
            with self.assertRaises(ValueError):
                load_frame(root / "manifest.json", 1002)
        with self.assertRaises(ValueError):
            tag_masks(masks, [7, 7], (h, w))
        kept = process(arrays, item, {"drop": [7, 42]})  # Unknown IDs are ignored.
        self.assertEqual(set(np.unique(kept["object_id"])), {0, 9})
        # Once any actor has an override row, only actors with rows are included.
        listed = process(arrays, item, {"only": [7]})
        self.assertEqual(set(np.unique(listed["object_id"])), {0, 7})


if __name__ == "__main__":
    unittest.main()


class EdgeExtensionTest(unittest.TestCase):
    def test_extension_grows_actor_points_at_actor_depth(self):
        arrays, item, pose = scene([7])  # Actor at 3 m in front of a 20 m wall
        base = process(arrays, item, dict(remove_background=True, inset=0, extension=0))
        grown = process(arrays, item, dict(remove_background=True, inset=0, extension=2))
        self.assertGreater(len(grown["P"]), len(base["P"]))
        added = ~np.isin(grown["source_pixel"], base["source_pixel"])
        depth = ((grown["P"] - pose[:3, 3]) @ pose[:3, :3])[:, 2]
        np.testing.assert_allclose(depth[added], 3, atol=1e-5)  # Grown points take actor depth, not the wall's
        self.assertNotIn("Alpha", grown)  # No Alpha attribute: the matte is the roto attribute


class DepthToleranceTest(unittest.TestCase):
    def test_outlier_depths_inside_mask_are_repaired_not_copied(self):
        arrays, item, pose = scene([7])  # Actor 7 sits at 3 m in front of a 20 m wall.
        cam = (arrays["P"] - pose[:3, 3]) @ pose[:3, :3]
        depth = cam[..., 2].copy()
        depth[6:8, 5:7] = 14.0  # Blended "flying" depths deep inside the mask.
        k = np.array(item["K"])
        arrays = {**arrays, "P": camera_points(depth, k) @ pose[:3, :3].T + pose[:3, 3]}

        def actor_depths(**cfg):
            result = process(arrays, item, dict(remove_background=True, inset=1, **cfg))
            return ((result["P"] - pose[:3, 3]) @ pose[:3, :3])[:, 2]

        self.assertGreater(actor_depths(extension=2).max(), 10)  # Without tolerance the outliers survive.
        repaired = actor_depths(extension=2, tolerance=1.0)
        np.testing.assert_allclose(repaired, 3, atol=1e-5)  # Outliers repaired, and extension copies only 3 m.
        self.assertEqual(len(repaired), len(actor_depths(extension=2)))  # Repaired, not deleted.


def flickering_shot(root, scales):
    """A wall 20 m away, filmed by a camera sliding sideways; frame i's depth is off by scales[i]."""
    h, w = 40, 60
    k = np.array([[50.0, 0, 30], [0, 50, 20], [0, 0, 1]])
    frames = []
    for i, scale in enumerate(scales):
        pose = np.eye(4)
        pose[0, 3] = 0.2 * i
        depth = np.full((h, w), 20.0 * scale, np.float32)
        actor = np.zeros((1, h, w), np.float32)
        actor[0, 10:30, 25:35] = 1  # An actor at 5 m: moving things must not steer the scale.
        depth[actor[0] > 0] = 5.0 * scale * (1 + 0.05 * i)
        labels, alpha = tag_masks(actor, [1], (h, w))
        np.savez(
            root / f"frame.{1001 + i}.npz",
            P=camera_points(depth, k) + pose[:3, 3],
            rgb=np.zeros((h, w, 3), np.uint8),
            masks=actor,
            actor_ids=np.array([1], np.int32),
            object_id=labels,
            alpha=alpha,
        )
        frames.append(dict(frame=1001 + i, file=f"frame.{1001 + i}.npz", c2w=pose.tolist(), K=k.tolist()))
    json_write(root / "manifest.json", dict(kind="image2point.point_package", version=1, frames=frames))
    return root / "manifest.json"


class DepthScaleTest(unittest.TestCase):
    def test_frame_to_frame_scale_flicker_is_removed(self):
        rng = np.random.default_rng(0)
        flicker = 1 + rng.uniform(-0.025, 0.025, 30)  # MoGe's per-frame scale guess wobbles 2-3%
        with tempfile.TemporaryDirectory() as directory:
            scales = depth_scales(flickering_shot(Path(directory), flicker))
        corrected = flicker * np.array([scales[1001 + i] for i in range(30)])
        jumps = np.abs(np.diff(corrected)) / corrected[1:]
        self.assertLess(np.median(jumps), 0.002, "frame-to-frame scale jumps remain")
        self.assertLess(abs(np.median(corrected) - np.median(flicker)), 0.01, "overall scale must be kept")

    def test_steady_shot_is_left_alone(self):
        with tempfile.TemporaryDirectory() as directory:
            scales = depth_scales(flickering_shot(Path(directory), np.ones(8)))
        np.testing.assert_allclose(list(scales.values()), 1, atol=1e-4)

    def test_process_scales_along_camera_rays(self):
        arrays, item, pose = scene([7])
        base = process(arrays, item, dict(inset=0))
        scaled = process(arrays, item, dict(inset=0, depth_scale=1.1))
        for actor in (0, 7):
            np.testing.assert_allclose(
                ray_distance(scaled, pose, actor), ray_distance(base, pose, actor) * 1.1, rtol=1e-5
            )


def noisy_shot(root, count, actor_depth, pixel_noise=0.0, scale_noise=0.0, seed=0):
    """Wall 20 m away and one actor, filmed by a sliding camera, with MoGe-like noise.

    scale_noise wobbles each whole frame; pixel_noise is independent per-pixel depth noise;
    actor_depth(i) is the actor's true depth on frame i.
    """
    rng = np.random.default_rng(seed)
    h, w = 40, 60
    k = np.array([[50.0, 0, 30], [0, 50, 20], [0, 0, 1]])
    frames = []
    for i in range(count):
        pose = np.eye(4)
        pose[0, 3] = 0.1 * i
        depth = np.full((h, w), 20.0, np.float32)
        actor = np.zeros((1, h, w), np.float32)
        actor[0, 10:30, 25:35] = 1
        depth[actor[0] > 0] = actor_depth(i)
        depth *= 1 + rng.uniform(-scale_noise, scale_noise)
        depth *= 1 + rng.normal(0, pixel_noise, (h, w)).astype(np.float32)
        labels, alpha = tag_masks(actor, [1], (h, w))
        np.savez(
            root / f"frame.{1001 + i}.npz",
            P=camera_points(depth, k) + pose[:3, 3],
            rgb=np.zeros((h, w, 3), np.uint8),
            masks=actor,
            actor_ids=np.array([1], np.int32),
            object_id=labels,
            alpha=alpha,
        )
        frames.append(dict(frame=1001 + i, file=f"frame.{1001 + i}.npz", c2w=pose.tolist(), K=k.tolist()))
    json_write(root / "manifest.json", dict(kind="image2point.point_package", version=1, frames=frames))
    return root / "manifest.json"


class ActorSmoothingTest(unittest.TestCase):
    def test_actor_wobble_is_smoothed_and_its_motion_kept(self):
        rng = np.random.default_rng(1)
        wobble = 1 + rng.uniform(-0.01, 0.01, 40)  # The actor's own 1% depth wobble
        truth = 5.0 + 0.05 * np.arange(40)  # Walking away from camera at a steady pace
        with tempfile.TemporaryDirectory() as directory:
            manifest = noisy_shot(Path(directory), 40, lambda i: truth[i] * wobble[i])
            scales, depths = measure_shot(manifest)
        factors = actor_factors(depths[1], scales, 3)
        smoothed = np.array([truth[i] * wobble[i] * factors[1001 + i] for i in range(40)])
        before = np.median(np.abs(np.diff(truth * wobble, 2)))
        after = np.median(np.abs(np.diff(smoothed, 2)))
        self.assertLess(after, before / 4, "actor wobble remains")
        np.testing.assert_allclose(smoothed[5:-5], truth[5:-5], rtol=0.01)  # The walk itself is kept
        self.assertEqual(actor_factors(depths[1], scales, 0), {})  # 0 = off


class BackgroundSmoothingTest(unittest.TestCase):
    def test_background_shimmer_is_averaged_with_neighbouring_frames(self):
        with tempfile.TemporaryDirectory() as directory:
            manifest = noisy_shot(Path(directory), 9, lambda i: 5.0, pixel_noise=0.005)
            scales, _ = measure_shot(manifest)
            _, item, arrays = load_frame(manifest, 1005)
            factor = background_factors(manifest, 1005, 2, scales)
        pose = np.asarray(item["c2w"])
        z = ((arrays["P"] - pose[:3, 3]) @ pose[:3, :3])[..., 2]
        wall = (arrays["object_id"] == 0)[:, 5:-5]  # Away from the frame edge, which neighbours don't cover
        before = np.std(z[:, 5:-5][wall] / 20 - 1)
        after = np.std((z * factor)[:, 5:-5][wall] / 20 - 1)
        self.assertLess(after, before * 0.6, "background shimmer remains")
        self.assertTrue(np.all(factor[arrays["object_id"] == 1] == 1), "actors are not touched")


class SceneScaleNeighboursTest(unittest.TestCase):
    def test_scale_is_measured_against_several_neighbours(self):
        # With per-pixel noise, one-neighbour chaining drifts; several neighbours keep the scale level.
        with tempfile.TemporaryDirectory() as directory:
            manifest = noisy_shot(Path(directory), 60, lambda i: 5.0, pixel_noise=0.01)
            single = depth_scales(manifest, neighbours=1)
            several = depth_scales(manifest, neighbours=3)

        def jump(scales):
            return np.median(np.abs(np.diff(np.log(list(scales.values())))))

        self.assertLessEqual(jump(several), jump(single))


class StraySpeckTest(unittest.TestCase):
    def test_specks_with_no_reliable_depth_nearby_are_dropped_not_errors(self):
        from unittest.mock import patch

        depth = np.full((60, 60), 20.0)
        mask = np.zeros((60, 60), np.float32)
        mask[5:25, 5:25] = 1  # The actor
        mask[50, 50] = 1  # A one-pixel stray speck far away: Edge Repair removes it entirely
        depth[mask > 0] = 3.0
        for scipy in (True, False):  # Houdini usually has no SciPy; both must agree
            with patch.dict("sys.modules", {} if scipy else {"scipy.ndimage": None}):
                filled, support = extend_depth(depth, mask, 1, 0)
            self.assertFalse(support[50, 50], f"speck kept (scipy={scipy})")
            self.assertTrue(support[5:25, 5:25].all())
            np.testing.assert_allclose(filled[support], 3)


class TagMasksTest(unittest.TestCase):
    def test_faint_mask_values_stay_background(self):
        # Video compression leaves faint colour around every edge of an RGB roto.
        mask = np.zeros((4, 4), np.float32)
        mask[0, 0] = 1.0
        mask[1, 1] = 0.6
        mask[2, 2] = 0.3  # Compression fringe: not part of the actor
        labels, _ = tag_masks(mask[None], [5], (4, 4))
        self.assertEqual(labels[0, 0], 5)
        self.assertEqual(labels[1, 1], 5)
        self.assertEqual(labels[2, 2], 0)


class RotoAttributeTest(unittest.TestCase):
    def test_linear_colour_and_legacy_srgb_keep_roto_as_data(self):
        arrays, item, _ = scene([7])
        arrays["roto_grid"] = np.full_like(arrays["P"], 0.5)
        legacy = process(arrays, item, dict(inset=0))
        np.testing.assert_allclose(legacy["Cd"], 0.2158605, atol=1e-6)  # sRGB 128/255
        arrays["linear_rgb"] = np.full_like(arrays["P"], 0.21404114)
        current = process(arrays, item, dict(inset=0))
        np.testing.assert_allclose(current["Cd"], 0.21404114, atol=1e-7)  # No second conversion
        np.testing.assert_array_equal(current["roto"], 0.5)

    def test_roto_is_its_own_attribute_on_every_point_and_alpha_is_unchanged(self):
        arrays, item, pose = scene([7])  # SAM mask: rows 3-12, cols 2-9
        roto = np.zeros(arrays["object_id"].shape, np.float16)  # Older packages: one greyscale matte
        roto[3:13, 3:10] = 0.75  # Artist matte one column tighter on the left, soft value
        roto[0, :4] = 0.5  # A soft patch over background
        without = process(arrays, item, dict(inset=0, extension=2))
        result = process({**arrays, "roto_grid": roto}, item, dict(inset=0, extension=2))
        for name in without:  # The roto only adds its own attribute; everything else is unchanged
            np.testing.assert_array_equal(result[name], without[name])
        self.assertNotIn("roto", without)  # No roto attribute without a roto
        w = arrays["object_id"].shape[1]
        value = {
            (int(o), int(p)): float(r[0])
            for o, p, r in zip(result["object_id"], result["source_pixel"], result["roto"])
        }
        self.assertEqual(value[(7, 5 * w + 5)], 0.75)  # Actor under the roto
        self.assertEqual(value[(7, 5 * w + 2)], 0)  # Actor (SAM) pixel outside the roto
        self.assertEqual(value[(7, 5 * w + 0)], 0)  # Edge Extend point: plain projection
        self.assertEqual(value[(0, 15 * w + 20)], 0)  # Background under black roto: still present
        self.assertEqual(value[(0, 0 * w + 1)], 0.5)  # Background under a soft roto value

    def test_rgb_roto_keeps_its_colours(self):
        arrays, item, pose = scene([7])
        h, w = arrays["object_id"].shape
        roto = np.zeros((h, w, 3), np.float16)
        roto[3:13, 2:10] = (1, 0, 0)  # Red matte on the actor
        roto[0, :4] = (0, 0.5, 1)  # Soft blue-green over background
        result = process({**arrays, "roto_grid": roto}, item, dict(inset=0))
        colour = {int(p): tuple(r) for p, r in zip(result["source_pixel"], result["roto"])}
        self.assertEqual(colour[5 * w + 5], (1, 0, 0))
        self.assertEqual(colour[0 * w + 1], (0, 0.5, 1))
        self.assertEqual(colour[15 * w + 20], (0, 0, 0))


class LegacyPackageTest(unittest.TestCase):
    def test_packages_from_before_the_rename_still_load(self):
        arrays, item, _ = scene([7])
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            np.savez(root / "frame.npz", **arrays)
            for kind in ("image2point.point_package", "image2depth.point_package"):
                json_write(root / "manifest.json", dict(kind=kind, version=1, frames=[item]))
                self.assertEqual(load_frame(root / "manifest.json", 1001)[1]["frame"], 1001)
            json_write(root / "manifest.json", dict(kind="something.else", version=1, frames=[item]))
            with self.assertRaises(ValueError):
                load_frame(root / "manifest.json", 1001)
