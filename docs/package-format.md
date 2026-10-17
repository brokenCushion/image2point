# Package format

Written by the ComfyUI node, read by the Houdini asset (`src/image2point/point_package.py`).

**`manifest.json`** (kind `image2point.point_package`, version 1): FPS, up axis, metres,
plate and grid resolution, model fingerprints, and one record per frame (frame number,
NPZ file, camera-to-world pose, intrinsics `K` and `K_plate`).

**`frame.####.npz`** (read with `allow_pickle=False`):
- `P`: world positions; `NaN` where depth is invalid.
- `rgb`: uint8 sRGB for model input and native PLY preview.
- `linear_rgb`: float32 linear sRGB point colour; `manifest.color_space` is `linear_srgb`.
  Houdini uses this directly for `Cd`; old packages without it are converted from `rgb`.
- `object_id`: 0 is background. Actors are their SAM track slot + 1, or without tracks
  the roto channel (red 1, green 2, blue 3).
- `alpha`: mask coverage.
- `masks`, `original_masks` and `actor_ids`: per-actor masks on the grid and at plate
  resolution.
- `roto_grid`: the RGB roto on the grid, only when a roto was connected.

**`points.####.ply`**: the same points for ComfyUI preview. Standard RGB remains sRGB;
float `linear_r`, `linear_g`, `linear_b` attributes carry linear colour.
Roto is data and receives no colour conversion. BGEO `Cd` is linear sRGB, labelled by
the detail attribute `i2p_color_space`; an ACEScg workflow needs a working-space transform.

**`stabilisation`** (in the manifest): per-frame depth scales and each actor's depth per
frame, measured once by ComfyUI so Houdini does not re-read the shot. Older packages without
it are measured by the Houdini asset on first cook.

**Custom output path:** with `output_path` connected, the files sit flat beside the path:
`<path>.####.npz`, `<path>.####.ply`, `<path>.manifest.json`. The manifest is written last
and carries a `package_key`; the same results at the same path are reused, different results
are refused, and frame files left without a manifest by an interrupted run are replaced.

**Rules:** a package is written to a staging folder and published only when every frame
succeeds. It is saved as `<name>_v001` and never overwritten: different results go to `_v002`,
`_v003` and so on, and
identical results reuse the existing folder. The MoGe depth cache sits in ComfyUI's temp
folder and is not part of the package.
