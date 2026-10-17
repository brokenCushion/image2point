# Changelog

## 0.1.1 — farm-ready Houdini asset, custom output path

- ComfyUI: optional **`output_path`** input. When connected, files are written as given
  (`<path>.####.npz`, `.ply`, `<path>.manifest.json`) with no versioning; Output Directory
  is ignored.
- ComfyUI: depth stabilisation is measured once and stored in the manifest, so Houdini and
  farm tasks no longer re-read the whole shot.
- Houdini asset: the node outputs the light viewport **Preview** (Preview Point Spacing,
  Roto/ID Check). The full-resolution, clean points are the internal **`OUT_RENDER`**, for a
  standard Geometry ROP and studio farm submission.
- Houdini asset: the Bake tab is now **Local Bake** with a typed **Output File**; it overwrites
  like a ROP and writes no bake manifest.
- **Breaking:** Prepare Farm Job, Publish Directory and `farm_bake.py` are removed. Scenes that
  used them need a Geometry ROP or Local Bake instead.

## 0.1.0 — first public version
