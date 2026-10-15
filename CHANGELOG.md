# Changelog

## 0.1.3 — shorter docs, mask checks

- Houdini asset: **Stabilize Depth** (on by default) removes MoGe's frame-to-frame depth
  scale flicker, measured on the static background. On a 100-frame test shot, frame-to-frame
  jitter of three static objects dropped from about 1% to 0.4-0.6%. Each frame's scale is
  measured against 3 neighbouring frames.
- Houdini asset: Global → **Stabilisation** group with Stabilize Depth and **Background
  Smoothing** (averages the static set with neighbouring frames; off by default). Per-Actor
  Overrides gain **Depth Smoothing** (steadies an actor's overall depth; off by default).
- Docs cut to essentials; model notes merged into the audit sheet.
- Release ZIP is now `image2point-<version>.zip`.
- Houdini asset: **Mask** folder with Roto Check and ID Check (viewport only); the Actor IDs
  field is replaced by a **Keep** toggle on each Per-Actor Override.
- Fixed: adding a second Per-Actor Override row errored (both rows were actor 1); a new row
  now takes the next unused ID.
- Packages start at `_v001` (the first one used to have no version suffix).
- The roto input is now an RGB image (`IMAGE`) instead of a `MASK`, and stays RGB: each
  channel can matte a different object. The `roto` point attribute is a colour, and Roto
  Check shows it as is.
- Without SAM tracks, the roto's channels are the actors: red = 1, green = 2, blue = 3
  (greyscale = 1). The `roto_actor_id` setting is removed.
- A pixel belongs to an actor only at mask coverage 0.5 or more, so compression fringes
  around an RGB roto video no longer become actor points.
- Edge Inset no longer errors on stray mask specks: specks with no reliable depth nearby
  are dropped, with the same limit with or without SciPy.
- Once any Per-Actor Override row exists, only actors with a row are included.
- Edge Inset and Edge Extension default to 1. Depth Tolerance defaults to 0; it flattened
  actors spanning a long depth.

## 0.1.2 — audit sheet

- `docs/audit.md`: provenance, deployment, licences of every direct dependency,
  security, verification, and after-approval steps (initial run, yearly review, removal).
- `requirements.txt` caps each package below its next untested major version
  (OpenCV 5 and huggingface-hub 2 were otherwise picked up).

## 0.1.1 — MIT license and audit prep

- Released under the MIT license.
- Release ZIP vendors only MoGe's inference code and includes the license.
- Tests: a checkpoint carrying hidden code is refused; node runs fail on any network access.
- Removed the Windows-only `.cmd` launchers; build and test the Houdini asset with
  `hython houdini/build_hda.py` and `hython houdini/smoke_test.py` on any platform.
  Docs now show Linux-friendly commands.

## 0.1.0 — first public version

- **ComfyUI nodes:** *Load MoGe-3 Model* (local `.pt` checkpoints from
  `models/image2point` or `extra_model_paths.yaml` folders; never downloads) and
  *Generate Point Cloud* (plates + tracked USD camera → per-frame point clouds).
- **Actors and roto:** SAM 3 video tracks give actor IDs and the masks used for edge
  repair; an artist roto (one matte for all actors) is stored on every point as the
  `roto` attribute for use as a matte in shaders.
- **Packages:** one self-contained folder per run (`output_directory`), never
  overwritten (`_v002`, `_v003`…); MoGe depth cached per ComfyUI session.
- **Houdini HDA (Image2Point Proxy):** depth offset, edge inset/extension, depth
  tolerance, point coverage and per-actor overrides; local bake (threaded) and farm
  bake to `points.####.bgeo.sc`.
- Manual, offline-friendly install (`docs/install.md`); tests for CPU, GPU and Houdini.
