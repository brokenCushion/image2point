# Install

Manual install; works on offline servers. Needs ComfyUI with a CUDA PyTorch.

1. **Packages.** Install `comfyui_image2point/requirements.txt` into ComfyUI's Python.
   Two of them are not on PyPI; `requirements-github.txt` has their sources:

   ```bash
   <ComfyUI python> -m pip install -r requirements-github.txt -r requirements.txt
   ```

   Offline: build a wheel folder on a connected machine with the **same OS and Python
   version**, copy it over, and install from it:

   ```bash
   python -m pip wheel -r requirements-github.txt -r requirements.txt -w wheels
   <ComfyUI python> -m pip install --no-index --find-links wheels -r requirements.txt
   ```

2. **Node.** Copy `comfyui_image2point/` into ComfyUI's `custom_nodes/`. Move any older
   copy out of `custom_nodes/` first.

3. **Model.** Put `model.pt` (see [AUDIT.md](audit.md)) in ComfyUI's
   `models/image2point/`, or add an existing folder to `extra_model_paths.yaml`:

   ```yaml
   studio_models:
     base_path: //server/models
     image2point: depth/moge3
   ```

4. **Check.** Restart ComfyUI. The **Image2Point** menu shows *Load MoGe-3 Model* and
   *Generate Point Cloud*, and the log has no `IMPORT FAILED` for `comfyui_image2point`.

Tested on Windows, Python 3.11. Not yet tested on Linux.
