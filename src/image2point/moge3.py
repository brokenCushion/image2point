"""MoGe-3 metric monocular depth using the supplied USD camera."""

import math
import os
import sys
import warnings
from pathlib import Path

import numpy as np

from .camera import require
from .point_package import camera_points


def model_paths():
    root = Path(__file__).resolve().parents[2] / "cache"  # Repository checkout: src/image2point/ -> cache/
    source = Path(os.environ.get("I2P_MOGE_SOURCE", root / "MoGe"))
    return source, [Path(os.environ.get("I2P_MOGE_MODEL", source / "checkpoints/moge3/model.pt"))]


class MoGe3Engine:
    def __init__(self, checkpoint=None, source_path=None, resolution_level=9, refine_steps=3, use_fp16=True):
        import torch

        require(torch.cuda.is_available(), "MoGe-3 requires a CUDA GPU")
        source, weights = model_paths()
        source = Path(source_path) if source_path else source
        self.resolution_level, self.refine_steps, self.use_fp16 = resolution_level, refine_steps, use_fp16
        require(source.is_dir(), "MoGe source missing; see docs/audit.md")
        sys.path.insert(0, str(source))
        import moge
        from moge.model.v3 import MoGeModel

        # Another 'moge' already imported in this Python (e.g. by a different ComfyUI node)
        # would silently replace the pinned source.
        loaded = Path(moge.__file__).resolve()
        require(
            loaded.is_relative_to(source.resolve()),
            f"A different 'moge' package is loaded from {loaded.parent}; remove it or the node that installs it",
        )
        model_ref = Path(checkpoint) if checkpoint else weights[0]
        # Must exist locally: for a missing path MoGe falls back to downloading from the internet.
        require(model_ref.is_file(), f"MoGe checkpoint missing: {model_ref}")
        # MoGe loads non-strictly and only warns about mismatched weights, leaving them random.
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            model = MoGeModel.from_pretrained(str(model_ref))
        mismatched = [str(w.message) for w in caught if "absent from the checkpoint" in str(w.message)]
        mismatched += [str(w.message) for w in caught if "no counterpart in the model" in str(w.message)]
        if mismatched:
            raise ValueError(f"Checkpoint does not match the MoGe-3 model: {mismatched[0][:300]}")
        self.torch = torch
        self.model = model.eval().to("cuda")

    def predict(self, rgb, k):
        torch = self.torch
        h, w = rgb.shape[:2]
        image = torch.from_numpy(rgb.astype(np.float32) / 255).permute(2, 0, 1).to("cuda")
        fov_x = math.degrees(2 * math.atan(w / (2 * float(k[0, 0]))))
        torch.cuda.reset_peak_memory_stats()
        with torch.inference_mode():
            out = self.model.infer(
                image,
                fov_x=fov_x,
                resolution_level=self.resolution_level,
                refine_steps=self.refine_steps,
                use_fp16=self.use_fp16,
            )
        points = camera_points(out["depth"].detach().float().cpu().numpy(), k)
        mask = out.get("mask")
        if mask is None:
            mask = np.ones(points.shape[:2], dtype=np.float32)
        else:
            mask = mask.detach().cpu().numpy().astype(np.float32)
        return (
            points,
            mask,
            {
                "gpu": torch.cuda.get_device_name(),
                "peak_vram_bytes": torch.cuda.max_memory_allocated(),
                "fov_x": fov_x,
                "model_intrinsics": out.get("intrinsics", None) is not None,
            },
        )
