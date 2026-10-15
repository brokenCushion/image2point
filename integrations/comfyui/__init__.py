"""Image2Point ComfyUI nodes. Studio input, segmentation and farm nodes stay external."""

import sys
from pathlib import Path

# Distribution includes the shared runner here; repository development uses its parent.
sys.path.insert(0, str(Path(__file__).resolve().parent))
from .nodes import GeneratePointCloud, LoadMoGe3

NODE_CLASS_MAPPINGS = {
    "Image2PointLoadMoGe3": LoadMoGe3,
    "Image2PointGeneratePointCloud": GeneratePointCloud,
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "Image2PointLoadMoGe3": "Image2Point · Load MoGe-3 Model",
    "Image2PointGeneratePointCloud": "Image2Point · Generate Point Cloud",
}
