"""Real MoGe-3 inference through the ComfyUI node; skipped without CUDA or local weights."""

import pytest
import torch
from comfy_host import CHECKPOINT, REPOSITORY, exercise


@pytest.mark.gpu
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA GPU required")
@pytest.mark.skipif(
    not CHECKPOINT.is_file() or not (REPOSITORY / "cache/MoGe").is_dir(),
    reason="MoGe source and checkpoint required under cache/MoGe",
)
def test_real_moge3_through_node():
    assert exercise(gpu=True) == [64, 48]
