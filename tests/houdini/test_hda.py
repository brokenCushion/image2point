"""Houdini HDA smoke test via hython; skipped when Houdini is not installed."""

import glob
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "tools"))
from sample_shot import sample  # noqa: E402


def hython():
    if os.environ.get("HFS"):
        candidate = Path(os.environ["HFS"]) / "bin" / ("hython.exe" if os.name == "nt" else "hython")
        if candidate.is_file():
            return str(candidate)
    installed = sorted(glob.glob(r"C:\Program Files\Side Effects Software\Houdini *\bin\hython.exe"))
    return installed[0] if installed else shutil.which("hython")


@pytest.mark.houdini
@pytest.mark.skipif(hython() is None, reason="Houdini (hython) not installed")
def test_hda_build_cook_and_bake(tmp_path):
    manifest = sample(tmp_path / "sample")
    result = subprocess.run(
        [hython(), str(Path(__file__).with_name("smoke_test.py")), str(manifest)],
        capture_output=True,
        text=True,
        timeout=600,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert "PASS:" in result.stdout
