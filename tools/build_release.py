"""Build the release ZIP dist/image2point-<version>.zip; no production inputs or weights are copied.

Release layout: comfyui_image2point/ (node + vendored image2point + MoGe source), houdini/,
sample/, examples/, INSTALL.md, USAGE.md, AUDIT.md, PACKAGE-FORMAT.md.
"""

import hashlib
import shutil
import subprocess
import tomllib
import zipfile
from pathlib import Path

from sample_shot import json_write, sample, workflow

ROOT = Path(__file__).resolve().parents[1]
MOGE_REVISION = "74fbce054ebed49800de42d0ad0e83495065719a"
MOGE_CHECKPOINT = "Ruicheng/moge-3-vitl@184008f877d7ad1ad4c2cd2182a9bd1f63d0e5be"
CORE_FILES = ("__init__.py", "point_package.py", "camera.py", "io.py", "moge3.py")
# Only what MoGe-3 inference imports; training, scripts, the web viewer and download helpers stay out.
# moge/model/v1.py is left out too: MoGe-1 is not used and its weight loading reaches the internet.
MOGE_FILES = (
    "moge/__init__.py",
    "moge/model",
    "moge/utils/__init__.py",
    "moge/utils/geometry_numpy.py",
    "moge/utils/geometry_torch.py",
    "moge/utils/tools.py",
)


def git(*args, cwd=ROOT):
    return subprocess.check_output(["git", "-C", str(cwd), *args], text=True).strip()


def build():
    version = tomllib.loads((ROOT / "pyproject.toml").read_text())["project"]["version"]
    target = ROOT / "dist" / ("image2point-" + version)
    target.mkdir(parents=True, exist_ok=False)
    ignore = shutil.ignore_patterns("__pycache__", "*.pyc", ".git", "checkpoints")

    node = target / "comfyui_image2point"
    shutil.copytree(ROOT / "integrations/comfyui", node, ignore=ignore)
    (node / "image2point").mkdir()
    for filename in CORE_FILES:
        shutil.copy2(ROOT / "src/image2point" / filename, node / "image2point" / filename)

    moge = ROOT / "cache/MoGe"
    if git("rev-parse", "HEAD", cwd=moge) != MOGE_REVISION:
        raise ValueError("Unexpected MoGe source revision in cache/MoGe")
    subprocess.run(["git", "-C", str(moge), "diff", "--exit-code", "HEAD"], check=True, stdout=subprocess.DEVNULL)
    vendor = node / "vendor/MoGe"
    for relative in MOGE_FILES:
        if (moge / relative).is_dir():
            shutil.copytree(moge / relative, vendor / relative, ignore=ignore)
        else:
            (vendor / relative).parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(moge / relative, vendor / relative)
    (vendor / "moge/model/v1.py").unlink()
    shutil.copy2(moge / "LICENSE", vendor / "LICENSE")
    (vendor / "SOURCE_REVISION").write_text(MOGE_REVISION)

    houdini = target / "houdini"
    shutil.copytree(ROOT / "integrations/houdini", houdini, ignore=ignore)
    shutil.copy2(ROOT / "src/image2point/point_package.py", houdini / "point_package.py")
    shutil.copy2(ROOT / "tests/houdini/smoke_test.py", houdini / "smoke_test.py")
    shutil.copy2(ROOT / "LICENSE", target / "LICENSE")
    shutil.copy2(ROOT / "LICENSE", node / "LICENSE")
    shutil.copy2(ROOT / "docs/install.md", target / "INSTALL.md")
    shutil.copy2(ROOT / "docs/usage.md", target / "USAGE.md")
    shutil.copy2(ROOT / "docs/audit.md", target / "AUDIT.md")
    shutil.copy2(ROOT / "docs/package-format.md", target / "PACKAGE-FORMAT.md")
    sample(target / "sample")
    workflow(target / "examples")

    json_write(
        target / "SOURCE-VERSIONS.json",
        dict(
            model_sources={"MoGe": MOGE_REVISION},
            moge_checkpoint=MOGE_CHECKPOINT,
            source_repository_commit=git("rev-parse", "HEAD"),
            includes_uncommitted_changes=bool(git("status", "--porcelain")),
        ),
    )
    hashes = {
        p.relative_to(target).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
        for p in target.rglob("*")
        if p.is_file()
    }
    json_write(target / "SHA256.json", hashes)

    archive = target.parent / (target.name + ".zip")
    with zipfile.ZipFile(archive, "w", zipfile.ZIP_DEFLATED) as out:
        for file in target.rglob("*"):
            if file.is_file():
                out.write(file, (Path("image2point") / file.relative_to(target)).as_posix())
    print(archive)
    return target


if __name__ == "__main__":
    build()
