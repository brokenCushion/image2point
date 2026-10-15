"""Studio farm command: hython farm_bake.py /shared/job_.../job.json"""

import json
import sys
from pathlib import Path

import hou


def run(job_path):
    path = Path(job_path).resolve()
    job = json.loads(path.read_text())
    hou.hda.installFile(str(path.parent / job["asset"]))
    hou.hipFile.load(str(path.parent / job["scene"]), suppress_save_prompt=True, ignore_load_warnings=False)
    node = hou.node(job["node"])
    if node is None:
        raise ValueError("Image2Point node missing from saved scene: " + job["node"])
    print(node.hdaModule().bake(node, job["publish_directory"]))


if __name__ == "__main__":
    run(sys.argv[1])
