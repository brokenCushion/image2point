"""File digests and binary PLY publication; no model dependencies."""

import hashlib

import numpy as np

from .point_package import atomic


def digest(path):
    with open(path, "rb") as f:
        return hashlib.file_digest(f, "sha256").hexdigest()


def write_ply(path, points, colors, confidence, *, coordinate_system="Z-up meters", attributes=None):
    count = len(points)
    attributes = attributes or {}
    extra = []
    for name, values in attributes.items():
        if not name.isidentifier() or name in ("x", "y", "z", "red", "green", "blue", "confidence", "xyz", "rgb"):
            raise ValueError("Invalid or reserved PLY attribute: " + name)
        values = np.asarray(values)
        if values.shape != (count,) or values.dtype.kind not in "ifu" or not np.isfinite(values).all():
            raise ValueError("PLY attributes must be finite scalar arrays matching the points")
        extra.append((name, "<i4" if values.dtype.kind in "iu" else "<f4"))
    records = np.empty(count, dtype=[("xyz", "<f4", (3,)), ("rgb", "u1", (3,)), ("confidence", "<f4")] + extra)
    records["xyz"], records["rgb"], records["confidence"] = points, colors, confidence
    for name in attributes:
        records[name] = attributes[name]
    header = (
        f"ply\nformat binary_little_endian 1.0\ncomment image2point {coordinate_system}\n"
        f"element vertex {count}\nproperty float x\nproperty float y\nproperty float z\n"
        "property uchar red\nproperty uchar green\nproperty uchar blue\nproperty float confidence\n"
        + "".join(f"property {'int' if dtype == '<i4' else 'float'} {name}\n" for name, dtype in extra)
        + "end_header\n"
    )
    atomic(path, lambda f: (f.write(header.encode("ascii")), f.write(records.tobytes())))
