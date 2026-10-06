"""Minimal, dependency-free PCD reader (no Open3D).

Handles the OpenCOOD PCD variant used here -- `DATA binary`, FIELDS `x y z intensity`,
TYPE F, SIZE 4 (little-endian float32) -- and the ascii and binary_compressed cases as
fallbacks. Returns just the XYZ (N,3) float32 points; intensity is ignored.

We parse the ASCII header generically (FIELDS/SIZE/TYPE/COUNT/POINTS/DATA) so a slightly
different field order still works: we locate the x,y,z fields by name and slice them out of
each point record. Non-finite points are dropped.
"""
from __future__ import annotations

import struct
from pathlib import Path

import numpy as np

_NP_TYPE = {("F", 4): np.float32, ("F", 8): np.float64,
            ("U", 1): np.uint8, ("U", 2): np.uint16, ("U", 4): np.uint32,
            ("I", 1): np.int8, ("I", 2): np.int16, ("I", 4): np.int32}


def _parse_header(f):
    """Read ASCII header lines from an open binary file; return (dict, data_offset)."""
    hdr = {}
    while True:
        line = f.readline()
        if not line:
            raise ValueError("PCD: reached EOF before DATA")
        text = line.decode("ascii", "replace").strip()
        if not text or text.startswith("#"):
            continue
        key, _, rest = text.partition(" ")
        key = key.upper()
        hdr[key] = rest.strip()
        if key == "DATA":
            return hdr, f.tell()


def read_pcd_xyz(path) -> np.ndarray:
    """Load a .pcd file -> (N,3) float32 XYZ, finite points only."""
    path = Path(path)
    with open(path, "rb") as f:
        hdr, off = _parse_header(f)
        fields = hdr.get("FIELDS", "x y z").split()
        sizes = [int(s) for s in hdr.get("SIZE", "4 4 4").split()]
        types = hdr.get("TYPE", "F F F").split()
        counts = [int(c) for c in hdr.get("COUNT", " ".join(["1"] * len(fields))).split()]
        npoints = int(hdr.get("POINTS", hdr.get("WIDTH", "0")))
        data = hdr.get("DATA", "ascii").lower()
        try:
            ix, iy, iz = fields.index("x"), fields.index("y"), fields.index("z")
        except ValueError:
            ix, iy, iz = 0, 1, 2

        if data == "ascii":
            arr = np.loadtxt(path, skiprows=_ascii_header_lines(path))
            arr = np.atleast_2d(arr)
            pts = arr[:, [ix, iy, iz]].astype(np.float32)
        else:
            # binary: one record per point = sum(size*count) bytes, fields in order
            rec_dtype = []
            for name, s, t, c in zip(fields, sizes, types, counts):
                base = _NP_TYPE.get((t.upper(), s), np.float32)
                rec_dtype.append((name, base, (c,) if c > 1 else ()))
            if data == "binary_compressed":
                pts = _read_binary_compressed(f, off, fields, sizes, types, counts,
                                              npoints, ix, iy, iz)
            else:
                rec = np.dtype(rec_dtype)
                f.seek(off)
                buf = f.read(rec.itemsize * npoints)
                a = np.frombuffer(buf, dtype=rec, count=npoints)
                pts = np.column_stack([a[fields[ix]], a[fields[iy]], a[fields[iz]]]).astype(np.float32)

    pts = pts[np.all(np.isfinite(pts), axis=1)]
    return np.ascontiguousarray(pts)


def _ascii_header_lines(path):
    n = 0
    with open(path, "rb") as f:
        for line in f:
            n += 1
            if line.decode("ascii", "replace").strip().upper().startswith("DATA"):
                return n
    return n


def _read_binary_compressed(f, off, fields, sizes, types, counts, npoints, ix, iy, iz):
    """LZF-compressed, struct-of-arrays PCD (rare). Uses python-lzf if available."""
    import lzf  # optional; only needed for binary_compressed PCDs
    f.seek(off)
    comp_size, uncomp_size = struct.unpack("<II", f.read(8))
    raw = lzf.decompress(f.read(comp_size), uncomp_size)
    out = {}
    p = 0
    for name, s, t, c in zip(fields, sizes, types, counts):
        base = _NP_TYPE.get((t.upper(), s), np.float32)
        w = s * c * npoints
        out[name] = np.frombuffer(raw[p:p + w], dtype=base).reshape(npoints, c)[:, 0]
        p += w
    return np.column_stack([out[fields[ix]], out[fields[iy]], out[fields[iz]]]).astype(np.float32)
