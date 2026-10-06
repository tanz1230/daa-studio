"""Ground removal from the field's Digital Terrain Model (DTM).

The DTM is a 1 m UTM elevation grid built from the UAV map (dtm.npz), with the 4x4 UTM -> map
transform (T_utm_to_local.txt). The two files are not part of this repository: they are looked up
in $DAA_STUDIO_DTM_DIR, the session folder, the dataset root, then studio/assets/ (see
find_dtm). A map-frame point maps to UTM with one matmul, the ground elevation is
looked up in its cell, and the difference is the point's height above ground. Working in UTM
keeps "up" truly vertical: the LiDAR map frame itself is tilted ~2 deg, so a flat z-cut would
leave sloped road behind.

Plain numpy, no scipy.
"""
from __future__ import annotations

import os
from pathlib import Path

import numpy as np

GROUND_HAG = 0.2            # points at most this far above the DTM count as ground
_ASSETS = Path(__file__).resolve().parent / "assets"
DTM_FILES = ("dtm.npz", "T_utm_to_local.txt")


def find_dtm(*folders):
    """The first folder holding both DTM files: $DAA_STUDIO_DTM_DIR, then `folders` (e.g. the
    session and the dataset root), then studio/assets/. None when there is none."""
    for d in (os.environ.get("DAA_STUDIO_DTM_DIR"), *folders, _ASSETS):
        if d and all((Path(d) / f).is_file() for f in DTM_FILES):
            return Path(d)
    return None


class GroundModel:
    """Height above the DTM ground; `ok` is False when no terrain model is found."""

    def __init__(self, dtm_npz=None, transform_txt=None, folders=()):
        self.ok = False
        self.source = None
        if dtm_npz is None or transform_txt is None:
            where = find_dtm(*folders)
            if where is None:
                return
            dtm_npz, transform_txt = where / DTM_FILES[0], where / DTM_FILES[1]
        dtm_npz, transform_txt = Path(dtm_npz), Path(transform_txt)
        if not (dtm_npz.exists() and transform_txt.exists()):
            return
        self.source = str(dtm_npz.parent)
        d = np.load(dtm_npz)
        self.grid = d["grid"].astype(np.float32)          # (ny, nx), NaN = no data
        self.x_min, self.y_min = float(d["x_min"]), float(d["y_min"])
        self.cell = float(d["cell_size"])
        self.ny, self.nx = self.grid.shape
        T = np.loadtxt(transform_txt)                     # UTM -> map
        if T.shape != (4, 4):
            return
        self.T_map_to_utm = np.linalg.inv(T)
        # the UTM vertical expressed in the map frame (unit vector; the map is slightly tilted)
        up = T[:3, :3] @ np.array([0.0, 0.0, 1.0])
        self.up = up / np.linalg.norm(up)
        self.ok = True

    def height_above(self, map_pts) -> np.ndarray:
        """(N, >=3) map-frame points -> (N,) metres above the DTM; NaN off the DTM footprint."""
        p = np.asarray(map_pts, np.float64)
        if len(p) == 0:
            return np.zeros(0)
        utm = np.column_stack([p[:, :3], np.ones(len(p))]) @ self.T_map_to_utm.T
        ix = np.floor((utm[:, 0] - self.x_min) / self.cell).astype(np.int64)
        iy = np.floor((utm[:, 1] - self.y_min) / self.cell).astype(np.int64)
        on = (ix >= 0) & (ix < self.nx) & (iy >= 0) & (iy < self.ny)
        gz = np.full(len(p), np.nan)
        gz[on] = self.grid[iy[on], ix[on]]
        return utm[:, 2] - gz

    def keep_mask(self, map_pts, hag: float = GROUND_HAG) -> np.ndarray:
        """True for points above the ground. Points off the DTM are kept (never silently lost)."""
        h = self.height_above(map_pts)
        return ~(np.isfinite(h) & (h <= hag))

    def ground_point(self, map_xyz) -> np.ndarray:
        """Drop a map-frame point vertically onto the DTM surface (for new boxes' default z)."""
        p = np.asarray(map_xyz, np.float64).reshape(1, 3)
        h = float(self.height_above(p)[0])
        return p[0] if not np.isfinite(h) else p[0] - h * self.up
