"""UAV camera model from the release's camera_<agent>.json.

OpenCV pinhole (x right, y down, z forward) with (k1, k2, p1, p2, k3) distortion and a per-frame
4x4 map -> camera extrinsic. Projecting full 3-D box corners (not a ground homography) means the
relief displacement of everything above the ground comes out of the geometry, so a box drawn on
the drone image sits on the vehicle as the drone actually sees it.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from .poses import box_corners


class Camera:
    def __init__(self, K, dist, frames: dict, size, ground_plane=None, name=""):
        self.K = np.asarray(K, float)
        self.dist = np.zeros(5)
        d = np.asarray(dist, float).ravel()
        self.dist[: len(d)] = d[:5]
        self.size = (int(size[0]), int(size[1]))          # (width, height)
        self.ground_plane = ground_plane
        self.name = name
        self._frames = frames                              # {int frame: {"E": 4x4, "H": 3x3, "pos": 3}}

    @classmethod
    def from_json(cls, path):
        d = json.loads(Path(path).read_text())
        frames = {}
        for key, f in d["frames"].items():
            frames[int(key)] = {"E": np.asarray(f["extrinsic"], float),
                                "H": np.asarray(f["homography_ground"], float),
                                "pos": np.asarray(f["position"], float)}
        return cls(d["intrinsic"], d["distortion"], frames, d["image_size"],
                   d.get("ground_plane"), d.get("sensor", ""))

    def has(self, frame: int) -> bool:
        return int(frame) in self._frames

    def position(self, frame: int) -> np.ndarray:
        return self._frames[int(frame)]["pos"]

    # ---------------------------------------------------------------- projection
    def project(self, frame: int, pts, distort: bool = True):
        """Map-frame points (N, 3) -> pixels (N, 2) and depth along the optical axis (N,)."""
        E = self._frames[int(frame)]["E"]
        p = np.asarray(pts, float).reshape(-1, 3)
        cam = p @ E[:3, :3].T + E[:3, 3]
        z = cam[:, 2]
        x = cam[:, 0] / z
        y = cam[:, 1] / z
        if distort:
            k1, k2, p1, p2, k3 = self.dist
            r2 = x * x + y * y
            radial = 1 + k1 * r2 + k2 * r2 * r2 + k3 * r2 * r2 * r2
            x, y = (x * radial + 2 * p1 * x * y + p2 * (r2 + 2 * x * x),
                    y * radial + p1 * (r2 + 2 * y * y) + 2 * p2 * x * y)
        u = self.K[0, 0] * x + self.K[0, 2]
        v = self.K[1, 1] * y + self.K[1, 2]
        return np.column_stack([u, v]), z

    def ground_from_pixel(self, frame: int, uv_undistorted) -> np.ndarray:
        """Undistorted pixels (N, 2) -> map (x, y, z) on the take's ground plane."""
        H = self._frames[int(frame)]["H"]
        uv = np.asarray(uv_undistorted, float).reshape(-1, 2)
        q = np.column_stack([uv, np.ones(len(uv))]) @ H.T
        xy = q[:, :2] / q[:, 2:3]
        gp = self.ground_plane or {"z0": 0.0, "gx": 0.0, "gy": 0.0}
        z = gp["z0"] + gp["gx"] * xy[:, 0] + gp["gy"] * xy[:, 1]
        return np.column_stack([xy, z])

    def pixel_to_map(self, frame: int, uv, height: float = 0.0) -> np.ndarray:
        """Image pixels (N, 2), as seen (distorted) -> the map points where their rays meet the take's
        ground plane raised by `height` metres. Inverse of `project` for points at that height, so a
        click on a vehicle's middle (height ~ H/2) lands on its centre, relief displacement included."""
        import cv2
        E = self._frames[int(frame)]["E"]
        R, t = E[:3, :3], E[:3, 3]
        uv = np.asarray(uv, float).reshape(-1, 1, 2)
        n = cv2.undistortPoints(uv, self.K, self.dist).reshape(-1, 2)      # normalised image coords
        d = np.column_stack([n, np.ones(len(n))]) @ R                       # ray directions (map)
        C = -R.T @ t                                                        # camera centre (map)
        gp = self.ground_plane or {"z0": 0.0, "gx": 0.0, "gy": 0.0}
        num = gp["z0"] + gp["gx"] * C[0] + gp["gy"] * C[1] + height - C[2]
        den = d[:, 2] - gp["gx"] * d[:, 0] - gp["gy"] * d[:, 1]
        return C + (num / den)[:, None] * d

    def box_corners_px(self, frame: int, box7) -> np.ndarray:
        """(8, 2) distorted pixel corners of a map-frame box (bottom face, then top face)."""
        uv, _ = self.project(frame, box_corners(box7))
        return uv

    def crop_window(self, frame: int, box7, pad: float = 1.6, min_px: int = 360, aspect: float = 1.0):
        """Pixel window (x0, y0, w, h) around a box with w / h = aspect, kept inside the image.

        The window is `pad` times the box's projected extent (at least `min_px` tall); a window
        larger than the image shrinks to fit, keeping the aspect."""
        W, H = self.size
        uv = self.box_corners_px(frame, box7)
        cx, cy = uv.mean(axis=0)
        ext = np.ptp(uv, axis=0)
        h = max(min_px, pad * float(ext[1]), pad * float(ext[0]) / aspect)
        h = min(h, H, W / aspect)
        w = h * aspect
        x0 = float(np.clip(cx - w / 2, 0, W - w))
        y0 = float(np.clip(cy - h / 2, 0, H - h))
        return int(round(x0)), int(round(y0)), int(round(w)), int(round(h))
