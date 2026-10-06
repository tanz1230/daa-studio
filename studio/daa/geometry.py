"""Geometry primitives. Self-contained; no imports from other refiner versions.

Canonical frame: origin = box centre, x = heading (yaw), y = left, z = world-up.
Box format everywhere in this package: [cx, cy, cz, L, W, H, yaw].
"""
from __future__ import annotations

import numpy as np


def to_canonical(points: np.ndarray, box) -> np.ndarray:
    """World/local (N,3+) -> canonical (N,3): z = R(yaw)^T (p - centre)."""
    p = np.asarray(points, np.float64)
    cx, cy, cz, yaw = float(box[0]), float(box[1]), float(box[2]), float(box[6])
    c, s = np.cos(yaw), np.sin(yaw)
    dx = p[:, 0] - cx
    dy = p[:, 1] - cy
    dz = p[:, 2] - cz
    xc = dx * c + dy * s          # heading axis
    yc = -dx * s + dy * c         # left axis
    return np.stack([xc, yc, dz], axis=1)


def canonical_xy_to_world(dxy_can: np.ndarray, yaw: float) -> np.ndarray:
    """Rotate a canonical xy vector back to world: world = R(yaw) @ can."""
    c, s = np.cos(yaw), np.sin(yaw)
    x, y = float(dxy_can[0]), float(dxy_can[1])
    return np.array([c * x - s * y, s * x + c * y])


def weighted_quantile(values: np.ndarray, weights: np.ndarray, q: float) -> float:
    """Weighted quantile (Hazen midpoint). Robust to zero/neg weights + empty input."""
    v = np.asarray(values, np.float64)
    w = np.clip(np.asarray(weights, np.float64), 0.0, None)
    if v.size == 0:
        return 0.0
    if w.sum() <= 0:
        return float(np.quantile(v, q))
    order = np.argsort(v)
    v = v[order]
    w = w[order]
    cw = np.cumsum(w) - 0.5 * w
    cw /= w.sum()
    return float(np.interp(q, cw, v))


def outside_distance(z_can: np.ndarray, half: np.ndarray) -> np.ndarray:
    """Per-point Euclidean distance OUTSIDE the box (0 inside). z_can (N,3), half (3,)."""
    out = np.maximum(np.abs(z_can[:, :3]) - half[None, :], 0.0)
    return np.linalg.norm(out, axis=1)


def make_box(uav_box, offset_xy, dims, ground_z=None) -> np.ndarray:
    """Compose a box from a UAV seed + xy offset + track dims (yaw kept from UAV)."""
    b = np.asarray(uav_box, np.float64).copy()
    b[0] += float(offset_xy[0])
    b[1] += float(offset_xy[1])
    b[3:6] = dims
    if ground_z is not None:
        b[2] = float(ground_z) + float(dims[2]) / 2.0
    return b
