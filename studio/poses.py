"""OpenCOOD pose conventions, YAML I/O and box conversions.

A session folder holds per-agent `<frame:06d>.yaml` files whose `lidar_pose` is
`[x, y, z, roll, yaw, pitch]` (degrees, yaw in the MIDDLE) and whose `vehicles` block is
`{vid: {location, angle=[roll, yaw, pitch], center, extent=[L/2, W/2, H/2], obj_type}}` in the
shared map frame.

Two pose storages exist and are auto-detected per session:
  "opencood"  standard OpenCOOD for every agent: R = Rz(yaw) Ry(-pitch) Rx(-roll)
              (the released dataset, agi_coop/take_<N>)
  "legacy"    the original release: egos rebuilt through the exact ZYX quaternion they were
              written from; the roadside unit stores its rotation as scipy "xyz" [roll, yaw, pitch]

"""
from __future__ import annotations

import math
import os
from pathlib import Path

import numpy as np
import yaml

ROADSIDE_AGENT = 2
OPENCOOD, LEGACY = "opencood", "legacy"


# ----------------------------------------------------------------------------- rotations
def _quat_to_rot(qx, qy, qz, qw):
    n = math.sqrt(qx * qx + qy * qy + qz * qz + qw * qw) or 1.0
    qx, qy, qz, qw = qx / n, qy / n, qz / n, qw / n
    return np.array([
        [1 - 2 * (qy * qy + qz * qz), 2 * (qx * qy - qz * qw), 2 * (qx * qz + qy * qw)],
        [2 * (qx * qy + qz * qw), 1 - 2 * (qx * qx + qz * qz), 2 * (qy * qz - qx * qw)],
        [2 * (qx * qz - qy * qw), 2 * (qy * qz + qx * qw), 1 - 2 * (qx * qx + qy * qy)],
    ], float)


def _euler_zyx_to_quat(roll, pitch, yaw):
    """(roll, pitch, yaw) rad -> (qx, qy, qz, qw); exact inverse of the ZYX quat->euler that wrote the poses."""
    cy, sy = math.cos(yaw / 2), math.sin(yaw / 2)
    cp, sp = math.cos(pitch / 2), math.sin(pitch / 2)
    cr, sr = math.cos(roll / 2), math.sin(roll / 2)
    return (sr * cp * cy - cr * sp * sy, cr * sp * cy + sr * cp * sy,
            cr * cp * sy - sr * sp * cy, cr * cp * cy + sr * sp * sy)


def _rot_xyz_extrinsic(rx, ry, rz):
    """scipy Rotation.from_euler('xyz', [rx, ry, rz]).as_matrix(), i.e. R = Rz @ Ry @ Rx."""
    cx, sx = math.cos(rx), math.sin(rx)
    cy, sy = math.cos(ry), math.sin(ry)
    cz, sz = math.cos(rz), math.sin(rz)
    Rx = np.array([[1, 0, 0], [0, cx, -sx], [0, sx, cx]])
    Ry = np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]])
    Rz = np.array([[cz, -sz, 0], [sz, cz, 0], [0, 0, 1]])
    return Rz @ Ry @ Rx


def pose_R_t(agent: int, lidar_pose, convention: str):
    """`lidar_pose` -> (R (3,3), t (3,)) mapping sensor coordinates into the map frame."""
    x, y, z, roll, yaw, pitch = (float(v) for v in lidar_pose)
    t = np.array([x, y, z], float)
    if convention == OPENCOOD:
        return _rot_xyz_extrinsic(-math.radians(roll), -math.radians(pitch), math.radians(yaw)), t
    if agent == ROADSIDE_AGENT:
        return _rot_xyz_extrinsic(math.radians(roll), math.radians(yaw), math.radians(pitch)), t
    q = _euler_zyx_to_quat(math.radians(roll), math.radians(pitch), math.radians(yaw))
    return _quat_to_rot(*q), t


def detect_convention(session_dir, n: int = 12):
    """-> OPENCOOD | LEGACY | None. The roadside unit is mounted nearly level and faces ~136 deg, so
    its yaw sits at index 4 in the standard storage and at index 5 in the legacy one."""
    ys = sorted((Path(session_dir) / str(ROADSIDE_AGENT)).glob("*.yaml"))
    if not ys:
        return OPENCOOD                       # no roadside agent: the egos read the same either way
    votes = set()
    for yp in ys[:: max(1, len(ys) // n)][:n]:
        p = load_yaml(yp).get("lidar_pose")
        if not p:
            continue
        a4, a5 = abs(float(p[4])), abs(float(p[5]))
        if a4 > 20 and a5 < 15:
            votes.add(OPENCOOD)
        elif a5 > 20 and a4 < 15:
            votes.add(LEGACY)
        else:
            votes.add(None)
    return votes.pop() if len(votes) == 1 else None


# ----------------------------------------------------------------------------- YAML
try:
    _LOADER = yaml.CSafeLoader                # libyaml: ~10x faster
except AttributeError:                        # pragma: no cover - depends on the PyYAML build
    _LOADER = yaml.SafeLoader


def load_yaml(path) -> dict:
    with open(path) as f:
        return yaml.load(f, Loader=_LOADER) or {}


def save_yaml(path, doc) -> None:
    """Atomic write: a reader never sees a half-written file."""
    tmp = f"{path}.tmp"
    with open(tmp, "w") as f:
        yaml.safe_dump(doc, f, default_flow_style=None, sort_keys=False)
    os.replace(tmp, path)


# ----------------------------------------------------------------------------- boxes
def box_from_vehicle(v) -> tuple:
    """OpenCOOD vehicle dict -> box7 (cx, cy, cz, L, W, H, yaw_rad)."""
    cx, cy, cz = (float(a) for a in v["location"])
    hl, hw, hh = (float(a) for a in v["extent"])
    return (cx, cy, cz, 2 * hl, 2 * hw, 2 * hh, math.radians(float(v["angle"][1])))


def vehicle_from_box(box7, obj_type: str = "Car") -> dict:
    """box7 -> OpenCOOD vehicle dict in the release schema."""
    cx, cy, cz, L, W, H, yaw = (float(v) for v in box7)
    return {
        "location": [round(cx, 4), round(cy, 4), round(cz, 4)],
        "angle": [0.0, round(math.degrees(yaw), 4), 0.0],
        "center": [0.0, 0.0, 0.0],
        "extent": [round(L / 2, 4), round(W / 2, 4), round(H / 2, 4)],
        "obj_type": obj_type,
    }


def box_corners(box7) -> np.ndarray:
    """box7 -> (8, 3) corners: bottom face first (counter-clockwise), then the top face."""
    cx, cy, cz, L, W, H, yaw = (float(v) for v in box7)
    c, s = math.cos(yaw), math.sin(yaw)
    xs = np.array([1, -1, -1, 1]) * L / 2
    ys = np.array([1, 1, -1, -1]) * W / 2
    out = []
    for dz in (-H / 2, H / 2):
        for x, y in zip(xs, ys):
            out.append((cx + c * x - s * y, cy + s * x + c * y, cz + dz))
    return np.array(out)


def to_box_frame(pts, box7) -> np.ndarray:
    """Map-frame points -> the box's own frame (x along the heading, origin at the centre)."""
    cx, cy, cz, _L, _W, _H, yaw = (float(v) for v in box7)
    c, s = math.cos(yaw), math.sin(yaw)
    d = np.asarray(pts, np.float64)[:, :3] - (cx, cy, cz)
    return np.column_stack([c * d[:, 0] + s * d[:, 1], -s * d[:, 0] + c * d[:, 1], d[:, 2]])
