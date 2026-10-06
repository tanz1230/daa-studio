"""The label model: one vehicle track = one shared shape + one pose per frame.

Vehicles are rigid, so length, width and height are shared by every frame of a track (DAA itself
shares them); only the pose (x, y, z, yaw) varies per frame. Every edit operation here is a pure
function: it returns a new Track and never mutates its input, which keeps undo, history and
optimistic saving simple and safe.
"""
from __future__ import annotations

import copy
import math
from dataclasses import asdict, dataclass, field

STATUSES = ("unreviewed", "accepted", "edited", "flagged", "deleted", "new")
CLASSES = ("Car", "Truck")
_AXIS = {"L": 0, "W": 1, "H": 2}


def wrap(a: float) -> float:
    """Angle -> (-pi, pi]."""
    return math.atan2(math.sin(a), math.cos(a))


@dataclass
class Track:
    tid: int
    cls: str = "Car"
    shape: list = field(default_factory=lambda: [4.5, 1.8, 1.5])      # L, W, H (m)
    poses: dict = field(default_factory=dict)                          # frame -> [x, y, z, yaw]
    status: str = "unreviewed"
    flags: dict = field(default_factory=dict)                          # frame -> ["edited", "keyframe"]
    note: str = ""
    seed: dict = field(default_factory=dict)                           # {"kind", "run", "conf", "diag"}
    version: int = 0
    updated: str = ""
    updated_by: str = ""

    # ------------------------------------------------------------------ access
    def frames(self) -> list:
        return sorted(self.poses)

    def box(self, frame: int) -> list:
        x, y, z, yaw = self.poses[frame]
        L, W, H = self.shape
        return [x, y, z, L, W, H, yaw]

    def nearest_frame(self, frame: int) -> int:
        return min(self.poses, key=lambda f: (abs(f - frame), f))

    # ------------------------------------------------------------------ JSON
    def to_json(self) -> dict:
        d = asdict(self)
        d["poses"] = {str(f): [round(float(v), 5) for v in p] for f, p in sorted(self.poses.items())}
        d["flags"] = {str(f): sorted(set(v)) for f, v in sorted(self.flags.items()) if v}
        d["shape"] = [round(float(v), 5) for v in self.shape]
        if "conf" in d["seed"]:
            d["seed"] = dict(d["seed"], conf={str(f): round(float(c), 4)
                                              for f, c in sorted(d["seed"]["conf"].items())})
        return d

    @classmethod
    def from_json(cls, d: dict) -> "Track":
        d = dict(d)
        d["poses"] = {int(f): [float(v) for v in p] for f, p in d.get("poses", {}).items()}
        d["flags"] = {int(f): list(v) for f, v in d.get("flags", {}).items()}
        seed = dict(d.get("seed") or {})
        if "conf" in seed:
            seed["conf"] = {int(f): float(c) for f, c in seed["conf"].items()}
        d["seed"] = seed
        known = {k: d[k] for k in cls.__dataclass_fields__ if k in d}
        return cls(**known)

    def summary(self) -> dict:
        fr = self.frames()
        conf = self.seed.get("conf") or {}
        return {"tid": self.tid, "cls": self.cls, "status": self.status, "version": self.version,
                "n_frames": len(fr), "first": fr[0] if fr else None, "last": fr[-1] if fr else None,
                "conf": round(sum(conf.values()) / len(conf), 3) if conf else None,
                "shape": [round(v, 2) for v in self.shape], "seed": self.seed.get("kind"),
                "diag": self.seed.get("diag", {}), "note": self.note,
                "edited_frames": sum(1 for v in self.flags.values() if "edited" in v)}


# ---------------------------------------------------------------------------- helpers
def _clone(t: Track) -> Track:
    return copy.deepcopy(t)


def _flag(t: Track, frame: int, flag: str, on: bool = True) -> None:
    cur = set(t.flags.get(frame, []))
    cur.add(flag) if on else cur.discard(flag)
    if cur:
        t.flags[frame] = sorted(cur)
    else:
        t.flags.pop(frame, None)


def _frames(t: Track, frames) -> list:
    return t.frames() if frames is None else [f for f in frames if f in t.poses]


def touched(t: Track) -> Track:
    """A geometric edit moves a reviewed/unreviewed track to 'edited' (new tracks stay 'new')."""
    if t.status in ("unreviewed", "accepted"):
        t.status = "edited"
    return t


# ---------------------------------------------------------------------------- shape
def resize(t: Track, axis: str, size: float, side: int = 0, min_size: float = 0.2) -> Track:
    """Set one shared dimension. side=+1 keeps the NEGATIVE face fixed (the positive face moves),
    side=-1 keeps the positive face fixed, side=0 resizes about the centre. Applied to every frame,
    each along its own heading."""
    i = _AXIS[axis]
    out = _clone(t)
    size = max(float(size), min_size)
    delta = size - t.shape[i]
    out.shape[i] = size
    if side:
        for f, (x, y, z, yaw) in t.poses.items():
            s = side * delta / 2
            if i == 0:
                x, y = x + s * math.cos(yaw), y + s * math.sin(yaw)
            elif i == 1:
                x, y = x - s * math.sin(yaw), y + s * math.cos(yaw)
            else:
                z = z + s
            out.poses[f] = [x, y, z, yaw]
    return touched(out)


def set_shape(t: Track, L=None, W=None, H=None) -> Track:
    out = t
    for axis, v in (("L", L), ("W", W), ("H", H)):
        if v is not None:
            out = resize(out, axis, v, side=0)
    return out


# ---------------------------------------------------------------------------- pose
def set_pose(t: Track, frame: int, x=None, y=None, z=None, yaw=None) -> Track:
    """Edit one frame's pose. A frame the track does not have yet is created from the nearest one."""
    out = _clone(t)
    base = list(t.poses[frame] if frame in t.poses else t.poses[t.nearest_frame(frame)])
    for i, v in enumerate((x, y, z, yaw)):
        if v is not None:
            base[i] = float(v)
    base[3] = wrap(base[3])
    out.poses[frame] = base
    _flag(out, frame, "edited")
    return touched(out)


def move(t: Track, frames=None, dx=0.0, dy=0.0, dz=0.0) -> Track:
    out = _clone(t)
    for f in _frames(t, frames):
        x, y, z, yaw = t.poses[f]
        out.poses[f] = [x + dx, y + dy, z + dz, yaw]
        if frames is not None:
            _flag(out, f, "edited")
    return touched(out)


def rotate(t: Track, frames=None, dyaw=0.0) -> Track:
    out = _clone(t)
    for f in _frames(t, frames):
        x, y, z, yaw = t.poses[f]
        out.poses[f] = [x, y, z, wrap(yaw + dyaw)]
        if frames is not None:
            _flag(out, f, "edited")
    return touched(out)


def flip(t: Track, frames=None) -> Track:
    """Turn the heading by 180 deg (a box looks the same, the direction of travel does not)."""
    return rotate(t, frames, math.pi)


def interpolate(t: Track, f0: int, f1: int, fill=()) -> Track:
    """Linear centre and heading between keyframes f0 and f1 for every frame strictly between.
    The heading takes the short way round modulo pi (box axis), so a flipped endpoint never makes
    the box spin. Frames in `fill` (e.g. the session's frames) that the track lacks are created."""
    if f0 > f1:
        f0, f1 = f1, f0
    if f0 not in t.poses or f1 not in t.poses:
        raise KeyError("both keyframes must belong to the track")
    out = _clone(t)
    x0, y0, z0, a0 = t.poses[f0]
    x1, y1, z1, a1 = t.poses[f1]
    d = wrap(a1 - a0)
    if d > math.pi / 2:
        d -= math.pi
    elif d < -math.pi / 2:
        d += math.pi
    for f in sorted(set(t.frames()) | {int(g) for g in fill}):
        if f0 < f < f1:
            w = (f - f0) / (f1 - f0)
            out.poses[f] = [x0 + w * (x1 - x0), y0 + w * (y1 - y0), z0 + w * (z1 - z0), wrap(a0 + w * d)]
            _flag(out, f, "edited")
    return touched(out)


def fill_gaps(t: Track, frames) -> Track:
    """Create the frames of `frames` that fall inside the track's gaps, interpolated."""
    out, fr = t, t.frames()
    for a, b in zip(fr, fr[1:]):
        if any(a < f < b for f in frames):
            out = interpolate(out, a, b, fill=[f for f in frames if a < f < b])
    return out


def copy_to(t: Track, src: int, frames) -> Track:
    """Copy the box at `src` into every frame of `frames` the track lacks (a parked vehicle)."""
    out = _clone(t)
    for f in frames:
        if f not in out.poses:
            out.poses[f] = list(t.poses[src])
            _flag(out, f, "edited")
    return touched(out)


def remove_frame(t: Track, frame: int) -> Track:
    out = _clone(t)
    if len(out.poses) > 1:
        out.poses.pop(frame, None)
        out.flags.pop(frame, None)
    return touched(out)


# ---------------------------------------------------------------------------- review
def mark(t: Track, frame: int, flag: str, on: bool = True) -> Track:
    out = _clone(t)
    _flag(out, frame, flag, on)
    return out


def set_status(t: Track, status: str, note: str | None = None) -> Track:
    if status not in STATUSES:
        raise ValueError(f"unknown status {status!r}")
    out = _clone(t)
    out.status = status
    if note is not None:
        out.note = note
    return out


def set_class(t: Track, cls: str) -> Track:
    out = _clone(t)
    out.cls = cls
    return touched(out)
