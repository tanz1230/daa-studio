"""UAV detections (Stage-1 boxes in the LiDAR map frame) -> per-track arrays for DAA.

Input is a label CSV in the DAA schema (frame_idx, track_id, cx, cy, cz, L, W, H, yaw[, conf]).
Rows are matched to the session's frames by frame id and the mismatch is reported, never silently
dropped. Two things DAA needs are not in the CSV and are derived here:
  * the vehicle class: DAA only distinguishes car (0) from non-car; the UAV prior's default height
    encodes it (cars 1.5-1.55 m, large classes 3.2-3.5 m), so a 2.0 m threshold is exact;
  * the UAV velocity (DAA's yaw prior): central differences of the track's own positions.
"""
from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field

import numpy as np

from ..importers import read_label_csv

FPS = 10.0                 # LiDAR / OpenCOOD frame rate of the release
LARGE_H = 2.0              # prior height at or above which the UAV called it a large vehicle


@dataclass
class TrackDets:
    tid: int
    frames: np.ndarray        # (T,) int, sorted
    boxes: np.ndarray         # (T, 7) cx, cy, cz, L, W, H, yaw (map frame)
    conf: np.ndarray          # (T,)
    veh_class: int            # 0 car, 1 large

    @property
    def cls_name(self) -> str:
        return "Car" if self.veh_class == 0 else "Truck"


@dataclass
class Detections:
    tracks: dict = field(default_factory=dict)        # tid -> TrackDets
    report: dict = field(default_factory=dict)

    def by_frame(self) -> dict:
        out = defaultdict(list)
        for t in self.tracks.values():
            for f, b in zip(t.frames, t.boxes):
                out[int(f)].append((t.tid, b))
        return out


def velocities(frames, xy, fps: float = FPS, half_window: int = 2) -> np.ndarray:
    """(T,) frame ids + (T, 2) positions -> (T, 2) m/s by central differences over +-half_window."""
    frames = np.asarray(frames, float)
    xy = np.asarray(xy, float)
    T = len(frames)
    v = np.zeros((T, 2))
    if T < 2:
        return v
    for i in range(T):
        j0, j1 = max(0, i - half_window), min(T - 1, i + half_window)
        dt = (frames[j1] - frames[j0]) / fps
        if dt > 0:
            v[i] = (xy[j1] - xy[j0]) / dt
    return v


def load_detections(path, session) -> Detections:
    rows = read_label_csv(path)
    session_frames = set(session.frames)
    per = defaultdict(dict)
    off_session = duplicates = 0
    for r in rows:
        f = int(r["frame_idx"])
        if f not in session_frames:
            off_session += 1
            continue
        tid = int(r["track_id"])
        if f in per[tid]:
            duplicates += 1
        box = [float(r[k]) for k in ("cx", "cy", "cz", "L", "W", "H", "yaw")]
        conf = float(r["conf"]) if r.get("conf") not in (None, "") else 1.0
        per[tid][f] = (box, conf)
    tracks = {}
    for tid, fr in per.items():
        frames = np.array(sorted(fr), dtype=np.int64)
        boxes = np.array([fr[f][0] for f in frames], float)
        conf = np.array([fr[f][1] for f in frames], float)
        veh_class = 0 if float(np.median(boxes[:, 5])) < LARGE_H else 1
        tracks[tid] = TrackDets(tid, frames, boxes, conf, veh_class)
    covered = {int(f) for t in tracks.values() for f in t.frames}
    report = {"source": str(path), "rows": len(rows), "rows_matched": len(rows) - off_session,
              "rows_off_session": off_session, "duplicate_rows": duplicates, "tracks": len(tracks),
              "large_vehicles": sum(1 for t in tracks.values() if t.veh_class),
              "session_frames": len(session.frames), "frames_with_detections": len(covered),
              "frames_without_detections": len(session_frames - covered)}
    return Detections(tracks, report)
