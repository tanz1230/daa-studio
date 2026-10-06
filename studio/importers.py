"""Turn label sources into Tracks: DAA / Stage-1 label CSVs and a session's own annotations."""
from __future__ import annotations

import csv
from collections import defaultdict

import numpy as np

from .labels import Track

CSV_FIELDS = ["session", "agent", "frame_idx", "lidar_time", "track_id",
              "cx", "cy", "cz", "L", "W", "H", "yaw", "conf", "dim_agree"]


def _shared_shape(dims: list) -> list:
    """Per-frame (L, W, H) -> one shared shape (median: robust to a few outlier frames)."""
    return [float(v) for v in np.median(np.asarray(dims, float), axis=0)]


def tracks_from_rows(rows, kind: str, run: str = "", frames=None, cls_of=None) -> list:
    """rows: iterable of dicts with frame_idx, track_id, cx..yaw[, conf] (DAA label CSV schema)."""
    keep = set(frames) if frames is not None else None
    per = defaultdict(lambda: {"poses": {}, "dims": [], "conf": {}})
    for r in rows:
        f = int(r["frame_idx"])
        if keep is not None and f not in keep:
            continue
        tid = int(r["track_id"])
        d = per[tid]
        d["poses"][f] = [float(r["cx"]), float(r["cy"]), float(r["cz"]), float(r["yaw"])]
        d["dims"].append((float(r["L"]), float(r["W"]), float(r["H"])))
        if r.get("conf") not in (None, ""):
            d["conf"][f] = float(r["conf"])
    out = []
    for tid, d in sorted(per.items()):
        cls = cls_of(tid) if cls_of else "Car"
        out.append(Track(tid=tid, cls=cls, shape=_shared_shape(d["dims"]), poses=d["poses"],
                         status="unreviewed", seed={"kind": kind, "run": run, "conf": d["conf"]}))
    return out


def read_label_csv(path) -> list:
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def tracks_from_label_csv(path, kind: str, run: str = "", frames=None) -> list:
    return tracks_from_rows(read_label_csv(path), kind, run, frames)


def write_label_csv(path, tracks, session_name: str = "", agent: int = 0) -> int:
    """Tracks -> DAA label CSV (one row per track and frame). Returns the row count."""
    n = 0
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        w.writeheader()
        for t in sorted(tracks, key=lambda t: t.tid):
            conf = (t.seed or {}).get("conf") or {}
            for fr in t.frames():
                x, y, z, L, W, H, yaw = t.box(fr)
                w.writerow({"session": session_name, "agent": agent, "frame_idx": fr, "lidar_time": "",
                            "track_id": t.tid, "cx": f"{x:.5f}", "cy": f"{y:.5f}", "cz": f"{z:.5f}",
                            "L": f"{L:.5f}", "W": f"{W:.5f}", "H": f"{H:.5f}", "yaw": f"{yaw:.6f}",
                            "conf": f"{conf.get(fr, 1.0):.4f}", "dim_agree": "0.0"})
                n += 1
    return n


def tracks_from_session(session, run: str = "session") -> list:
    """The session's own annotations (agent-0 YAML `vehicles`) as tracks."""
    per = defaultdict(lambda: {"poses": {}, "dims": [], "cls": "Car"})
    for f in session.frames:
        for vid, (box, cls) in session.labels(f).items():
            d = per[vid]
            d["poses"][f] = [float(box[0]), float(box[1]), float(box[2]), float(box[6])]
            d["dims"].append(tuple(float(v) for v in box[3:6]))
            d["cls"] = cls
    return [Track(tid=vid, cls=d["cls"], shape=_shared_shape(d["dims"]), poses=d["poses"],
                  status="unreviewed", seed={"kind": "import", "run": run})
            for vid, d in sorted(per.items())]
