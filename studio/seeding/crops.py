"""Session LiDAR + UAV detections -> DAA's per-track inputs (`TrackFrames`).

Frame-major and parallel: each worker opens the session itself, loads the selected sensors' sweeps
for its frames (map frame, DTM ground removed), and crops every detection's neighbourhood in the
detection's own box frame at the refiner's crop scale. The parent then regroups the crops by track.
This is DAA's input contract (daa/types.py): a generous ground-removed candidate crop per frame,
the UAV prior box, its velocity, the ground height under the box (the prior box's bottom) and the
sensor origin.
"""
from __future__ import annotations

import os
from concurrent.futures import ProcessPoolExecutor
from multiprocessing import get_context

import numpy as np
from scipy.spatial import cKDTree

from ..daa.types import TrackFrames
from .detections import velocities


def _crop_frames(args):
    """Worker: crops for a chunk of frames -> {tid: [(frame, pts, ego)]}, {frame: origin}."""
    session_path, convention, sensors, frame_dets, crop = args
    from ..session import Session
    s = Session(session_path, pose_convention=convention)
    scale, zscale, rmin = crop
    out, origins = {}, {}
    for frame, dets in frame_dets:
        parts, egos, origin = [], [], None
        for a in sensors:
            if not s.has(a, frame):
                continue
            p = s.points(a, frame, drop_ground=True)
            if origin is None:
                origin = s.sensor_origin(a, frame)
            if len(p):
                parts.append(p)
                egos.append(np.full(len(p), a, np.int8))
        origins[frame] = origin
        if not parts:
            for tid, _box in dets:
                out.setdefault(tid, []).append((frame, np.zeros((0, 3), np.float32), np.zeros(0, np.int8)))
            continue
        pts = np.concatenate(parts).astype(np.float64)
        ego = np.concatenate(egos)
        tree = cKDTree(pts[:, :2])
        for tid, box in dets:
            cx, cy, cz, L, W, H, yaw = box
            rx, ry = max(scale * L / 2, rmin), max(scale * W / 2, rmin)
            idx = np.asarray(tree.query_ball_point((cx, cy), float(np.hypot(rx, ry))), dtype=np.int64)
            if len(idx):
                c, sn = np.cos(yaw), np.sin(yaw)
                d = pts[idx] - (cx, cy, cz)
                lx, ly = c * d[:, 0] + sn * d[:, 1], -sn * d[:, 0] + c * d[:, 1]
                keep = (np.abs(lx) <= rx) & (np.abs(ly) <= ry) & (np.abs(d[:, 2]) <= zscale * H / 2)
                idx = idx[keep]
            out.setdefault(tid, []).append((frame, pts[idx].astype(np.float32), ego[idx]))
    return out, origins


def build_trackframes(session, dets, sensors, cfg, workers: int | None = None,
                      progress=None, chunk: int = 24) -> dict:
    """-> {tid: TrackFrames}. `progress(done_frames, total_frames)` is called as chunks finish."""
    by_frame = sorted(dets.by_frame().items())
    crop = (cfg.data_crop_scale, cfg.data_crop_z_scale, cfg.data_crop_min_radius)
    chunks = [by_frame[i:i + chunk] for i in range(0, len(by_frame), chunk)]
    jobs = [(str(session.path), session.convention, list(sensors), c, crop) for c in chunks]
    crops, origins = {}, {}
    workers = workers or max(1, min(8, (os.cpu_count() or 2) - 1))
    done = 0

    def absorb(res, n):
        nonlocal done
        part, org = res
        for tid, items in part.items():
            crops.setdefault(tid, []).extend(items)
        origins.update(org)
        done += n
        if progress:
            progress(done, len(by_frame))

    if workers == 1 or len(jobs) == 1:
        for j in jobs:
            absorb(_crop_frames(j), len(j[3]))
    else:
        with ProcessPoolExecutor(workers, mp_context=get_context("spawn")) as ex:
            futs = [(ex.submit(_crop_frames, j), len(j[3])) for j in jobs]
            for fut, n in futs:
                absorb(fut.result(), n)

    try:
        session_no = int(session.name.split("_")[-1])
    except ValueError:
        session_no = 0
    out = {}
    for tid, td in dets.tracks.items():
        items = {f: (p, e) for f, p, e in crops.get(tid, [])}
        vel = velocities(td.frames, td.boxes[:, :2])
        tf = TrackFrames(track_id=int(tid), veh_class=int(td.veh_class), session=session_no,
                         agent=int(sensors[0]) if len(sensors) == 1 else -1)
        for k, f in enumerate(td.frames):
            p, e = items.get(int(f), (np.zeros((0, 3), np.float32), np.zeros(0, np.int8)))
            box = td.boxes[k]
            tf.frame_ids.append(int(f))
            tf.cand_pts.append(p)
            tf.cand_ego.append(e)
            tf.uav_boxes.append(box.astype(np.float64))
            tf.velocities.append(vel[k])
            tf.ground_z.append(float(box[2] - box[5] / 2.0))
            o = origins.get(int(f))
            tf.lidar_origin.append(np.zeros(3) if o is None else np.asarray(o, float))
        out[int(tid)] = tf
    return out
