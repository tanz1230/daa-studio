"""The seeding job: produce a seed run for a project.

    python -m studio seed --project P --mode daa --detections stage1.csv [--sensors 0,1,2]

Modes
  daa      UAV detections -> LiDAR crops -> DAA (paper configuration) per track
  uav      UAV detections as they are (default heights), no refinement
  session  the session's own annotations (e.g. to review or correct existing ground truth)
  import   any label CSV in the DAA schema

Output, under <project>/seeds/<run>/: batches/NNNN.json (finished tracks, written as they
complete), labels.csv and meta.json at the end; per-track crops under <project>/cache/crops/ for
the aggregate view; progress in <project>/jobs/<job>/progress.json. The job never writes tracks:
the server imports the batches, so the project keeps a single writer. Run standalone (no server)
with --import and it imports the run itself at the end.
"""
from __future__ import annotations

import datetime as _dt
import json
import os
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from multiprocessing import get_context
from pathlib import Path

import numpy as np

from .. import __version__
from ..importers import tracks_from_label_csv, tracks_from_session, write_label_csv
from ..labels import Track
from ..project import Project
from .config import DELTA, adopted_config
from .crops import build_trackframes
from .detections import load_detections

BATCH = 25


class Progress:
    """Writes the job's progress file (atomically) and an ETA."""

    def __init__(self, path):
        self.path = Path(path) if path else None
        self.state = {"state": "running", "phase": "starting", "done": 0, "total": 0, "message": "",
                      "started": time.time(), "updated": time.time(), "eta_s": None}
        self._phase_t0 = time.time()
        self.flush()

    def phase(self, name, total, message=""):
        self._phase_t0 = time.time()
        self.state.update(phase=name, done=0, total=int(total), message=message, eta_s=None)
        self.flush()

    def step(self, done, message=None):
        self.state["done"] = int(done)
        if message is not None:
            self.state["message"] = message
        tot = self.state["total"]
        el = time.time() - self._phase_t0
        self.state["eta_s"] = round(el / done * (tot - done), 1) if done and tot else None
        self.flush()

    def finish(self, state="done", **extra):
        self.state.update(state=state, **extra)
        self.flush()

    def flush(self):
        if not self.path:
            return
        self.state["updated"] = time.time()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_suffix(".tmp")
        tmp.write_text(json.dumps(self.state))
        os.replace(tmp, self.path)


# ---------------------------------------------------------------------------- DAA worker
def _diagnostics(tf, r) -> dict:
    """Per-track review signals from the refined result."""
    boxes = np.asarray(r.boxes)
    pts_in, ys = [], []
    for t, p in enumerate(r.fg_points):
        p = np.asarray(p, float)
        pts_in.append(len(p))
        if p.ndim == 2 and len(p):
            c, s = np.cos(boxes[t, 6]), np.sin(boxes[t, 6])
            d = p[:, :2] - boxes[t, :2]
            ys.append(-s * d[:, 0] + c * d[:, 1])
    y = np.concatenate(ys) if ys else np.zeros(0)
    two = float(min((y < 0).mean(), (y > 0).mean())) if len(y) else 0.0
    uav = np.asarray(tf.uav_boxes, float)
    shift = np.hypot(*(boxes[:, :2] - uav[:, :2]).T)
    conf = float(np.mean(r.conf)) if len(r.conf) else 0.0
    pts_med = float(np.median(pts_in)) if pts_in else 0.0
    risk = (min(max(1.0 - conf, 0.0), 1.0) + (0.25 if pts_med < 20 else 0.0)
            + (0.25 if two < 0.1 else 0.0) + 0.25 * min(float(np.median(shift)), 1.0))
    return {"conf": round(conf, 3), "pts": round(pts_med, 1), "two_sided": round(two, 3),
            "shift_m": round(float(np.median(shift)), 3),
            "dL": round(float(boxes[0, 3] - np.median(uav[:, 3])), 3),
            "dW": round(float(boxes[0, 4] - np.median(uav[:, 4])), 3),
            "risk": round(risk, 3), "failed": False}


def _refine(item):
    tid, tf = item
    from ..daa.boxfit_ov import box_fit_ov        # imported in the worker: hermetic OV_* handling
    from ..daa.harvest import em_harvest
    cfg = adopted_config()
    try:
        r = box_fit_ov(em_harvest(tf, cfg), tf, cfg)
        return tid, np.asarray(r.boxes), np.asarray(r.conf), _diagnostics(tf, r), None
    except Exception as e:                       # the track falls back to its UAV seed
        return tid, None, None, None, f"{type(e).__name__}: {e}"


def _track_from_result(tid, tf, boxes, conf, diag, cls, run) -> Track:
    poses, dims = {}, []
    for k, f in enumerate(tf.frame_ids):
        b = boxes[k]
        poses[int(f)] = [float(b[0]), float(b[1]), float(b[2]), float(b[6])]
        dims.append(b[3:6])
    shape = [float(v) for v in np.median(np.asarray(dims, float), axis=0)]
    return Track(tid=int(tid), cls=cls, shape=shape, poses=poses, status="unreviewed",
                 seed={"kind": "daa", "run": run,
                       "conf": {int(f): float(c) for f, c in zip(tf.frame_ids, conf)}, "diag": diag})


def _fallback(tid, tf, cls, run, err) -> Track:
    uav = np.asarray(tf.uav_boxes, float)
    t = _track_from_result(tid, tf, uav, np.zeros(len(uav)), {}, cls, run)
    t.seed["kind"] = "uav-fallback"
    t.seed["diag"] = {"failed": True, "error": err, "risk": 2.0, "conf": 0.0}
    return t


# ---------------------------------------------------------------------------- crops cache
def save_crops(project_path, tfs) -> None:
    out = Path(project_path) / "cache" / "crops"
    out.mkdir(parents=True, exist_ok=True)
    for tid, tf in tfs.items():
        lens = np.array([len(p) for p in tf.cand_pts], np.int64)
        pts = (np.concatenate(tf.cand_pts).astype(np.float32) if lens.sum()
               else np.zeros((0, 3), np.float32))
        ego = np.concatenate(tf.cand_ego).astype(np.int8) if lens.sum() else np.zeros(0, np.int8)
        np.savez(out / f"{tid}.npz", frames=np.asarray(tf.frame_ids, np.int64),
                 offsets=np.concatenate([[0], np.cumsum(lens)]), pts=pts, ego=ego)


# ---------------------------------------------------------------------------- the job
def run_seeding(project_path, mode="daa", detections=None, labels_csv=None, sensors=None,
                workers=None, limit=None, run_id=None, progress_path=None, do_import=False) -> dict:
    t0 = time.time()
    project = Project.open(project_path)
    session = project.session
    sensors = list(sensors or project.meta.get("sensors") or session.lidar_agents)
    run = run_id or f"{mode}-{_dt.datetime.now().strftime('%Y%m%d-%H%M%S')}"
    rdir = project.path / "seeds" / run
    (rdir / "batches").mkdir(parents=True, exist_ok=True)
    prog = Progress(progress_path)
    prog.state["run"] = run
    # DAA_STUDIO_WORKERS caps the default, to share a machine politely (e.g. while a model trains)
    workers = workers or int(os.environ.get("DAA_STUDIO_WORKERS") or 0) or max(1, min(12, (os.cpu_count() or 2) - 1))
    meta = {"run": run, "mode": mode, "created": _dt.datetime.now().isoformat(timespec="seconds"),
            "app_version": __version__, "sensors": sensors, "workers": workers}
    tracks, nb = [], 0

    def emit(batch):
        nonlocal nb
        if batch:
            nb += 1
            (rdir / "batches" / f"{nb:04d}.json").write_text(json.dumps([t.to_json() for t in batch]))

    try:
        if mode in ("daa", "uav"):
            if not detections:
                raise ValueError("--detections is required for mode daa / uav")
            prog.phase("reading", 1, "Reading UAV detections")
            dets = load_detections(detections, session)
            if limit:
                keep = sorted(dets.tracks, key=lambda k: -len(dets.tracks[k].frames))[:limit]
                dets.tracks = {k: dets.tracks[k] for k in keep}
            meta.update(detections=str(detections), report=dets.report)
            if not dets.tracks:
                raise ValueError("no UAV detection matches a frame of this session")
            from ..daa.config import RefinerConfig
            prog.phase("crops", len(dets.by_frame()), "Cropping LiDAR around each detection")
            tfs = build_trackframes(session, dets, sensors, RefinerConfig(), workers=workers,
                                    progress=lambda d, n: prog.step(d))
            save_crops(project.path, tfs)
            cls = {tid: td.cls_name for tid, td in dets.tracks.items()}
            if mode == "uav":
                prog.phase("import", len(tfs), "Loading UAV boxes")
                batch = []
                for i, (tid, tf) in enumerate(sorted(tfs.items()), 1):
                    t = _track_from_result(tid, tf, np.asarray(tf.uav_boxes, float),
                                           np.asarray(dets.tracks[tid].conf), {}, cls[tid], run)
                    t.seed["kind"] = "uav"
                    tracks.append(t)
                    batch.append(t)
                    if len(batch) >= BATCH:
                        emit(batch)
                        batch = []
                    prog.step(i)
                emit(batch)
            else:
                meta["config"] = {"delta_m": DELTA, "width_from_mstep": True,
                                  "source": "DAA paper configuration"}
                prog.phase("refine", len(tfs), "Refining with DAA")
                items = sorted(tfs.items(), key=lambda kv: -kv[1].n_frames)    # long first: better ETA
                batch, failed, done = [], 0, 0
                with ProcessPoolExecutor(workers, mp_context=get_context("spawn")) as ex:
                    futs = {ex.submit(_refine, it): it[0] for it in items}
                    for fut in as_completed(futs):
                        tid, boxes, conf, diag, err = fut.result()
                        tf = tfs[tid]
                        if err:
                            failed += 1
                            t = _fallback(tid, tf, cls[tid], run, err)
                        else:
                            t = _track_from_result(tid, tf, boxes, conf, diag, cls[tid], run)
                        tracks.append(t)
                        batch.append(t)
                        done += 1
                        if len(batch) >= BATCH:
                            emit(batch)
                            batch = []
                        prog.step(done, f"Refining with DAA ({failed} fell back to the UAV box)"
                                  if failed else None)
                emit(batch)
                meta["failed"] = failed
        elif mode == "session":
            prog.phase("import", 1, "Reading the session's annotations")
            tracks = tracks_from_session(session, run)
            for i in range(0, len(tracks), BATCH):
                emit(tracks[i:i + BATCH])
        elif mode == "import":
            if not labels_csv:
                raise ValueError("--labels is required for mode import")
            prog.phase("import", 1, "Reading labels")
            tracks = tracks_from_label_csv(labels_csv, "import", run, frames=session.frames)
            meta["labels"] = str(labels_csv)
            for i in range(0, len(tracks), BATCH):
                emit(tracks[i:i + BATCH])
        else:
            raise ValueError(f"unknown mode {mode!r}")

        write_label_csv(rdir / "labels.csv", tracks, session.name)
        meta.update(tracks=len(tracks), batches=nb, seconds=round(time.time() - t0, 1))
        (rdir / "meta.json").write_text(json.dumps(meta, indent=1))
        if do_import:
            prog.phase("import", len(tracks), "Importing into the project")
            counts = project.import_seed(run, tracks)
            meta["imported"] = counts
            (rdir / "meta.json").write_text(json.dumps(meta, indent=1))
        prog.finish("done", tracks=len(tracks), seconds=meta["seconds"], message="Done")
        return meta
    except Exception as e:
        prog.finish("failed", error=f"{type(e).__name__}: {e}")
        raise
