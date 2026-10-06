"""Background jobs for the server.

Seeding runs as a subprocess (`python -m studio seed ...`): a 10-minute DAA run can never block the
server, and a crash stays contained. The job writes finished tracks in batches; the manager's
poller thread imports each new batch into the project as it appears, so the server stays the
project's single writer and annotators can review tracks while seeding is still running.
Exports are quick and run as threads.
"""
from __future__ import annotations

import datetime as _dt
import json
import os
import subprocess
import sys
import threading
import time
import traceback
from pathlib import Path

from .labels import Track

STUDIO_ROOT = Path(__file__).resolve().parent.parent          # the folder that holds `studio/`


class JobManager:
    def __init__(self, project, poll_s: float = 1.0):
        self.project = project
        self.jobs: dict = {}
        self._lock = threading.RLock()
        self._stop = threading.Event()
        self._poller = threading.Thread(target=self._poll_loop, args=(poll_s,), daemon=True)
        self._poller.start()

    # ------------------------------------------------------------------ seeding (subprocess)
    def start_seed(self, mode: str, detections=None, labels=None, sensors=None, limit=None,
                   workers=None) -> str:
        stamp = _dt.datetime.now().strftime("%Y%m%d-%H%M%S")
        job_id, run = f"seed-{stamp}", f"{mode}-{stamp}"
        jdir = self.project.path / "jobs" / job_id
        jdir.mkdir(parents=True, exist_ok=True)
        cmd = [sys.executable, "-m", "studio", "seed", "--project", str(self.project.path),
               "--mode", mode, "--run-id", run, "--progress", str(jdir / "progress.json")]
        if detections:
            cmd += ["--detections", str(detections)]
        if labels:
            cmd += ["--labels", str(labels)]
        if sensors:
            cmd += ["--sensors", ",".join(str(int(s)) for s in sensors)]
        if limit:
            cmd += ["--limit", str(int(limit))]
        if workers:
            cmd += ["--workers", str(int(workers))]
        env = dict(os.environ)
        env["PYTHONPATH"] = str(STUDIO_ROOT) + os.pathsep + env.get("PYTHONPATH", "")
        log = open(jdir / "log.txt", "w")
        proc = subprocess.Popen(cmd, cwd=str(STUDIO_ROOT), stdout=log, stderr=subprocess.STDOUT, env=env)
        with self._lock:
            self.jobs[job_id] = {"id": job_id, "kind": "seed", "mode": mode, "run": run, "dir": jdir,
                                 "proc": proc, "log": log, "imported": set(), "counts":
                                 {"added": 0, "replaced": 0, "kept": 0}, "created": time.time(),
                                 "cancelled": False}
        return job_id

    def _import_batches(self, job) -> None:
        bdir = self.project.path / "seeds" / job["run"] / "batches"
        if not bdir.is_dir():
            return
        for p in sorted(bdir.glob("*.json")):
            if p.name in job["imported"]:
                continue
            try:
                tracks = [Track.from_json(d) for d in json.loads(p.read_text())]
            except (json.JSONDecodeError, OSError):
                continue                                          # still being written; next poll
            c = self.project.import_seed(job["run"], tracks)
            for k, v in c.items():
                job["counts"][k] += v
            job["imported"].add(p.name)

    # ------------------------------------------------------------------ export (thread)
    def start_thread(self, kind: str, fn, *args, **kwargs) -> str:
        job_id = f"{kind}-{_dt.datetime.now().strftime('%Y%m%d-%H%M%S-%f')}"
        job = {"id": job_id, "kind": kind, "state": "running", "result": None, "error": None,
               "created": time.time()}

        def target():
            try:
                job["result"] = fn(*args, **kwargs)
                job["state"] = "done"
            except Exception as e:                              # reported to the UI, not swallowed
                job["state"], job["error"] = "failed", f"{type(e).__name__}: {e}"
                traceback.print_exc()

        with self._lock:
            self.jobs[job_id] = job
        threading.Thread(target=target, daemon=True).start()
        return job_id

    # ------------------------------------------------------------------ status
    def status(self, job_id: str) -> dict:
        with self._lock:
            job = self.jobs.get(job_id)
            if job is None:
                raise KeyError(job_id)
            if job["kind"] != "seed":
                return {k: job[k] for k in ("id", "kind", "state", "result", "error")}
            self._import_batches(job)
            prog = {}
            pp = job["dir"] / "progress.json"
            if pp.exists():
                try:
                    prog = json.loads(pp.read_text())
                except json.JSONDecodeError:
                    prog = {}
            rc = job["proc"].poll()
            state = prog.get("state", "running")
            if job["cancelled"]:
                state = "cancelled"
            elif rc is not None and state == "running":
                state = "failed" if rc else "done"
            if rc is not None and not job["log"].closed:
                job["log"].close()
            err = prog.get("error")
            if state == "failed" and not err:
                err = self._tail(job["dir"] / "log.txt")
            return {"id": job_id, "kind": "seed", "mode": job["mode"], "run": job["run"], "state": state,
                    "phase": prog.get("phase"), "done": prog.get("done", 0), "total": prog.get("total", 0),
                    "message": prog.get("message", ""), "eta_s": prog.get("eta_s"), "error": err,
                    "imported": dict(job["counts"]), "batches": len(job["imported"])}

    def active(self) -> list:
        with self._lock:
            ids = list(self.jobs)
        out = [self.status(j) for j in ids]
        return [s for s in out if s["state"] == "running"] + \
               [s for s in out if s["state"] != "running"][-3:]

    def cancel(self, job_id: str) -> dict:
        with self._lock:
            job = self.jobs[job_id]
            if job["kind"] == "seed" and job["proc"].poll() is None:
                job["cancelled"] = True
                job["proc"].terminate()
        return self.status(job_id)

    def _poll_loop(self, poll_s: float) -> None:
        """Import new batches while a seed job runs, once more when it exits, then stop watching."""
        while not self._stop.wait(poll_s):
            with self._lock:
                seeds = [j for j in self.jobs.values() if j["kind"] == "seed" and not j.get("final")]
            for job in seeds:
                try:
                    finished = job["proc"].poll() is not None      # checked BEFORE the import pass
                    with self._lock:
                        self._import_batches(job)
                    if finished:
                        job["final"] = True
                except Exception:                                  # keep polling the other jobs
                    traceback.print_exc()

    def shutdown(self) -> None:
        self._stop.set()
        with self._lock:
            for job in self.jobs.values():
                if job["kind"] == "seed" and job["proc"].poll() is None:
                    job["proc"].terminate()

    @staticmethod
    def _tail(path: Path, n: int = 12) -> str:
        try:
            return "\n".join(path.read_text().splitlines()[-n:])
        except OSError:
            return ""
