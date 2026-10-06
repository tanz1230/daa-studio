"""The project workspace: everything a labeling job produces, kept out of the dataset.

    <project>/
        project.json        name, session path, sensors, UAV camera, active seed run, app version
        tracks/<tid>.json   the editable labels (one file per track, versioned)
        tracks/.prev/       the previous version of every track (one-step server-side revert)
        seeds/<run>/        immutable seed runs: labels.csv + meta.json
        cache/crops/        per-track LiDAR crops written by seeding (aggregate view)
        jobs/<id>/          background job progress + logs
        exports/            export outputs
        history.jsonl       append-only edit log: who, when, which track, which operation

Saving is optimistic: every track carries a version, and a save that names a stale version is
refused, so two annotators can share a project without silently overwriting each other.
"""
from __future__ import annotations

import datetime as _dt
import json
import os
import threading
from collections import defaultdict
from pathlib import Path

from . import __version__
from .labels import Track
from .session import Session

FORMAT = 1


class StaleVersion(Exception):
    def __init__(self, tid: int, expected: int, current: int):
        super().__init__(f"track {tid}: saved over version {expected}, current is {current}")
        self.tid, self.expected, self.current = tid, expected, current


class ProjectError(ValueError):
    pass


def _now() -> str:
    return _dt.datetime.now().isoformat(timespec="seconds")


def _write_json(path: Path, obj) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, indent=1))
    os.replace(tmp, path)


class Project:
    def __init__(self, path: Path, meta: dict):
        self.path = Path(path)
        self.meta = meta
        self._lock = threading.RLock()
        self._session = None
        self._tracks: dict = {}
        self._by_frame = defaultdict(set)
        for p in sorted((self.path / "tracks").glob("*.json")):
            t = Track.from_json(json.loads(p.read_text()))
            self._tracks[t.tid] = t
            self._index(t)

    # ------------------------------------------------------------------ lifecycle
    @classmethod
    def create(cls, path, session_path, name: str | None = None, sensors=None,
               uav_agent: int | None = None) -> "Project":
        path = Path(path).expanduser().resolve()
        if (path / "project.json").exists():
            raise ProjectError(f"a project already exists at {path}")
        session = Session(session_path)                       # validates the session up front
        for sub in ("tracks/.prev", "seeds", "cache/crops", "jobs", "exports"):
            (path / sub).mkdir(parents=True, exist_ok=True)
        meta = {"format": FORMAT, "app_version": __version__, "name": name or session.name,
                "session": str(session.path), "created": _now(),
                "sensors": list(sensors) if sensors else list(session.lidar_agents),
                "uav_agent": uav_agent if uav_agent is not None else
                (session.uav_agents[0] if session.uav_agents else None),
                "active_run": None}
        _write_json(path / "project.json", meta)
        p = cls(path, meta)
        p._session = session
        return p

    @classmethod
    def open(cls, path) -> "Project":
        path = Path(path).expanduser().resolve()
        mp = path / "project.json"
        if not mp.exists():
            raise ProjectError(f"no project at {path}")
        meta = json.loads(mp.read_text())
        if meta.get("format", 1) > FORMAT:
            raise ProjectError("this project was written by a newer DAA Studio")
        return cls(path, meta)

    @property
    def session(self) -> Session:
        if self._session is None:
            self._session = Session(self.meta["session"])
        return self._session

    def save_meta(self) -> None:
        with self._lock:
            _write_json(self.path / "project.json", self.meta)

    # ------------------------------------------------------------------ tracks
    def _index(self, t: Track) -> None:
        for f in t.poses:
            self._by_frame[f].add(t.tid)

    def _unindex(self, t: Track) -> None:
        for f in t.poses:
            self._by_frame[f].discard(t.tid)

    def _store(self, t: Track) -> None:
        tp = self.path / "tracks" / f"{t.tid}.json"
        if tp.exists():
            os.replace(tp, self.path / "tracks" / ".prev" / f"{t.tid}.json")
        _write_json(tp, t.to_json())

    def tracks(self) -> list:
        with self._lock:
            return [t.summary() for t in sorted(self._tracks.values(), key=lambda t: t.tid)]

    def get(self, tid: int) -> Track:
        with self._lock:
            if tid not in self._tracks:
                raise KeyError(f"no track {tid}")
            return Track.from_json(self._tracks[tid].to_json())      # a private copy

    def tids_at(self, frame: int) -> list:
        with self._lock:
            return sorted(self._by_frame.get(frame, ()))

    def save(self, track: Track, expected_version: int, user: str = "", op: str = "edit") -> Track:
        with self._lock:
            cur = self._tracks.get(track.tid)
            current = cur.version if cur else 0
            if cur is not None and expected_version != current:
                raise StaleVersion(track.tid, expected_version, current)
            t = Track.from_json(track.to_json())
            t.version = current + 1
            t.updated, t.updated_by = _now(), user
            if cur is not None:
                self._unindex(cur)
            self._tracks[t.tid] = t
            self._index(t)
            self._store(t)
            self._log({"ts": t.updated, "user": user, "tid": t.tid, "op": op,
                       "from": current, "to": t.version, "status": t.status})
            return Track.from_json(t.to_json())

    def add(self, track: Track, user: str = "", op: str = "add") -> Track:
        with self._lock:
            if track.tid is None or track.tid in self._tracks:
                track.tid = self.next_tid()
            return self.save(track, 0, user, op)

    def next_tid(self) -> int:
        with self._lock:
            return (max(self._tracks) + 1) if self._tracks else 1

    def boxes_at(self, frame: int) -> list:
        """Every live track's box at a frame (context rendering)."""
        with self._lock:
            out = []
            for tid in self._by_frame.get(frame, ()):
                t = self._tracks[tid]
                if t.status == "deleted":
                    continue
                out.append({"tid": tid, "box": [round(v, 4) for v in t.box(frame)],
                            "cls": t.cls, "status": t.status})
            return out

    # ------------------------------------------------------------------ seeds
    def import_seed(self, run: str, tracks: list, user: str = "seeding") -> dict:
        """Load seed tracks. Untouched (unreviewed) tracks are replaced by the new seed; any track a
        person has already accepted, edited, flagged or added is left exactly as it is."""
        counts = {"added": 0, "replaced": 0, "kept": 0}
        with self._lock:
            for t in tracks:
                cur = self._tracks.get(t.tid)
                if cur is None:
                    self.save(t, 0, user, f"seed:{run}")
                    counts["added"] += 1
                elif cur.status == "unreviewed":
                    self.save(t, cur.version, user, f"seed:{run}")
                    counts["replaced"] += 1
                else:
                    counts["kept"] += 1
            self.meta["active_run"] = run
            self.save_meta()
        return counts

    def seed_runs(self) -> list:
        out = []
        for m in sorted((self.path / "seeds").glob("*/meta.json")):
            try:
                out.append(json.loads(m.read_text()))
            except json.JSONDecodeError:
                continue
        return out

    # ------------------------------------------------------------------ reporting
    def stats(self) -> dict:
        with self._lock:
            by = defaultdict(int)
            for t in self._tracks.values():
                by[t.status] += 1
            total = len(self._tracks) - by["deleted"]
            done = by["accepted"] + by["edited"] + by["new"]
            return {"total": total, "reviewed": done, "by_status": dict(by)}

    def _log(self, rec: dict) -> None:
        with open(self.path / "history.jsonl", "a") as f:
            f.write(json.dumps(rec) + "\n")

    def history(self, tid: int | None = None, limit: int = 200) -> list:
        p = self.path / "history.jsonl"
        if not p.exists():
            return []
        rows = [json.loads(line) for line in p.read_text().splitlines() if line.strip()]
        if tid is not None:
            rows = [r for r in rows if r.get("tid") == tid]
        return rows[-limit:]
