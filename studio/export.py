"""Export project labels: OpenCOOD YAMLs, DAA label CSV, or (explicitly) into the dataset itself.

The YAML export mirrors the session folder YAML-for-YAML: every agent's file is copied, and the
`vehicles` block of the label-carrying agents (0 and 1 in the release) is replaced by the project's
labels, so the output is a drop-in label set for the same point clouds.
"""
from __future__ import annotations

import datetime as _dt
import json
import shutil
from pathlib import Path

from . import poses
from .importers import write_label_csv

REVIEWED = ("accepted", "edited", "new")                 # default: human-verified only
ALL_LIVE = ("unreviewed", "accepted", "edited", "flagged", "new")
LABEL_AGENTS = (0, 1)


def _select(project, statuses):
    statuses = set(statuses)
    return [project.get(s["tid"]) for s in project.tracks() if s["status"] in statuses]


def _vehicles_by_frame(tracks) -> dict:
    out = {}
    for t in tracks:
        for f in t.frames():
            out.setdefault(f, {})[t.tid] = poses.vehicle_from_box(t.box(f), t.cls)
    return out


def _provenance(project, tracks, statuses, target) -> dict:
    by = {}
    for t in tracks:
        by[t.status] = by.get(t.status, 0) + 1
    return {"exported": _dt.datetime.now().isoformat(timespec="seconds"), "target": str(target),
            "project": project.meta.get("name"), "session": project.meta.get("session"),
            "seed_run": project.meta.get("active_run"), "statuses": sorted(statuses),
            "tracks": len(tracks), "by_status": by,
            "reviewers": sorted({t.updated_by for t in tracks if t.updated_by})}


def export_opencood(project, out_dir, statuses=REVIEWED) -> dict:
    """Write <out_dir>/<session>/<agent>/<frame>.yaml for every frame of the session."""
    s = project.session
    tracks = _select(project, statuses)
    veh = _vehicles_by_frame(tracks)
    root = Path(out_dir) / s.name
    n = 0
    for agent in sorted(int(p.name) for p in s.path.iterdir() if p.is_dir() and p.name.isdigit()):
        src = s.path / str(agent)
        if not any(src.glob("*.yaml")):
            continue
        (root / str(agent)).mkdir(parents=True, exist_ok=True)
        for yp in sorted(src.glob("*.yaml")):
            doc = poses.load_yaml(yp)
            if agent in LABEL_AGENTS:
                doc["vehicles"] = veh.get(int(yp.stem), {})
            poses.save_yaml(root / str(agent) / yp.name, doc)
            n += 1
    for extra in ("camera_3.json", "camera_4.json", "annotation_area.json"):
        if (s.path / extra).exists():
            shutil.copy2(s.path / extra, root / extra)
    prov = _provenance(project, tracks, statuses, root)
    prov["yaml_files"] = n
    (root / "provenance.json").write_text(json.dumps(prov, indent=1))
    return prov


def export_csv(project, path, statuses=REVIEWED) -> dict:
    tracks = _select(project, statuses)
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    rows = write_label_csv(path, tracks, project.session.name)
    prov = _provenance(project, tracks, statuses, path)
    prov["rows"] = rows
    Path(str(path) + ".provenance.json").write_text(json.dumps(prov, indent=1))
    return prov


def apply_to_dataset(project, statuses=REVIEWED) -> dict:
    """Write the labels INTO the session's own YAMLs. Each file is backed up once to
    <project>/backups/<agent>/<frame>.yaml.preEdit before its first overwrite."""
    s = project.session
    tracks = _select(project, statuses)
    veh = _vehicles_by_frame(tracks)
    n = 0
    for agent in LABEL_AGENTS:
        if not (s.path / str(agent)).is_dir():
            continue
        bdir = project.path / "backups" / str(agent)
        bdir.mkdir(parents=True, exist_ok=True)
        for yp in sorted((s.path / str(agent)).glob("*.yaml")):
            bak = bdir / (yp.name + ".preEdit")
            if not bak.exists():
                shutil.copy2(yp, bak)
            doc = poses.load_yaml(yp)
            doc["vehicles"] = veh.get(int(yp.stem), {})
            poses.save_yaml(yp, doc)
            n += 1
    prov = _provenance(project, tracks, statuses, s.path)
    prov["yaml_files"] = n
    (project.path / "backups" / "applied.json").write_text(json.dumps(prov, indent=1))
    return prov
