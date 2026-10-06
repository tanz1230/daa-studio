"""HTTP API handlers. Each takes (app, params, query, body) and returns a Response.

JSON in and out, except the bulk geometry routes, which stream little-endian binary:
    points     uint32 N | float32 N*3 (x, y, z, map frame) | uint8 N (agent id)
    aggregate  uint32 N | float32 N*3 (box frame)          | uint16 N (index into the track's sorted frames) | uint8 N (agent)
"""
from __future__ import annotations

import json
import math
import struct
import threading
from pathlib import Path

import numpy as np

from . import __version__, export as ex, labels as L
from .labels import Track
from .project import Project, ProjectError, StaleVersion
from .session import AGENT_NAMES, Session, SessionError


class Response:
    def __init__(self, status=200, body=b"", ctype="application/json", headers=None):
        self.status, self.body, self.ctype, self.headers = status, body, ctype, headers or {}


def _default(o):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, Path):
        return str(o)
    if isinstance(o, set):
        return sorted(o)
    raise TypeError(type(o).__name__)


def ok(obj, status=200) -> Response:
    return Response(status, json.dumps(obj, default=_default).encode())


def err(msg, status=400, **extra) -> Response:
    return ok(dict({"error": msg}, **extra), status)


def _need_project(app):
    if app.project is None:
        raise LookupError("no project is open")
    return app.project


# ============================================================================ app + projects
def app_info(app, p, q, body):
    pr = app.project
    return ok({"version": __version__, "user": app.user, "recent": app.settings.recent(),
               "project": _project_payload(app, pr) if pr else None,
               "agent_names": AGENT_NAMES})


def _project_payload(app, pr):
    s = pr.session
    return {"path": str(pr.path), "meta": pr.meta, "session": s.info(), "stats": pr.stats(),
            "seed_runs": pr.seed_runs(), "jobs": app.jobs.active() if app.jobs else [],
            "area": s.annotation_area(), "frames": s.frames}


def projects_create(app, p, q, body):
    path, session = body.get("path"), body.get("session")
    if not path or not session:
        return err("path and session are required")
    try:
        pr = Project.create(path, session, body.get("name") or None, body.get("sensors") or None,
                            body.get("uav_agent"))
    except (ProjectError, SessionError, OSError) as e:
        return err(str(e))
    app.open_project(pr)
    return ok(_project_payload(app, pr), 201)


def projects_open(app, p, q, body):
    try:
        pr = Project.open(body.get("path", ""))
        pr.session                                         # validate the session is still reachable
    except (ProjectError, SessionError, OSError) as e:
        return err(str(e))
    app.open_project(pr)
    return ok(_project_payload(app, pr))


def projects_close(app, p, q, body):
    app.close_project()
    return ok({"closed": True})


def project_get(app, p, q, body):
    return ok(_project_payload(app, _need_project(app)))


def project_update(app, p, q, body):
    pr = _need_project(app)
    for k in ("name", "sensors", "uav_agent"):
        if k in body:
            pr.meta[k] = body[k]
    pr.save_meta()
    return ok(pr.meta)


# ============================================================================ tracks
def tracks_list(app, p, q, body):
    return ok(_need_project(app).tracks())


def track_get(app, p, q, body):
    return ok(_need_project(app).get(int(p["tid"])).to_json())


def track_put(app, p, q, body):
    pr = _need_project(app)
    t = Track.from_json(body["track"])
    t.tid = int(p["tid"])
    try:
        saved = pr.save(t, int(body.get("expected_version", t.version)), app.user_of(body), body.get("op", "edit"))
    except StaleVersion as e:
        return err(str(e), 409, current=pr.get(t.tid).to_json())
    return ok(saved.to_json())


def track_post(app, p, q, body):
    pr = _need_project(app)
    raw = body["track"]
    H = float(raw.get("shape", [4.6, 1.85, 1.55])[2])
    s = pr.session
    plane = next((c.ground_plane for c in map(s.camera, s.uav_agents) if c and c.ground_plane), None)
    for f, pose in raw.get("poses", {}).items():        # z = null: stand the box on the ground
        if pose[2] is None:
            z = 0.0
            if s.ground.ok:                               # the terrain model
                for _ in range(3):
                    z = float(s.ground.ground_point([pose[0], pose[1], z])[2])
            elif plane:                                   # else the take's ground plane (UAV camera file)
                z = plane["z0"] + plane["gx"] * pose[0] + plane["gy"] * pose[1]
            pose[2] = z + H / 2
    t = Track.from_json(raw)
    t.tid = None                                      # the project assigns the id
    t.status = "new"
    t.seed = {"kind": "manual"}
    saved = pr.add(t, app.user_of(body), "add")
    return ok(saved.to_json(), 201)


def history(app, p, q, body):
    tid = q.get("tid")
    return ok(_need_project(app).history(int(tid) if tid else None, int(q.get("limit", 200))))


# ============================================================================ geometry
def _agents(q, pr):
    if q.get("agents"):
        return [int(a) for a in q["agents"].split(",") if a.strip()]
    return list(pr.meta.get("sensors") or pr.session.lidar_agents)


def frame_points(app, p, q, body):
    pr = _need_project(app)
    f = int(p["frame"])
    ground = q.get("ground", "0") == "1"
    parts, ids = [], []
    for a in _agents(q, pr):
        if a not in pr.session.lidar_agents:
            continue
        pts = pr.session.points(a, f, drop_ground=not ground)
        if "cx" in q and len(pts):
            r = float(q.get("r", 80))
            d2 = (pts[:, 0] - float(q["cx"])) ** 2 + (pts[:, 1] - float(q["cy"])) ** 2
            pts = pts[d2 <= r * r]
        parts.append(pts)
        ids.append(np.full(len(pts), a, np.uint8))
    pts = np.concatenate(parts) if parts else np.zeros((0, 3), np.float32)
    ego = np.concatenate(ids) if ids else np.zeros(0, np.uint8)
    blob = struct.pack("<I", len(pts)) + pts.astype("<f4").tobytes() + ego.tobytes()
    return Response(200, blob, "application/octet-stream")


def frame_boxes(app, p, q, body):
    return ok(_need_project(app).boxes_at(int(p["frame"])))


_AGG_LOCK = threading.Lock()


def track_aggregate(app, p, q, body):
    """Every frame's crop of this track, in the box frame of THAT frame's current box: a crisp blob
    means the per-frame boxes are right, a smeared one means a frame is off."""
    pr = _need_project(app)
    t = pr.get(int(p["tid"]))
    cur = int(q["frame"]) if q.get("frame") else None
    L_, W_, H_ = t.shape
    mx, mz = 1.5, 0.8
    cache = pr.path / "cache" / "crops" / f"{t.tid}.npz"
    chunks, egos, curs = [], [], []
    if cache.exists():
        with _AGG_LOCK:
            d = np.load(cache)
            frames, offs, pts_all, ego_all = d["frames"], d["offsets"], d["pts"], d["ego"]
        items = [(int(f), pts_all[offs[i]:offs[i + 1]], ego_all[offs[i]:offs[i + 1]])
                 for i, f in enumerate(frames)]
    else:                                   # no seeding crops: sample frames live from the session
        fr = t.frames()
        if cur is not None:
            fr = sorted(fr, key=lambda f: abs(f - cur))[:40]
        else:
            fr = fr[:: max(1, len(fr) // 40)]
        items = []
        for f in fr:
            ps, es = [], []
            for a in pr.meta.get("sensors") or pr.session.lidar_agents:
                pp = pr.session.points(a, f)
                x, y = t.poses[f][:2]
                m = (pp[:, 0] - x) ** 2 + (pp[:, 1] - y) ** 2 <= (L_ / 2 + 4) ** 2
                ps.append(pp[m]); es.append(np.full(int(m.sum()), a, np.int8))
            items.append((f, np.concatenate(ps), np.concatenate(es)))
    order = {f: i for i, f in enumerate(t.frames())}
    for f, pts, ego in items:
        if f not in t.poses or not len(pts):
            continue
        x, y, z, yaw = t.poses[f]
        c, s = math.cos(yaw), math.sin(yaw)
        dx, dy, dz = pts[:, 0] - x, pts[:, 1] - y, pts[:, 2] - z
        lx, ly = c * dx + s * dy, -s * dx + c * dy
        keep = (np.abs(lx) <= L_ / 2 + mx) & (np.abs(ly) <= W_ / 2 + mx) & (dz >= -H_ / 2 - 0.3) & (dz <= H_ / 2 + mz)
        if keep.any():
            chunks.append(np.column_stack([lx[keep], ly[keep], dz[keep]]).astype("<f4"))
            egos.append(ego[keep].astype(np.uint8))
            curs.append(np.full(int(keep.sum()), order[f], "<u2"))
    if chunks:
        pts, ego, fidx = np.concatenate(chunks), np.concatenate(egos), np.concatenate(curs)
        cap = 150_000
        if len(pts) > cap:                  # thin uniformly; the client highlights any frame
            keep = np.random.default_rng(0).choice(len(pts), cap, replace=False)
            keep.sort()
            pts, ego, fidx = pts[keep], ego[keep], fidx[keep]
    else:
        pts, ego, fidx = np.zeros((0, 3), "<f4"), np.zeros(0, np.uint8), np.zeros(0, "<u2")
    blob = struct.pack("<I", len(pts)) + pts.tobytes() + fidx.tobytes() + ego.tobytes()
    return Response(200, blob, "application/octet-stream")


# ============================================================================ UAV view
def uav_view(app, p, q, body):
    """Crop window around the selected box + every box projected into the drone image (full-image px)."""
    pr = _need_project(app)
    a, f = int(p["agent"]), int(p["frame"])
    cam = pr.session.camera(a)
    if cam is None or not cam.has(f) or pr.session.image_path(a, f) is None:
        return ok({"available": False})
    tid = int(q["tid"]) if q.get("tid") else None
    boxes = pr.boxes_at(f)
    focus = next((b for b in boxes if b["tid"] == tid), None)
    if focus is None and tid is not None:
        t = pr.get(tid)
        focus = {"tid": tid, "box": t.box(t.nearest_frame(f))}
    aspect = min(4.0, max(0.25, float(q.get("aspect", 1.0))))
    if focus is not None:
        x0, y0, w, h = cam.crop_window(f, focus["box"], pad=float(q.get("pad", 2.4)),
                                       min_px=int(q.get("min", 420)), aspect=aspect)
    else:                                   # nothing selected: the whole frame, centred
        W, H = cam.size
        h = min(H, W / aspect)
        w = h * aspect
        x0, y0, w, h = int((W - w) // 2), int((H - h) // 2), int(w), int(h)
    out = []
    for b in boxes:
        uv = cam.box_corners_px(f, b["box"])
        c = uv.mean(axis=0)
        if x0 - w * 0.2 <= c[0] <= x0 + w * 1.2 and y0 - h * 0.2 <= c[1] <= y0 + h * 1.2:
            out.append({"tid": b["tid"], "status": b["status"], "corners": np.round(uv, 1).tolist()})
    return ok({"available": True, "window": [x0, y0, w, h], "size": list(cam.size), "boxes": out,
               "camera": cam.name})


def uav_locate(app, p, q, body):
    """A pixel of the drone image -> the map point under it at mid-vehicle height (default 0.78 m)."""
    pr = _need_project(app)
    a, f = int(p["agent"]), int(p["frame"])
    cam = pr.session.camera(a)
    if cam is None or not cam.has(f):
        return err("no camera for this frame", 404)
    xyz = cam.pixel_to_map(f, [[float(q["u"]), float(q["v"])]], height=float(q.get("h", 0.78)))[0]
    return ok({"xy": [round(float(xyz[0]), 3), round(float(xyz[1]), 3)]})


def uav_image(app, p, q, body):
    pr = _need_project(app)
    a, f = int(p["agent"]), int(p["frame"])
    path = pr.session.image_path(a, f)
    if path is None:
        return err("no image", 404)
    import cv2
    img = app.images.get((a, f), lambda: cv2.imread(str(path), cv2.IMREAD_COLOR))
    x0, y0 = int(q.get("x0", 0)), int(q.get("y0", 0))
    w = int(q.get("w", q.get("side", img.shape[1])))
    h = int(q.get("h", q.get("side", img.shape[0])))
    outw = max(16, min(2400, int(q.get("out", 640))))               # output width in px
    crop = img[max(0, y0):y0 + h, max(0, x0):x0 + w]
    if crop.size == 0:
        return err("empty crop")
    outh = max(16, round(outw * crop.shape[0] / crop.shape[1]))
    crop = cv2.resize(crop, (outw, outh), interpolation=cv2.INTER_AREA)
    okk, enc = cv2.imencode(".jpg", crop, [cv2.IMWRITE_JPEG_QUALITY, 86])
    return Response(200, enc.tobytes(), "image/jpeg", {"Cache-Control": "max-age=3600"})


# ============================================================================ jobs
def seed_start(app, p, q, body):
    pr = _need_project(app)
    mode = body.get("mode", "daa")
    if mode in ("daa", "uav") and not body.get("detections"):
        return err("choose a UAV detections file")
    if mode in ("daa", "uav") and not Path(body["detections"]).exists():
        return err(f"not found: {body['detections']}")
    app.settings.remember("detections", body.get("detections"))
    jid = app.jobs.start_seed(mode, body.get("detections"), body.get("labels"),
                              body.get("sensors") or pr.meta.get("sensors"), body.get("limit"),
                              body.get("workers"))
    return ok(app.jobs.status(jid), 202)


def jobs_list(app, p, q, body):
    return ok(app.jobs.active() if app.jobs else [])


def job_get(app, p, q, body):
    try:
        return ok(app.jobs.status(p["id"]))
    except KeyError:
        return err("no such job", 404)


def job_cancel(app, p, q, body):
    return ok(app.jobs.cancel(p["id"]))


def export_start(app, p, q, body):
    pr = _need_project(app)
    fmt = body.get("format", "yaml")
    statuses = tuple(body.get("statuses") or ex.REVIEWED)
    if body.get("apply"):
        jid = app.jobs.start_thread("apply", ex.apply_to_dataset, pr, statuses)
    else:
        out = body.get("out") or str(pr.path / "exports" / ("labels" if fmt == "yaml" else "labels.csv"))
        fn = ex.export_opencood if fmt == "yaml" else ex.export_csv
        jid = app.jobs.start_thread("export", fn, pr, out, statuses)
    return ok(app.jobs.status(jid), 202)


# ============================================================================ misc
def fs_list(app, p, q, body):
    """Directory listing for the file pickers: folders, plus .csv/.json files. Names only."""
    path = Path(q.get("path") or str(Path.home())).expanduser()
    if not path.is_dir():
        path = path.parent
    try:
        entries = []
        for e in sorted(path.iterdir(), key=lambda e: (not e.is_dir(), e.name.lower())):
            if e.name.startswith("."):
                continue
            if e.is_dir():
                is_session = (e / "0").is_dir()
                is_project = (e / "project.json").exists()
                entries.append({"name": e.name, "dir": True, "session": is_session, "project": is_project})
            elif e.suffix.lower() in (".csv", ".json"):
                entries.append({"name": e.name, "dir": False})
    except PermissionError:
        return err(f"permission denied: {path}", 403)
    return ok({"path": str(path), "parent": str(path.parent), "entries": entries[:2000],
               "is_session": (path / "0").is_dir(), "is_project": (path / "project.json").exists()})


def session_info(app, p, q, body):
    """Validate a folder as a session for the wizard: frames, sensors, UAV cameras."""
    try:
        return ok(Session(q.get("path", "")).info())
    except (SessionError, OSError) as e:
        return err(str(e))


def suggest(app, p, q, body):
    """Smart defaults for the new-project wizard (detections file, project folder)."""
    session = q.get("session", "")
    return ok(app.settings.suggest(session))


def client_log(app, p, q, body):
    app.log_client(body)
    return ok({"logged": True})


ROUTES = [
    ("GET", r"/api/app", app_info),
    ("POST", r"/api/projects", projects_create),
    ("POST", r"/api/projects/open", projects_open),
    ("POST", r"/api/projects/close", projects_close),
    ("GET", r"/api/project", project_get),
    ("PUT", r"/api/project", project_update),
    ("GET", r"/api/tracks", tracks_list),
    ("POST", r"/api/tracks", track_post),
    ("GET", r"/api/tracks/(?P<tid>\d+)", track_get),
    ("PUT", r"/api/tracks/(?P<tid>\d+)", track_put),
    ("GET", r"/api/tracks/(?P<tid>\d+)/aggregate", track_aggregate),
    ("GET", r"/api/history", history),
    ("GET", r"/api/frames/(?P<frame>\d+)/points", frame_points),
    ("GET", r"/api/frames/(?P<frame>\d+)/boxes", frame_boxes),
    ("GET", r"/api/uav/(?P<agent>\d+)/(?P<frame>\d+)/view", uav_view),
    ("GET", r"/api/uav/(?P<agent>\d+)/(?P<frame>\d+)/image", uav_image),
    ("GET", r"/api/uav/(?P<agent>\d+)/(?P<frame>\d+)/locate", uav_locate),
    ("POST", r"/api/seed", seed_start),
    ("GET", r"/api/jobs", jobs_list),
    ("GET", r"/api/jobs/(?P<id>[\w\-]+)", job_get),
    ("POST", r"/api/jobs/(?P<id>[\w\-]+)/cancel", job_cancel),
    ("POST", r"/api/export", export_start),
    ("GET", r"/api/fs", fs_list),
    ("GET", r"/api/suggest", suggest),
    ("GET", r"/api/session-info", session_info),
    ("POST", r"/api/client-log", client_log),
]
