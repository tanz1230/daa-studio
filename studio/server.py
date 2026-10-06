"""The DAA Studio server: stdlib only (http.server), one process, one open project.

Binds to 127.0.0.1 by default. To let annotators on other machines connect, start it with
--host 0.0.0.0 and share http://<this-machine>:<port>/ (everyone then works in the same project;
saves are versioned, so concurrent edits of one track are refused rather than lost).
"""
from __future__ import annotations

import datetime as _dt
import json
import mimetypes
import os
import re
import socket
import sys
import threading
import traceback
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qsl, unquote, urlparse

from . import __version__, api
from .jobs import JobManager
from .project import Project, ProjectError, StaleVersion
from .session import SessionError, _LRU

WEB = Path(__file__).resolve().parent.parent / "web"
HOME = Path(os.environ.get("DAA_STUDIO_HOME") or Path.home() / ".daa_studio")   # settings + client log

# where to look for a session's UAV detections; {n} = session number, {name} = session folder name,
# {dir} = the session folder. First existing file wins. Add your own as "detection_patterns" in
# ~/.daa_studio/settings.json.
DEFAULT_DETECTION_PATTERNS = [
    "{dir}/uav_detections.csv",
    "{dir}/../uav_detections/{name}.csv",
]


class Settings:
    def __init__(self, path: Path = HOME / "settings.json"):
        self.path = path
        self.data = {}
        if path.exists():
            try:
                self.data = json.loads(path.read_text())
            except json.JSONDecodeError:
                self.data = {}
        self._lock = threading.Lock()

    def _save(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(json.dumps(self.data, indent=1))

    def recent(self) -> list:
        return [r for r in self.data.get("recent", []) if Path(r["path"], "project.json").exists()]

    def add_recent(self, project: Project) -> None:
        with self._lock:
            rec = [r for r in self.data.get("recent", []) if r["path"] != str(project.path)]
            st = project.stats()
            rec.insert(0, {"path": str(project.path), "name": project.meta.get("name"),
                           "session": project.meta.get("session"), "tracks": st["total"],
                           "reviewed": st["reviewed"],
                           "opened": _dt.datetime.now().isoformat(timespec="minutes")})
            self.data["recent"] = rec[:12]
            self._save()

    def remember(self, key, value) -> None:
        if value:
            with self._lock:
                self.data.setdefault("last", {})[key] = value
                self._save()

    def suggest(self, session: str) -> dict:
        out = {"detections": None, "project": None}
        if not session:
            return out
        sp = Path(session).expanduser()
        name = sp.name
        n = name.split("_")[-1]
        for pat in self.data.get("detection_patterns", DEFAULT_DETECTION_PATTERNS):
            cand = Path(pat.format(n=n, name=name, dir=str(sp)))
            if cand.exists():
                out["detections"] = str(cand)
                break
        root = self.data.get("project_root")          # optional: where new projects go by default
        out["project"] = str(Path(root).expanduser() / name if root
                             else sp.parent.parent / f"{sp.parent.name}_studio" / name)
        return out


class App:
    def __init__(self, user: str, settings: Settings):
        self.user = user
        self.settings = settings
        self.project = None
        self.jobs = None
        self.images = _LRU(6)
        self._lock = threading.Lock()

    def open_project(self, project: Project) -> None:
        with self._lock:
            self.close_project()
            self.project = project
            self.jobs = JobManager(project)
            self.settings.add_recent(project)

    def close_project(self) -> None:
        if self.jobs:
            self.jobs.shutdown()
        self.project, self.jobs = None, None

    def user_of(self, body) -> str:
        return (body or {}).get("user") or self.user or "annotator"

    def log_client(self, body) -> None:
        HOME.mkdir(parents=True, exist_ok=True)
        line = json.dumps({"ts": _dt.datetime.now().isoformat(timespec="seconds"), **(body or {})})
        with open(HOME / "client.log", "a") as f:
            f.write(line + "\n")
        if (body or {}).get("level") == "error":
            print(f"[client error] {body.get('message')}", file=sys.stderr)


ROUTES = [(m, re.compile("^" + pat + "$"), fn) for m, pat, fn in api.ROUTES]


class Handler(BaseHTTPRequestHandler):
    server_version = f"DAAStudio/{__version__}"
    app: App = None

    def log_message(self, fmt, *args):            # quiet: errors only
        pass

    def _send(self, r: api.Response) -> None:
        try:
            self._write(r)
        except (BrokenPipeError, ConnectionResetError):      # the browser cancelled the request
            pass

    def _write(self, r: api.Response) -> None:
        self.send_response(r.status)
        self.send_header("Content-Type", r.ctype)
        self.send_header("Content-Length", str(len(r.body)))
        for k, v in r.headers.items():
            self.send_header(k, v)
        if "Cache-Control" not in r.headers:
            self.send_header("Cache-Control", "no-store")
        self.end_headers()
        if self.command != "HEAD":
            self.wfile.write(r.body)

    def _body(self):
        n = int(self.headers.get("Content-Length") or 0)
        if not n:
            return {}
        raw = self.rfile.read(n)
        try:
            return json.loads(raw.decode() or "{}")
        except json.JSONDecodeError:
            return {}

    def _dispatch(self, method: str) -> None:
        url = urlparse(self.path)
        path = unquote(url.path)
        if path.startswith("/api/"):
            q = dict(parse_qsl(url.query))
            body = self._body() if method in ("POST", "PUT") else {}
            if isinstance(body, dict) and not body.get("user"):
                body["user"] = self.headers.get("X-User", "")
            for m, rx, fn in ROUTES:
                mt = rx.match(path)
                if mt and m == method:
                    try:
                        resp = fn(self.app, mt.groupdict(), q, body)
                    except LookupError as e:                 # no project / unknown track
                        resp = api.err(str(e).strip("'\""), 404)
                    except StaleVersion as e:
                        resp = api.err(str(e), 409)
                    except (ValueError, ProjectError, SessionError) as e:
                        resp = api.err(str(e), 400)
                    except Exception as e:                   # bug: report, keep serving
                        traceback.print_exc()
                        resp = api.err(f"{type(e).__name__}: {e}", 500)
                    return self._send(resp)
            return self._send(api.err(f"no route {method} {path}", 404))
        if method != "GET":
            return self._send(api.err("not found", 404))
        return self._static(path)

    def _static(self, path: str) -> None:
        rel = "index.html" if path in ("", "/") else path.lstrip("/")
        f = (WEB / rel).resolve()
        if WEB not in f.parents or not f.is_file():         # traversal or unknown: SPA fallback
            f = WEB / "index.html"                         # single-page app fallback
        ctype = mimetypes.guess_type(str(f))[0] or "application/octet-stream"
        if f.suffix == ".js":
            ctype = "text/javascript"
        cache = "no-cache" if f.suffix in (".html", ".js", ".css") else "max-age=86400"
        self._send(api.Response(200, f.read_bytes(), ctype, {"Cache-Control": cache}))

    def do_GET(self):
        self._dispatch("GET")

    def do_HEAD(self):
        self._dispatch("GET")

    def do_POST(self):
        self._dispatch("POST")

    def do_PUT(self):
        self._dispatch("PUT")


def _free_port(host: str, port: int) -> int:
    for p in range(port, port + 20):
        with socket.socket() as s:
            s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)     # as the server itself binds
            try:
                s.bind((host, p))
                return p
            except OSError:
                continue
    raise OSError(f"no free port in {port}-{port + 19}")


def make_server(app: App, host: str = "127.0.0.1", port: int = 8765) -> ThreadingHTTPServer:
    handler = type("BoundHandler", (Handler,), {"app": app})
    srv = ThreadingHTTPServer((host, _free_port(host, port)), handler)
    srv.daemon_threads = True
    return srv


def serve(project_path=None, host="127.0.0.1", port=8765, open_browser=True, user="") -> int:
    app = App(user, Settings())
    if project_path:
        app.open_project(Project.open(project_path))
    srv = make_server(app, host, port)
    url = f"http://{'127.0.0.1' if host in ('127.0.0.1', '0.0.0.0', 'localhost') else host}:{srv.server_address[1]}/"
    print(f"\n  DAA Studio {__version__}  ->  {url}\n  (Ctrl+C to stop)\n", flush=True)
    if open_browser:
        threading.Timer(0.6, lambda: webbrowser.open(url)).start()
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        app.close_project()
        srv.server_close()
    return 0
