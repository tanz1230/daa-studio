"""DAA Studio command line.

    python -m studio serve   [--project P] [--port 8765] [--host 127.0.0.1] [--no-browser] [--user NAME]
    python -m studio new     --project P --session S [--name N] [--sensors 0,1,2] [--uav 3]
    python -m studio seed    --project P --mode daa|uav|session|import [--detections CSV] [--labels CSV]
                             [--sensors 0,1,2] [--workers N] [--limit N] [--import]
    python -m studio export  --project P --format yaml|csv --out PATH [--all | --statuses a,b]
    python -m studio info    SESSION
"""
from __future__ import annotations

import argparse
import json
import sys


def _ints(s):
    return [int(x) for x in s.split(",") if x.strip()] if s else None


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="studio", description="DAA Studio: UAV-assisted 3D labeling")
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("serve", help="start the labeling server and open the browser")
    p.add_argument("--project")
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8765)
    p.add_argument("--no-browser", action="store_true")
    p.add_argument("--user", default="")

    p = sub.add_parser("new", help="create a project for a session")
    p.add_argument("--project", required=True)
    p.add_argument("--session", required=True)
    p.add_argument("--name")
    p.add_argument("--sensors", type=_ints)
    p.add_argument("--uav", type=int)

    p = sub.add_parser("seed", help="generate seeds for a project (DAA, UAV, session or CSV)")
    p.add_argument("--project", required=True)
    p.add_argument("--mode", choices=("daa", "uav", "session", "import"), default="daa")
    p.add_argument("--detections")
    p.add_argument("--labels")
    p.add_argument("--sensors", type=_ints)
    p.add_argument("--workers", type=int)
    p.add_argument("--limit", type=int)
    p.add_argument("--run-id")
    p.add_argument("--progress")
    p.add_argument("--import", dest="do_import", action="store_true",
                   help="import the run into the project at the end (standalone use)")

    p = sub.add_parser("export", help="export labels")
    p.add_argument("--project", required=True)
    p.add_argument("--format", choices=("yaml", "csv"), default="yaml")
    p.add_argument("--out", required=True)
    g = p.add_mutually_exclusive_group()
    g.add_argument("--all", action="store_true",
                   help="every live track, including unreviewed seeds (e.g. DAA pseudo labels as generated)")
    g.add_argument("--statuses")

    p = sub.add_parser("info", help="describe a session")
    p.add_argument("session")

    a = ap.parse_args(argv)
    if a.cmd == "serve":
        from .server import serve
        return serve(a.project, a.host, a.port, open_browser=not a.no_browser, user=a.user)
    if a.cmd == "new":
        from .project import Project
        pr = Project.create(a.project, a.session, a.name, a.sensors, a.uav)
        print(json.dumps(pr.meta, indent=1))
        return 0
    if a.cmd == "seed":
        from .seeding.run import run_seeding
        meta = run_seeding(a.project, a.mode, a.detections, a.labels, a.sensors, a.workers,
                           a.limit, a.run_id, a.progress, a.do_import)
        print(json.dumps({k: meta[k] for k in ("run", "mode", "tracks", "seconds") if k in meta}))
        return 0
    if a.cmd == "export":
        from . import export as ex
        from .project import Project
        pr = Project.open(a.project)
        statuses = ex.ALL_LIVE if a.all else (tuple(a.statuses.split(",")) if a.statuses else ex.REVIEWED)
        prov = (ex.export_opencood(pr, a.out, statuses) if a.format == "yaml"
                else ex.export_csv(pr, a.out, statuses))
        print(json.dumps(prov, indent=1))
        return 0
    if a.cmd == "info":
        from .session import Session
        print(json.dumps(Session(a.session).info(), indent=1))
        return 0
    return 1


if __name__ == "__main__":
    sys.exit(main())
