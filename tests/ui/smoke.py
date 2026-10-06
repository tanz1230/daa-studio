"""End-to-end UI smoke test: a real server, headless Chrome, the main review workflows.

    python tests/ui/smoke.py --project /path/to/a/seeded/project [--out shots/] [--keep]

The project is COPIED next to itself first; the copy is what the test edits, and it is removed at
the end (--keep leaves it). Every step checks the result on the server, not just in the page, and
screenshots go to --out. Exit status 0 = all steps passed.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import shutil
import socket
import subprocess
import sys
import tempfile
import time
import traceback
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from cdp import Chrome  # noqa: E402

STUDIO = Path(__file__).resolve().parents[2]


class Server:
    def __init__(self, project: Path, home: Path):
        with socket.socket() as s:
            s.bind(("127.0.0.1", 0))
            self.port = s.getsockname()[1]
        env = dict(os.environ, DAA_STUDIO_HOME=str(home), PYTHONPATH=str(STUDIO))
        self.proc = subprocess.Popen([sys.executable, "-m", "studio", "serve", "--project", str(project),
                                      "--no-browser", "--user", "smoke", "--port", str(self.port)],
                                     cwd=str(STUDIO), env=env, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
        self.base = f"http://127.0.0.1:{self.port}"
        for _ in range(100):
            try:
                self.get("/api/app")
                return
            except OSError:
                time.sleep(0.1)
        raise RuntimeError("server did not start")

    def get(self, path):
        return json.load(urllib.request.urlopen(self.base + path, timeout=30))

    def stop(self):
        self.proc.terminate()
        try:
            self.proc.wait(10)
        except subprocess.TimeoutExpired:
            self.proc.kill()
        return self.proc.stderr.read().decode(errors="replace")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--project", required=True)
    ap.add_argument("--out", default=None)
    ap.add_argument("--keep", action="store_true")
    a = ap.parse_args()
    src = Path(a.project).resolve()
    copy = src.parent / f"{src.name}_smoke_{os.getpid()}"
    out = Path(a.out or tempfile.mkdtemp(prefix="daa-smoke-shots-"))
    out.mkdir(parents=True, exist_ok=True)
    home = Path(tempfile.mkdtemp(prefix="daa-smoke-home-"))
    shutil.copytree(src, copy)
    srv = Server(copy, home)
    c = Chrome(1600, 1000)
    results = []

    def step(name):
        def deco(fn):
            t0 = time.time()
            try:
                fn()
                results.append((name, True, f"{time.time() - t0:.1f}s"))
            except Exception as e:                          # report and continue with the next step
                results.append((name, False, f"{type(e).__name__}: {e}"))
                traceback.print_exc()
            try:
                c.shot(str(out / f"{len(results):02d}_{name.replace(' ', '_')}.png"))
            except Exception:
                pass
            return fn
        return deco

    js = c.js
    sel = lambda: js("__daa.S.tid")                         # noqa: E731
    track = lambda tid: srv.get(f"/api/tracks/{tid}")      # noqa: E731

    def settle(s=1.0):
        c.sleep(s)
        c.wait_for("document.getElementById('savestate')?.textContent === 'All saved'", 20)

    def at_canvas(selector, fx, fy):
        r = js(f"(() => {{ const r = document.querySelector({json.dumps(selector)}).getBoundingClientRect();"
               f" return [r.left + r.width * {fx}, r.top + r.height * {fy}]; }})()")
        return r

    try:
        c.goto(srv.base + "/", 1.0)
        js("localStorage.setItem('daa.user', 'Smoke test')")
        c.goto(srv.base + "/", 3.0)

        @step("editor loads")
        def _():
            c.wait_for("document.querySelectorAll('.trow').length > 0", 40)
            queue = [t for t in srv.get("/api/tracks") if t["status"] == "unreviewed"]
            assert js("document.querySelectorAll('.trow').length") == len(queue)

        st = {}

        @step("select top of queue")
        def _():
            c.key("n")
            c.wait_for("__daa.S.track !== null", 20)
            c.sleep(3)
            st["tid"], st["frame"] = sel(), js("__daa.S.frame")
            rows = js("[...document.querySelectorAll('.trow')].map((r) => +r.dataset.tid)")
            assert rows[0] == st["tid"], "N must select the highest-priority track"
            assert js("document.querySelector('.uav canvas').style.display !== 'none'"), "UAV image shown"

        @step("drag box in top view")
        def _():
            v0 = track(st["tid"])["version"]
            c.key("v")
            c.sleep(1.5)
            x, y = at_canvas(".viewport canvas", 0.5, 0.5)       # Top view is fitted on the selected box
            c.drag(x, y, x + 40, y, steps=10)
            settle()
            t = track(st["tid"])
            assert t["version"] == v0 + 1 and t["status"] == "edited", (v0, t["version"], t["status"])
            st["after_drag"] = t

        @step("keys edit then undo")
        def _():
            L0 = st["after_drag"]["shape"][0]
            for _ in range(3):
                c.key("ArrowRight")
            c.key("i")
            c.key("i")
            settle()
            t = track(st["tid"])
            assert abs(t["shape"][0] - (L0 + 0.10)) < 1e-6, t["shape"]
            c.key("z", ctrl=True)
            c.sleep(0.3)
            c.key("z", ctrl=True)
            settle()
            t = track(st["tid"])
            assert abs(t["shape"][0] - L0) < 1e-6
            assert t["poses"][str(st["frame"])] == st["after_drag"]["poses"][str(st["frame"])]

        @step("shape panel face drag")
        def _():
            L0 = track(st["tid"])["shape"][0]
            # the right face handle of the Top tab: centre + L/2 * scale (same maths as the panel)
            x, y = js("""(() => {
              const c = document.querySelector('.ortho canvas').getBoundingClientRect();
              const [L, W] = __daa.S.track.shape, pad = 1.4;
              const scale = Math.min((c.width - 40) / (2 * (L / 2 + pad)), (c.height - 40) / (2 * (W / 2 + pad)));
              return [c.left + c.width / 2 + (L / 2) * scale, c.top + c.height / 2]; })()""")
            c.drag(x, y, x + 25, y, steps=8)
            settle()
            L1 = track(st["tid"])["shape"][0]
            assert L1 > L0 + 0.05, (L0, L1)

        @step("enter moves on")
        def _():
            before = sel()
            c.key("Enter")
            c.sleep(2.5)
            assert sel() != before
            assert track(before)["status"] == "edited"            # an edited track is already reviewed

        @step("accept")
        def _():
            tid = sel()
            assert track(tid)["status"] == "unreviewed"
            c.key("Enter")
            settle(2.0)
            assert track(tid)["status"] == "accepted"

        @step("flag with reason")
        def _():
            tid = sel()
            c.key("F", shift=True)
            c.sleep(0.5)
            c.click_text("Two vehicles merged", ".popover .item")
            settle(2.0)
            t = track(tid)
            assert t["status"] == "flagged" and t["note"] == "Two vehicles merged", (t["status"], t["note"])
            assert sel() != tid

        @step("delete and undo")
        def _():
            tid = sel()
            c.key("Delete")
            c.sleep(0.5)
            c.click_text("Delete", ".modal-f button")
            settle(2.0)
            assert track(tid)["status"] == "deleted"
            c.key("z", ctrl=True)
            settle(2.0)
            assert track(tid)["status"] == "unreviewed" and sel() == tid

        @step("add a vehicle")
        def _():
            n0 = len(srv.get("/api/tracks"))
            c.key("a")
            c.sleep(0.6)
            assert js("__daa.S.placing") is True
            x, y = at_canvas(".viewport canvas", 0.3, 0.35)
            c.click(x, y)
            c.wait_for("__daa.S.track && __daa.S.track.status === 'new'", 20)
            c.sleep(1.0)
            tracks = srv.get("/api/tracks")
            assert len(tracks) == n0 + 1
            st["new"] = sel()
            assert js(f"!!document.querySelector('.trow.pinned[data-tid=\"{st['new']}\"]')"), "new track pinned"
            t = track(st["new"])
            z = t["poses"][str(js('__daa.S.frame'))][2]
            assert -5 < z < 5, z                                # stood on the ground by the server

        @step("parked copy to all frames")
        def _():
            c.click_text("Frames", ".sec button")
            c.sleep(0.4)
            c.click_text("Parked: copy", ".popover .item")
            settle()
            n = len(track(st["new"])["poses"])
            assert n == len(srv.get("/api/project")["frames"]), n

        @step("timeline scrub")
        def _():
            f0 = js("__daa.S.frame")
            x, y = at_canvas(".track-canvas", 0.75, 0.4)
            c.click(x, y)
            c.sleep(1.5)
            assert js("__daa.S.frame") != f0

        @step("export reviewed")
        def _():
            target = copy / "exports" / "smoke_csv.csv"
            c.click_text("Export", ".header button")
            c.sleep(0.6)
            c.click_text("DAA CSV", ".modal .seg button")
            js(f"(() => {{ const i = document.querySelector('.modal input.mono'); i.value = {json.dumps(str(target))};"
               " i.dispatchEvent(new Event('input')); })()")
            c.click_text("Export", ".modal-f button")
            c.wait_for("!!document.querySelector('.modal .infobox.ok')", 60)
            text = target.read_text()
            assert text.count("\n") > 10, "exported rows"
            c.key("Escape")

        @step("dialogs open and close")
        def _():
            c.click_text("Seed", ".header button")
            c.sleep(0.6)
            assert js("!!document.querySelector('.modal')")
            c.key("Escape")
            c.key("?")
            c.sleep(0.4)
            assert js("document.querySelector('.modal h3').textContent") == "Keyboard shortcuts"
            c.key("Escape")
            c.sleep(0.3)
            assert not js("!!document.querySelector('.scrim')")

        @step("close project")
        def _():
            c.click_sel(".header .btn.icon.ghost:last-child")
            c.sleep(0.4)
            c.click_text("Close project", ".popover .item")
            c.wait_for("!!document.querySelector('.start .rrow')", 20)

        errors = [m for lv, m in c.console if lv in ("error", "exception")]
        results.append(("no console errors", not errors, "; ".join(e[:160] for e in errors[:3])))
    finally:
        c.close()
        log = srv.stop()
        if not a.keep:
            shutil.rmtree(copy, ignore_errors=True)
        shutil.rmtree(home, ignore_errors=True)
    width = max(len(n) for n, _, _ in results)
    for name, ok, info in results:
        print(f"  {'PASS' if ok else 'FAIL'}  {name:<{width}}  {info}")
    if "Traceback" in log:
        print("\nserver errors:\n" + log[-3000:])
    print(f"\nscreenshots: {out}")
    return 0 if all(ok for _, ok, _ in results) and "Traceback" not in log else 1


if __name__ == "__main__":
    sys.exit(main())
