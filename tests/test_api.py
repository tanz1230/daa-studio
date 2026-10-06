"""Smoke test: a live server on a free port, every route exercised once on real data."""
import json
import math
import struct
import tempfile
import threading
import unittest
import urllib.error
import urllib.request
from pathlib import Path

from studio import labels as L
from studio.project import Project
from studio.server import App, Settings, make_server
from studio.session import Session
from tests._data import SESSION, needs_session


@needs_session
class APISmoke(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        root = Path(cls.tmp.name)
        cls.app = App("tester", Settings(root / "settings.json"))
        cls.srv = make_server(cls.app, "127.0.0.1", 18765)
        cls.base = f"http://127.0.0.1:{cls.srv.server_address[1]}"
        threading.Thread(target=cls.srv.serve_forever, daemon=True).start()
        s = Session(SESSION)
        cls.f0 = s.frames[0]
        box, _ = s.labels(cls.f0)[next(iter(s.labels(cls.f0)))]
        cls.box = box

    @classmethod
    def tearDownClass(cls):
        cls.srv.shutdown()
        cls.app.close_project()
        cls.tmp.cleanup()

    def call(self, method, path, body=None, raw=False):
        data = json.dumps(body).encode() if body is not None else None
        req = urllib.request.Request(self.base + path, data=data, method=method,
                                     headers={"Content-Type": "application/json", "X-User": "tester"})
        try:
            with urllib.request.urlopen(req, timeout=60) as r:
                payload = r.read()
                return r.status, (payload if raw else json.loads(payload or b"null")), dict(r.headers)
        except urllib.error.HTTPError as e:
            return e.code, json.loads(e.read() or b"null"), {}

    def test_full_flow(self):
        st, info, _ = self.call("GET", "/api/app")
        self.assertEqual((st, info["project"]), (200, None))
        st, body, _ = self.call("GET", "/api/project")
        self.assertEqual(st, 404)                                       # nothing open yet

        proj = str(Path(self.tmp.name) / "p")
        st, body, _ = self.call("POST", "/api/projects", {"path": proj, "session": str(SESSION)})
        self.assertEqual(st, 201, body)
        self.assertEqual(body["session"]["frames"], len(Session(SESSION).frames))

        x, y, z = (float(v) for v in self.box[:3])
        t = L.Track(tid=0, shape=[4.5, 1.8, 1.5], poses={self.f0: [x, y, z, float(self.box[6])]})
        st, made, _ = self.call("POST", "/api/tracks", {"track": t.to_json()})
        self.assertEqual((st, made["status"], made["version"]), (201, "new", 1))
        tid = made["tid"]
        self.assertGreaterEqual(tid, 1)                     # the server assigns ids, never the client's 0

        # z = null: the server stands the box on the DTM ground (centre = ground + H / 2)
        raw = t.to_json()
        raw["poses"] = {str(self.f0): [x + 30.0, y, None, 0.0]}
        st, stood, _ = self.call("POST", "/api/tracks", {"track": raw})
        z_new = stood["poses"][str(self.f0)][2]
        self.assertLess(abs(z_new - z), 1.5)                # same road surface, nearby
        self.assertNotEqual(stood["tid"], tid)
        stood["status"] = "deleted"
        self.call("PUT", f"/api/tracks/{stood['tid']}", {"track": stood, "expected_version": 1})

        moved = L.move(L.Track.from_json(made), None, 0.3)
        st, saved, _ = self.call("PUT", f"/api/tracks/{tid}", {"track": moved.to_json(), "expected_version": 1, "op": "move"})
        self.assertEqual((st, saved["version"]), (200, 2))
        st, stale, _ = self.call("PUT", f"/api/tracks/{tid}", {"track": moved.to_json(), "expected_version": 1})
        self.assertEqual(st, 409)
        self.assertEqual(stale["current"]["version"], 2)

        st, lst, _ = self.call("GET", "/api/tracks")
        self.assertEqual([r["tid"] for r in lst if r["status"] != "deleted"], [tid])
        st, boxes, _ = self.call("GET", f"/api/frames/{self.f0}/boxes")
        self.assertEqual(boxes[0]["tid"], tid)

        st, blob, _ = self.call("GET", f"/api/frames/{self.f0}/points?agents=0,2", raw=True)
        n = struct.unpack("<I", blob[:4])[0]
        self.assertGreater(n, 1000)
        self.assertEqual(len(blob), 4 + n * 12 + n)

        st, blob, _ = self.call("GET", f"/api/tracks/{tid}/aggregate?frame={self.f0}", raw=True)
        n = struct.unpack("<I", blob[:4])[0]
        self.assertEqual(len(blob), 4 + n * 12 + 2 * n + n)
        self.assertGreater(n, 10)

        st, view, _ = self.call("GET", f"/api/uav/3/{self.f0}/view?tid={tid}&aspect=1.6")
        self.assertTrue(view["available"])
        x0, y0, w, h = view["window"]
        self.assertAlmostEqual(w / h, 1.6, delta=0.02)
        st, jpg, hdr = self.call("GET", f"/api/uav/3/{self.f0}/image?x0={x0}&y0={y0}&w={w}&h={h}&out=320", raw=True)
        self.assertEqual(jpg[:2], b"\xff\xd8")                           # a JPEG

        # a click on the vehicle in the drone image lands on its map position
        mine = next(b for b in view["boxes"] if b["tid"] == tid)
        u, v = (sum(c[i] for c in mine["corners"]) / 8 for i in (0, 1))
        st, loc, _ = self.call("GET", f"/api/uav/3/{self.f0}/locate?u={u}&v={v}")
        self.assertLess(math.hypot(loc["xy"][0] - (x + 0.3), loc["xy"][1] - y), 0.35)   # moved 0.3 m in x

        st, hist, _ = self.call("GET", f"/api/history?tid={tid}")
        self.assertEqual([h["op"] for h in hist], ["add", "move"])        # only this track's history
        self.assertEqual(hist[-1]["user"], "tester")

        st, job, _ = self.call("POST", "/api/export", {"format": "csv"})
        self.assertEqual(st, 202)
        import time
        for _ in range(60):
            st, job, _ = self.call("GET", f"/api/jobs/{job['id']}")
            if job["state"] != "running":
                break
            time.sleep(0.5)
        self.assertEqual(job["state"], "done", job)

        st, fs, _ = self.call("GET", f"/api/fs?path={SESSION.parent}")
        self.assertTrue(any(e["session"] for e in fs["entries"]))
        st, sug, _ = self.call("GET", f"/api/suggest?session={SESSION}")
        self.assertIn("project", sug)
        st, _b, _ = self.call("POST", "/api/client-log", {"level": "info", "message": "smoke"})
        self.assertEqual(st, 200)
        st, page, _ = self.call("GET", "/", raw=True)
        self.assertIn(b"DAA Studio", page)


if __name__ == "__main__":
    unittest.main()
