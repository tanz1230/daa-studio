"""End-to-end seeding: the server's JobManager runs the real subprocess on real data."""
import json
import subprocess
import sys
import tempfile
import time
import unittest
from pathlib import Path

from studio.jobs import STUDIO_ROOT, JobManager
from studio.project import Project
from tests._data import DETECTIONS, SESSION, needs_detections, needs_session


@needs_session
@needs_detections
class SeedingJob(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name) / "proj"
        self.p = Project.create(self.root, SESSION)

    def tearDown(self):
        self.tmp.cleanup()

    def test_daa_job_streams_tracks_into_the_project(self):
        jm = JobManager(self.p, poll_s=0.5)
        jid = jm.start_seed("daa", detections=DETECTIONS, limit=4, workers=4)
        deadline = time.time() + 900
        st = jm.status(jid)
        while st["state"] == "running" and time.time() < deadline:
            time.sleep(2)
            st = jm.status(jid)
        jm.shutdown()
        self.assertEqual(st["state"], "done", st)
        tracks = self.p.tracks()
        self.assertEqual(len(tracks), 4)
        kinds = {t["seed"] for t in tracks}
        self.assertLessEqual(kinds, {"daa", "uav-fallback"})
        self.assertIn("daa", kinds)
        self.assertTrue(all((self.root / "cache" / "crops" / f"{t['tid']}.npz").exists() for t in tracks))
        meta = json.loads((self.root / "seeds" / st["run"] / "meta.json").read_text())
        self.assertEqual(meta["tracks"], 4)
        self.assertEqual(meta["config"]["delta_m"], 0.3)
        diag = self.p.get(tracks[0]["tid"]).seed["diag"]
        self.assertIn("risk", diag)

    def test_cli_uav_mode_imports_standalone(self):
        r = subprocess.run([sys.executable, "-m", "studio", "seed", "--project", str(self.root),
                            "--mode", "uav", "--detections", str(DETECTIONS), "--limit", "3",
                            "--workers", "2", "--import"], cwd=STUDIO_ROOT, capture_output=True, text=True)
        self.assertEqual(r.returncode, 0, r.stderr[-2000:])
        again = Project.open(self.root)
        self.assertEqual(len(again.tracks()), 3)
        self.assertEqual({t["seed"] for t in again.tracks()}, {"uav"})


if __name__ == "__main__":
    unittest.main()
