import tempfile
import unittest
from pathlib import Path

from studio import labels as L
from studio.importers import tracks_from_label_csv, tracks_from_session, write_label_csv
from studio.project import Project, ProjectError, StaleVersion
from tests._data import DETECTIONS, SESSION, needs_detections, needs_session


def toy(tid, status="unreviewed"):
    return L.Track(tid=tid, shape=[4.5, 1.8, 1.5], status=status,
                   poses={5746: [1.0, 2.0, 0.8, 0.1], 5747: [1.5, 2.0, 0.8, 0.1]})


@needs_session
class ProjectTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name) / "proj"
        self.p = Project.create(self.root, SESSION, name="test")

    def tearDown(self):
        self.tmp.cleanup()

    def test_create_open_round_trip(self):
        self.assertEqual(self.p.meta["sensors"], [0, 1, 2])
        self.assertEqual(self.p.meta["uav_agent"], 3)
        self.p.save(toy(1), 0, "ann", "seed")
        again = Project.open(self.root)
        self.assertEqual([t["tid"] for t in again.tracks()], [1])
        self.assertEqual(again.get(1).version, 1)
        with self.assertRaises(ProjectError):
            Project.create(self.root, SESSION)

    def test_save_versions_history_and_stale_refusal(self):
        t = self.p.save(toy(1), 0, "ann", "seed")
        t2 = self.p.save(L.move(t, None, 0.5), t.version, "ann", "move")
        self.assertEqual(t2.version, 2)
        self.assertEqual(t2.status, "edited")
        with self.assertRaises(StaleVersion):
            self.p.save(L.move(t, None, 1.0), t.version, "other", "move")     # stale
        ops = [h["op"] for h in self.p.history(1)]
        self.assertEqual(ops, ["seed", "move"])
        self.assertTrue((self.root / "tracks" / ".prev" / "1.json").exists())

    def test_import_keeps_human_work(self):
        self.p.save(toy(1), 0, "s", "seed")
        accepted = self.p.save(L.set_status(self.p.get(1), "accepted"), 1, "ann", "accept")
        self.p.save(toy(2), 0, "s", "seed")
        new_seed = [L.move(toy(1), None, 9.0), L.move(toy(2), None, 9.0), toy(3)]
        counts = self.p.import_seed("run2", new_seed)
        self.assertEqual(counts, {"added": 1, "replaced": 1, "kept": 1})
        self.assertEqual(self.p.get(1).poses, accepted.poses)                   # untouched
        self.assertAlmostEqual(self.p.get(2).poses[5746][0], 10.0)             # replaced
        self.assertEqual(self.p.meta["active_run"], "run2")

    def test_boxes_at_and_stats(self):
        self.p.save(toy(1), 0)
        self.p.save(toy(2, "deleted"), 0)
        self.assertEqual([b["tid"] for b in self.p.boxes_at(5746)], [1])
        st = self.p.stats()
        self.assertEqual(st["total"], 1)
        self.assertEqual(st["by_status"]["deleted"], 1)

    def test_label_csv_round_trip(self):
        tracks = [toy(1), toy(2)]
        csv_path = Path(self.tmp.name) / "labels.csv"
        self.assertEqual(write_label_csv(csv_path, tracks, "s"), 4)
        back = tracks_from_label_csv(csv_path, "import")
        self.assertEqual([t.tid for t in back], [1, 2])
        for a, b in zip(tracks, back):
            for f in a.frames():
                for x, y in zip(a.box(f), b.box(f)):
                    self.assertAlmostEqual(x, y, places=4)


@needs_session
@needs_detections
class ImportersOnRealData(unittest.TestCase):
    def test_session_annotations_and_stage1(self):
        from studio.session import Session
        s = Session(SESSION)
        gt = tracks_from_session(s)
        self.assertGreater(len(gt), 100)
        t = gt[0]
        self.assertTrue(set(t.poses) <= set(s.frames))
        uav = tracks_from_label_csv(DETECTIONS, "uav", frames=s.frames)
        self.assertGreater(len(uav), 100)


if __name__ == "__main__":
    unittest.main()
