import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from studio import labels as L
from studio.export import ALL_LIVE, export_csv, export_opencood
from studio.importers import tracks_from_label_csv
from studio.project import Project
from studio.session import Session
from tests._data import SESSION, needs_session


@needs_session
class ExportTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.p = Project.create(Path(self.tmp.name) / "proj", SESSION)
        f0, f1 = Session(SESSION).frames[:2]
        self.f0 = f0
        self.p.save(L.Track(tid=11, cls="Truck", shape=[7.2, 2.5, 3.1], status="accepted",
                            poses={f0: [10.0, -20.0, 1.2, 0.4], f1: [10.5, -20.0, 1.2, 0.4]}), 0)
        self.p.save(L.Track(tid=12, shape=[4.4, 1.8, 1.5], status="unreviewed",
                            poses={f0: [30.0, -40.0, 0.7, -1.2]}), 0)

    def tearDown(self):
        self.tmp.cleanup()

    def test_yaml_export_round_trips_through_the_reader(self):
        out = Path(self.tmp.name) / "export"
        prov = export_opencood(self.p, out)                              # reviewed only
        self.assertEqual(prov["tracks"], 1)
        exported = Session(out / SESSION.name, pose_convention="opencood")
        got = exported.labels(self.f0)
        self.assertEqual(set(got), {11})
        box, cls = got[11]
        np.testing.assert_allclose(box, self.p.get(11).box(self.f0), atol=1e-4)
        self.assertEqual(cls, "Truck")
        # the rest of the YAML (poses) is preserved, and every agent is mirrored
        src = Session(SESSION)
        self.assertEqual(exported._doc(0, self.f0)["lidar_pose"], src._doc(0, self.f0)["lidar_pose"])
        self.assertTrue((out / SESSION.name / "2").is_dir())
        self.assertTrue((out / SESSION.name / "provenance.json").exists())

    def test_csv_export_with_all_live_statuses(self):
        path = Path(self.tmp.name) / "labels.csv"
        prov = export_csv(self.p, path, ALL_LIVE)
        self.assertEqual(prov["rows"], 3)
        back = {t.tid: t for t in tracks_from_label_csv(path, "import")}
        self.assertEqual(set(back), {11, 12})
        self.assertTrue(json.loads(Path(str(path) + ".provenance.json").read_text())["tracks"] == 2)


if __name__ == "__main__":
    unittest.main()
