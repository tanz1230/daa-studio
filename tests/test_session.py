import unittest

import numpy as np

from studio.session import Session, SessionError
from tests._data import FRAME, SESSION, needs_session


@needs_session
class SessionTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.s = Session(SESSION)

    def test_frames_and_agents(self):
        n_yaml = len(list((SESSION / "0").glob("*.yaml")))
        self.assertEqual(len(self.s.frames), n_yaml)
        self.assertEqual(self.s.frames[0], FRAME)
        self.assertEqual(self.s.lidar_agents, [0, 1, 2])
        self.assertEqual(self.s.uav_agents, [3, 4])
        self.assertEqual(self.s.info()["frames"], n_yaml)

    def test_points_are_map_frame_around_the_sensor(self):
        for agent in (0, 2):
            pts = self.s.points(agent, FRAME)
            o = self.s.sensor_origin(agent, FRAME)
            self.assertGreater(len(pts), 500)
            d = np.hypot(pts[:, 0] - o[0], pts[:, 1] - o[1])
            self.assertLess(np.median(d), 60.0)
        full = self.s.points(0, FRAME, drop_ground=False)
        self.assertGreater(len(full), len(self.s.points(0, FRAME)))   # ground really removed

    def test_labels_are_box7(self):
        labels = self.s.labels(FRAME)
        self.assertGreater(len(labels), 10)
        box, cls = next(iter(labels.values()))
        self.assertEqual(box.shape, (7,))
        self.assertTrue(2.0 < box[3] < 20.0 and 1.0 < box[4] < 4.0)
        self.assertIn(cls, ("Car", "Truck"))

    def test_uav_assets(self):
        self.assertIsNotNone(self.s.image_path(3, FRAME))
        self.assertIsNotNone(self.s.camera(3))
        self.assertGreater(len(self.s.annotation_area()), 3)

    def test_rejects_non_session(self):
        with self.assertRaises(SessionError):
            Session(SESSION.parent)


if __name__ == "__main__":
    unittest.main()
