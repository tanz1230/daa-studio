import unittest

import numpy as np

from studio.seeding.config import adopted_config
from studio.seeding.crops import build_trackframes
from studio.seeding.detections import Detections, TrackDets, load_detections, velocities
from tests._data import DETECTIONS, SESSION, needs_detections, needs_session


class Velocities(unittest.TestCase):
    def test_constant_velocity_is_exact_and_gaps_use_frame_time(self):
        frames = np.array([0, 1, 2, 5, 6])                  # a 3-frame gap
        xy = np.column_stack([frames * 0.8, -frames * 0.2])   # 8 m/s east, -2 m/s north at 10 Hz
        v = velocities(frames, xy)
        np.testing.assert_allclose(v, np.tile([8.0, -2.0], (5, 1)), atol=1e-9)
        np.testing.assert_allclose(velocities([3], [[1.0, 2.0]]), [[0.0, 0.0]])


@needs_session
@needs_detections
class Seeding(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from studio.session import Session
        cls.s = Session(SESSION)
        cls.dets = load_detections(DETECTIONS, cls.s)

    def test_detection_report(self):
        r = self.dets.report
        self.assertGreater(r["tracks"], 100)
        self.assertEqual(r["rows"], r["rows_matched"] + r["rows_off_session"])
        self.assertLessEqual(r["frames_with_detections"], r["session_frames"])
        cls = {t.veh_class for t in self.dets.tracks.values()}
        self.assertEqual(cls, {0, 1})

    def test_crops_hold_the_vehicle(self):
        """Vehicles near a sensor must arrive with their points: compare each crop with an
        independent count of the merged sweep inside the vehicle's GT box (the crop window is
        deliberately generous, so it must hold at least that many)."""
        from studio.poses import to_box_frame
        f0 = self.s.frames[0]
        o = [self.s.sensor_origin(a, f0) for a in (0, 2)]
        starts = [t for t in self.dets.tracks.values() if t.frames[0] == f0]
        near = sorted(starts, key=lambda t: min(np.hypot(*(t.boxes[0, :2] - x[:2])) for x in o))[:3]
        sub = Detections({t.tid: TrackDets(t.tid, t.frames[:10], t.boxes[:10], t.conf[:10], t.veh_class)
                          for t in near})
        tfs = build_trackframes(self.s, sub, [0, 1, 2], adopted_config(), workers=1)
        self.assertEqual(set(tfs), {t.tid for t in near})
        gt = self.s.labels(f0)
        merged = np.concatenate([self.s.points(a, f0) for a in (0, 1, 2)])
        for tid, tf in tfs.items():
            n = tf.n_frames
            for arr in (tf.cand_pts, tf.cand_ego, tf.uav_boxes, tf.velocities, tf.ground_z, tf.lidar_origin):
                self.assertEqual(len(arr), n)
            self.assertGreater(len(tf.cand_pts[0]), 200)
            if tid in gt:
                box = gt[tid][0]
                loc = to_box_frame(merged, box)
                inside = ((np.abs(loc[:, 0]) <= box[3] / 2) & (np.abs(loc[:, 1]) <= box[4] / 2)
                          & (np.abs(loc[:, 2]) <= box[5] / 2)).sum()
                self.assertGreaterEqual(len(tf.cand_pts[0]), inside)

if __name__ == "__main__":
    unittest.main()
