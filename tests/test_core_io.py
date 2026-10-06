import unittest

import numpy as np

from studio import poses
from studio.ground import GroundModel
from studio.pcdio import read_pcd_xyz
from tests._data import FRAME, SESSION, needs_session


@needs_session
class CoreIO(unittest.TestCase):
    def test_pcd_reads_finite_xyz(self):
        pts = read_pcd_xyz(SESSION / "0" / f"{FRAME:06d}.pcd")
        self.assertEqual(pts.shape[1], 3)
        self.assertGreater(len(pts), 1000)
        self.assertTrue(np.isfinite(pts).all())

    def test_convention_detected_and_rotation_orthonormal(self):
        conv = poses.detect_convention(SESSION)
        self.assertEqual(conv, poses.OPENCOOD)
        for agent in (0, 1, 2):
            pose = poses.load_yaml(SESSION / str(agent) / f"{FRAME:06d}.yaml")["lidar_pose"]
            R, t = poses.pose_R_t(agent, pose, conv)
            np.testing.assert_allclose(R @ R.T, np.eye(3), atol=1e-9)
            self.assertAlmostEqual(np.linalg.det(R), 1.0, places=9)

    def test_ground_model_puts_road_near_zero(self):
        g = GroundModel()
        self.assertTrue(g.ok)
        conv = poses.detect_convention(SESSION)
        pose = poses.load_yaml(SESSION / "0" / f"{FRAME:06d}.yaml")["lidar_pose"]
        R, t = poses.pose_R_t(0, pose, conv)
        pts = read_pcd_xyz(SESSION / "0" / f"{FRAME:06d}.pcd") @ R.T + t
        h = g.height_above(pts)
        h = h[np.isfinite(h)]
        hist, edges = np.histogram(h, bins=np.arange(-2, 4, 0.1))
        mode = edges[np.argmax(hist)] + 0.05
        self.assertLess(abs(mode), 0.3)
        self.assertGreater(g.keep_mask(pts).sum(), 0)

    def test_box_roundtrip_and_corners(self):
        box = (1.0, 2.0, 0.8, 4.6, 1.9, 1.5, 0.7)
        v = poses.vehicle_from_box(box, "Car")
        back = poses.box_from_vehicle(v)
        np.testing.assert_allclose(back, box, atol=1e-4)
        c = poses.box_corners(box)
        local = poses.to_box_frame(c, box)
        np.testing.assert_allclose(np.abs(local).max(axis=0), [2.3, 0.95, 0.75], atol=1e-9)


if __name__ == "__main__":
    unittest.main()
