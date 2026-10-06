import unittest

import numpy as np

from studio.camera import Camera
from tests._data import FRAME, SESSION, needs_session


@needs_session
class CameraTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.cam = Camera.from_json(SESSION / "camera_3.json")

    def test_ground_points_round_trip_through_the_homography(self):
        """The ground homography and the full 3-D projection are two views of one camera: a
        ground point from the homography must project back onto its own (undistorted) pixel."""
        u, v = np.meshgrid(np.linspace(200, 3600, 9), np.linspace(200, 1900, 7))
        uv = np.column_stack([u.ravel(), v.ravel()])
        ground = self.cam.ground_from_pixel(FRAME, uv)
        back, depth = self.cam.project(FRAME, ground, distort=False)
        self.assertTrue((depth > 50).all())                          # ~120 m flying height
        self.assertLess(np.abs(back - uv).max(), 1.0)

    def test_distortion_is_applied_but_small(self):
        ground = self.cam.ground_from_pixel(FRAME, [[3600.0, 1900.0]])
        d, _ = self.cam.project(FRAME, ground, distort=True)
        u, _ = self.cam.project(FRAME, ground, distort=False)
        shift = float(np.hypot(*(d - u)[0]))
        self.assertGreater(shift, 0.1)
        self.assertLess(shift, 40.0)

    def test_gt_box_lands_inside_the_image_and_crop(self):
        from studio.session import Session
        s = Session(SESSION)
        box, _ = next(iter(s.labels(FRAME).values()))
        uv = self.cam.box_corners_px(FRAME, box)
        w, h = self.cam.size
        self.assertTrue(((uv[:, 0] > 0) & (uv[:, 0] < w) & (uv[:, 1] > 0) & (uv[:, 1] < h)).all())
        for aspect in (1.0, 1.9):                      # square, and a wide panel
            x0, y0, cw, ch = self.cam.crop_window(FRAME, box, aspect=aspect)
            self.assertAlmostEqual(cw / ch, aspect, delta=0.01)
            self.assertTrue(0 <= x0 and x0 + cw <= w and 0 <= y0 and y0 + ch <= h)
            self.assertTrue(x0 <= uv[:, 0].min() and uv[:, 0].max() <= x0 + cw)
            self.assertTrue(y0 <= uv[:, 1].min() and uv[:, 1].max() <= y0 + ch)


    def test_pixel_to_map_inverts_projection(self):
        from studio.session import Session
        s = Session(SESSION)
        gp = self.cam.ground_plane or {"z0": 0.0, "gx": 0.0, "gy": 0.0}
        for box, _ in list(s.labels(FRAME).values())[:10]:
            c = np.asarray(box[:3], float)
            h = c[2] - (gp["z0"] + gp["gx"] * c[0] + gp["gy"] * c[1])        # centre height above the plane
            uv, _ = self.cam.project(FRAME, c[None])
            back = self.cam.pixel_to_map(FRAME, uv, height=h)[0]
            self.assertLess(np.linalg.norm(back - c), 0.02)


if __name__ == "__main__":
    unittest.main()
