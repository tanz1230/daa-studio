import math
import unittest

import numpy as np

from studio import labels as L
from studio.poses import box_corners, to_box_frame


def track():
    poses = {f: [10.0 + f, 5.0 + 0.5 * f, 0.8, 0.3 + 0.1 * f] for f in range(5)}
    return L.Track(tid=7, shape=[4.6, 1.9, 1.5], poses=poses)


def face(t, f, axis, sign):
    """Map-frame position of one box face along its own axis, for frame f."""
    box = t.box(f)
    c = box_corners(box)
    i = {"L": 0, "W": 1, "H": 2}[axis]
    local = to_box_frame(c, track().box(f))      # measured in the ORIGINAL box frame
    return (local[:, i].max() if sign > 0 else local[:, i].min())


class Labels(unittest.TestCase):
    def test_one_sided_resize_keeps_the_opposite_face(self):
        t = track()
        for axis in ("L", "W", "H"):
            for side in (+1, -1):
                new = L.resize(t, axis, t.shape["LWH".index(axis)] + 0.8, side=side)
                for f in t.frames():
                    fixed_before = face(t, f, axis, -side)
                    fixed_after = face(new, f, axis, -side)
                    self.assertAlmostEqual(fixed_before, fixed_after, places=9,
                                           msg=f"{axis} side={side} frame={f}")
                self.assertEqual(new.status, "edited")
        self.assertEqual(t.shape, [4.6, 1.9, 1.5])          # input untouched

    def test_interpolate_is_linear_and_short_way_modulo_pi(self):
        t = track()
        # flip the last keyframe: the box axis is unchanged, so no spin may appear
        t = L.set_pose(t, 4, yaw=t.poses[4][3] + math.pi)
        out = L.interpolate(t, 0, 4)
        for f in (1, 2, 3):
            w = f / 4
            self.assertAlmostEqual(out.poses[f][0], t.poses[0][0] + w * (t.poses[4][0] - t.poses[0][0]))
            step = abs(L.wrap(out.poses[f][3] - out.poses[f - 1][3]))
            self.assertLess(step, 0.2)                       # 0.1 rad per frame, not ~pi/4
            self.assertIn("edited", out.flags[f])

    def test_fill_gaps_creates_the_missing_frames_interpolated(self):
        t = track()
        full = L.interpolate(t, 0, 4)
        holed = t
        for f in (1, 2, 3):
            holed = L.remove_frame(holed, f)
        out = L.fill_gaps(holed, range(0, 10))
        self.assertEqual(out.frames(), [0, 1, 2, 3, 4])      # only inside the span, never beyond
        for f in (1, 2, 3):
            np.testing.assert_allclose(out.poses[f], full.poses[f], atol=1e-12)
            self.assertIn("edited", out.flags[f])

    def test_copy_to_fills_only_missing_frames(self):
        t = track()
        out = L.copy_to(t, 0, [3, 5, 6])
        self.assertEqual(out.frames(), [0, 1, 2, 3, 4, 5, 6])
        self.assertEqual(out.poses[3], t.poses[3])             # existing frame untouched
        self.assertEqual(out.poses[5], t.poses[0])
        self.assertIn("edited", out.flags[6])

    def test_flip_twice_is_identity(self):
        t = track()
        back = L.flip(L.flip(t))
        for f in t.frames():
            np.testing.assert_allclose(back.poses[f], t.poses[f], atol=1e-12)

    def test_set_pose_adds_a_missing_frame_from_the_nearest(self):
        t = L.set_pose(track(), 9, x=99.0)
        self.assertEqual(t.poses[9][0], 99.0)
        self.assertEqual(t.poses[9][1], track().poses[4][1])

    def test_json_round_trip(self):
        t = L.mark(L.set_status(track(), "flagged", "occluded"), 2, "keyframe")
        t.seed = {"kind": "daa", "run": "r1", "conf": {0: 0.9, 1: 0.8}, "diag": {"pts": 40}}
        back = L.Track.from_json(t.to_json())
        self.assertEqual(back.to_json(), t.to_json())
        self.assertEqual(back.summary()["conf"], 0.85)

    def test_unknown_status_rejected(self):
        with self.assertRaises(ValueError):
            L.set_status(track(), "done")


if __name__ == "__main__":
    unittest.main()
