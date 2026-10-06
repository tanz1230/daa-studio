"""Gate for the seeding engine: DAA with the relative prior floor (delta = 0.3 m) must reproduce
a reference label file exactly from the same track cache (fixtures: tests/_data.py)."""
import csv
import unittest

from studio.daa.boxfit_ov import box_fit_ov
from studio.daa.harvest import em_harvest
from studio.daa.types import load_tracks
from studio.seeding.config import adopted_config
from tests._data import PARITY_CACHE, PARITY_EXPECT, needs_parity

N_TRACKS = 40


@needs_parity
class DAAParity(unittest.TestCase):
    def test_vendored_delta_floor_reproduces_the_paper_run(self):
        expected = {}
        with open(PARITY_EXPECT, newline="") as f:
            for r in csv.DictReader(f):
                expected[(int(r["frame"]), int(r["track_id"]))] = r
        tracks = load_tracks(PARITY_CACHE)
        items = sorted((tid, tf) for tid, tf in tracks.items() if tf.n_frames >= 2)
        sample = items[:: max(1, len(items) // N_TRACKS)][:N_TRACKS]
        cfg = adopted_config()
        checked = worst = 0
        for tid, tf in sample:
            try:
                r = box_fit_ov(em_harvest(tf, cfg), tf, cfg)
            except Exception:
                self.assertFalse(any(k[1] == tid for k in expected), f"track {tid} failed here only")
                continue
            for k, fr in enumerate(tf.frame_ids):
                row = expected.get((int(fr), tid))
                if row is None:          # DAA runs on every cached frame; the export kept dataset frames
                    continue
                b = r.boxes[k]
                for i, key in enumerate(("x", "y", "z", "l", "w", "h")):
                    d = abs(round(float(b[i]), 4) - float(row[key]))
                    worst = max(worst, d)
                    self.assertLessEqual(d, 1.01e-4, f"track {tid} frame {fr} {key}")
                self.assertLessEqual(abs(float(b[6]) - float(row["yaw"])), 1e-5,
                                     f"track {tid} frame {fr} yaw")
                checked += 1
        print(f"\n  parity: {len(sample)} tracks, {checked} boxes, worst |diff| {worst:.1e} m")
        self.assertGreater(checked, 1000)


if __name__ == "__main__":
    unittest.main()
