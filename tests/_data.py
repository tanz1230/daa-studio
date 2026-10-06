"""Real-data fixtures, given by environment variables; tests that need them skip when unset.

    DAA_STUDIO_TEST_SESSION        an OpenCOOD session folder (e.g. .../agi_coop/take_3)
    DAA_STUDIO_TEST_DETECTIONS     the UAV detections CSV for that session
    DAA_STUDIO_TEST_PARITY_CACHE   a DAA track cache (pickled {track_id: TrackFrames})
    DAA_STUDIO_TEST_PARITY_EXPECT  the label CSV that DAA must reproduce from that cache
"""
import os
import unittest
from pathlib import Path


def _path(var: str) -> Path:
    value = os.environ.get(var, "")
    return Path(value) if value else Path("/nonexistent") / var


SESSION = _path("DAA_STUDIO_TEST_SESSION")
DETECTIONS = _path("DAA_STUDIO_TEST_DETECTIONS")
PARITY_CACHE = _path("DAA_STUDIO_TEST_PARITY_CACHE")
PARITY_EXPECT = _path("DAA_STUDIO_TEST_PARITY_EXPECT")


def _first_frame() -> int:
    frames = [int(p.stem) for p in (SESSION / "0").glob("*.yaml")] if SESSION.is_dir() else []
    return min(frames) if frames else 0


FRAME = _first_frame()             # the session's first frame

needs_session = unittest.skipUnless(SESSION.exists(), "set DAA_STUDIO_TEST_SESSION to run this test")
needs_detections = unittest.skipUnless(DETECTIONS.exists(), "set DAA_STUDIO_TEST_DETECTIONS to run this test")
needs_parity = unittest.skipUnless(PARITY_CACHE.exists() and PARITY_EXPECT.exists(),
                                   "set DAA_STUDIO_TEST_PARITY_CACHE and _EXPECT to run this test")
