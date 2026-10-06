"""Input/output contracts of the refiner, plus a loader for legacy track caches.

The refiner is data-source agnostic: anything that fills a `TrackFrames` (one per
UAV-tracked vehicle; per-frame candidate crops in a shared world/map frame) can be
refined. Older track caches store the same fields under the class name `TrackFramesV4`;
`load_tracks` reads those too.
"""
from __future__ import annotations

import pickle
from dataclasses import dataclass, field
from typing import Dict, List

import numpy as np


@dataclass
class TrackFrames:
    """Per-track input: the UAV prior boxes + the cropped candidate points, per frame.

    All geometry is in one shared (map/world) frame. `cand_pts[t]` is a generous
    padded crop around the UAV box (see config data_crop_*), ground-removed.
    """
    track_id: int
    veh_class: int = 0                                             # UAV class: 0 = car
    frame_ids: List[int] = field(default_factory=list)
    cand_pts: List[np.ndarray] = field(default_factory=list)       # (N,3) candidate crop
    cand_ego: List[np.ndarray] = field(default_factory=list)       # (N,) point source id
    uav_boxes: List[np.ndarray] = field(default_factory=list)      # (7,) UAV prior box
    velocities: List[np.ndarray] = field(default_factory=list)     # (2,) UAV velocity
    ground_z: List[float] = field(default_factory=list)            # ground height under box
    lidar_origin: List[np.ndarray] = field(default_factory=list)   # (3,) sensor position
    session: int = 0
    agent: int = 0

    @property
    def n_frames(self) -> int:
        return len(self.frame_ids)


@dataclass
class TrackResult:
    """Per-track output: one refined box per frame (shared dims, per-frame pose)."""
    boxes: np.ndarray              # (T,7) refined boxes
    dims: np.ndarray               # (3,) shared L, W, H
    xy_offsets: np.ndarray         # (T,2) refined centre minus UAV centre (world xy)
    fg_mass: np.ndarray            # (T,) responsibility mass per frame (diagnostic)
    conf: np.ndarray               # (T,) frame quality x aggregation quality
    fg_points: List[np.ndarray]    # (n,3) harvested foreground points per frame
    fg_r: List[np.ndarray]         # (n,) their responsibilities
    diagnostics: List[dict] = field(default_factory=list)


class _CompatUnpickler(pickle.Unpickler):
    """Map the older TrackFramesV4 class (from whatever module it was pickled) onto TrackFrames."""

    def find_class(self, module, name):
        if name == "TrackFramesV4":
            return TrackFrames
        return super().find_class(module, name)


def load_tracks(path: str) -> Dict[int, TrackFrames]:
    """Load a {track_id: TrackFrames} cache, accepting legacy TrackFramesV4 pickles."""
    with open(path, "rb") as f:
        return _CompatUnpickler(f).load()
