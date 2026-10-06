"""The refiner: converging EM harvest -> corner consensus -> box-fit.

A planar wall is SLIDE-AMBIGUOUS (the aperture problem): a point on it carries almost
no information about extent/position. The information is at CORNERS and EDGES. The
per-frame L-corner is a stable, recurring feature — it clusters to ~3-5 cm across
hundreds of frames (vs the ~30 cm point-mode) because a corner is the INTERSECTION of
two line fits, which averages out per-point noise. So after the harvest converges we
detect the per-frame L-corner, build a cross-frame consensus, and let a reliable END
anchor override the silhouette end in the box-fit. (The near-WALL analogue was
evaluated and rejected: the corner only fires where an L is visible, i.e. well-observed
cases the box-fit already centres well.)
"""
from __future__ import annotations

from typing import Optional

import numpy as np

from .boxfit import box_fit
from .config import RefinerConfig
from .geometry import to_canonical
from .harvest import em_harvest
from .types import TrackResult

# ---- corner-consensus knobs ----
MIN_PTS = 20          # min frame points to attempt an L-fit
EDGE_BAND = 0.15      # a point is "on" an edge within this band (m)
MIN_CORNERS = 25      # need at least this many per-frame corners to trust a consensus
MIN_END = 15          # an END is reliable only with at least this many corners on it
END_TOL = 0.15        # ... and cross-frame std < this (m)
END_MIN_OFFSET = 0.6  # an end must be at least this far from the UAV centre


def _edge_dens(P, axis, val, lo, hi):
    other = 1 - axis
    return int(((np.abs(P[:, axis] - val) < EDGE_BAND) & (P[:, other] >= lo) & (P[:, other] <= hi)).sum())


def _lcorner(P):
    """L-shape corner of a BEV point set P (N,2), canonical (~axis-aligned by the UAV
    heading). The corner is the bbox vertex whose TWO adjacent edges are both dense — a
    real L (side wall + end face), not a lone wall or an occlusion boundary.
    Returns (x, y) or None."""
    if len(P) < MIN_PTS:
        return None
    xmn, xmx = np.quantile(P[:, 0], [0.02, 0.98])
    ymn, ymx = np.quantile(P[:, 1], [0.02, 0.98])
    best, bs = None, -1
    for cx in (xmn, xmx):
        for cy in (ymn, ymx):
            de = _edge_dens(P, 0, cx, ymn, ymx)        # end face at x=cx
            ds = _edge_dens(P, 1, cy, xmn, xmx)        # side wall at y=cy
            if min(de, ds) > bs:
                bs, best = min(de, ds), (cx, cy, de, ds)
    cx, cy, de, ds = best
    if min(de, ds) < max(8, 0.04 * len(P)):            # require a real L (both walls observed)
        return None
    return (cx, cy)


def _corner_consensus(res, tf):
    """Per-frame L-corners -> a cross-frame consensus END anchor in the canonical frame:
    {'xlo', 'xhi'} (each None if not reliable). The ends are split by the UAV CENTRE (not
    their own median, which degenerates when only one end is seen); an end is reliable if
    its corners are tight, well-supported, and clearly off the centre."""
    cs = []
    for t in range(tf.n_frames):
        p = np.asarray(res.fg_points[t], np.float64)
        if p.ndim != 2 or len(p) < MIN_PTS:
            continue
        c = _lcorner(to_canonical(p[:, :3], res.boxes[t])[:, :2])
        if c is not None:
            cs.append(c)
    if len(cs) < MIN_CORNERS:
        return None
    C = np.array(cs)
    out = {"xlo": None, "xhi": None}

    # NEAR side = the y-sign the ego sees most; only its corners feed the end consensus.
    near_sign = 1.0 if (C[:, 1] > 0).sum() >= (C[:, 1] < 0).sum() else -1.0
    nearC = C[np.sign(C[:, 1]) == near_sign]
    if len(nearC):
        ucx = float(np.median([to_canonical(np.asarray(tf.uav_boxes[t], np.float64)[None, :3],
                                            res.boxes[t])[0, 0] for t in range(tf.n_frames)]))
        for side, key in ((-1.0, "xlo"), (1.0, "xhi")):
            endx = nearC[(np.sign(nearC[:, 0] - ucx) == side)
                         & (np.abs(nearC[:, 0] - ucx) > END_MIN_OFFSET), 0]
            if len(endx) >= MIN_END and np.std(endx) < END_TOL:
                out[key] = float(np.median(endx))
    return out


def _is_moving(tf, cfg) -> bool:
    uav = np.stack([np.asarray(b, np.float64)[:2] for b in tf.uav_boxes])
    plen = float(np.sum(np.linalg.norm(np.diff(uav, axis=0), axis=1))) if tf.n_frames > 1 else 0.0
    return plen >= cfg.cf_parked_max


def refine_track(tf, cfg: Optional[RefinerConfig] = None) -> TrackResult:
    """Refine one track: EM harvest -> (moving: Stage-1.5 registration) ->
    corner consensus -> box-fit."""
    cfg = cfg or RefinerConfig()
    res = em_harvest(tf, cfg)                    # Stage 1: the harvest converges
    if cfg.register_moving and _is_moving(tf, cfg):
        from .registration import register_track
        res = register_track(tf, res, cfg)       # Stage 1.5: per-frame pose refinement
    corner = _corner_consensus(res, tf) if cfg.corner_consensus else None
    return box_fit(res, tf, cfg, corner=corner)  # Stage 2: measure the final box


def refine_track_integrated(tf, cfg: Optional[RefinerConfig] = None) -> TrackResult:
    """Variant (not the default): feed registration back into the harvest.
    Movers run the alternation harvest -> register -> re-harvest on the
    de-smeared poses -> re-register -> box-fit, so the EM operates on the
    well-aggregated cloud. One outer iteration; parked tracks are single-pass."""
    import copy as _copy
    cfg = cfg or RefinerConfig()
    res = em_harvest(tf, cfg)
    if cfg.register_moving and _is_moving(tf, cfg):
        from .registration import register_track
        off0 = res.xy_offsets.copy()
        res = register_track(tf, res, cfg)
        corr = res.xy_offsets - off0
        if np.any(np.abs(corr) > 1e-9):
            tf2 = _copy.copy(tf)
            tf2.uav_boxes = []
            for t in range(tf.n_frames):
                b = np.asarray(tf.uav_boxes[t], np.float64).copy()
                b[:2] += corr[t]
                tf2.uav_boxes.append(b)
            tf = tf2
            res = em_harvest(tf, cfg)
            res = register_track(tf, res, cfg)
    corner = _corner_consensus(res, tf) if cfg.corner_consensus else None
    return box_fit(res, tf, cfg, corner=corner)


def refine_track_noem(tf, cfg: Optional[RefinerConfig] = None) -> TrackResult:
    """Variant (not the default): skip the EM harvest entirely — feed the raw candidate
    points and the raw UAV prior box straight to Stage 2. Keeps only the EM-independent
    pieces: the static ceiling filter, the Stage-1.5 motion gate, and (moving)
    registration."""
    cfg = cfg or RefinerConfig()
    T = tf.n_frames
    uav = np.stack([np.asarray(b, np.float64) for b in tf.uav_boxes])
    ground_z = np.asarray(tf.ground_z, np.float64)
    med = np.median(uav[:, 3:6], axis=0)
    veh_class = int(getattr(tf, "veh_class", 0))
    moving = _is_moving(tf, cfg)

    # static, EM-independent pre-filters (identical to harvest.py's pre-loop)
    ceil = cfg.sky_max_height if cfg.sky_max_height < 1e3 else 1e18
    if cfg.dim_overhead_reject and veh_class == 0:
        ceil = min(ceil, float(cfg.dim_overhead_car))
    cand = []
    for t in range(T):
        p = np.asarray(tf.cand_pts[t], np.float64)
        if len(p) and ceil < 1e17:
            p = p[p[:, 2] <= ground_z[t] + ceil]
        cand.append(p)
    if cfg.motion_gate and moving:
        from .registration import motion_gate
        cand, _ = motion_gate(tf, cand, cfg)

    # raw box = UAV prior (median dims, ground-anchored, zero offset);
    # foreground = ALL filtered candidates with uniform weight (no E-step)
    boxes = uav.copy()
    boxes[:, 2] = ground_z + med[2] / 2.0
    boxes[:, 3:6] = med
    fg_points = [p[:, :3] if len(p) else np.zeros((0, 3)) for p in cand]
    fg_r = [np.ones(len(p)) for p in cand]
    res = TrackResult(boxes=boxes, dims=med.copy(), xy_offsets=np.zeros((T, 2)),
                      fg_mass=np.array([len(p) for p in cand], float),
                      conf=np.zeros(T), fg_points=fg_points, fg_r=fg_r)

    if cfg.register_moving and moving:
        from .registration import register_track
        res = register_track(tf, res, cfg)
    corner = _corner_consensus(res, tf) if cfg.corner_consensus else None
    return box_fit(res, tf, cfg, corner=corner)


def v48_config() -> RefinerConfig:
    return RefinerConfig()


refine_track_v48 = refine_track
TrackResultV4 = TrackResult
