"""Stage 1.5: motion-consistency gating + prior-anchored
sequential shape registration for MOVING tracks.

Inspired by NSFP++/Auto-Meta-Labeling (Najibi et al., ECCV 2022), adapted to
the aerial-prior setting where flow needs no estimation — the prior supplies
each track's velocity, so motion consistency becomes a *test*:

MOTION GATE — a candidate point observed near a moving target that ALSO has a
close counterpart at the same MAP location in a frame where the target had
displaced by more than its own length is STATIC WORLD (parked neighbour,
vegetation, curb): the target has vacated that spot, so the point cannot be
its. Conservative by construction: it only rejects points with positive
static evidence, only on frames where the prior says the target is moving,
and never empties a frame.

REGISTRATION — the canonical aggregation is currently rigid under the prior's
per-frame pose, so per-frame residuals (sync quantization = dt*v along-track,
prior jitter) smear a mover's aggregate. Sequential 2-D ICP registers each
frame's foreground into the running aggregate (densest frame first), with the
prior as the anchor: translation-only, correction shrunk toward zero by a
prior weight (MAP), clipped to a sync-scale bound, and skipped entirely on
frames too sparse to register (they keep the prior pose). The corrections ARE
the per-frame pose refinement; the sharper aggregate is the by-product the
box-fit then measures.
"""
from __future__ import annotations

import numpy as np
from scipy.spatial import cKDTree

from .geometry import canonical_xy_to_world, to_canonical


def motion_gate(tf, cand, cfg):
    """Drop map-static points from a MOVING track's candidate clouds.

    For each frame t with prior speed >= cfg.motion_min_speed, build the
    static-evidence set from candidate clouds of frames whose prior centre is
    displaced by [L+1, L+5] m from frame t's (the target has fully vacated its
    footprint, while the crop regions still overlap); a point with a neighbour
    within cfg.motion_static_r in that set is static world.
    Returns (filtered cand list, fraction dropped)."""
    T = tf.n_frames
    uav = np.stack([np.asarray(b, np.float64) for b in tf.uav_boxes])
    c = uav[:, :2]
    L = float(np.median(uav[:, 3]))
    lo, hi = L + 1.0, L + 5.0
    # per-frame prior speed
    t_axis = np.arange(T, dtype=np.float64)
    if T > 2:
        vx = np.gradient(c[:, 0], t_axis)
        vy = np.gradient(c[:, 1], t_axis)
        spd = np.hypot(vx, vy) * 10.0          # frames are ~10 Hz
    else:
        spd = np.zeros(T)
    out = list(cand)
    n_all = n_drop = 0
    for t in range(T):
        p = np.asarray(cand[t], np.float64)
        if len(p) < 10 or spd[t] < cfg.motion_min_speed:
            continue
        d = np.hypot(c[:, 0] - c[t, 0], c[:, 1] - c[t, 1])
        nb = np.flatnonzero((d >= lo) & (d <= hi))
        if len(nb) == 0:
            continue
        # up to 2 evidence frames per side, nearest in time
        nb = sorted(nb, key=lambda k: abs(k - t))
        pool = []
        n_before = n_after = 0
        for k in nb:
            if k < t and n_before < 2:
                pool.append(np.asarray(cand[k], np.float64)[:, :2]); n_before += 1
            elif k > t and n_after < 2:
                pool.append(np.asarray(cand[k], np.float64)[:, :2]); n_after += 1
            if n_before >= 2 and n_after >= 2:
                break
        pool = [q for q in pool if len(q)]
        if not pool:
            continue
        tree = cKDTree(np.concatenate(pool))
        dist, _ = tree.query(p[:, :2], k=1, distance_upper_bound=cfg.motion_static_r)
        static = np.isfinite(dist)
        if static.all():                        # never empty a frame
            continue
        n_all += len(p)
        n_drop += int(static.sum())
        out[t] = p[~static]
    frac = n_drop / max(n_all, 1)
    return out, frac


def register_track(tf, res, cfg):
    """Prior-anchored sequential 2-D registration of a moving track's
    foreground into its canonical aggregate. Updates res.boxes / res.xy_offsets
    per frame (the pose refinement) and appends a diagnostics record."""
    T = tf.n_frames
    Z = []
    for t in range(T):
        p = np.asarray(res.fg_points[t], np.float64)
        Z.append(to_canonical(p[:, :3], res.boxes[t])[:, :2] if p.ndim == 2 and len(p)
                 else np.zeros((0, 2)))
    order = sorted(range(T), key=lambda t: -len(Z[t]))
    if len(Z[order[0]]) < cfg.reg_min_pts:
        return res
    agg_np = Z[order[0]]
    corr = np.zeros((T, 2))
    tree = cKDTree(agg_np)
    pending = []
    for t in order[1:]:
        z = Z[t]
        if len(z) < cfg.reg_min_pts:
            continue
        delta = np.zeros(2)
        n_eff = 0.0
        for _ in range(4):
            q = z + delta
            dist, idx = tree.query(q, k=1, distance_upper_bound=cfg.reg_pair_r)
            m = np.isfinite(dist)
            n_eff = float(m.sum())
            if n_eff < 15:
                break
            step = (agg_np[idx[m]] - q[m]).mean(axis=0)
            delta = delta + step
            if np.hypot(*step) < 0.01:
                break
        # MAP shrinkage toward the prior pose + hard sync-scale bound
        delta = delta * (n_eff / (n_eff + cfg.reg_prior_kappa))
        nrm = float(np.hypot(*delta))
        if nrm > cfg.reg_max_corr:
            delta = delta * (cfg.reg_max_corr / nrm)
        corr[t] = delta
        pending.append(z + delta)
        if len(pending) >= 5:                 # fold into the aggregate in batches
            agg_np = np.concatenate([agg_np] + pending)
            tree = cKDTree(agg_np)
            pending = []
    # apply corrections: canonical (x,y) shift -> world, per frame
    boxes = res.boxes.copy()
    offs = res.xy_offsets.copy()
    n_moved = 0
    for t in range(T):
        if abs(corr[t, 0]) < 1e-9 and abs(corr[t, 1]) < 1e-9:
            continue
        w = canonical_xy_to_world(corr[t], float(boxes[t, 6]))
        boxes[t, 0:2] = boxes[t, 0:2] + w
        offs[t] = offs[t] + w
        n_moved += 1
    res.boxes = boxes
    res.xy_offsets = offs
    if isinstance(res.diagnostics, list):
        nrm = np.hypot(corr[:, 0], corr[:, 1])
        res.diagnostics.append({"register": {
            "n_moved": int(n_moved), "corr_med": round(float(np.median(nrm[nrm > 0])), 3)
            if (nrm > 0).any() else 0.0, "corr_max": round(float(nrm.max()), 3)}})
    return res
