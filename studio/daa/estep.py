"""E-step: box-conditioned responsibilities, graph smoothing, and the coherent mask.

1. `compute_responsibilities` — crop candidates loosely under the current box and score
   each point by box compatibility + height plausibility + an ego-view prior, mapped to a
   soft responsibility r0 in [0,1].
2. `graph_smooth` — closed-form Laplacian regularisation over a kNN graph: propagates
   responsibility along the connected vehicle surface (pulls up the far end of a long
   vehicle, pulls down isolated salt-and-pepper clutter).
3. `coherent_mask` — clutter rejection on an aggregated canonical cloud: a BEV density
   floor (a parked vehicle is densely sampled; over-harvested clutter is a sparse spread)
   then the dominant connected BEV component (drops separate neighbour clusters).
"""
from __future__ import annotations

import numpy as np
import scipy.sparse as sp
from scipy import ndimage
from scipy.sparse.linalg import spsolve
from scipy.spatial import cKDTree

from .geometry import to_canonical, outside_distance


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(x, -40.0, 40.0)))


# ============================================================================================
# 1. box-conditioned unary responsibilities
# ============================================================================================

def unary_scores(z: np.ndarray, box, lidar_origin, cfg):
    """Per-point unary log-score terms (box, height, view) for canonical points z."""
    L, W, H = float(box[3]), float(box[4]), float(box[5])
    half = np.array([L / 2.0, W / 2.0, H / 2.0])

    # 1. box compatibility: 0 inside, decays with Euclidean outside-distance
    od = outside_distance(z, half)
    score_box = np.clip(-od / max(cfg.sigma_outside, 1e-6), -cfg.outside_clip, 0.0)

    # 2. height plausibility (ground already removed): penalize below floor / above roof
    z_low = -H / 2.0 - cfg.height_min_extra
    z_high = H / 2.0 + cfg.height_max_extra
    below = np.maximum(z_low - z[:, 2], 0.0)
    above = np.maximum(z[:, 2] - z_high, 0.0)
    score_height = -(below + above) / max(cfg.sigma_outside, 1e-6)

    # 3. ego-view consistency: reward points near a wall on the side facing the ego
    if cfg.use_view_prior:
        view = to_canonical(np.asarray(lidar_origin, np.float64)[None, :], box)[0]
        vx, vy = abs(view[0]), abs(view[1])
        wx = vx / (vx + vy + 1e-6)
        wy = vy / (vx + vy + 1e-6)
        face_y = np.exp(-((np.abs(z[:, 1]) - W / 2.0) ** 2) / (2 * cfg.face_sigma ** 2))
        vis_y = _sigmoid(cfg.view_visible_k * np.sign(view[1]) * z[:, 1])
        face_x = np.exp(-((np.abs(z[:, 0]) - L / 2.0) ** 2) / (2 * cfg.face_sigma ** 2))
        vis_x = _sigmoid(cfg.view_visible_k * np.sign(view[0]) * z[:, 0])
        score_view = cfg.view_weight * (wy * face_y * vis_y + wx * face_x * vis_x)
    else:
        score_view = np.zeros(z.shape[0])
    return score_box, score_height, score_view, od


# ============================================================================================
# 1b. ray-geometry evidence fields (witnessed-empty corridors, both BEV axes)
# ============================================================================================

def build_ray_fields(frames, sensors, dims, cfg):
    """Aggregate witnessed-empty evidence along the two BEV axes of the canonical frame.

    frames: per-frame (z_canonical, r) from the PREVIOUS iteration; sensors: canonical
    sensor origins per frame. For each axis, each 1-D cell scores
    u = [never occupied by any return] x (fraction of frames whose beams crossed it) --
    the eq:wsig cell evidence, reused at the point level. Frame t crosses cell c when its
    sensor stood on one side and its returns continue past c toward the body, so a
    never-scanned occluded cell reads 0 (unknown), a scanned-through empty cell reads
    high (proven empty). Returns per axis (lo, cell, prefix-sum of u) or None."""
    L, W = float(dims[0]), float(dims[1])
    cell = cfg.ray_cell
    out = []
    for ax in (0, 1):
        perp = 1 - ax
        half_perp = (W if ax == 0 else L) / 2.0 + 0.4
        vals = []
        for z, _ in frames:
            if len(z):
                v = z[np.abs(z[:, perp]) <= half_perp, ax]
                if len(v):
                    vals.append(v)
        if not vals:
            out.append(None)
            continue
        cat = np.concatenate(vals)
        lo = float(cat.min())
        n = max(int(np.ceil((float(cat.max()) - lo) / cell)) + 1, 1)
        xs = lo + (np.arange(n) + 0.5) * cell
        occ_any = np.zeros(n, bool)
        wit = np.zeros(n)
        T_eff = 0
        for (z, _), s in zip(frames, sensors):
            if not len(z):
                continue
            T_eff += 1
            v = z[np.abs(z[:, perp]) <= half_perp, ax]
            if not len(v):
                continue
            ci = np.clip(((v - lo) / cell).astype(int), 0, n - 1)
            occ_any[np.unique(ci)] = True
            sx = float(s[ax])
            wit += ((sx < xs) & (float(v.max()) > xs)) | ((sx > xs) & (float(v.min()) < xs))
        if T_eff == 0:
            out.append(None)
            continue
        u = (~occ_any) * (wit / T_eff)          # witnessed-empty evidence per cell
        # the anchor-connected BODY RUN: contiguous occupied cells containing the box
        # centre (canonical 0), absorbing sub-0.3 m holes -- the wrap's run, as a prior
        c0 = int(np.clip((0.0 - lo) / cell, 0, n - 1))
        occ_idx = np.where(occ_any)[0]
        if not len(occ_idx):
            out.append(None)
            continue
        if not occ_any[c0]:
            c0 = int(occ_idx[np.argmin(np.abs(occ_idx - c0))])
        hole = 3
        run_hi = c0
        while True:
            nxt = occ_idx[(occ_idx > run_hi) & (occ_idx <= run_hi + hole + 1)]
            if not len(nxt):
                break
            run_hi = int(nxt.max())
        run_lo = c0
        while True:
            nxt = occ_idx[(occ_idx < run_lo) & (occ_idx >= run_lo - hole - 1)]
            if not len(nxt):
                break
            run_lo = int(nxt.min())
        out.append((lo, cell, np.concatenate([[0.0], np.cumsum(u)]), run_lo, run_hi))
    return out


def ray_break_distance(z: np.ndarray, fields, cfg) -> np.ndarray:
    """Per-point metres of PROVEN-EMPTY space separating the point from the anchor-
    connected body run (max over the two BEV axes). Occupied corridor cells contribute
    0 (connection is the graph's judgement), never-scanned cells contribute 0 (unknown
    is not empty) -- only cells the sensor crossed and found empty add distance."""
    n = z.shape[0]
    d = np.zeros(n)
    if fields is None:
        return d
    for ax in (0, 1):
        f = fields[ax]
        if f is None:
            continue
        lo, cell, P, run_lo, run_hi = f
        ncell = len(P) - 1
        c_pt = np.clip(((z[:, ax] - lo) / cell).astype(int), 0, ncell - 1)
        d_ax = np.zeros(n)
        hi_sel = c_pt > run_hi
        if hi_sel.any():
            d_ax[hi_sel] = (P[c_pt[hi_sel] + 1] - P[run_hi + 1]) * cell
        lo_sel = c_pt < run_lo
        if lo_sel.any():
            d_ax[lo_sel] = (P[run_lo] - P[c_pt[lo_sel]]) * cell
        d = np.maximum(d, d_ax)
    return d


def compute_responsibilities(points, box, lidar_origin, cfg, rng=None, ray_fields=None):
    """Loose-crop candidates under `box`; return (cand_pts, z_can, r0, stats)."""
    pts = np.asarray(points, np.float64)
    empty = (np.zeros((0, 3)), np.zeros((0, 3)), np.zeros(0))
    if pts.shape[0] == 0:
        return (*empty, {"n_cand": 0, "mass": 0.0})
    L, W, H = float(box[3]), float(box[4]), float(box[5])
    half = np.array([L / 2.0, W / 2.0, H / 2.0])
    z = to_canonical(pts, box)
    keep = ((np.abs(z[:, 0]) <= cfg.search_x_scale * half[0])
            & (np.abs(z[:, 1]) <= cfg.search_y_scale * half[1])
            & (np.abs(z[:, 2]) <= cfg.search_z_scale * half[2]))
    pts, z = pts[keep], z[keep]
    if pts.shape[0] == 0:
        return (*empty, {"n_cand": 0, "mass": 0.0})
    if pts.shape[0] > cfg.max_candidate_points:
        rng = rng or np.random.RandomState(0)
        idx = rng.choice(pts.shape[0], cfg.max_candidate_points, replace=False)
        pts, z = pts[idx], z[idx]

    sb, sh, sv, od = unary_scores(z, box, lidar_origin, cfg)
    unary = sb + sh + sv
    if ray_fields is not None and cfg.ray_resp:
        # 4. ray-geometry evidence: membership decays with the metres of sensor-PROVEN
        # empty space separating the point from the body run (unknown space is free)
        d_ray = ray_break_distance(z, ray_fields, cfg)
        unary = unary - np.clip(d_ray / max(cfg.sigma_outside, 1e-6), 0.0, cfg.outside_clip)
    r0 = _sigmoid(cfg.resp_sharpness * (unary - cfg.bg_thr))
    stats = {
        "n_cand": int(pts.shape[0]),
        "mass": float(r0.sum()),
        "mean_outside": float(np.average(od, weights=r0)) if r0.sum() > 0 else 0.0,
        "inside_frac": float((od <= 1e-6).mean()),
    }
    return pts, z, r0, stats


# ============================================================================================
# 2. graph-regularized responsibilities
# ============================================================================================

def graph_smooth(points_xyz: np.ndarray, r0: np.ndarray, cfg) -> np.ndarray:
    """Return graph-regularized responsibilities. `points_xyz` (N,3) in any frame."""
    r0 = np.asarray(r0, np.float64)
    n = r0.shape[0]
    if not cfg.use_graph_smoothing or n < 3:
        return r0
    pts = np.asarray(points_xyz, np.float64)
    tree = cKDTree(pts)
    dist, idx = tree.query(pts, k=min(cfg.knn_k + 1, n))
    dist, idx = dist[:, 1:], idx[:, 1:]                    # drop self
    wij = np.exp(-(dist ** 2) / max(cfg.graph_sigma ** 2, 1e-9))
    rows = np.repeat(np.arange(n), idx.shape[1])
    cols = idx.ravel()
    W = sp.coo_matrix((wij.ravel(), (rows, cols)), shape=(n, n)).tocsr()
    W = (W + W.T) * 0.5                                    # symmetrize
    d = np.asarray(W.sum(1)).ravel()
    Lap = sp.diags(d) - W                                  # graph Laplacian
    A = (sp.eye(n) + cfg.graph_lambda * Lap).tocsc()
    r = spsolve(A, r0)
    return np.clip(r, 0.0, 1.0)


# ============================================================================================
# 3. coherent mask (clutter rejection on the aggregate)
# ============================================================================================

def _bev_grid(P: np.ndarray, R: np.ndarray, cell: float):
    """Responsibility-mass BEV grid + each point's (i,j) cell index."""
    mn = P[:, :2].min(0)
    ij = np.floor((P[:, :2] - mn) / cell).astype(int)
    nx, ny = int(ij[:, 0].max()) + 1, int(ij[:, 1].max()) + 1
    grid = np.zeros((nx, ny))
    np.add.at(grid, (ij[:, 0], ij[:, 1]), R)
    return grid, ij[:, 0], ij[:, 1]


def _dense_mask(grid, ii, jj, frac: float) -> np.ndarray:
    """Keep points whose BEV cell mass >= frac * the robust peak (90th pct of occupied cells)."""
    occ = grid[grid > 0]
    peak = float(np.percentile(occ, 90)) if occ.size else 0.0
    return grid[ii, jj] >= frac * peak


def _dominant_component(grid, ii, jj, close_iter: int) -> np.ndarray:
    """Mask of the dominant connected BEV component (drops separate neighbour clusters)."""
    occ = grid > 0
    if close_iter > 0:
        occ = ndimage.binary_closing(occ, structure=np.ones((3, 3)), iterations=close_iter)
    lab, ncomp = ndimage.label(occ, structure=np.ones((3, 3)))
    if ncomp <= 1:
        return np.ones(len(ii), bool)
    masses = ndimage.sum(grid, lab, index=np.arange(1, ncomp + 1))   # mass from REAL points only
    keep = 1 + int(np.argmax(masses))
    return (lab == keep)[ii, jj]


def coherent_mask(P: np.ndarray, R: np.ndarray, cfg) -> np.ndarray:
    """Boolean mask over P of the clutter-rejected coherent inliers: a density floor, then
    the dominant connected BEV component. Boundary/corner points of the dense vehicle
    survive (an aggressive scatter filter under-reads the extent)."""
    n = len(P)
    if n < cfg.min_points_for_dims:
        return np.ones(n, bool)
    grid, ii, jj = _bev_grid(P, R, cfg.cf_cell)
    keep = _dense_mask(grid, ii, jj, cfg.cf_dens_frac)
    if keep.sum() < cfg.min_points_for_dims:
        keep = np.ones(n, bool)                       # density too aggressive -> skip it
    sub = np.where(keep)[0]
    g2, i2, j2 = _bev_grid(P[sub], R[sub], cfg.cf_cell)
    comp = _dominant_component(g2, i2, j2, cfg.cf_close)
    if comp.sum() >= cfg.min_points_for_dims:
        out = np.zeros(n, bool)
        out[sub[comp]] = True
        return out
    return keep
