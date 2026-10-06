"""Stage 2: the VISIBILITY-GRADED AMODAL WRAP.

Stage 2 does exactly two things per box face:

  THE WRAP   the face is the outer edge of the CONTIGUOUS run of vehicle
             cells from the anchor (holes < 0.3 m absorbed). Membership is
             the shipped definition (occupied / grounded / sub-roof / no
             canopy). The wrap never guesses across a genuine gap.

  THE GRADE  one number per face: w = the witnessed-empty fraction of the
             1.5 m beyond it — how thoroughly rays PROVED that space empty
             (a ray that hit the body traversed its sensor-side cells; a
             frame witnesses a cell only if its sensor is outward AND its
             returns continue past the cell, so a never-scanned occluded
             interior reads unobserved, not empty). w >= 0.5 -> a proven
             wall (sigma 0.15 m); w <= 0.05 -> open (sigma inf: the prior
             supplies the face, floored OUTWARD at the wrap — amodal
             completion never retracts inside observed body); linear
             0.15..0.55 m between.

The faces then fuse with the calibrated prior ends by precision weighting
(prior.py, unchanged); the prior band caps L, liftable by the wrap's own
directly-observed run. Roof, height, centre logic, the width path (coverage
quantile + asymmetric band), and the de-harvest come from boxfit.py.

Alternative formulations selected by OV_* environment variables are disabled in DAA Studio
(see __init__.py).
"""
from __future__ import annotations

import os

import numpy as np

from .boxfit import (CELL, DF_MIN, GROUND, GROW_CELL,
                                              GROW_FRAC, GROW_MIN_BODY,
                                              GROW_MIN_DENS, MARGIN_L,
                                              MARGIN_W, NMIN, ROOF_MARGIN,
                                              ROOF_Q, WQ, _box_roof,
                                              _canopy_mask, _cells,
                                              _centre_pull, _first_gap,
                                              _profile)
from .geometry import (canonical_xy_to_world,
                                                to_canonical,
                                                weighted_quantile)
from .prior import TrackPrior, gauss_fuse

# ---- change-point log-likelihood-ratio weights (the 4 constants that replace
#      the rule thresholds; see the module docstring for what each state means)
B_MEMBER = 2.0     # body model: a member cell inward of the cut
B_WEAK = 0.5       # body model: a weak (raw dense) cell inward
B_EMPTY = 4.0      # body model: an observed-empty cell inward (times v)
O_MEMBER = 1.5     # outside model: a member cell outward (clutter cost)
O_WEAK = 0.3       # outside model: a weak cell outward
O_EMPTY = 0.5      # outside model: an observed-empty cell outward (times v)

SIG_FLOOR = 0.15   # sigma floor = boundary resolution (matches the shipped
                   # dense-boundary grade in prior.end_sigma)
CONF = 0.20        # a fused end tighter than this is "confident" (shipped)
RUN_GAP = 2        # strong-run contiguity tolerance (cells)
PAD = 5            # grid padding (cells) past the data range
MIN_GAP_CELLS = 3  # a >=0.3 m non-member run is a real break (run-edge floor)

WEAK_RAWOCC = os.environ.get("OV_WEAK_RAWOCC", "0") == "1"
FLOOR_RUNEDGE = os.environ.get("OV_FLOOR_RUNEDGE", "0") == "1"

SIMPLE = os.environ.get("OV_SIMPLE", "1") == "1"
WRAP_WIN = 15        # cells beyond the wrap examined for proven-emptiness
WRAP_W_HI = 0.5
WRAP_W_LO = 0.05
# Optional: apply the wrap+grade+fusion to the lateral axis too (off by default;
# the width is the coverage quantile + prior band).
WRAP_WIDTH = os.environ.get("OV_WRAP_WIDTH", "0") == "1"

CORNER = os.environ.get("OV_CORNER", "0") == "1"
CORNER_MIDY = os.environ.get("OV_CORNER_MIDY", "1") == "1"   # lateral de-bias arm
SIG_CORNER = 0.07    # per-axis sigma of a consensus corner (self-calibrated)
CORNER_MIN_ALL = 25  # frame-corners needed to attempt any consensus
CORNER_MIN = 15      # corners per (x-side, y-side) cluster
CORNER_STD = 0.15    # max per-axis std of a cluster (m)
CORNER_MIN_OFF = 0.6 # a corner must sit at least this far from the anchor (x)
SIG_UAV_LAT = 0.20   # trust scale of the aerial lateral centre in the de-bias
CORNER_GATE_X = 0.4  # a corner is REAL only at the wrap's own face (m) --
                     # the wrap arbitrates between multiple corner clusters
CORNER_GATE_Y = 0.3  # ... and only if its wall agrees with the cover edge
# OV_LEN_ONLY=1 -> Stage-3 simplification test: keep only the length wrap + its
# longitudinal centre de-bias; revert W, H, and lateral centre to Stage 2 (harvest).
LEN_ONLY = os.environ.get("OV_LEN_ONLY", "0") == "1"
# OV_NO_MIDX=1 -> drop the Stage-3 longitudinal centre de-bias: keep full
# L/W/H and the lateral guardrail, but leave the along-length centre at Stage 2's seat.
NO_MIDX = os.environ.get("OV_NO_MIDX", "0") == "1"
# OV_WH_ONLY=1 -> "Stage-2-integrated" candidate: keep Stage 2's length + longitudinal
# centre (grow-only, prior-floored) and apply ONLY the natural final read-outs — the
# width band fusion + class-capped roof + lateral anchor. No wrap/grade/end-fusion.
WH_ONLY = os.environ.get("OV_WH_ONLY", "1") == "1"
# OV_RO_LCLIP=1 -> readout also re-measures L: span of the final foreground clipped
# into the one-sided prior band (floor at the prior = amodal completion; cap at the
# calibrated band / class limit). Margin 0.1 = the M-step margin.
RO_LCLIP = os.environ.get("OV_RO_LCLIP", "0") == "1"
RO_MARGIN = 0.1
# OV_RO_SYM=1 -> fully uniform readout: EVERY planar dimension = final-foreground
# span + margin, clipped into prior +/- 1 sigma (subsumes the one-sided fallback,
# whose floor coincides with the band floor).
RO_SYM = os.environ.get("OV_RO_SYM", "0") == "1"
RO_NOH = os.environ.get("OV_RO_NOH", "1") == "1"
# readout W band depth (lower edge = prior - RO_KDOWN * sigma_W).
# 3.0 = bracketed optimum (2: 0.766/0.559, 3: 0.768/0.563, 4: 0.768/0.537).
RO_KDOWN = float(os.environ.get("OV_RO_KDOWN", "3.0"))
# OV_RO_WBAND=<metres> -> flat symmetric width band: W = clip(span+mu, prior-b, prior+b).
# No sigma scaling, no one-sided branch (the flat floor subsumes it).
RO_WBAND = float(os.environ.get("OV_RO_WBAND", "0"))
# OV_RO_WBAND_LO/HI=<metres> -> asymmetric flat band [prior-LO, prior+HI].
RO_WBAND_LO = float(os.environ.get("OV_RO_WBAND_LO", "0"))
RO_WBAND_HI = float(os.environ.get("OV_RO_WBAND_HI", "0"))
# OV_RO_GATEBAND=1 -> observability-gated flat bands: far flank unobserved -> tight
# clip prior+/-0.2 (stay near the prior, both directions bounded); both flanks
# observed -> wide clip prior+/-0.4 (the data may correct either way).
RO_GATEBAND = os.environ.get("OV_RO_GATEBAND", "1") == "1"
# OV_RO_GATE2=1 -> gated hybrid: blind +/-0.2, seen -0.5/+0.2 (round-metre sigma shape)
RO_GATE2 = os.environ.get("OV_RO_GATE2", "0") == "1"


def _axis_field(A_h, B_h, Hg_h, F_h, raw_a, raw_b, bc, gate, roof_cap,
                s_axis, nfr, parked, canopy_from=None, use_weak=True):
    """Per-cell evidence along one axis. A/B = along/across coordinates;
    raw_a/raw_b = PER-FRAME lists of raw vehicle-plausible coordinates;
    s_axis = per-frame sensor coordinate on this axis.

    Observedness is DIRECTIONAL AND WITNESSED: frame t certifies cell i on a
    side only when its sensor sits outward of i on that side AND the frame's
    gated returns extend past i toward the body — the ray demonstrably
    traversed the cell's slab (a never-scanned occluded interior therefore
    reads UNOBSERVED, not observed-empty; the S8 guard, by construction)."""
    ch = _cells(A_h, B_h, Hg_h, bc, gate, F=F_h)
    cxh, cnh, chh, cdfh = ch
    occ = (cnh >= NMIN) if parked else ((cnh >= NMIN) | (cdfh >= DF_MIN))
    memberh = occ & (chh > GROUND) & (chh <= roof_cap)
    if canopy_from is not None:
        cxg, chg = canopy_from
        canopyh = _canopy_mask(cxh, cxg, chg, roof_cap)
    else:
        canopyh = np.zeros(len(cxh), bool)

    # per-frame gated raw coords + their extents (for the witnessed traversal)
    ars, amins, amaxs, svals = [], [], [], []
    for a_t, b_t, s_t in zip(raw_a, raw_b, s_axis):
        if len(a_t):
            m = np.abs(b_t - bc) <= gate
            if m.any():
                at = a_t[m]
                ars.append(at)
                amins.append(float(at.min())); amaxs.append(float(at.max()))
                svals.append(float(s_t))
    ar = np.concatenate(ars) if ars else np.zeros(0)

    idx_parts = []
    if len(ar):
        idx_parts.append(np.round(ar / CELL).astype(int))
    if len(cxh):
        idx_parts.append(np.round(cxh / CELL).astype(int))
    if not idx_parts:
        return None
    imin = int(min(p.min() for p in idx_parts)) - PAD
    imax = int(max(p.max() for p in idx_parts)) + PAD
    n = imax - imin + 1
    if n < 3 or n > 200000:
        return None
    x = np.arange(imin, imax + 1) * CELL

    member = np.zeros(n, bool)
    cnt = np.zeros(n)
    ambig = np.zeros(n, bool)
    if len(cxh):
        hi = np.round(cxh / CELL).astype(int) - imin
        member[hi] = memberh
        cnt[hi] = cnh
        ambig[hi] = canopyh
    rawocc = np.zeros(n, bool)
    if len(ar):
        rawocc[np.bincount(np.round(ar / CELL).astype(int) - imin,
                           minlength=n) > 0] = True

    weak = np.zeros(n, bool)
    if use_weak and len(ar) and member.any():
        e0 = np.floor(ar.min() / GROW_CELL)
        edges = np.arange(e0, np.ceil(ar.max() / GROW_CELL) + 1) * GROW_CELL
        if len(edges) >= 3:
            cw, _ = np.histogram(ar, edges)
            dens = cw / max(nfr, 1)
            cenw = 0.5 * (edges[:-1] + edges[1:])
            mx = x[member]
            inb = (cenw >= mx.min()) & (cenw <= mx.max())
            body = float(np.median(dens[inb])) if inb.any() else 0.0
            if body >= GROW_MIN_BODY:
                wb = dens >= max(GROW_FRAC * body, GROW_MIN_DENS)
                j = np.clip(np.digitize(x, edges) - 1, 0, len(wb) - 1)
                weak = wb[j] & (x >= edges[0]) & (x <= edges[-1])
                if WEAK_RAWOCC:
                    weak = weak & rawocc

    nT = max(nfr, 1)
    if svals:
        S = np.asarray(svals)[:, None]                  # (nf, 1)
        AMIN = np.asarray(amins)[:, None]
        AMAX = np.asarray(amaxs)[:, None]
        xr = x[None, :]                                 # (1, n)
        v_lo = ((S < xr) & (AMAX > xr)).sum(axis=0) / nT
        v_hi = ((S > xr) & (AMIN < xr)).sum(axis=0) / nT
    else:
        v_lo = np.zeros(n)
        v_hi = np.zeros(n)
    return {"x": x, "member": member, "cnt": cnt, "weak": weak,
            "rawocc": rawocc, "ambig": ambig, "v_lo": v_lo, "v_hi": v_hi}


def _side_scores(fld, side):
    """Per-cell (body, outside) log-likelihood-ratio scores for one side."""
    v = fld["v_lo"] if side < 0 else fld["v_hi"]
    member, weak = fld["member"], fld["weak"]
    quiet = fld["ambig"] | (~member & ~weak & fld["rawocc"])   # ambiguous: 0/0
    empty = ~member & ~weak & ~quiet
    b = np.where(member, B_MEMBER,
                 np.where(weak & ~member, B_WEAK, 0.0)) - np.where(
                     empty, B_EMPTY * v, 0.0)
    o = np.where(member, -O_MEMBER,
                 np.where(weak & ~member, -O_WEAK, 0.0)) + np.where(
                     empty, O_EMPTY * v, 0.0)
    b[quiet] = 0.0
    o[quiet] = 0.0
    return b, o


def _scan_side(fld, c, step):
    """Change-point scan from anchor cell c outward (step = -1 lo / +1 hi).
    Returns (cut, sigma, edge, support, dense_edge_or_nan_flagless):
      cut     the face estimate (cell boundary)
      sigma   read off the score curve; inf = unobserved beyond (open)
      edge    outermost member cell face (the observed-body edge)
      support mean count of the outermost member cells (diagnostic)"""
    x, member, cnt = fld["x"], fld["member"], fld["cnt"]
    n = len(x)
    b, o = _side_scores(fld, step)
    if step > 0:
        ks = np.arange(c, n)
        inward = np.cumsum(b[c:])                       # cells c..k
        out_suf = np.concatenate((np.cumsum(o[c + 1:][::-1])[::-1], [0.0]))
        S = inward + out_suf
    else:
        ks = np.arange(c, -1, -1)                       # c down to 0
        inward = np.cumsum(b[:c + 1][::-1])             # cells k..c
        out_pre = np.concatenate((np.cumsum(o[:c])[::-1], [0.0]))  # cells 0..k-1
        S = inward + out_pre
    mem_idx = [k for k in ks if member[k]]
    if not mem_idx:
        return float(x[c]), float("inf"), float(x[c]), 0.0, float("nan")
    kbest = int(np.argmax(S))
    cut = float(x[ks[kbest]] + step * CELL / 2.0)
    good = S >= S.max() - 1.0
    lo_g, hi_g = kbest, kbest
    while lo_g - 1 >= 0 and good[lo_g - 1]:
        lo_g -= 1
    while hi_g + 1 < len(S) and good[hi_g + 1]:
        hi_g += 1
    open_end = hi_g == len(S) - 1                       # flat to the data edge
    sigma = float("inf") if open_end else max(
        SIG_FLOOR, CELL * (hi_g - lo_g) / 2.0)
    edge_k = mem_idx[-1]
    edge = float(x[edge_k] + step * CELL / 2.0)
    sup_cells = [cnt[k] for k in (edge_k, edge_k - step, edge_k - 2 * step)
                 if 0 <= k < n and member[k]]
    support = float(np.mean(sup_cells)) if sup_cells else 0.0
    return cut, sigma, edge, support, float("nan")


def _dense_edge(fld, c, step):
    """Parked amodal floor: outermost member cell still carrying body-level
    occupancy (shipped semantics, computed on the field grid)."""
    x, member, cnt = fld["x"], fld["member"], fld["cnt"]
    n = len(x)
    body = [cnt[k] for k in range(c, n if step > 0 else -1, step) if member[k]]
    if not body:
        return float("nan")
    ref = float(np.median(body))
    ks = range(c, n, 1) if step > 0 else range(c, -1, -1)
    out = float("nan")
    for k in ks:
        if member[k] and cnt[k] >= max(NMIN, 0.5 * ref):
            out = float(x[k] + step * CELL / 2.0)
    return out


def _run_edge_k(fld, c, step):
    """Index of the outermost member cell of the contiguous run from the
    anchor, absorbing non-member runs shorter than MIN_GAP_CELLS (the same
    'a sub-0.3 m hole is not a break' convention as the strong run).
    Returns -1 if the anchor's run holds no member cell."""
    member = fld["member"]
    n = len(member)
    edge, gap, k = c, 0, c
    while 0 <= k + step < n:
        k += step
        if member[k]:
            edge, gap = k, 0
        else:
            gap += 1
            if gap >= MIN_GAP_CELLS:
                break
    return edge if member[edge] else -1


def _run_edge(fld, c, step):
    """Marginal member RUN edge as a face position (nan if no member)."""
    k = _run_edge_k(fld, c, step)
    if k < 0:
        return float("nan")
    return float(fld["x"][k] + step * CELL / 2.0)


def _wrap_face(fld, c, step):
    """SIMPLE mode: the face is the wrap edge; sigma is the proven-emptiness
    of the window beyond it. Returns (face, sigma, edge, support, w)."""
    x, member, cnt = fld["x"], fld["member"], fld["cnt"]
    n = len(x)
    k = _run_edge_k(fld, c, step)
    if k < 0:
        return float(x[c]), float("inf"), float(x[c]), 0.0, 0.0
    face = float(x[k] + step * CELL / 2.0)
    v = fld["v_lo"] if step < 0 else fld["v_hi"]
    empty = ~member & ~fld["rawocc"] & ~fld["ambig"]
    k0, k1 = k + step, k + step * WRAP_WIN
    lo_k, hi_k = max(min(k0, k1), 0), min(max(k0, k1), n - 1)
    if hi_k >= lo_k:
        sl = slice(lo_k, hi_k + 1)
        w = float((empty[sl] * v[sl]).mean())
    else:
        w = 0.0
    if w >= WRAP_W_HI:
        sigma = SIG_FLOOR
    elif w <= WRAP_W_LO:
        sigma = float("inf")
    else:
        sigma = SIG_FLOOR + 0.40 * (WRAP_W_HI - w) / (WRAP_W_HI - WRAP_W_LO)
    sup_cells = [cnt[j] for j in (k, k - step, k - 2 * step)
                 if 0 <= j < n and member[j]]
    sup = float(np.mean(sup_cells)) if sup_cells else 0.0
    return face, sigma, face, sup, w


def _strong_run(fld, c):
    """Span of the member run through the anchor, tolerating RUN_GAP cells."""
    x, member = fld["x"], fld["member"]
    n = len(x)

    def reach(step):
        edge, gap, k = c, 0, c
        while 0 <= k + step < n:
            k += step
            if member[k]:
                edge, gap = k, 0
            else:
                gap += 1
                if gap > RUN_GAP:
                    break
        return edge
    lo_k, hi_k = reach(-1), reach(+1)
    return float(x[hi_k] - x[lo_k]) + CELL


def _corners_2d(res, tf, ucx, ucy):
    """Cross-frame consensus L-corners in the canonical frame, keyed by
    (x-side, y-side) relative to the anchor. Each value = (x, y, n): the
    median corner position and its support. A cluster qualifies only when
    well-populated AND tight on BOTH axes — the 2-D analogue of the retired
    end-only consensus gates."""
    from .refiner import _lcorner
    cs = []
    for t in range(tf.n_frames):
        p = np.asarray(res.fg_points[t], np.float64)
        if p.ndim != 2 or len(p) < 20:
            continue
        c = _lcorner(to_canonical(p[:, :3], res.boxes[t])[:, :2])
        if c is not None:
            cs.append(c)
    if len(cs) < CORNER_MIN_ALL:
        return {}
    C = np.asarray(cs)
    out = {}
    for sx in (-1, 1):
        for sy in (-1, 1):
            m = ((np.sign(C[:, 0] - ucx) == sx)
                 & (np.sign(C[:, 1] - ucy) == sy)
                 & (np.abs(C[:, 0] - ucx) > CORNER_MIN_OFF))
            sub = C[m]
            if (len(sub) >= CORNER_MIN and sub[:, 0].std() < CORNER_STD
                    and sub[:, 1].std() < CORNER_STD):
                out[(sx, sy)] = (float(np.median(sub[:, 0])),
                                 float(np.median(sub[:, 1])), int(len(sub)))
    return out


def _fuse_end(cut, sigma, prior_end, sig_prior, outward, floor):
    """Precision-weighted fusion of one face with its prior end. An unobserved
    face anchors on the prior, floored OUTWARD at the parked dense-body edge
    (amodal completion, shipped semantics)."""
    if not np.isfinite(sigma):
        pe = prior_end
        if np.isfinite(floor):
            pe = outward * max(outward * pe, outward * floor)
        return float(pe), float(sig_prior)
    s_eff = (sigma ** -2 + sig_prior ** -2) ** -0.5
    return float(gauss_fuse(cut, sigma, prior_end, sig_prior)), float(s_eff)


def box_fit_ov(res, tf, cfg):
    """OV Stage 2: same contract as boxfit.box_fit (res modified in place)."""
    T = tf.n_frames
    uav = np.stack([np.asarray(b, np.float64) for b in tf.uav_boxes])
    med = np.median(uav[:, 3:6], axis=0)
    veh_class = int(getattr(tf, "veh_class", 0))

    X, Y, Hg, R, Fr = [], [], [], [], []
    for t in range(T):
        p = np.asarray(res.fg_points[t], np.float64)
        if p.ndim != 2 or len(p) == 0:
            continue
        zc = to_canonical(p[:, :3], res.boxes[t])
        X.append(zc[:, 0]); Y.append(zc[:, 1]); Hg.append(p[:, 2] - tf.ground_z[t])
        r = np.asarray(res.fg_r[t], np.float64)
        R.append(r if r.shape[0] == p.shape[0] else np.ones(p.shape[0]))
        Fr.append(np.full(zc.shape[0], t))
    if not X:
        return res
    X, Y, Hg, R, Fr = (np.concatenate(X), np.concatenate(Y), np.concatenate(Hg),
                       np.concatenate(R), np.concatenate(Fr))
    L0, W0, H0 = (float(v) for v in res.dims)
    plen = float(np.sum(np.linalg.norm(np.diff(res.boxes[:, :2], axis=0), axis=1))) if T > 1 else 0.0
    parked = plen < cfg.cf_parked_max

    # ---- UAV centre, roof (shipped machinery, reused) ----
    uc = np.array([to_canonical(np.asarray(tf.uav_boxes[t], np.float64)[None, :3],
                                res.boxes[t])[0, :2] for t in range(T)])
    ucx, ucy = float(np.median(uc[:, 0])), float(np.median(uc[:, 1]))
    inu = (np.abs(X - ucx) < med[0] / 2.0 * 0.7) & (np.abs(Y - ucy) < med[1] / 2.0)
    roof = float(np.quantile(Hg[inu], ROOF_Q)) if inu.sum() >= 20 else 1e18
    rr = _box_roof(X, Y, Hg, L0, W0)
    if rr is not None:
        cap_h = float(med[2]) * (1.4 if veh_class != 0 else 1.0)
        capped = min(rr, cap_h)
        roof = capped if roof > 1e17 else max(roof, capped)
    roof_cap = (roof + ROOF_MARGIN) if roof < 1e17 else 1e18

    # ---- raw candidate cloud in canonical (per frame, vehicle-plausible);
    #      sensor canonical coords per frame ----
    XRawF, YRawF, SXl, SYl = [], [], [], []
    for t in range(T):
        s = to_canonical(np.asarray(tf.lidar_origin[t], np.float64)[None, :3],
                         res.boxes[t])[0]
        SXl.append(float(s[0])); SYl.append(float(s[1]))
        p = np.asarray(tf.cand_pts[t], np.float64)
        if len(p):
            zc = to_canonical(p[:, :3], res.boxes[t])
            hg = p[:, 2] - tf.ground_z[t]
            m = hg > GROUND
            if roof < 1e17:
                m &= hg <= roof_cap
            XRawF.append(zc[m, 0]); YRawF.append(zc[m, 1])
        else:
            XRawF.append(np.zeros(0)); YRawF.append(np.zeros(0))

    tp = TrackPrior(L=float(med[0]), W=float(med[1]),
                    sig_L=cfg.prior_sig_l, sig_W=cfg.prior_sig_w,
                    sig_ctr=cfg.prior_sig_ctr)
    plo_end, phi_end = ucx - tp.L / 2.0, ucx + tp.L / 2.0

    # ---- LENGTH: field -> per-side change-point -> fusion ----
    canopy_from = None
    if roof < 1e17:
        cxg, _cg, chg = _cells(X, Y, Hg, ucy, max(med[1], 2.0))
        canopy_from = (cxg, chg)
    fld = _axis_field(X, Y, Hg, Fr, XRawF, YRawF, ucy, W0 / 2.0 + 0.4, roof_cap,
                      SXl, T, parked, canopy_from=canopy_from, use_weak=not SIMPLE)
    if fld is None:
        return res
    c = int(np.argmin(np.abs(fld["x"] - ucx)))
    w_lo = w_hi = float("nan")
    if SIMPLE:
        cut_lo, sig_lo, edge_lo, sup_lo, w_lo = _wrap_face(fld, c, -1)
        cut_hi, sig_hi, edge_hi, sup_hi, w_hi = _wrap_face(fld, c, +1)
    else:
        cut_lo, sig_lo, edge_lo, sup_lo, _ = _scan_side(fld, c, -1)
        cut_hi, sig_hi, edge_hi, sup_hi, _ = _scan_side(fld, c, +1)
    if parked:
        if SIMPLE:
            de_lo, de_hi = edge_lo, edge_hi     # the wrap is its own floor
        else:
            floor_fn = _run_edge if FLOOR_RUNEDGE else _dense_edge
            de_lo = floor_fn(fld, c, -1)
            de_hi = floor_fn(fld, c, +1)
    else:
        de_lo = de_hi = float("nan")
    # ---- L-corner anchors: the wrap confirms which corner cluster is real
    #      (must sit at the wrap's own face); the surviving corner enters the
    #      face fusion as a third Gaussian at corner grade ----
    corners = _corners_2d(res, tf, ucx, ucy) if (CORNER and SIMPLE) else {}
    diag_c = {}

    def _corner_x(sx, face):
        hits = [(v[0], v[2]) for (kx, ky), v in corners.items()
                if kx == sx and abs(v[0] - face) <= CORNER_GATE_X]
        if not hits:
            return None
        wsum = sum(n for _x, n in hits)
        return sum(x * n for x, n in hits) / max(wsum, 1)

    cx_lo = _corner_x(-1, cut_lo)
    cx_hi = _corner_x(+1, cut_hi)
    if cx_lo is not None:
        if np.isfinite(sig_lo):
            cut_lo, sig_lo = (gauss_fuse(cut_lo, sig_lo, cx_lo, SIG_CORNER),
                              (sig_lo ** -2 + SIG_CORNER ** -2) ** -0.5)
        else:
            cut_lo, sig_lo = cx_lo, SIG_CORNER
        diag_c["corner_lo"] = round(cx_lo, 3)
    if cx_hi is not None:
        if np.isfinite(sig_hi):
            cut_hi, sig_hi = (gauss_fuse(cut_hi, sig_hi, cx_hi, SIG_CORNER),
                              (sig_hi ** -2 + SIG_CORNER ** -2) ** -0.5)
        else:
            cut_hi, sig_hi = cx_hi, SIG_CORNER
        diag_c["corner_hi"] = round(cx_hi, 3)
    flo, s_lo = _fuse_end(cut_lo, sig_lo, plo_end, tp.sig_end, -1.0, de_lo)
    fhi, s_hi = _fuse_end(cut_hi, sig_hi, phi_end, tp.sig_end, +1.0, de_hi)

    cap_l = max(tp.L + cfg.prior_cap_k * tp.sig_L, _strong_run(fld, c) + MARGIN_L)
    L = float(np.clip(fhi - flo,
                      max(cfg.dim_min_l, tp.L - cfg.prior_floor_k * tp.sig_L),
                      min(cap_l, cfg.dim_max_l)))

    # centre: shipped logic (fused ends when a confident end exists; else the
    # outward-floored extent midpoint bounded by the calibrated centre scale)
    ev_lo = cut_lo if np.isfinite(sig_lo) else edge_lo
    ev_hi = cut_hi if np.isfinite(sig_hi) else edge_hi
    if min(s_lo, s_hi) > CONF:
        mlo = min(plo_end, ev_lo)
        mhi = max(phi_end, ev_hi)
        midx = ucx + float(np.clip(0.5 * (mlo + mhi) - ucx, -tp.sig_ctr, tp.sig_ctr))
    else:
        midx = _centre_pull(0.5 * (flo + fhi), flo, fhi, L, X, R)

    if WH_ONLY:
        # flag-2 removal: the W/H point-selection window is the HARVEST box's own
        # extent, not the wrap's fused window -- no wrap influence in the readout.
        flo, fhi = -L0 / 2.0, L0 / 2.0

    # ---- HEIGHT (shipped) ----
    keepz = (X >= flo) & (X <= fhi) & ((Hg <= roof_cap) if roof < 1e17 else True)
    Hk = Hg[keepz] if keepz.sum() >= 20 else Hg
    H = float(min(H0, np.quantile(Hk, WQ) + 0.10))

    # ---- WIDTH: shipped path verbatim (quantile cover, parked valley cut,
    #      one-sided fallback, asymmetric prior band). The lateral axis never
    #      held the rule zoo OV replaces, and its cover quantile is already
    #      the right wall statistic; OV owns the length axis. ----
    diag_w = {}
    inl = (X >= flo) & (X <= fhi)
    if inl.sum() >= 20:
        Xk, Yk, Hgk, Rk = X[inl], Y[inl], Hg[inl], R[inl]
    else:
        Xk, Yk, Hgk, Rk = X, Y, Hg, R

    # ---- optional: wrap the WIDTH axis (member-run edge + grade + fusion),
    #      symmetric to the length wrap; else the coverage-quantile path ----
    wrapw_done = False
    if WRAP_WIDTH:
        midx_box = 0.5 * (flo + fhi)
        half_len = (fhi - flo) / 2.0 + 0.4
        fldw = _axis_field(Y, X, Hg, Fr, YRawF, XRawF, midx_box, half_len,
                           roof_cap, SYl, T, parked, canopy_from=None, use_weak=False)
        if fldw is not None:
            cw = int(np.argmin(np.abs(fldw["x"] - ucy)))
            wlo_cut, wlo_sig, wlo_edge, _, _ = _wrap_face(fldw, cw, -1)
            whi_cut, whi_sig, whi_edge, _, _ = _wrap_face(fldw, cw, +1)
            w_eff = tp.W - tp.sig_W               # de-bias the prior's shadow margin
            sig_end_w = float(np.hypot(tp.sig_W / 2.0, 0.10))
            de_lo_w = wlo_edge if parked else float("nan")
            de_hi_w = whi_edge if parked else float("nan")
            wlo, _ = _fuse_end(wlo_cut, wlo_sig, ucy - w_eff / 2.0, sig_end_w, -1.0, de_lo_w)
            whi, _ = _fuse_end(whi_cut, whi_sig, ucy + w_eff / 2.0, sig_end_w, +1.0, de_hi_w)
            W = float(np.clip(whi - wlo, cfg.dim_min_w, cfg.dim_max_w))
            ylo, yhi = wlo, whi
            midy = _centre_pull(0.5 * (wlo + whi), wlo, whi, W)
            diag_w = {"wrapW": 1,
                      "wsig_lo": round(wlo_sig, 3) if np.isfinite(wlo_sig) else -1,
                      "wsig_hi": round(whi_sig, 3) if np.isfinite(whi_sig) else -1}
            wrapw_done = True
    if not wrapw_done:
        if parked:
            yl, hyl = _profile(Yk, Xk, Hgk, max((fhi - flo) / 2.0, 2.0))
            glo_w, ghi_w = _first_gap(yl, hyl, GROUND, med[1] / 2.0)
            kw = (Yk >= glo_w - 0.05) & (Yk <= ghi_w + 0.05)
            if kw.sum() >= 20:
                Yk, Rk = Yk[kw], Rk[kw]
        ylo = weighted_quantile(Yk, Rk, 1.0 - WQ)
        yhi = weighted_quantile(Yk, Rk, WQ)
        W_obs = (yhi - ylo) + MARGIN_W
        frac_minor = min(float((Yk < ucy).mean()), float((Yk > ucy).mean())) if len(Yk) else 0.0
        if frac_minor < cfg.onesided_thr:              # far wall occluded
            W = max(W_obs, tp.W - tp.sig_W)
        else:
            W = float(np.clip(W_obs, tp.W - RO_KDOWN * tp.sig_W, tp.W + tp.sig_W))
        W = float(np.clip(W, cfg.dim_min_w, cfg.dim_max_w))
        if RO_SYM:
            # uniform symmetric band: prior +/- 1 sigma (one-sided fallback subsumed)
            W = float(np.clip(np.clip(W_obs, tp.W - tp.sig_W, tp.W + tp.sig_W),
                              cfg.dim_min_w, cfg.dim_max_w))
        if RO_WBAND > 0:
            # flat metre band, single clip, no branches
            W = float(np.clip(np.clip(W_obs, tp.W - RO_WBAND, tp.W + RO_WBAND),
                              cfg.dim_min_w, cfg.dim_max_w))
        if RO_WBAND_LO > 0 or RO_WBAND_HI > 0:
            # asymmetric flat metre band, single clip
            W = float(np.clip(np.clip(W_obs, tp.W - RO_WBAND_LO, tp.W + RO_WBAND_HI),
                              cfg.dim_min_w, cfg.dim_max_w))
        if RO_GATEBAND:
            # observability-gated flat bands: blind -> tight, seen -> wide
            b = 0.2 if frac_minor < cfg.onesided_thr else 0.4
            W = float(np.clip(np.clip(W_obs, tp.W - b, tp.W + b),
                              cfg.dim_min_w, cfg.dim_max_w))
        if RO_GATE2:
            # gated hybrid, round metres: blind -> clip +/-0.2; seen -> clip -0.5/+0.2
            if frac_minor < cfg.onesided_thr:
                W = float(np.clip(W_obs, tp.W - 0.2, tp.W + 0.2))
            else:
                W = float(np.clip(W_obs, tp.W - 0.5, tp.W + 0.2))
            W = float(np.clip(W, cfg.dim_min_w, cfg.dim_max_w))
        # corner wall coordinates de-bias the lateral centre: each corner whose
        # wall agrees with the observed cover edge implies a centre at
        # wall -/+ W/2; fused against the aerial lateral centre by precision,
        # then bounded by the usual guardrail.
        midy_target = ucy
        if corners and CORNER_MIDY:
            imps = []
            for (kx, ky), (cx_, cy_, nC) in corners.items():
                edge_y = yhi if ky > 0 else ylo
                if abs(cy_ - edge_y) <= CORNER_GATE_Y:
                    imps.append(cy_ - ky * W / 2.0)
            if imps:
                yc = float(np.mean(imps))
                sig_imp = float(np.hypot(SIG_CORNER, tp.sig_W / 2.0))
                midy_target = float(gauss_fuse(yc, sig_imp, ucy, SIG_UAV_LAT))
                diag_c["midy_corner"] = round(yc, 3)
        midy = _centre_pull(midy_target, ylo, yhi, W)

    # ---- apply + de-harvest (shipped) ----
    boxes = res.boxes.copy()
    offs = res.xy_offsets.copy()
    W2, H2 = float(boxes[0, 4]), float(boxes[0, 5])   # Stage-2 (harvest) W,H
    L2 = float(boxes[0, 3])                           # Stage-2 (harvest) L
    if LEN_ONLY:
        midy = 0.0                                    # revert lateral centre to Stage 2
    if NO_MIDX:
        midx = 0.0                                    # drop the longitudinal centre de-bias
    if WH_ONLY:
        midx = 0.0                                    # Stage-2 longitudinal centre
        L = L2                                        # Stage-2 grow-only length (prior-floored)
        if RO_LCLIP:
            # readout L: final-foreground span, floored at the prior (amodal
            # completion) and capped at the calibrated band / class limit
            xlo_ro = weighted_quantile(X, R, 1.0 - WQ)
            xhi_ro = weighted_quantile(X, R, WQ)
            cap_ro = max(cfg.dim_max_l_car if veh_class == 0 else cfg.dim_max_l,
                         tp.L + cfg.prior_cap_k * tp.sig_L)
            L = float(np.clip(max((xhi_ro - xlo_ro) + RO_MARGIN, tp.L),
                              cfg.dim_min_l, cap_ro))
        if RO_SYM:
            # uniform symmetric band: span + margin clipped into prior +/- 1 sigma
            xlo_ro = weighted_quantile(X, R, 1.0 - WQ)
            xhi_ro = weighted_quantile(X, R, WQ)
            L = float(np.clip(np.clip((xhi_ro - xlo_ro) + RO_MARGIN,
                                      tp.L - tp.sig_L, tp.L + tp.sig_L),
                              cfg.dim_min_l, cfg.dim_max_l))
    for t in range(T):
        shift = canonical_xy_to_world(np.array([midx, midy]), float(boxes[t, 6]))
        boxes[t, 0:2] = boxes[t, 0:2] + shift
        offs[t] = offs[t] + shift
    boxes[:, 3] = L
    if LEN_ONLY:
        # keep Stage-2 W, H, z (already in the copy); only the length axis is refined
        W, H = W2, H2
    else:
        if RO_NOH:
            H = H2                                    # keep Stage-2 (harvest) height
        if getattr(cfg, "width_from_mstep", False):
            W = W2
        boxes[:, 4] = W
        boxes[:, 5] = H
        boxes[:, 2] = np.asarray(tf.ground_z, np.float64) + H / 2.0
    res.boxes = boxes
    res.xy_offsets = offs
    res.dims = np.array([L, W, H])

    hx, hy, hz = L / 2.0 + 0.3, W / 2.0 + 0.3, H / 2.0 + 0.3
    fg_p, fg_r = [], []
    for t in range(T):
        p = np.asarray(res.fg_points[t], np.float64)
        if p.ndim == 2 and len(p):
            zc = to_canonical(p[:, :3], boxes[t]); hgf = p[:, 2] - tf.ground_z[t]
            inb = ((np.abs(zc[:, 0]) <= hx) & (np.abs(zc[:, 1]) <= hy)
                   & (np.abs(zc[:, 2]) <= hz) & (hgf <= roof_cap))
            fg_p.append(p[inb]); r = np.asarray(res.fg_r[t])
            fg_r.append(r[inb] if r.shape[0] == p.shape[0] else r)
        else:
            fg_p.append(res.fg_points[t]); fg_r.append(res.fg_r[t])
    res.fg_points, res.fg_r = fg_p, fg_r
    if isinstance(res.diagnostics, list):
        res.diagnostics.append({"boxfit_ov": {
            "cut_lo": round(float(cut_lo), 3), "cut_hi": round(float(cut_hi), 3),
            "sig_lo": round(sig_lo, 3) if np.isfinite(sig_lo) else -1,
            "sig_hi": round(sig_hi, 3) if np.isfinite(sig_hi) else -1,
            "sup_lo": round(sup_lo, 1), "sup_hi": round(sup_hi, 1),
            "flo": round(float(flo), 3), "fhi": round(float(fhi), 3),
            "plo": round(float(plo_end), 3), "phi": round(float(phi_end), 3),
            "cap_l": round(float(cap_l), 2), "parked": bool(parked),
            "w_lo": round(w_lo, 3) if np.isfinite(w_lo) else -1,
            "w_hi": round(w_hi, 3) if np.isfinite(w_hi) else -1,
            "L": round(L, 3), "W": round(W, 3), "H": round(H, 3),
            **diag_w, **diag_c}})
    return res
