"""Stage 2: the deterministic box-fit on a converged harvest — the FUSED variant.

The silhouette machinery is the unified EM refiner's (vehicle-cell membership, reachability
walk, gap bridging, canopy guard, dense extension/cap-lift, roof, width valley).
What changed is HOW the prior enters. the unified EM refiner's prior was 34% under-scaled, so
it was safe as a one-sided AMODAL FLOOR ("at least the UAV extent") with hard
class caps above. The GSD-fixed prior is a real measurement (unbiased, calibrated
sigma) whose error is SYMMETRIC — so the floor/cap asymmetry is replaced by ONE
mechanism (see prior.py):

  * each box END is fused per-end: the silhouette walk reports its edge together
    with a STATUS (closed / open / canopy) and a boundary support; corner
    consensus supplies a tighter end where available; the prior end contributes
    its own calibrated Gaussian. Precision-weighted fusion replaces the amodal
    floor, the corner override, AND most of `_centre_pull` on the length axis —
    with two confident ends the centre follows the evidence (de-biasing the
    body-fixed UAV centre error), with none it stays on the prior.
  * the S8 gap ambiguity (sparse occluded far end vs clutter) is arbitrated by
    the prior band instead of the parked/moving gate alone: a sparse 'closed'
    boundary gets a wide sigma, so the fused end lands between the evidence and
    the (now trustworthy) prior end.
  * caps are the prior band (L <= prior + k*sigma_L for every class), liftable
    by the dense directly-observed run — the per-class cap constants become
    global backstops only.
  * WIDTH fuses the quantile cover against the prior band (the prior W runs
    ~+0.3 m over — shadow margin — so the band is asymmetric); the one-sided
    case falls back to prior - 1*sigma instead of a hand class floor.

HEIGHT is unchanged (the prior carries no measured height).
"""
from __future__ import annotations

import numpy as np

from .geometry import (canonical_xy_to_world, to_canonical,
                                                  weighted_quantile)
from .prior import TrackPrior, fuse_track_ends

# ---- structural scale constants (data resolution 0.1 m; swept and frozen) ----
CELL = 0.1           # BEV pillar size for the silhouette
VBIN = 0.15          # vertical bin for the robust per-pillar height (density-band top)
VGAP = 0.5           # the vertical density band tolerates a gap up to this (underbody)
GROUND = 0.3         # membership lower bound: a vehicle cell must rise above ground-clearance
ROOF_MARGIN = 0.4    # membership upper bound: roof + this = "taller than the vehicle"
NMIN = 5             # membership occupancy: a vehicle cell holds at least this many points
VEH_H = 1.1          # reachability: cross a gap only toward a cabin-height (>= this) cell
MIN_GAP = 0.30       # reachability: a non-vehicle run >= this is a gap (else absorbed)
Q_PILLAR = 0.98      # pillar height = band top capped at this quantile (noise-spike robust)
MARGIN_L = 0.10
MARGIN_W = 0.0
WQ = 0.99            # width cover quantile / height roof quantile
ROOF_Q = 0.95        # roof = this quantile of heights inside the UAV interior window
DF_MIN = 3           # range-adaptive occupancy: >= this many DISTINCT frames (moving only)
ROOF_BOX_MARGIN = 0.30   # robust roof: margin around the frozen harvest-box footprint
ROOF_CAP_LARGE = 1.4     # roof cap = class height for cars, x this for large classes
# ---- unified dense-extent (length extension + car cap-lift) ----
GROW_FRAC = 0.30     # extend while a cell's per-frame density >= this x the body density
GROW_CELL = 0.5      # coarse bin for the grow density (0.1 m is too fine — single-cell gaps)
GROW_GAP = 1         # tolerate this many below-threshold cells (a window/door gap)
GROW_MIN_BODY = 2.0  # skip the grow unless the body itself is dense (cand pts/frame/cell)
GROW_MIN_DENS = 1.0  # absolute density floor on a grown cell
UNI_WIN = 0.5        # cap-lift sliding window: 0.5 m occupancy robustness ...
UNI_STEP = 0.1       # ... stepped at 0.1 m boundary resolution
UNI_GAP_CNT = 2      # a window with >= this many TOTAL candidate points is occupied


def _band_top(h, q=None):
    """Robust pillar height = top of the connected vertical density band (rejects a sparse
    canopy above a gap), not the single highest point. If `q` is given, ALSO cap the band
    top at the q-quantile of the heights so a few noise points above the roof don't
    inflate it."""
    if len(h) < 2:
        return float(h.max()) if len(h) else 0.0
    lo, hi = max(0.0, float(h.min())), float(h.max())
    edges = np.arange(lo, hi + VBIN, VBIN)
    if len(edges) < 2:
        return min(hi, float(np.quantile(h, q))) if q is not None else hi
    cnt, _ = np.histogram(h, bins=edges)
    top, miss = lo, 0.0
    for i in range(len(cnt)):
        if cnt[i] >= 1:
            top, miss = edges[i + 1], 0.0
        else:
            miss += VBIN
            if miss > VGAP and edges[i] > lo:
                break
    return min(top, float(np.quantile(h, q))) if q is not None else top


def _profile(A, B, Hg, gate, q=None):
    """1-D pillar-height silhouette along A (gate |B| <= gate), on a COMPLETE grid (empty
    bins = 0) so a drop is visible; each bin = max robust pillar height."""
    m = np.abs(B) <= gate
    a, h = A[m], Hg[m]
    if len(a) == 0:
        return np.array([]), np.array([])
    ai = np.round(a / CELL).astype(int)
    amin, amax = int(ai.min()), int(ai.max())
    grid = np.zeros(amax - amin + 1)
    order = np.argsort(ai); ai, h = ai[order], h[order]
    for g in np.split(np.arange(len(h)), np.where(np.diff(ai))[0] + 1):
        idx = ai[g[0]] - amin
        grid[idx] = max(grid[idx], _band_top(h[g], q))
    return (np.arange(amin, amax + 1)) * CELL, grid


def _first_gap(xc, h, thr, uav_half, veh=0.9):
    """Lateral over-harvest boundary (lo, hi). A side is cut ONLY at a VALLEY — a ground
    gap (h < thr) FOLLOWED by a separate structure rising back to >= `veh` (a second
    object). A gap with nothing rising beyond it (the sparse occluded far wall, or the
    vehicle tapering off) does NOT cut — one-sided vehicles are not shrunk. A side with
    no valley returns +-inf (keep all points there). Sub-bin interpolated."""
    if len(xc) < 3:
        return -1e18, 1e18
    n = len(xc)
    near = np.abs(xc) <= uav_half + CELL
    c = int(np.where(near)[0][np.argmax(h[near])]) if near.any() else int(np.argmax(h))

    def side(step):
        prev_high, j = c, c
        while 0 <= j + step < n:
            j += step
            if h[j] >= thr:
                prev_high = j
            else:
                k = j                                   # extent of this ground gap
                while 0 <= k + step < n and h[k + step] < thr:
                    k += step
                if 0 <= k + step < n and h[k + step] >= veh:   # a SEPARATE structure beyond
                    den = h[prev_high] - h[j]
                    fr = (h[prev_high] - thr) / den if den > 1e-9 else 0.0
                    return xc[prev_high] + (xc[j] - xc[prev_high]) * min(max(fr, 0.0), 1.0)
                j = k                                   # nothing beyond -> bridge, keep going
        return step * 1e18                              # no valley on this side -> keep all
    return side(-1), side(+1)


def _cells(A, B, Hg, bc, gate, F=None):
    """1-D cells of size CELL along axis A, over points with |B - bc| <= gate, on a
    COMPLETE grid. Returns (centers, count, height[, distinct_frames]). height is the
    q-capped robust pillar height; distinct_frames (only when F is given) is the number
    of distinct source frames per cell — a range-robust occupancy signal."""
    m = np.abs(B - bc) <= gate
    a, h = A[m], Hg[m]
    f = F[m] if F is not None else None
    if len(a) == 0:
        empty = (np.array([]), np.array([]), np.array([]))
        return empty + (np.array([]),) if F is not None else empty
    ai = np.round(a / CELL).astype(int)
    amin, amax = int(ai.min()), int(ai.max())
    n = amax - amin + 1
    cnt = np.bincount(ai - amin, minlength=n).astype(float)
    hgt = np.zeros(n)
    dfr = np.zeros(n)
    order = np.argsort(ai); ais, hs = ai[order], h[order]
    fs = f[order] if f is not None else None
    for g in np.split(np.arange(len(hs)), np.where(np.diff(ais))[0] + 1):
        idx = ais[g[0]] - amin
        hgt[idx] = _band_top(hs[g], Q_PILLAR)
        if fs is not None:
            dfr[idx] = len(np.unique(fs[g]))
    centers = np.arange(amin, amax + 1) * CELL
    return (centers, cnt, hgt, dfr) if F is not None else (centers, cnt, hgt)


def _canopy_mask(cx, cxg, chg, thr):
    """Boolean over the tight-gate cells cx: True where a TALLER-than-thr canopy sits at
    this x over the GENEROUS gate — a laterally-offset tree whose tall column the tight
    gate misses but whose low base would otherwise pass membership."""
    if len(cxg) == 0:
        return np.zeros(len(cx), bool)
    gmap = {int(round(g / CELL)): h for g, h in zip(cxg, chg)}
    return np.array([gmap.get(int(round(x / CELL)), 0.0) > thr for x in cx], bool)


def _box_roof(X, Y, Hg, rbl, rbw, margin=ROOF_BOX_MARGIN, q=0.90):
    """Robust roof = q-quantile of the PER-CELL pillar tops over the dense cells inside
    the (frozen) harvest-box footprint. Pillar tops, NOT a raw point-height quantile: a
    vehicle cloud is bottom-heavy, so a raw quantile in the box window is diluted; the
    per-cell tops aren't, and they match the membership test `_vehicle_extent` uses.
    Returns None if the footprint holds too few points."""
    m = (np.abs(X) <= rbl / 2.0 + margin) & (np.abs(Y) <= rbw / 2.0 + margin)
    if int(m.sum()) < 20:
        return None
    Xm, Ym, Hm = X[m], Y[m], Hg[m]
    xi = np.round(Xm / CELL).astype(int); yi = np.round(Ym / CELL).astype(int)
    x0, y0 = int(xi.min()), int(yi.min())
    nx, ny = int(xi.max()) - x0 + 1, int(yi.max()) - y0 + 1
    if nx < 1 or ny < 1 or nx * ny > 400000:
        return None
    flat = (xi - x0) * ny + (yi - y0)
    cnt = np.bincount(flat, minlength=nx * ny).astype(float)
    hgt = np.zeros(nx * ny)
    order = np.argsort(flat, kind="stable"); fs = flat[order]; hs = Hm[order]
    for g in np.split(np.arange(len(hs)), np.where(np.diff(fs))[0] + 1):
        hgt[fs[g[0]]] = _band_top(hs[g], Q_PILLAR)
    dense = (cnt >= NMIN) & (hgt > GROUND)
    if int(dense.sum()) < 8:
        return None
    return float(np.quantile(hgt[dense], q))


def _vehicle_extent(centers, cnt, hgt, ac, uav_half, roof, canopy=None, parked=False,
                    df=None, df_min=0, naive=False, edge_ctx=None, low_bridge=None,
                    fade_thr=0.3, fade_sup_max=float('inf'),
                    gap_close_sup_min=0.0):
    """Box extent along the length axis = the run of VEHICLE cells reachable from the
    UAV-centre cell. Returns ((lo, st_lo, sup_lo), (hi, st_hi, sup_hi)) — per end the
    evidence edge, HOW it was established, and the boundary support:

      'closed' — the walk SAW the boundary (a gap with nothing tall beyond it);
                 support = mean occupancy of the outermost member cells, grading a
                 dense wall vs a sparse fade-out (the S8 case — the fusion lets the
                 prior arbitrate exactly there).
      'open'   — the walk ran off the observed grid (occlusion / edge of data):
                 no boundary evidence; the caller falls back to the prior end.
      'canopy' — a taller-than-roof canopy beyond the prior end on this side (an
                 offset tree whose grounded inter-region would connect through):
                 use the prior end exactly — the the unified EM refiner guard, kept verbatim.

    NO amodal floor here anymore: the prior enters through the per-end fusion,
    weighted by these statuses, not as a hard one-sided floor.

    The gap-bridge itself is unchanged: MOVING bridges on height alone (occluded
    rear), PARKED bridges only to a dense member; the edge advances on members
    either way, so a tall clutter stray can never grow the box."""
    le, re = ac - uav_half, ac + uav_half
    if len(centers) < 3:
        return (le, "open", 0.0, le), (re, "open", 0.0, re)
    roof_cap = (roof + ROOF_MARGIN) if roof < 1e17 else 1e18
    occ = (cnt >= NMIN) if df is None else ((cnt >= NMIN) | (df >= df_min))
    member = occ & (hgt > GROUND) & (hgt <= roof_cap)
    if canopy is not None:
        member = member & ~canopy
    # bridge target: MEMBER cells at cabin height. the unified EM refiner bridged moving
    # vehicles on raw height alone — but that lets a distant TREE column keep the
    # walk alive to the grid edge (an 'open' end where a real boundary existed).
    # The df-aided membership already admits the sparse occluded rear (the S8
    # case), so members-only bridging keeps the recovery and drops the clutter.
    tall = (hgt >= VEH_H) & member
    c = int(np.argmin(np.abs(centers - ac)))
    gtol = max(1, int(round(MIN_GAP / CELL)))

    if naive:
        # ABLATION: raw membership extent — no reachability walk, no gap logic,
        # no closed/open grading. Every edge counts as observed evidence.
        idx = np.where(member)[0]
        if not len(idx):
            return (le, "open", 0.0, le), (re, "open", 0.0, re)
        out = []
        for edge, step in ((int(idx[0]), -1), (int(idx[-1]), +1)):
            sup_cells = [cnt[k] for k in (edge, edge - step, edge - 2 * step)
                         if 0 <= k < len(centers) and member[k]]
            sup = float(np.mean(sup_cells)) if sup_cells else 0.0
            out.append((centers[edge] + step * CELL / 2.0, "closed", sup,
                        float(centers[edge] + step * CELL / 2.0) if parked else np.nan))
        return out[0], out[1]

    def _low_body(j, step):
        """A LOW but substantial continuation just past the gap — a pickup bed, a
        trailer deck, a trunk at range. The cabin-height bridge test cannot see
        these, so the walk would stop at the cab/bed gap and call it the vehicle's
        end. Bridge to it only when it is CONTIGUOUS (clutter is patchy) and still
        inside the prior's footprint band (clutter is not)."""
        if low_bridge is None:
            return False
        reach, min_run, win = low_bridge
        k0, k1 = j + step, j + step * max(int(round(win / CELL)), 1)
        lo_k, hi_k = max(min(k0, k1), 0), min(max(k0, k1), len(centers) - 1)
        if hi_k <= lo_k:
            return False
        sl = slice(lo_k, hi_k + 1)
        seg = member[sl] & (np.abs(centers[sl] - ac) <= reach)
        best = cur = 0
        for v in seg:
            cur = cur + 1 if v else 0
            best = max(best, cur)
        return best >= min_run

    def side(step):
        edge, j, run, closed = c, c, 0, False
        while 0 <= j + step < len(centers):
            j += step
            if member[j]:
                edge, run = j, 0
            else:
                run += 1
                if run >= gtol:                                  # a real gap
                    beyond = tall[j:] if step > 0 else tall[:j + 1]
                    if not np.any(beyond):                       # nothing cabin-height past it
                        if _low_body(j, step):                   # ... but a low body continues
                            run = 0
                            continue
                        closed = True
                        break
                    run = 0                                       # tall vehicle beyond -> bridge
        certified = False
        if not closed and edge_ctx is not None:
            # Ran off the raster while on (or near) body: the harvested raster
            # cannot say whether the region beyond was scanned-empty (a wall)
            # or unobserved. Ask the RAW candidate cloud.
            outward, plaus = edge_ctx(centers[edge] + step * CELL / 2.0, step)
            if outward >= 0.3 and plaus <= 0.3:                  # viewed face-on AND empty -> wall
                closed = certified = True
        sup_cells = [cnt[k] for k in (edge, edge - step, edge - 2 * step)
                     if 0 <= k < len(centers) and member[k]]
        sup = float(np.mean(sup_cells)) if sup_cells else 0.0
        # FADE test: a 'closed' boundary is only a real wall if the cells beyond
        # the gap hold no VEHICLE-PLAUSIBLE occupancy (grounded, sub-roof). A
        # sparse plausible tail trailing outward is the S8 fade (occluded far end)
        # — no wall was seen, the prior must arbitrate. Above-roof clutter (the
        # tree the walk just refused to bridge to) does NOT make a fade.
        k0, k1 = j + step, j + step * int(round(1.5 / CELL))
        lo_k, hi_k = max(min(k0, k1), 0), min(max(k0, k1), len(centers) - 1)
        if hi_k >= lo_k:
            sl = slice(lo_k, hi_k + 1)
            plaus = (cnt[sl] > 0) & (hgt[sl] > GROUND) & (hgt[sl] <= roof_cap)
            fade = float(plaus.mean())
        else:
            fade = 0.0
        if closed and fade > fade_thr and sup < fade_sup_max:
            closed = False
        # A GAP closure (as opposed to a ray-certified one) is right barely half
        # the time; require real boundary support before trusting it.
        if closed and not certified and sup < gap_close_sup_min:
            closed = False
        # DENSE-BODY EDGE: the outermost member cell still carrying body-level
        # occupancy (>= half the robust body density). The marginal member edge
        # rides the aggregation/motion smear; the dense edge does not — it is
        # the smear-robust "body observed at least to here" bound.
        # Smear-robust outward floor for the prior anchor — PARKED ONLY. A parked
        # aggregate's smear is registration jitter (~10 cm, sparse): the dense-body
        # edge bounds the true body. A MOVING aggregate's smear is ego-motion /
        # deskew error — long AND dense — so no occupancy statistic separates it
        # from body; a moving open end gets NO evidence floor (the prior and the
        # confident-end coupling carry it).
        if parked:
            body = [cnt[k] for k in range(c, edge + step, step) if member[k]]
            ref = float(np.median(body)) if body else float(NMIN)
            dense_edge = centers[c]
            for k in range(edge, c - step, -step):
                if member[k] and cnt[k] >= max(NMIN, 0.5 * ref):
                    dense_edge = centers[k] + step * CELL / 2.0
                    break
        else:
            dense_edge = np.nan
        return (centers[edge] + step * CELL / 2.0,
                ("closed" if closed else "open"), sup, float(dense_edge))

    lo_end, hi_end = side(-1), side(+1)
    if canopy is not None and np.any(canopy & (centers < le)):
        lo_end = (le, "canopy", 0.0, le)
    if canopy is not None and np.any(canopy & (centers > re)):
        hi_end = (re, "canopy", 0.0, re)
    return lo_end, hi_end


def _dense_grow(tf, res, ucy, w_half, roof, lo, hi, frac=GROW_FRAC):
    """EXTEND the length extent [lo,hi] outward into DENSE, CONTIGUOUS candidate cells —
    the un-harvested far body of a long vehicle the UAV under-detected. Slices the
    candidate cloud (sub-roof, in the lateral band) into GROW_CELL bins and uses
    PER-FRAME density: in the canonical frame the vehicle concentrates (20+/frame) while
    smear spreads thin (<1/frame). Grows while density >= frac x body density, tolerating
    GROW_GAP below-threshold cells, stopping at a sustained valley. EXTEND-ONLY — never
    trims, so the sparse-far-end recovery is kept. Only fires on a dense+contiguous far
    body, so it can't bridge a gap to a lead vehicle or grow into a moving-ego smear."""
    nfr = max(tf.n_frames, 1)
    XS = []
    for t in range(tf.n_frames):
        p = np.asarray(tf.cand_pts[t], np.float64)
        if not len(p):
            continue
        zc = to_canonical(p[:, :3], res.boxes[t])
        hg = p[:, 2] - tf.ground_z[t]
        m = (np.abs(zc[:, 1] - ucy) < w_half) & (hg > GROUND)
        if roof < 1e17:
            m &= hg < roof + ROOF_MARGIN
        if m.any():
            XS.append(zc[m, 0])
    if not XS:
        return lo, hi
    XS = np.concatenate(XS)
    e0, e1 = np.floor(XS.min() / GROW_CELL), np.ceil(XS.max() / GROW_CELL)
    edges = np.arange(e0, e1 + 1) * GROW_CELL
    if len(edges) < 3:
        return lo, hi
    cnt, _ = np.histogram(XS, edges)
    cen = 0.5 * (edges[:-1] + edges[1:])
    dens = cnt / nfr
    inb = (cen >= lo) & (cen <= hi)
    if not inb.any():
        return lo, hi
    body = float(np.median(dens[inb]))
    if body < GROW_MIN_BODY:                 # sparse track -> no reliable dense body
        return lo, hi
    thr = max(frac * body, GROW_MIN_DENS)    # absolute floor: a low body can't admit clutter

    def walk(start, step):
        edge, run, j = start, 0, start
        while 0 <= j + step < len(cen):
            j += step
            if dens[j] >= thr:
                edge, run = j, 0
            else:
                run += 1
                if run > GROW_GAP:
                    break
        return float(cen[edge])

    glo = walk(int(np.argmin(np.abs(cen - lo))), -1)
    ghi = walk(int(np.argmin(np.abs(cen - hi))), +1)
    return min(lo, glo), max(hi, ghi)   # EXTEND-ONLY


def _dense_extent(tf, res, ucy, w_half, roof, lo, hi, cap_l):
    """The unified length rule, two parts:

    CAP-LIFT (every class — the prior band replaced the per-class cap constants):
      lift the size-cap to the CONTINUOUS directly-observed run covering the
      validated silhouette [lo,hi] — a run longer than the prior band IS a longer
      vehicle than the UAV thought. A 0.5 m window stepped at 0.1 m gives occupancy
      robustness AND boundary resolution; occupancy is ABSOLUTE (>= UNI_GAP_CNT total
      points), so a sparse distinct-frame-bridged merge can't lift. Bounded by [lo,hi]:
      never credits length the harvest didn't see.
    EXTENSION: reach outward from [lo,hi] into a dense un-harvested far body — delegated
      to `_dense_grow` (its non-overlapping bins place the far boundary correctly; an
      overlapping window dilutes at the body edge and stops short).

    Returns (xlo, xhi, cap_l). Extend-only; the cap lifts only upward."""
    XS = []
    for t in range(tf.n_frames):
        p = np.asarray(tf.cand_pts[t], np.float64)
        if not len(p):
            continue
        zc = to_canonical(p[:, :3], res.boxes[t])
        hg = p[:, 2] - tf.ground_z[t]
        m = (np.abs(zc[:, 1] - ucy) < w_half) & (hg > GROUND)
        if roof < 1e17:
            m &= hg < roof + ROOF_MARGIN
        if m.any():
            XS.append(zc[m, 0])
    if XS:
        XS = np.concatenate(XS)
        e0, e1 = np.floor(XS.min() / UNI_STEP), np.ceil(XS.max() / UNI_STEP)
        edges = np.arange(e0, e1 + 2) * UNI_STEP
        if len(edges) >= 3:
            cnt, _ = np.histogram(XS, edges)
            cen = 0.5 * (edges[:-1] + edges[1:])
            k = max(1, int(round(UNI_WIN / UNI_STEP)))
            # centered windowed count via cumsum (always len(cnt), unlike convolve)
            h = k // 2
            cs = np.concatenate(([0.0], np.cumsum(cnt.astype(np.float64))))
            ii = np.arange(len(cnt))
            cw = cs[np.minimum(ii + h + 1, len(cnt))] - cs[np.maximum(ii - h, 0)]
            inb = (cen >= lo) & (cen <= hi)
            if inb.any():
                core = int(np.flatnonzero(inb)[np.argmax(cw[inb])])
                gl = gr = core
                while gl - 1 >= 0 and cen[gl - 1] >= lo and cw[gl - 1] >= UNI_GAP_CNT:
                    gl -= 1
                while gr + 1 < len(cen) and cen[gr + 1] <= hi and cw[gr + 1] >= UNI_GAP_CNT:
                    gr += 1
                cap_l = max(cap_l, (cen[gr] - cen[gl]) + MARGIN_L)

    xlo, xhi = _dense_grow(tf, res, ucy, w_half, roof, lo, hi)
    return xlo, xhi, cap_l


def _centre_pull(uav_c, lo, hi, dim, coords=None, w=None):
    """Centre a box of size `dim` over the extent [lo, hi]. Default: anchor on the
    reliable UAV centre `uav_c`, clamped so a wall is never pushed past the extent (a
    midpoint would follow a contaminated extent toward a neighbour). DENSE-MASS override
    (length axis only): when the box is SHORTER than the extent (a size-capped vehicle),
    anchor on the dense mass (weighted median) so the box covers the body and clips a
    sparse tail."""
    a, b = hi - dim / 2.0, lo + dim / 2.0
    if coords is not None and len(coords) and (hi - lo) > dim:
        dense = weighted_quantile(coords, w, 0.5) if w is not None else float(np.median(coords))
        return float(np.clip(dense, min(a, b), max(a, b)))
    return float(np.clip(uav_c, min(a, b), max(a, b)))


def box_fit(res, tf, cfg, corner=None):
    """Fit the final box on a converged harvest `res` (modified in place and returned).

    `corner`: optional {'xlo': x, 'xhi': x} cross-frame-consistent END anchors in the
    canonical frame (see refiner._corner_consensus) — a corner is a 2-D-localised,
    noise-robust anchor, so a reliable one overrides the silhouette end."""
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
    L0, W0, H0 = (float(x) for x in res.dims)
    plen = float(np.sum(np.linalg.norm(np.diff(res.boxes[:, :2], axis=0), axis=1))) if T > 1 else 0.0
    parked = plen < cfg.cf_parked_max

    # ---- the reliable UAV centre in canonical + the vehicle's own roof scale ----
    uc = np.array([to_canonical(np.asarray(tf.uav_boxes[t], np.float64)[None, :3], res.boxes[t])[0, :2]
                   for t in range(T)])
    ucx, ucy = float(np.median(uc[:, 0])), float(np.median(uc[:, 1]))
    inu = (np.abs(X - ucx) < med[0] / 2.0 * 0.7) & (np.abs(Y - ucy) < med[1] / 2.0)
    roof = float(np.quantile(Hg[inu], ROOF_Q)) if inu.sum() >= 20 else 1e18
    # Robust roof: per-cell pillar tops inside the FROZEN harvest-box footprint, capped at
    # the UAV class height (the harvest box over-reads when it has grown into a tree). The
    # harvest box already fits the vehicle, so the roof lands ON it — the small UAV-centred
    # window instead lands OFF a one-sided mis-placed box and under-reads, which would trip
    # the tall-clutter guard and pin the length to the prior.
    rr = _box_roof(X, Y, Hg, L0, W0)
    if rr is not None:
        cap_h = float(med[2]) * (ROOF_CAP_LARGE if veh_class != 0 else 1.0)
        capped = min(rr, cap_h)
        roof = capped if roof > 1e17 else max(roof, capped)

    # ---- LENGTH: silhouette extent -> PER-END FUSION against the calibrated prior ----
    tp = TrackPrior(L=float(med[0]), W=float(med[1]),
                    sig_L=cfg.prior_sig_l, sig_W=cfg.prior_sig_w, sig_ctr=cfg.prior_sig_ctr)
    cx, cn, ch, cdf = _cells(X, Y, Hg, ucy, W0 / 2.0 + 0.4, F=Fr)
    if roof < 1e17:
        cxg, _cg, chg = _cells(X, Y, Hg, ucy, max(med[1], 2.0))
        canopy = _canopy_mask(cx, cxg, chg, roof + ROOF_MARGIN)
    else:
        canopy = None
    ectx = None
    if cfg.closed_at_edge:
        # RAY-GEOMETRY observedness: an end face is only visible from OUTWARD of
        # it, and every ray that hit it traversed the region beyond — so a high
        # outward-sensor fraction certifies that region as observed-empty
        # (ground returns not needed; the cache crop is ground-filtered).
        XR, HR, SX = [], [], []
        _gate = W0 / 2.0 + 0.4
        for t in range(T):
            SX.append(float(to_canonical(
                np.asarray(tf.lidar_origin[t], np.float64)[None, :3],
                res.boxes[t])[0, 0]))
            p = np.asarray(tf.cand_pts[t], np.float64)
            if not len(p):
                continue
            zc = to_canonical(p[:, :3], res.boxes[t])
            m = np.abs(zc[:, 1] - ucy) < _gate
            if m.any():
                XR.append(zc[m, 0])
                HR.append(p[m, 2] - tf.ground_z[t])
        if XR:
            XR, HR = np.concatenate(XR), np.concatenate(HR)
            SX = np.asarray(SX, np.float64)
            _rcap = roof + ROOF_MARGIN if roof < 1e17 else 1e18

            def ectx(x0, step, _XR=XR, _HR=HR, _SX=SX, _rc=_rcap):
                # observedness = fraction of frames with the sensor outward of
                # this edge (their rays traversed the beyond-region to hit it)
                outward = float((np.sign(_SX - x0) == np.sign(step)).mean())
                lo, hi = sorted((x0 + step * CELL, x0 + step * 1.5))
                nbin = max(int(round((hi - lo) / CELL)), 1)
                m = (_XR >= lo) & (_XR < hi)
                if not m.any():
                    return outward, 0.0
                b = np.floor((_XR[m] - lo) / CELL).astype(int)
                pm = (_HR[m] > GROUND) & (_HR[m] <= _rc)
                return outward, (len(np.unique(b[pm])) / nbin) if pm.any() else 0.0
    (xlo, st_lo, sup_lo, de_lo), (xhi, st_hi, sup_hi, de_hi) = _vehicle_extent(
        cx, cn, ch, ucx, tp.L / 2.0, roof, canopy=canopy, parked=parked,
        df=(cdf if not parked else None), df_min=DF_MIN,
        naive=cfg.abl_naive_extent, edge_ctx=ectx,
        low_bridge=((tp.L / 2.0 + cfg.prior_cap_k * tp.sig_L,
                     cfg.low_bridge_min_cells, cfg.low_bridge_win)
                    if cfg.low_bridge else None),
        fade_thr=cfg.fade_thr, fade_sup_max=cfg.fade_sup_max,
        gap_close_sup_min=cfg.gap_close_sup_min)
    # the dense extension reaches into directly-observed body: an end it moves is
    # evidence-closed by construction (the grow threshold IS a dense-body test)
    glo, ghi, cap_l = _dense_extent(tf, res, ucy, W0 / 2.0 + 0.4, roof, xlo, xhi,
                                    tp.L + cfg.prior_cap_k * tp.sig_L)
    if glo < xlo - CELL:
        xlo, st_lo, sup_lo, de_lo = glo, "closed", max(sup_lo, 3.0 * NMIN), glo
    if ghi > xhi + CELL:
        xhi, st_hi, sup_hi, de_hi = ghi, "closed", max(sup_hi, 3.0 * NMIN), ghi
    if corner is not None:              # a reliable corner IS the strongest end evidence
        if corner.get("xlo") is not None:
            xlo, st_lo = corner["xlo"], "corner"
        if corner.get("xhi") is not None:
            xhi, st_hi = corner["xhi"], "corner"
    plo_end, phi_end = ucx - tp.L / 2.0, ucx + tp.L / 2.0
    if st_lo == "canopy":
        lo_arg = (plo_end, "corner", 0.0, plo_end)   # prior end, taken exactly
    else:
        lo_arg = (xlo, st_lo, sup_lo, de_lo)
    if st_hi == "canopy":
        hi_arg = (phi_end, "corner", 0.0, phi_end)
    else:
        hi_arg = (xhi, st_hi, sup_hi, de_hi)
    if cfg.abl_ends_prior_only:           # ABLATION: evidence ignored, floors kept
        lo_arg = (lo_arg[0], "open", 0.0, lo_arg[3])
        hi_arg = (hi_arg[0], "open", 0.0, hi_arg[3])
    sig_end_eff = tp.sig_end
    if cfg.prior_span_gate and lo_arg[1] == "closed" and hi_arg[1] == "closed":
        span_delta = abs((hi_arg[0] - lo_arg[0]) - tp.L)
        if span_delta > 3.0 * tp.sig_L:      # prior length contradicted by evidence
            sig_end_eff = tp.sig_end * (span_delta / (3.0 * tp.sig_L))
    flo, fhi, s_lo, s_hi = fuse_track_ends(lo_arg, hi_arg, plo_end, phi_end,
                                           sig_end_eff, tp.sig_L,
                                           flat_sigma=cfg.abl_flat_end_sigma,
                                           no_couple=not cfg.end_coupling,
                                           no_amodal=cfg.abl_no_amodal,
                                           visible_ends=cfg.abl_visible_ends)
    L = float(np.clip(fhi - flo,
                      max(cfg.dim_min_l, tp.L - cfg.prior_floor_k * tp.sig_L),
                      min(cap_l, cfg.dim_max_l)))
    # centre: from the fused ends when at least one end is confident (the de-bias
    # emerges there). When BOTH ends are prior-anchored their midpoint is just the
    # biased prior centre — but the RAW extent edges still carry the direction the
    # prior is off (the body pokes past one prior end more than the other; the
    # smear rides on top as roughly symmetric noise). So the centre — and ONLY the
    # centre — uses the outward-floored extent midpoint, bounded by the calibrated
    # centre-bias scale; the length stays on the fused (unbiased) ends.
    if cfg.abl_centre_prior:              # ABLATION: no centre de-bias
        midx = ucx
    elif min(s_lo, s_hi) > 0.20:
        mlo = min(plo_end, float(xlo))
        mhi = max(phi_end, float(xhi))
        midx = ucx + float(np.clip(0.5 * (mlo + mhi) - ucx, -tp.sig_ctr, tp.sig_ctr))
    else:
        midx = _centre_pull(0.5 * (flo + fhi), flo, fhi, L, X, R)

    # ---- HEIGHT: drop the canopy (cells above roof are not vehicle), then roof quantile ----
    keepz = (X >= flo) & (X <= fhi) & ((Hg <= roof + ROOF_MARGIN) if roof < 1e17 else True)
    Hk = Hg[keepz] if keepz.sum() >= 20 else Hg
    H = float(min(H0, np.quantile(Hk, WQ) + 0.10))

    # ---- WIDTH: quantile cover on points within the fused length, band-fused ----
    # The width axis is fully side-observed (no occluded far end across a gap), so a
    # silhouette walk balloons into lateral neighbours; the cover quantile stays. The
    # prior band replaces the hard floor + per-class cap: the prior W is calibrated
    # (~+0.3 m over — shadow margin), so the band is ASYMMETRIC (down 2 sigma, up 1).
    # A one-sided vehicle (far wall occluded, under-reading cover) falls back to
    # prior - 1 sigma instead of a hand-tuned class floor.
    inl = (X >= flo) & (X <= fhi)
    if inl.sum() >= 20:
        Xk, Yk, Hgk, Rk = X[inl], Y[inl], Hg[inl], R[inl]
    else:
        Xk, Yk, Hgk, Rk = X, Y, Hg, R
    # The ONE parked-gated width rule: reject lateral clutter past a ground-gap-then-rise
    # VALLEY. It can't be un-gated — a moving vehicle's occluded FAR WALL sits past the
    # cabin gap and looks identical; moving vehicles are exempt.
    if parked:
        yl, hyl = _profile(Yk, Xk, Hgk, max((fhi - flo) / 2.0, 2.0))
        glo_w, ghi_w = _first_gap(yl, hyl, GROUND, med[1] / 2.0)
        kw = (Yk >= glo_w - 0.05) & (Yk <= ghi_w + 0.05)
        if kw.sum() >= 20:
            Yk, Rk = Yk[kw], Rk[kw]
    ylo, yhi = weighted_quantile(Yk, Rk, 1.0 - WQ), weighted_quantile(Yk, Rk, WQ)
    W_obs = (yhi - ylo) + MARGIN_W
    frac_minor = min(float((Yk < ucy).mean()), float((Yk > ucy).mean())) if len(Yk) else 0.0
    far_witnessed = False
    if cfg.far_wall_test and parked and len(Yk):   # PARKED only: a mover's smear
        # can fake a consistent far edge (same reason the valley cut is parked-only)
        minority_pos = float((Yk > ucy).mean()) <= float((Yk < ucy).mean())
        farp = Yk[Yk > ucy] if minority_pos else Yk[Yk < ucy]
        if len(farp) >= 30:                                 # enough far-side mass to define an edge
            edge = np.quantile(np.abs(farp - ucy), 0.95)
            sup_far = int((np.abs(farp - ucy) > edge - 0.15).sum())
            far_witnessed = sup_far >= 25                   # a supported, consistent far edge
    if frac_minor < cfg.onesided_thr and not far_witnessed:  # far wall occluded: cover under-reads
        W = max(W_obs, tp.W - tp.sig_W)
    elif cfg.abl_width_sym:                                # ABLATION: symmetric band
        W = float(np.clip(W_obs, tp.W - 1.5 * tp.sig_W, tp.W + 1.5 * tp.sig_W))
    else:                                                  # both walls seen: band-fused cover
        W = float(np.clip(W_obs, tp.W - 2.0 * tp.sig_W, tp.W + tp.sig_W))
    W = float(np.clip(W, cfg.dim_min_w, cfg.dim_max_w))
    # LATERAL POSITION = the reliable UAV centre, with the points only as a GUARDRAIL
    # (never to derive the centre): on a contaminated extent the clamp is loose so the box
    # stays on the UAV; on a clean vehicle the reliably-observed near wall corrects a
    # slightly-off prior.
    midy = _centre_pull(ucy, ylo, yhi, W)

    # ---- apply: the (midx, midy) centre shift is a per-frame pose change ----
    boxes = res.boxes.copy()
    offs = res.xy_offsets.copy()
    for t in range(T):
        shift = canonical_xy_to_world(np.array([midx, midy]), float(boxes[t, 6]))
        boxes[t, 0:2] = boxes[t, 0:2] + shift
        offs[t] = offs[t] + shift
    boxes[:, 3] = L
    boxes[:, 4] = W
    boxes[:, 5] = H
    boxes[:, 2] = np.asarray(tf.ground_z, np.float64) + H / 2.0
    res.boxes = boxes
    res.xy_offsets = offs
    res.dims = np.array([L, W, H])

    # ---- de-harvest to the corrected box ----
    hx, hy, hz = L / 2.0 + 0.3, W / 2.0 + 0.3, H / 2.0 + 0.3
    ceil_h = (roof + ROOF_MARGIN) if roof < 1e17 else 1e18
    fg_p, fg_r = [], []
    for t in range(T):
        p = np.asarray(res.fg_points[t], np.float64)
        if p.ndim == 2 and len(p):
            zc = to_canonical(p[:, :3], boxes[t]); hgf = p[:, 2] - tf.ground_z[t]
            inb = ((np.abs(zc[:, 0]) <= hx) & (np.abs(zc[:, 1]) <= hy)
                   & (np.abs(zc[:, 2]) <= hz) & (hgf <= ceil_h))
            fg_p.append(p[inb]); r = np.asarray(res.fg_r[t])
            fg_r.append(r[inb] if r.shape[0] == p.shape[0] else r)
        else:
            fg_p.append(res.fg_points[t]); fg_r.append(res.fg_r[t])
    res.fg_points, res.fg_r = fg_p, fg_r
    if isinstance(res.diagnostics, list):     # end-status record for bias decomposition
        res.diagnostics.append({"boxfit": {
            "st_lo": st_lo, "st_hi": st_hi,
            "sup_lo": round(sup_lo, 1), "sup_hi": round(sup_hi, 1),
            "xlo": round(float(xlo), 3), "xhi": round(float(xhi), 3),
            "flo": round(float(flo), 3), "fhi": round(float(fhi), 3),
            "plo": round(float(plo_end), 3), "phi": round(float(phi_end), 3),
            "cap_l": round(float(cap_l), 2), "parked": bool(parked),
            "L": round(L, 3), "W": round(W, 3), "H": round(H, 3)}})
    return res
