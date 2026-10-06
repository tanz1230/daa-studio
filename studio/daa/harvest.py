"""Stage 1: the converging EM harvest.

Alternates a box-conditioned E-step (soft point responsibilities + graph smoothing)
with a converging M-step (a robust observed fit of the shared dims, hard-floored at
the UAV prior, plus a bounded re-seat of the harvest centre) until the box settles
(max |ddims| < eps). The output is a stable per-track harvest: one box per frame
(shared dims, per-frame offset from the UAV centre) plus the foreground points it
collected — Stage 2 (`boxfit`) then measures the final box on that harvest.

Design facts this loop rests on (established empirically):
  * The per-frame pose is already consistent and the LiDAR is sparse (6-300 pts/frame),
    so per-frame pose refinement is a no-op; the pose stays UAV centre + a bounded seat.
  * Grow-only dims have no fixed point (they inflate every iteration); fitting observed
    walls with a UAV-footprint floor gives one: observed walls fit the points, occluded
    walls fall back to the prior.
  * Stage 1 trusts the POINTS to collect them (seat the box on the harvest); Stage 2
    trusts the UAV to place the final box. The asymmetry per axis emerges from each
    axis's measured two-sidedness, it is not hard-coded.
"""
from __future__ import annotations

from typing import Optional

import numpy as np

from .config import RefinerConfig
from .estep import (coherent_mask, compute_responsibilities,
                                               graph_smooth)
from .geometry import (canonical_xy_to_world, make_box,
                                                  to_canonical, weighted_quantile)
from .types import TrackResult


def _sigmoid(x):
    return 1.0 / (1.0 + np.exp(-np.clip(x, -40.0, 40.0)))


def _frame_quality(mass, mean_outside, cfg) -> float:
    """Soft per-frame quality: responsibility mass x how tightly the mass hugs the box.

    A frame that harvests no foreground still gets a small nonzero score: an empty crop
    reports no `mean_outside`, so its residual factor is exp(0) = 1."""
    q_mass = _sigmoid((mass - cfg.min_resp_mass) / 5.0)
    q_res = np.exp(-mean_outside / 0.5)
    return float(q_mass * q_res)


def _clamp_offsets(offsets: np.ndarray, max_off: float) -> np.ndarray:
    nrm = np.hypot(offsets[:, 0], offsets[:, 1])
    return offsets * np.minimum(1.0, max_off / np.maximum(nrm, 1e-9))[:, None]


def _yaw_align(P, w, bound):
    """Wall-alignment yaw residual for an aggregated canonical BEV cloud (N,2).

    A parked vehicle's aggregate has crisp walls; a yaw error rotates them, and
    the axis-aligned robust footprint area grows with the rotation. The residual
    is the rotation minimizing that area (coarse grid then a fine pass), bounded
    to +/-`bound`. Weighted quantiles keep clutter from steering the search."""
    if len(P) > 6000:
        idx = np.random.RandomState(0).choice(len(P), 6000, replace=False)
        P, w = P[idx], w[idx]

    def area(d):
        c, s = np.cos(d), np.sin(d)
        x = c * P[:, 0] - s * P[:, 1]
        y = s * P[:, 0] + c * P[:, 1]
        return ((weighted_quantile(x, w, 0.98) - weighted_quantile(x, w, 0.02))
                * (weighted_quantile(y, w, 0.98) - weighted_quantile(y, w, 0.02)))

    coarse = np.linspace(-bound, bound, 21)
    d0 = coarse[int(np.argmin([area(d) for d in coarse]))]
    step = coarse[1] - coarse[0]
    fine = np.linspace(d0 - step, d0 + step, 11)
    return float(fine[int(np.argmin([area(d) for d in fine]))])


def _harvest_seat(coords, lo, hi, u, tau, twosided, dim=None, w=None):
    """Harvest-centre seat on ONE axis. Seat the box on the observed points (the midpoint
    of the extent [lo, hi]) so the E-step's distance-to-box unary overlaps them — but
    anchor on the reliable UAV centre `u` and bound the seat to +/-`tau` of it. `alpha`
    in [0,1] gates the seat by how two-sided the points are about `u` on this axis: a
    one-sided axis (occluded far flank/end) keeps the UAV centre (its point-midpoint is
    biased toward the visible side); a two-sided axis seats on the midpoint and
    self-corrects a mis-placed UAV box. When the harvest spans more than the box covers
    (span > `dim`, length axis only) the midpoint can land on a sparse tail, so seat on
    the dense mass (weighted median) instead. Clean 1-step fixed point:
    new_centre = u + clip(alpha * (seat - u), +/-tau), independent of the current centre."""
    if len(coords) == 0:
        return u
    if dim is not None and (hi - lo) > dim:
        seat = weighted_quantile(coords, w, 0.5) if w is not None else float(np.median(coords))
    else:
        seat = 0.5 * (lo + hi)
    fmin = min(float((coords < u).mean()), float((coords > u).mean()))
    alpha = float(np.clip(fmin / max(twosided, 1e-6), 0.0, 1.0))
    return u + float(np.clip(alpha * (seat - u), -tau, tau))


def _evaluate_aggregation(per_frame, dims, offsets, cfg):
    """Aggregation-quality score Q_agg over the pose-corrected canonical aggregate:
    compactness (extent vs dims) x cross-frame support (do different frames agree in
    canonical space) x offset plausibility. Feeds the per-frame confidence output."""
    from collections import defaultdict

    def ext(a, w, lo=0.05, hi=0.95):
        return weighted_quantile(a, w, hi) - weighted_quantile(a, w, lo)

    zs = [z for z, r in per_frame if len(z) > 0]
    rs = [r for z, r in per_frame if len(r) > 0]
    if not zs:
        return 0.0, {"compact": 0.0, "cross": 0.0, "offset": 0.0, "n": 0}
    Z = np.concatenate(zs)
    R = np.concatenate(rs)
    if Z.shape[0] < cfg.min_points_for_dims:
        return 0.0, {"compact": 0.0, "cross": 0.0, "offset": 0.0, "n": int(Z.shape[0])}
    L, W, H = [float(d) for d in dims]

    # 1. compactness: <1 when the responsibility-weighted extent exceeds the growth limit
    Ex, Ey, Ez = ext(Z[:, 0], R), ext(Z[:, 1], R), ext(Z[:, 2], R)
    cx = min(cfg.compact_growth_limit_l * L / max(Ex, 1e-6), 1.0)
    cy = min(cfg.compact_growth_limit_w * W / max(Ey, 1e-6), 1.0)
    cz = min(cfg.compact_growth_limit_h * H / max(Ez, 1e-6), 1.0)
    compact = cx * cy * cz

    # 2. cross-frame support: mass in voxels seen by >=2 frames / total mass
    v = cfg.cross_frame_voxel
    vox_frames = defaultdict(set)
    vox_mass = defaultdict(float)
    total = 0.0
    for fi, (z, r) in enumerate(per_frame):
        if len(z) == 0:
            continue
        keys = np.round(z / v).astype(np.int64)
        for kk, rr in zip(map(tuple, keys), r):
            vox_frames[kk].add(fi)
            vox_mass[kk] += rr
            total += rr
    multi = sum(m for kk, m in vox_mass.items() if len(vox_frames[kk]) >= 2)
    cross = multi / max(total, 1e-6)
    cross_score = min(cross / max(cfg.min_cross_frame_support, 1e-6), 1.0)

    # 3. offset plausibility: penalize large / jagged offsets
    off = np.asarray(offsets, np.float64)
    nrm = np.hypot(off[:, 0], off[:, 1])
    med_off = float(np.median(nrm)) if nrm.size else 0.0
    if off.shape[0] >= 3:
        jag = float(np.mean(np.hypot(*(off[2:] - 2 * off[1:-1] + off[:-2]).T)))
    else:
        jag = 0.0
    offset_score = float(np.exp(-med_off / max(cfg.pose_max_offset_m, 1e-6) - jag))

    Q_agg = compact * cross_score * offset_score
    return float(Q_agg), {"compact": float(compact), "cross": float(cross),
                          "offset": offset_score, "n": int(Z.shape[0])}


def em_harvest(tf, cfg: Optional[RefinerConfig] = None) -> TrackResult:
    """Run the converging EM harvest on one track. Returns the settled harvest boxes,
    shared dims, per-frame offsets/confidence, and the foreground points collected."""
    cfg = cfg or RefinerConfig()
    T = tf.n_frames
    uav = np.stack([np.asarray(b, np.float64) for b in tf.uav_boxes])
    ground_z = np.asarray(tf.ground_z, np.float64)
    med = np.median(uav[:, 3:6], axis=0)
    veh_class = int(getattr(tf, "veh_class", 0))   # UAV class: 0=car -> tight caps
    # geometry-based class correction: a vehicle whose measured aerial length
    # exceeds the car-class ceiling is not a car regardless of its label. The
    # aerial length is a reliable measurement; the class is not. Promoting it
    # relaxes the class-based caps (esp. HEIGHT, where the aerial gives no
    # measurement that could lift a mislabelled cap on its own).
    _lthr = cfg.dim_lcap_int if cfg.dim_lcap_int > 0 else cfg.dim_max_l_car
    if (cfg.dim_hcap or cfg.dim_lcap_flat or cfg.dim_lcap_int > 0) \
            and veh_class == 0 and float(med[0]) > _lthr:
        veh_class = 1
    dims = med.astype(np.float64).copy()
    offsets = np.zeros((T, 2))
    one_sided = None    # occlusion side is a structural constant -> latch once, never unlatch
                        # (recomputing per-iter flip-flops the lateral re-centre).
    yaw_corr = 0.0      # cumulative parked yaw correction (see the yaw M-step below)
    step_psi = 0.0
    diagnostics = []
    fg_mass = np.zeros(T)
    conf = np.zeros(T)
    # PARKED = completely observed (box-centre path short) -> the coherent clutter mask
    # applies; moving/one-sided vehicles keep the amodal recovery untouched.
    path_len = (float(np.sum(np.linalg.norm(np.diff(uav[:, 0:2], axis=0), axis=1)))
                if T > 1 else 0.0)
    parked = path_len < cfg.cf_parked_max
    # NET displacement (max excursion from the start), for the yaw trust bound:
    # a stationary track with a jittery aerial path accumulates a huge CUMULATIVE
    # path (jitter x thousands of frames) yet goes nowhere -- net displacement
    # separates the jitterer (D ~ 0, junk tangent yaw) from the true mover.
    net_disp = (float(np.max(np.linalg.norm(uav[:, 0:2] - uav[0, 0:2], axis=1)))
                if T > 1 else 0.0)
    dims_prev = dims.copy()

    # ---- data prep: height-ceiling removal (static, EM-independent -> applied ONCE) ----
    # Ground is stripped upstream by the DTM. Here we strip everything above a fixed,
    # ground-referenced ceiling: SKY for all classes (removes trees/wires/walls-above),
    # and for cars a LOWER ceiling (canopy/structure over a parked car cannot be car).
    cand = list(tf.cand_pts)
    ceil = cfg.sky_max_height if cfg.sky_max_height < 1e3 else 1e18
    if cfg.dim_overhead_reject and veh_class == 0:
        ceil = min(ceil, float(cfg.dim_overhead_car))
    if ceil < 1e17:
        kept = []
        for t in range(T):
            p = np.asarray(tf.cand_pts[t], np.float64)
            if len(p):
                p = p[p[:, 2] <= ground_z[t] + ceil]
            kept.append(p)
        cand = kept
    # Stage-1.5 MOTION GATE (moving tracks): a candidate that persists at the
    # same MAP spot after the target has displaced past its own length is
    # static world — it cannot belong to the mover. Static-, class- and
    # box-independent like the ceilings above, so it runs once, pre-loop.
    if cfg.motion_gate and not parked:
        from .registration import motion_gate
        cand, gate_frac = motion_gate(tf, cand, cfg)
        diagnostics.append({"motion_gate_dropped": round(gate_frac, 3)})

    per_frame = []
    ray_prev = None     # previous iteration's canonical geometry -> evidence fields
    for it in range(cfg.num_iters):
        step_psi = 0.0
        # ---- ray-evidence fields (from the PREVIOUS iteration's geometry) ----
        ray_fields = None
        if cfg.ray_resp and it >= cfg.ray_warmup and ray_prev is not None:
            from .estep import build_ray_fields
            ray_fields = build_ray_fields(ray_prev[0], ray_prev[1], dims, cfg)
        # ---- E-step: responsibilities under the current box, graph-smoothed ----
        per_frame = []
        fq = np.zeros(T)
        for t in range(T):
            box = make_box(uav[t], offsets[t], dims, ground_z[t])
            pts, z, r0, st = compute_responsibilities(cand[t], box, tf.lidar_origin[t], cfg,
                                                      ray_fields=ray_fields)
            r = graph_smooth(pts, r0, cfg) if pts.shape[0] else r0
            per_frame.append((pts, z, r))
            fg_mass[t] = float(r.sum())
            fq[t] = _frame_quality(float(r.sum()), st.get("mean_outside", 0.0), cfg)

        # ---- re-canonicalize under the current pose ----
        refined = []
        sens_can = []
        for t in range(T):
            pts, _, r = per_frame[t]
            box = make_box(uav[t], offsets[t], dims, ground_z[t])
            sens_can.append(to_canonical(
                np.asarray(tf.lidar_origin[t], np.float64)[None, :3], box)[0])
            if pts.shape[0] == 0:
                refined.append((np.zeros((0, 3)), np.zeros(0)))
                continue
            refined.append((to_canonical(pts, box), r))
        if cfg.ray_resp:
            ray_prev = (refined, sens_can)

        Q_agg, parts = _evaluate_aggregation(refined, dims, offsets, cfg)

        # ---- converging M-step: fit each wall to the points, hard-floored at the UAV
        # footprint; length gated by observation completeness; width/height by robust
        # quantiles. Then re-seat the box on the support (bounded to the UAV centre). ----
        fgZ = [z[r >= cfg.dim_fg_thr] for z, r in refined if len(z)]
        fgR = [r[r >= cfg.dim_fg_thr] for z, r in refined if len(z)]
        if fgZ and sum(len(p) for p in fgZ) >= cfg.min_points_for_dims:
            Z = np.concatenate(fgZ)
            Rr = np.concatenate(fgR)
            q = cfg.dim_extent_q
            # WIDTH + HEIGHT: robust quantile + UAV/class floor. The lateral axis has no
            # clean gaps to bound an amodal recovery, so width abstains to the floor for
            # the occluded flank. PARKED fits the clutter-rejected coherent points.
            if parked:
                keep = coherent_mask(Z, Rr, cfg)
                Zc, Rc = (Z[keep], Rr[keep]) if keep.sum() >= cfg.min_points_for_dims else (Z, Rr)
                ylo, yhi = weighted_quantile(Zc[:, 1], Rc, 1 - q), weighted_quantile(Zc[:, 1], Rc, q)
                if cfg.dim_onesided_fix and one_sided is not True:
                    frac_minor = min(float((Z[:, 1] < 0).mean()), float((Z[:, 1] > 0).mean()))
                    if frac_minor < cfg.onesided_thr:   # one wall occluded -> width unmeasurable
                        one_sided = True
                Ho = weighted_quantile(Zc[:, 2], Rc, cfg.dim_height_q)
            else:
                Zc, Rc = Z, Rr
                ylo, yhi = weighted_quantile(Z[:, 1], Rr, 1 - q), weighted_quantile(Z[:, 1], Rr, q)
                Ho = weighted_quantile(Z[:, 2], Rr, cfg.dim_height_q)
            Wo = (yhi - ylo) + cfg.dim_margin_w
            if one_sided and veh_class == 0:
                Wo = max(Wo, cfg.dim_min_w_car)   # one-sided car: class floor
            # LENGTH: the same robust quantile — Stage 1 just sizes the harvest box; the
            # occluded far-end recovery is Stage 2's job (box_fit's silhouette walk).
            xlo, xhi = weighted_quantile(Zc[:, 0], Rc, 1 - q), weighted_quantile(Zc[:, 0], Rc, q)
            Lo = (xhi - xlo) + cfg.dim_margin_l
            Ho = Ho + dims[2] / 2.0 + cfg.dim_margin_h
            # DAA: the class cap yields to the calibrated prior band — the prior is
            # now a trustworthy size, so the HARVEST box must be allowed to cover e.g. a
            # 6.3 m van whose class label says "car". Stage 1 keeps the prior FLOOR (its
            # job is collecting the vehicle's points; slightly generous is correct here —
            # Stage 2's per-end fusion measures the final box and may come in under).
            class_l_cap = cfg.dim_max_l_car if veh_class == 0 else cfg.dim_max_l
            if cfg.dim_lcap_int > 0:
                cap_l = cfg.dim_lcap_int if veh_class == 0 else cfg.dim_max_l   # flat integer ceiling
            elif cfg.dim_lcap_flat:
                cap_l = class_l_cap                     # flat class ceiling (class geometry-corrected)
            elif cfg.dim_lcap_med:
                cap_l = max(class_l_cap, cfg.dim_width_cap_factor * med[0])
            else:
                cap_l = max(class_l_cap, med[0] + cfg.prior_cap_k * cfg.prior_sig_l)
            if cfg.band_obs:
                # unified band form: floor = prior, cap = class ceiling v prior band.
                # No absolute-metre bounds; constants = class table + calibrated sigmas.
                cap_w = max(cfg.dim_max_w_car if veh_class == 0 else cfg.dim_max_w,
                            med[1] + cfg.prior_cap_k * cfg.prior_sig_w)
                L = float(np.clip(Lo, med[0], cap_l))
                W = float(np.clip(Wo, med[1], cap_w))
            else:
                cap_w = (cfg.dim_max_w_car if veh_class == 0 else cfg.dim_max_w) \
                    if cfg.dim_wcap_flat else \
                    min(cfg.dim_max_w_car if veh_class == 0 else cfg.dim_max_w,
                        cfg.dim_width_cap_factor * med[1])
                L = float(np.clip(max(Lo, med[0] - cfg.prior_floor_dl), cfg.dim_min_l, cap_l))    # floor UAV, cap banded
                W = float(np.clip(max(Wo, med[1] - cfg.prior_floor_dw), cfg.dim_min_w, cap_w))
            # H is grow-only from the UAV only for drone-detected TALL vehicles: the UAV H
            # is a class default that over-floors short cars, so cars trust the LiDAR roof.
            H = max(Ho, med[2]) if med[2] > cfg.dim_truck_h_floor else Ho
            # height, like L and W, is capped at the aerial CLASS roof (relaxed for
            # large classes) -- a car-labelled box cannot grow into overhanging
            # structure. Absorbs the exit roof read-out into the M-step.
            cap_h = (min(cfg.dim_max_h, float(med[2]) * (1.4 if veh_class != 0 else cfg.dim_hcap_k))
                     if cfg.dim_hcap else cfg.dim_max_h)
            H = float(np.clip(H, cfg.dim_min_h, cap_h))
            dims_obs = np.array([L, W, H])
            a = cfg.converge_damp
            dims = (1.0 - a) * dims + a * dims_obs
            # RE-SEAT the harvest box on the observed points, anchored on the UAV centre
            # (one unified rule for both axes; see _harvest_seat).
            uc = np.array([to_canonical(uav[t][None, :3],
                           make_box(uav[t], offsets[t], dims, ground_z[t]))[0, :2] for t in range(T)])
            ucx, ucy = float(np.median(uc[:, 0])), float(np.median(uc[:, 1]))
            tau, two = cfg.stage1_centre_tau, cfg.stage1_centre_twosided
            dx_can = _harvest_seat(Z[:, 0], xlo, xhi, ucx, tau, two, float(dims[0]), Rr)
            dy_can = _harvest_seat(Z[:, 1], ylo, yhi, ucy, tau, two)
            if abs(dx_can) > 1e-6 or abs(dy_can) > 1e-6:
                for t in range(T):
                    offsets[t] = offsets[t] + canonical_xy_to_world(
                        np.array([dx_can, dy_can]), float(uav[t, 6]))
                offsets = _clamp_offsets(offsets, cfg.pose_max_offset_m)
            # ---- yaw M-step: align the box to the aggregated walls ----
            # (canonical points of a mis-yawed box appear rotated by +e; the
            # area search returns -e). Damped and bounded to 2.5 sigma of the
            # prior-yaw error. The bound is the yaw prior's own reliability:
            # PARKED = the ortho prior (~5 deg). With yaw_disp, one formula
            # covers every track: the tangent prior's error shrinks with the
            # genuine displacement (sigma ~ k/path), so a jittery-path
            # stationary vehicle (junk tangent, crisp walls) gets a wide
            # bound, and a genuinely fast mover (tangent already ~0.01 deg
            # median) disables the search on its own.
            if cfg.yaw_all:
                # no sigma_eff: wall-align every track with the fixed parked
                # bound (no displacement tightening, no gate).
                sig_yaw = cfg.prior_sig_yaw
                yaw_on = True
            elif cfg.yaw_disp:
                sig_yaw = min(cfg.prior_sig_yaw,
                              cfg.yaw_tangent_k / max(net_disp, 1e-6))
                yaw_on = sig_yaw >= cfg.yaw_min_sig
            else:
                sig_yaw = cfg.prior_sig_yaw if parked else cfg.prior_sig_yaw_moving
                yaw_on = parked or cfg.yaw_moving
            if (cfg.yaw_refine and yaw_on
                    and Z.shape[0] >= cfg.yaw_min_pts
                    and (dims[0] - dims[1]) >= cfg.yaw_min_elong):
                d = _yaw_align(Z[:, :2], Rr, 2.5 * sig_yaw)
                step_psi = float(np.clip(-cfg.converge_damp * d,
                                         -cfg.yaw_step_max, cfg.yaw_step_max))
                if abs(yaw_corr + step_psi) <= 2.5 * sig_yaw:
                    yaw_corr += step_psi
                    uav[:, 6] = uav[:, 6] + step_psi
        else:
            dims_obs = dims.copy()

        conf = fq * Q_agg
        delta = float(max(np.max(np.abs(dims - dims_prev)),
                          abs(step_psi) * float(dims[0])))
        diagnostics.append({"iter": it, "dims": dims.round(3).tolist(), "Q_agg": round(Q_agg, 3),
                            "dims_obs": np.round(dims_obs, 3).tolist(),
                            "off_med": float(np.median(np.hypot(offsets[:, 0], offsets[:, 1]))),
                            "ddims": round(delta, 4), "yaw_corr_deg": round(float(np.degrees(yaw_corr)), 2),
                            **{k: round(v, 3) for k, v in parts.items() if k != "n"}})
        # convergence stop: once the box stops moving, num_iters is just a safety cap
        if it > 0 and delta < cfg.converge_eps:
            diagnostics[-1]["converged"] = True
            break
        dims_prev = dims.copy()

    # ---- compose boxes + collect the harvested foreground ----
    boxes = uav.copy()
    boxes[:, 0:2] = uav[:, 0:2] + offsets
    boxes[:, 2] = ground_z + dims[2] / 2.0
    boxes[:, 3:6] = dims
    fg_points, fg_r = [], []
    for t in range(T):
        pts, _, r = per_frame[t]
        keep = r >= cfg.dim_fg_thr
        fg_points.append(pts[keep] if pts.shape[0] else np.zeros((0, 3)))
        fg_r.append(r[keep] if r.shape[0] else np.zeros(0))

    return TrackResult(boxes=boxes, dims=dims, xy_offsets=offsets, fg_mass=fg_mass,
                       conf=conf, fg_points=fg_points, fg_r=fg_r, diagnostics=diagnostics)
