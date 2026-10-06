"""Refiner configuration: one flat dataclass of the DAA refiner's parameters.

DAA Studio runs `seeding.config.adopted_config()`, which starts from these defaults and adds the
relative prior floor and the width from the M-step.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass
class RefinerConfig:
    data_crop_scale: float = 3.0          # canonical crop at data_crop_scale * box half-extent (xy)
    data_crop_z_scale: float = 3.0
    data_crop_min_radius: float = 5.0     # m, floor on the xy crop radius

    sky_max_height: float = 1.0e9
    dim_overhead_reject: bool = False     # cars only: a LOWER ceiling — points above a car's
    dim_overhead_car: float = 2.1         # possible roof are canopy/structure over a parked car.

    # ---- E-step: loose candidate search around the current box ----
    search_x_scale: float = 2.5           # longitudinal: let the far end of a long vehicle in
    search_y_scale: float = 2.0           # lateral
    search_z_scale: float = 2.0
    max_candidate_points: int = 6000

    # ---- E-step: geometric unary ----
    sigma_outside: float = 0.6
    outside_clip: float = 4.0
    height_max_extra: float = 0.0
    height_min_extra: float = 0.0   # floor at the box bottom (ground); no below-ground slack

    # ---- E-step: ray-geometry evidence scaling (witnessed-empty corridors) ----
    band_obs: bool = False          # M-step bounds in band form (prior floor, class v prior+2sig cap)
    dim_hcap: bool = False          # M-step: cap H at the class roof (med_H * 1.4 large / k car)
    dim_hcap_k: float = 1.0         # car-class height cap = med_H * this
    dim_lcap_flat: bool = False     # L cap = flat class ceiling (+ geometry class correction)
    dim_lcap_med: bool = False      # L cap = max(class, 1.5*med_L) instead of prior+2sigma
    dim_lcap_int: float = 6.0
    dim_wcap_flat: bool = True
    ray_resp: bool = False          # scale per-axis overshoots by corridor evidence
    ray_warmup: int = 2             # iterations before the evidence term activates
    ray_sig_hard: float = 0.3       # effective sigma beyond a witnessed-empty corridor
    ray_sig_soft: float = 1.0       # effective sigma beyond a never-scanned corridor
    ray_w_hi: float = 0.5           # corridor evidence >= this -> firm wall (suppress)
    ray_w_lo: float = 0.05          # corridor evidence <= this -> unknown (protect)
    ray_cell: float = 0.1           # 1-D field cell size (m)

    use_view_prior: bool = False
    view_weight: float = 0.3
    face_sigma: float = 0.4
    view_visible_k: float = 4.0           # sharpness of the sigmoid(sign(view)*coord) gate

    # ---- E-step: responsibility sigmoid r = sigmoid(k * (unary - bg_thr)) ----
    bg_thr: float = -1.0                  # unary level at the 50/50 fg/bg boundary (unary <= 0)
    resp_sharpness: float = 2.0

    # ---- E-step: graph smoothing (closed-form Laplacian over a kNN graph) ----
    # min ||r-U||^2 + lambda r'Lr  ->  (I + lambda L) r = U : propagates responsibility along
    # the connected vehicle surface (grows the far end, kills salt-and-pepper clutter).
    use_graph_smoothing: bool = True
    knn_k: int = 8
    graph_sigma: float = 0.5
    graph_lambda: float = 1.0

    # ---- responsibility thresholds ----
    min_resp_mass: float = 5.0
    min_points_for_dims: int = 15
    dim_fg_thr: float = 0.5               # responsibility threshold defining the foreground

    # ---- M-step: pose bound ----
    pose_max_offset_m: float = 2.0        # hard cap on |refined centre - UAV centre|

    # ---- M-step: converging dims (robust observed fit, hard-floored at the UAV prior) ----
    dim_extent_q: float = 0.99            # extent quantile for L/W (0.01/0.99)
    dim_height_q: float = 0.99            # roof quantile for H
    dim_margin_l: float = 0.1
    dim_margin_w: float = 0.1
    prior_floor_dl: float = 0.0
    prior_floor_dw: float = 0.0
    # with the relative floor, the width comes from the M-step; the separate read-out is skipped
    width_from_mstep: bool = False
    dim_margin_h: float = 0.1
    dim_min_l: float = 3.0
    dim_min_w: float = 1.4
    dim_min_h: float = 1.2
    dim_max_l: float = 18.0               # semi+trailer ceiling (runaway backstop)
    dim_max_w: float = 3.0
    dim_max_h: float = 4.2                # tall trucks/buses
    dim_max_l_car: float = 5.6            # car-class caps (veh_class 0); large classes keep the
    dim_max_w_car: float = 2.2            # global maxima above.
    dim_width_cap_factor: float = 1.5     # cap W at this * UAV footprint width: far above the
                                          # top-down footprint is mirror-captured clutter.
    dim_truck_h_floor: float = 1.8        # floor H at the UAV H only for drone-detected TALL
                                          # vehicles (the ego LiDAR can sit below their roof).

    dim_onesided_fix: bool = False
    onesided_thr: float = 0.25            # one-sided if the minority wall has < this point fraction
    dim_min_w_car: float = 1.7            # class-typical car width floor when a wall is occluded

    corner_consensus: bool = False

    end_coupling: bool = False

    closed_at_edge: bool = True

    fade_thr: float = 0.30
    fade_sup_max: float = float("inf")

    gap_close_sup_min: float = 30

    low_bridge: bool = False
    low_bridge_min_cells: int = 4      # >= 0.4 m of contiguous low body to bridge to
    low_bridge_win: float = 1.5        # look this far (m) past the gap

    prior_span_gate: bool = False

    far_wall_test: bool = False

    abl_ends_prior_only: bool = False     # ignore end evidence: both ends anchor on the prior
    abl_flat_end_sigma: float = 0.0       # >0: closed-end sigma = this constant (ungraded)
    abl_centre_prior: bool = False        # centre stays on the UAV centre (no de-bias)
    abl_width_sym: bool = False           # symmetric width band +/-1.5 sigma (no shadow asymmetry)
    abl_no_amodal: bool = False           # open ends: observed edge as weak evidence, no prior anchor
    abl_visible_ends: bool = False        # open ends: LAST OBSERVED BODY exactly (true no-amodal)
    abl_naive_extent: bool = False        # raw membership extent: no walk, no gap logic, no grading

    # ---- M-step: harvest-centre seat ----
    # Seat the harvest box on the observed points (so the next E-step overlaps them), anchored
    # on — and bounded to +/-tau of — the reliable UAV centre. alpha gates the seat by the
    # axis's measured two-sidedness: a one-sided axis stays on the UAV centre.
    stage1_centre_tau: float = 1.0        # max drift (m) of the harvest box centre from the UAV
    stage1_centre_twosided: float = 0.2   # minority-side fraction at which an axis is two-sided

    # ---- aggregation quality (feeds the per-frame confidence, diagnostics) ----
    compact_growth_limit_l: float = 1.6
    compact_growth_limit_w: float = 1.8
    compact_growth_limit_h: float = 1.6
    cross_frame_voxel: float = 0.3
    min_cross_frame_support: float = 0.15

    # ---- coherent (clutter) mask: BEV density floor + dominant connected component ----
    cf_cell: float = 0.30                 # BEV voxel for the density grid + connectivity
    cf_dens_frac: float = 0.05            # keep cells with mass >= this * robust peak cell mass
    cf_close: int = 0                     # morphological-closing iterations (0 = off)
    cf_parked_max: float = 2.5            # PARKED gate: box-centre path length < this (m).
                                          # Parked vehicles are completely observed (the ego
                                          # passes by) -> coherent extent is trustworthy; moving
                                          # vehicles keep the amodal recovery.

    # ---- EM loop ----
    num_iters: int = 15                   # safety cap; the loop stops at ddims < converge_eps
    converge_damp: float = 0.5            # blend toward the observed fit each iteration
    converge_eps: float = 0.01            # convergence stop on max |ddims| (m)

    prior_sig_l: float = 0.36             # 1-sigma of the prior length (self-calibrated)
    prior_sig_w: float = 0.185            # 1-sigma of the prior width (self-calibrated)
    prior_sig_ctr: float = 0.27           # body-fixed along-track centre bias scale (self-calibrated)
    prior_cap_k: float = 2.0              # length cap = prior + k*sigma (liftable by evidence)
    prior_floor_k: float = 3.0            # length floor = prior - k*sigma (sanity backstop)

    yaw_refine: bool = True
    prior_sig_yaw: float = 0.09
    yaw_step_max: float = 0.05            # rad, per-iteration cap (damped like dims)
    yaw_min_pts: int = 150                # need this much aggregate fg to search
    yaw_min_elong: float = 0.8            # L - W (m) must exceed this (orientation defined)
    yaw_moving: bool = False
    prior_sig_yaw_moving: float = 0.044   # rad (~2.5 deg): tangent-yaw error scale
    yaw_disp: bool = True
    yaw_tangent_k: float = 0.2         # yaw window kappa
    yaw_min_sig: float = 0.015            # rad (~0.9 deg): below this, skip
    yaw_all: bool = False                 # drop sigma_eff, wall-align every track at the
                                          # fixed parked bound (off)

    motion_gate: bool = True
    motion_static_r: float = 0.3          # m: neighbour radius = "same map spot"
    motion_min_speed: float = 1.5         # m/s prior speed for a frame to be gated
    register_moving: bool = False
    reg_min_pts: int = 25                 # frames sparser than this keep the prior pose
    reg_pair_r: float = 0.6               # m: ICP pairing radius (sync-scale)
    reg_max_corr: float = 0.8             # m: hard bound on a per-frame correction
    reg_prior_kappa: float = 10.0         # MAP prior weight (point-equivalents)


# Drop-in alias for callers written against the legacy config name.
BoxPointMatchConfig = RefinerConfig
