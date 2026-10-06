"""The calibrated dimension prior and the ONE fusion primitive DAA adds.

The UAV prior is a measurement: per-track constant dims with a SELF-CALIBRATED
error model, measured by the pipeline against its own high-confidence LiDAR
geometry, without ground truth: corner-consensus end anchors give length and
centre, both-side-wall silhouette spans give width:

    length:  sigma ~ 0.36 m  (near-end corner residuals, centre-deconvolved)
    width :  bias +0.5,      sigma ~ 0.185 m  (shadow margin -> runs over)
    centre:  body-fixed along-track bias, per-track constant, sigma ~ 0.27 m

Treating the prior as a one-sided FLOOR ("at least the prior") is only safe for a prior
that under-estimates. With an unbiased prior that asymmetry is
wrong: sometimes the prior over-estimates and the box must be allowed to come in
UNDER it — but only when the LiDAR actually saw the boundary. The mechanism:

  * every box END is a Gaussian estimate. Evidence ends get a sigma from HOW
    they were established (corner consensus ~7 cm; a dense silhouette boundary
    ~15 cm; a sparse one ~0.5 m); the prior contributes its own end estimate
    (uav_centre +/- prior_L/2) whose sigma combines the dim noise AND the
    centre bias. Precision-weighted fusion per end replaces the amodal floor,
    the corner override, and the hard dim floor in one rule.
  * an OPEN end (the walk ran out of observation - occluded/edge of data) has
    no evidence: it falls back to the prior end, floored outward at the last
    observed body cell (you can never end INSIDE observed body).
  * length caps become the prior band (prior + k*sigma) instead of per-class
    constants; the dense cap-lift can still raise the band with decisive
    directly-observed evidence.

The per-track centre de-bias EMERGES from this: with two confident ends the
fused midpoint replaces the biased UAV centre; with one confident end the
midpoint is pulled halfway; with none it stays on the prior.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class TrackPrior:
    """Per-track calibrated prior (dims from the track's median UAV box)."""
    L: float
    W: float
    sig_L: float = 0.36      # self-calibrated (see module docstring), NOT tuned
    sig_W: float = 0.185
    sig_ctr: float = 0.27    # body-fixed along-track centre bias scale

    @property
    def sig_end(self) -> float:
        """Sigma of a prior END estimate = dim noise on a half-length + the
        centre bias (the two prior ends share the centre error, but for a
        single end they add in quadrature)."""
        return float(np.hypot(self.sig_L / 2.0, self.sig_ctr))


def gauss_fuse(m1: float, s1: float, m2: float, s2: float) -> float:
    """Precision-weighted mean of two Gaussian estimates."""
    w1, w2 = 1.0 / max(s1, 1e-6) ** 2, 1.0 / max(s2, 1e-6) ** 2
    return (m1 * w1 + m2 * w2) / (w1 + w2)


def end_sigma(status: str, support: float, flat: float = 0.0) -> float:
    """Sigma of an EVIDENCE end estimate.

    status  'corner' — cross-frame L-corner consensus (already std-gated < 15 cm)
            'closed' — the silhouette walk saw a REAL boundary: a gap with
                       nothing tall beyond AND a clean drop (near-zero occupancy
                       past the gap — a sparse trailing fade demotes the end to
                       'open' inside the walk, because no wall was seen there).
                       Trust scales with the boundary cells' occupancy.
            'open'   — no boundary evidence (grid edge, occlusion, or fade):
                       returns inf; the caller anchors on the prior, floored
                       outward at the DENSE-BODY edge.
    support  mean point count of the outermost member cells (closed only).
    """
    if status == "corner":
        return 0.07
    if status == "closed":
        if flat > 0.0:                      # ABLATION: ungraded evidence weight
            return float(flat)
        # dense boundary (>=3x NMIN pts/cell) -> 0.15 m; fades to 0.55 m as the
        # boundary cells thin out toward single-hit cells.
        return float(0.15 + 0.40 * np.exp(-max(support, 0.0) / 10.0))
    return float("inf")


def fuse_track_ends(lo_end, hi_end, prior_lo: float, prior_hi: float,
                    sig_prior: float, sig_L: float,
                    flat_sigma: float = 0.0, no_couple: bool = False,
                    no_amodal: bool = False, visible_ends: bool = False):
    """Fuse both box ends. Each of lo_end / hi_end = (evi, status, support, floor)
    where `floor` is the dense-body edge (smear-robust "body observed at least
    to here"). Returns (flo, fhi).

    Per end: evidence-graded Gaussian fusion against the prior end; an end with
    no evidence anchors on the prior, floored OUTWARD at the dense edge (never
    at the marginal member edge — that rides the aggregation/motion smear).

    ONE-CONFIDENT-END COUPLING (off by default, cfg.end_coupling): when exactly
    one end is confident, re-anchor the other at `confident_end -/+ prior_L` with
    sigma_L. Designed for corner-grade (~exact) ends; without corners it propagates
    the confident end's inward bias to the far end, and under the self-calibrated
    sigmas its anchor is looser than the standalone prior end (sig_L > sig_end)."""
    CONF = 0.20

    def est(end, prior_end, outward, sig_anchor=None, anchor=None):
        evi, st, sup, floor = end
        if visible_ends and st == "open" and np.isfinite(evi):
            # ABLATION: true no-amodal — the end IS the last observed body,
            # no prior participation (reported weak so the centre logic is unchanged)
            return float(evi), end_sigma("closed", 0.0, flat_sigma)
        if no_amodal and st == "open" and np.isfinite(evi):
            st, sup = "closed", 0.0       # ABLATION: observed edge as weak evidence
        pe = anchor if anchor is not None else prior_end
        sp = sig_anchor if sig_anchor is not None else sig_prior
        se = end_sigma(st, sup, flat_sigma)
        if not np.isfinite(se):
            if np.isfinite(floor):        # parked: dense-body edge bounds the anchor
                return outward * max(outward * pe, outward * floor), sp
            return pe, sp                 # moving: smear is dense — no evidence floor
        s_eff = (se ** -2 + sp ** -2) ** -0.5
        return gauss_fuse(evi, se, pe, sp), s_eff

    flo, slo = est(lo_end, prior_lo, -1.0)
    fhi, shi = est(hi_end, prior_hi, +1.0)
    prior_L = prior_hi - prior_lo
    if no_couple:                # ABLATION: ends stay independent
        return float(flo), float(fhi), float(slo), float(shi)
    if slo <= CONF < shi:        # confident low end -> re-anchor the high end
        fhi, shi = est(hi_end, prior_hi, +1.0, sig_anchor=sig_L, anchor=flo + prior_L)
    elif shi <= CONF < slo:      # confident high end -> re-anchor the low end
        flo, slo = est(lo_end, prior_lo, -1.0, sig_anchor=sig_L, anchor=fhi - prior_L)
    return float(flo), float(fhi), float(slo), float(shi)
