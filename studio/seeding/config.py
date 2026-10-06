"""The DAA configuration DAA Studio runs: the configuration of the DAA paper.

Refiner defaults plus the relative prior floor delta = 0.3 m on length and width in the M-step,
with the width taken from the M-step (no separate read-out). tests/test_daa_parity.py checks
that it reproduces a reference label file exactly.
"""
from ..daa.config import RefinerConfig

DELTA = 0.3


def adopted_config() -> RefinerConfig:
    cfg = RefinerConfig()
    cfg.prior_floor_dl = DELTA
    cfg.prior_floor_dw = DELTA
    cfg.width_from_mstep = True
    return cfg
