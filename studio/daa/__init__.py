"""DAA refiner (numpy + scipy only): the method of the DAA paper (arXiv:2609.06819).

DAA Studio additions: package-relative imports, the relative prior floor
(config.prior_floor_dl/dw) and the width_from_mstep switch (harvest.py, boxfit_ov.py).
Alternative formulations behind OV_* environment switches are cleared at import, so DAA Studio
always runs exactly the configuration it asks for.
"""
import os as _os

for _k in [k for k in _os.environ if k.startswith("OV_")]:
    _os.environ.pop(_k)
