"""External validation entrypoint.

Loads the pre-trained model from `shared/ddqn_cql_reviewed.pt` and runs
the evaluation portion of the pipeline against the site's own data.

This script sets OHCA_RL_MODE=validate and re-executes 06_model_cql_fqe.py,
which (when in validate mode) skips the training loop and uses the shared
model checkpoint.

External sites should run:
    01_cohort.py → 02_wide.py → 03_sofa.py → 04_mdp.py → 05_reward.py
    → external_validation.py → make_tableone.py

Outputs land in output/final/<site_id>/.
"""

import os
import runpy
from pathlib import Path

os.environ["OHCA_RL_MODE"] = "validate"

_here = Path(__file__).resolve().parent
_target = _here / "06_model_cql_fqe.py"
assert _target.exists(), f"Cannot find {_target}"

runpy.run_path(str(_target), run_name="__main__")
