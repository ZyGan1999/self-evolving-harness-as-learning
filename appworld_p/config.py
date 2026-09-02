"""Path configuration.

APPWORLD_ROOT layout (created by `appworld download data --root ...`):
    {APPWORLD_ROOT}/data/...                 benchmark data
    {APPWORLD_ROOT}/experiments/outputs/...  AppWorld episode outputs

Our own artifacts (task pool, session outputs, manifests) live under OUTPUTS_DIR.
"""

import os
from pathlib import Path

EXPERIMENTS_DIR = Path(__file__).resolve().parent.parent

APPWORLD_ROOT = Path(os.environ.get("APPWORLD_ROOT", EXPERIMENTS_DIR / "appworld_root"))
APPWORLD_DATA_DIR = APPWORLD_ROOT / "data"

OUTPUTS_DIR = EXPERIMENTS_DIR / "outputs"
TASK_POOL_PATH = OUTPUTS_DIR / "task_pool.json"
PERSONA_DIR = EXPERIMENTS_DIR / "configs" / "personas"


def set_appworld_root() -> None:
    """Point the appworld package at our root. Call before importing appworld."""
    os.environ["APPWORLD_ROOT"] = str(APPWORLD_ROOT)
