"""Register protocol-adapted baselines in the fixed-strategy registry.

sagpool needs a scorer checkpoint (with device) and is constructed specially
by eval_baselines; random-walk takes no arguments and is usable right after
registration via create_fixed_reader('random-walk').
"""

from pathlib import Path
import sys

_REPO_ROOT = Path(__file__).resolve().parents[1]
_CODE_DIR = _REPO_ROOT / 'code'
for _p in (_CODE_DIR, _REPO_ROOT):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))


def register_baseline_strategies() -> None:
    """Register random-walk in the fixed-strategy registry (idempotent)."""
    from active_graph_reader.reader.fixed import register_strategy
    from baseline.randwalk import RandomWalkStrategy
    register_strategy('random-walk', RandomWalkStrategy)
