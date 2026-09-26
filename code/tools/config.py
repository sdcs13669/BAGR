"""Config file system: --config JSON loading, CLI-override merge, effective
config persistence.

The effective config = --config JSON as the baseline, overridden by
explicitly-set CLI values; every run writes it to <output>/config.json and
training scripts embed the same dict in every epoch checkpoint; resume/eval
always rebuild from the embedded checkpoint config. merge_args_with_config is
the single implementation of the "CLI value counts only when it differs from
the argparse default" rule."""

import json
import subprocess
from pathlib import Path
from typing import Dict, Optional


def load_json(path) -> dict:
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)


def save_json(obj, path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)


def git_commit() -> Optional[str]:
    """Current git commit (None on failure; never blocks training)."""
    try:
        return subprocess.check_output(
            ['git', 'rev-parse', 'HEAD'],
            cwd=str(Path(__file__).resolve().parents[2]),
            stderr=subprocess.DEVNULL, text=True).strip()
    except Exception:
        return None


def merge_args_with_config(args, cfg: Dict,
                           key_map: Optional[Dict[str, str]] = None) -> Dict:
    """Merge CLI args with the config JSON into the effective config dict.


    For each cfg key k (mapped to an args dest via key_map): a value equal
    to the argparse default is replaced by the cfg value; an explicitly set
    CLI value wins. Keys present only in args are kept as-is; keys present
    only in cfg are also kept (used for model building, e.g. per-layer lists).

    args must be an argparse.Namespace carrying args._defaults = {dest:
    default} (attached by tools/cli.py) so parser defaults are available
    without passing the parser itself.
    """
    key_map = key_map or {}
    defaults = getattr(args, '_defaults', {})
    effective: Dict[str, object] = {}
    effective.update(cfg)
    for dest in vars(args):
        if dest.startswith('_'):
            continue
        val = getattr(args, dest)
        if dest in defaults and val == defaults[dest]:
            continue
        cfg_key = key_map.get(dest, dest)
        effective[cfg_key] = val
    return effective


def build_experiment_config(args, extra: Optional[Dict] = None) -> Dict:
    """Experiment meta config: full CLI values + git commit + extra fields."""
    cfg = {k: v for k, v in vars(args).items() if not k.startswith('_')}
    cfg['git_commit'] = git_commit()
    if extra:
        cfg.update(extra)
    return cfg


def require_args(args, names) -> None:
    """Validate required args after the --config merge.

    Fails if any required arg is still empty (None or empty string).
    """
    missing = [n for n in names if not getattr(args, n, None)]
    if missing:
        flags = ', '.join('--' + n.replace('_', '-') for n in missing)
        raise SystemExit(f"missing required args: {flags} (provide via CLI or --config)")
