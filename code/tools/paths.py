"""Path bootstrap: call setup_path() before imports in every script.

Puts <repo>/code at sys.path[0] and removes the script own directory (so
package names cannot shadow each other)."""

import sys
from pathlib import Path

CODE_DIR = Path(__file__).resolve().parents[1]   # <repo>/code
REPO_ROOT = CODE_DIR.parent


def resolve(p) -> Path:
    """Resolve an input path against two bases.

    Relative paths resolve against code/ first; if not found there but
    present under the repo root, the repo root is used; otherwise code/ is
    assumed (default output location). Absolute paths pass through."""
    path = Path(p)
    if path.is_absolute():
        return path
    for base in (CODE_DIR, REPO_ROOT):
        cand = base / path
        if cand.exists():
            return cand
    return CODE_DIR / path


def setup_path() -> Path:
    """Ensure code/ is at the front of sys.path; return the repo root."""
    s = str(CODE_DIR)
    if s in sys.path:
        sys.path.remove(s)
    sys.path.insert(0, s)
    script_dir = str(Path(__file__).resolve().parent)
    here = str(Path().resolve())
    for d in (script_dir, here):
        if d != s and d in sys.path and Path(d).parent == CODE_DIR:
            sys.path.remove(d)
    return REPO_ROOT
