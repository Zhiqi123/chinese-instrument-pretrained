"""Safe UMAP import, bypassing TensorFlow SIGSEGV and Numba cache issues."""

from __future__ import annotations

import os
import sys
from pathlib import Path

# Project-level cache dir (under artifacts/, gitignored)
_EXP_ROOT = Path(__file__).resolve().parents[2]
_NUMBA_CACHE_DIR = _EXP_ROOT / "artifacts" / ".numba_cache"


def _setup_numba_cache() -> None:
    """Redirect NUMBA_CACHE_DIR to a writable project directory."""
    _NUMBA_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    os.environ["NUMBA_CACHE_DIR"] = str(_NUMBA_CACHE_DIR)


def get_umap_class():
    """Return umap.UMAP class, safely bypassing TensorFlow and Numba issues.

    Usage::

        from src.utils.umap_import import get_umap_class
        UMAP = get_umap_class()
        reducer = UMAP(n_components=2, ...)
    """
    # 1. Set up writable Numba cache
    _setup_numba_cache()

    # 2. Insert placeholder to skip parametric_umap (avoids TF SIGSEGV)
    _tf_placeholder = False
    if "tensorflow" not in sys.modules:
        sys.modules["tensorflow"] = None  # type: ignore[assignment]
        _tf_placeholder = True

    import umap  # noqa: E402

    # 3. Restore so real tensorflow can be imported later
    if _tf_placeholder:
        del sys.modules["tensorflow"]

    return umap.UMAP
