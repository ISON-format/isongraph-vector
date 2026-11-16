"""Shared helpers for the benchmark scripts (package import, small utilities)."""

from __future__ import annotations

import importlib.util
import platform
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
_PKG_NAME = "isongraph_vector"
_PKG_DIR = REPO_ROOT / "isongraph_vector-py"


def import_package():
    """Import ``isongraph_vector`` the way the test suite does.

    The package directory is ``isongraph_vector-py`` (not a legal module
    name), so when no installed copy exists it is bound explicitly, exactly
    as ``conftest.py`` does. An installed copy wins. The package is only
    imported, never modified.
    """
    if _PKG_NAME in sys.modules:
        return sys.modules[_PKG_NAME]
    if importlib.util.find_spec(_PKG_NAME) is not None:
        import isongraph_vector  # noqa: WPS433
        return isongraph_vector
    spec = importlib.util.spec_from_file_location(
        _PKG_NAME, _PKG_DIR / "__init__.py",
        submodule_search_locations=[str(_PKG_DIR)],
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules[_PKG_NAME] = module
    spec.loader.exec_module(module)
    return module


def environment_info() -> dict:
    info = {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "processor": platform.processor(),
    }
    try:
        import numpy
        info["numpy"] = numpy.__version__
    except ImportError:
        info["numpy"] = None
    try:
        import psutil
        info["psutil"] = psutil.__version__
        info["cpu_count"] = psutil.cpu_count(logical=True)
        info["ram_gb"] = round(psutil.virtual_memory().total / 2**30, 1)
    except ImportError:
        info["psutil"] = None
    try:
        import sqlite_vec
        info["sqlite_vec"] = getattr(sqlite_vec, "__version__", "installed")
    except ImportError:
        info["sqlite_vec"] = None
    try:
        import ison_graph
        info["ison_graph"] = getattr(ison_graph, "__version__", "installed")
    except ImportError:
        info["ison_graph"] = None
    return info
