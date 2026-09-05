"""
Make `import isongraph_vector` work when running the test suite from a
checkout, without requiring `pip install .` first.

The package directory is named `isongraph_vector-py` to line up with the
other language ports (`isongraph-vector-{ts,js,rs,cpp}`). A hyphen is not
legal in a Python module name, so the directory cannot simply be put on
sys.path - the import name has to be bound to it explicitly. `pyproject.toml`
does the same thing for installs via `[tool.setuptools.package-dir]`.

An installed copy wins: if `isongraph_vector` already imports, this leaves
it alone, so `pip install .` followed by `pytest` tests the installed package.
"""

import importlib.util
import sys
from pathlib import Path

_PKG_NAME = "isongraph_vector"
_PKG_DIR = Path(__file__).parent / "isongraph_vector-py"


def _bind_local_package() -> None:
    if _PKG_NAME in sys.modules:
        return
    if importlib.util.find_spec(_PKG_NAME) is not None:
        return  # an installed copy is importable; prefer it
    init = _PKG_DIR / "__init__.py"
    if not init.is_file():
        return
    spec = importlib.util.spec_from_file_location(
        _PKG_NAME, init, submodule_search_locations=[str(_PKG_DIR)]
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules[_PKG_NAME] = module
    spec.loader.exec_module(module)


_bind_local_package()
