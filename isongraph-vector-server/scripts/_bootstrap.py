"""Make `isongraph_vector` and `isongraph_vector_server` importable from a checkout
(same trick as the repo's root conftest.py: the library directory is `isongraph_vector-py`)."""

import importlib.util
import sys
from pathlib import Path

SERVER_ROOT = Path(__file__).resolve().parent.parent
REPO_ROOT = SERVER_ROOT.parent

if str(SERVER_ROOT) not in sys.path:
    sys.path.insert(0, str(SERVER_ROOT))

if "isongraph_vector" not in sys.modules and importlib.util.find_spec("isongraph_vector") is None:
    pkg_dir = REPO_ROOT / "isongraph_vector-py"
    spec = importlib.util.spec_from_file_location(
        "isongraph_vector", pkg_dir / "__init__.py", submodule_search_locations=[str(pkg_dir)])
    module = importlib.util.module_from_spec(spec)
    sys.modules["isongraph_vector"] = module
    spec.loader.exec_module(module)
