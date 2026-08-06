import importlib.util
import os
import sys


def bootstrap_repo_pythonpath():
    # Prefer an installed tecoops package. Fall back to the repo api/ tree only
    # when tecoops is not importable from the current environment.
    if importlib.util.find_spec("tecoops") is not None:
        return

    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    api_dir = os.path.join(repo_root, "api")
    if api_dir not in sys.path:
        sys.path.append(api_dir)


bootstrap_repo_pythonpath()
