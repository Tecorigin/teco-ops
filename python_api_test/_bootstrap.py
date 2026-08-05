import os
import sys


def bootstrap_repo_pythonpath():
    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    api_dir = os.path.join(repo_root, "api")
    if api_dir not in sys.path:
        sys.path.insert(0, api_dir)


bootstrap_repo_pythonpath()
