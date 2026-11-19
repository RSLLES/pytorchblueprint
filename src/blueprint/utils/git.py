# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""Utility functions for git."""

import subprocess


def get_head_commit_hash(repo_path: str = ".") -> str:
    """Return the hash of the HEAD commit for the given repository."""
    return subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=repo_path, text=True
    ).strip()
