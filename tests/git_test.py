# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import re
import subprocess
from pathlib import Path

import pytest

from blueprint.utils.git import get_head_commit_hash


def test_head_commit_hash_is_a_hash():
    """Inside a repository, the hash is the bare hexadecimal digest."""
    repo = Path(__file__).parents[1]
    assert re.fullmatch(r"[0-9a-f]{40}", get_head_commit_hash(str(repo)))


def test_head_commit_hash_raises_outside_a_repository(tmp_path):
    """Outside a repository, no error message may pass for a hash."""
    with pytest.raises(subprocess.CalledProcessError):
        get_head_commit_hash(str(tmp_path))
