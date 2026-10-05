# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import pytest
import torch

from blueprint.datasets.sample2Ddataset import Sample2DDataset
from blueprint.datasets.utils import ExtendDataset


def test_extend_dataset_loops_over_the_base_dataset():
    """Indices past the base length wrap around to its samples."""
    ds = Sample2DDataset(length=10, seed=0, dist="gaussian")
    extended = ExtendDataset(ds, length=25)
    assert len(extended) == 25
    assert torch.equal(extended[12], ds[2])


def test_extend_dataset_rejects_a_shorter_target_length():
    """Extending to fewer samples would silently drop part of the dataset."""
    ds = Sample2DDataset(length=10, seed=0, dist="gaussian")
    with pytest.raises(ValueError, match="Cannot extend"):
        ExtendDataset(ds, length=5)
