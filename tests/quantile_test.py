# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch

from blueprint.metrics import KLLOnlineQuantile


def test_kll_quantile():
    """The streamed KLL sketch lands near the exact per-channel quantiles."""
    torch.manual_seed(0)
    x = torch.randn(20_000, 2)
    q = (0.1, 0.5, 0.9)
    metric = KLLOnlineQuantile(n_channels=2, q=q)
    for batch in x.split(1000):
        metric.update(batch)
    expected = torch.quantile(x, torch.tensor(q), dim=0)
    torch.testing.assert_close(metric.compute(), expected, atol=0.1, rtol=0.0)
