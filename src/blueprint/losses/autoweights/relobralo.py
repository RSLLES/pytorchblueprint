# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch
import torch.distributed as dist
import torch.nn as nn
from torch import Tensor


class ReLoBRaLo(nn.Module):
    """Relative Loss Balancing with Random Lookback (Bischof & Kraus, 2022).

    No extra backward passes — uses only loss values.
    """

    L0: Tensor
    L_prev: Tensor
    w: Tensor
    initialized: Tensor

    def __init__(
        self,
        n_losses: int,
        alpha: float = 0.999,
        beta: float = 0.999,
        temperature: float = 1.0,
        eps: float = 1e-8,
    ) -> None:
        super().__init__()
        self.n_losses = n_losses
        self.alpha = alpha
        self.beta = beta
        self.temperature = temperature
        self.eps = eps
        self.register_buffer("L0", torch.zeros(n_losses))
        self.register_buffer("L_prev", torch.zeros(n_losses))
        self.register_buffer("w", torch.ones(n_losses))
        self.register_buffer("initialized", torch.tensor(False))

    @torch.compiler.disable
    def _sync_avg(self, x: Tensor) -> Tensor:
        if dist.is_available() and dist.is_initialized():
            x = x.clone()
            dist.all_reduce(x, op=dist.ReduceOp.AVG)
        return x

    @torch.compiler.disable
    def _sync_bcast(self, x: Tensor) -> Tensor:
        """Broadcast x from rank 0 so all workers share the same value."""
        if dist.is_available() and dist.is_initialized():
            x = x.clone()
            dist.broadcast(x, src=0)
        return x

    def _balanced_weights(self, vals: Tensor, anchor: Tensor) -> Tensor:
        """Compute λ_bal = m·softmax(L(t)/(T·L(t'))) per Eq. 11."""
        ratios = vals / (anchor + self.eps)
        return self.n_losses * torch.softmax(ratios / self.temperature, dim=0)

    @torch.compiler.disable
    def _update(self, vals: Tensor) -> Tensor:
        # Sync losses across workers first so all subsequent computations are identical.
        synced = self._sync_avg(vals)
        if not self.initialized:
            self.L0.copy_(synced)
            self.L_prev.copy_(synced)
            self.initialized.fill_(True)

        w_bal_0 = self._balanced_weights(synced, self.L0)
        w_bal_prev = self._balanced_weights(synced, self.L_prev)
        # Broadcast rho from rank 0: all workers must share the same lookback coin flip.
        rho = self._sync_bcast(
            torch.bernoulli(torch.full((), self.beta, device=vals.device))
        )
        w_new = (
            self.alpha * (rho * self.w + (1.0 - rho) * w_bal_0)
            + (1.0 - self.alpha) * w_bal_prev
        )
        self.w.copy_(w_new)
        self.L_prev.copy_(synced)
        return self.w

    def step(self, losses: list[Tensor]) -> Tensor:
        """Return weighted total loss; update weights in place."""
        vals = torch.stack([loss.detach() for loss in losses])
        w = self._update(vals).detach()
        return (w * torch.stack(losses)).sum()
