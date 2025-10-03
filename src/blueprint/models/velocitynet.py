# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch
import torch.nn.functional as F
from torch import Tensor, nn


def rk4_step(func: callable, x: Tensor, t: Tensor, dt: float) -> Tensor:
    """Perform one RK4 step."""
    k1 = func(x=x, t=t)
    k2 = func(x=x + dt / 2 * k1, t=t + dt / 2)
    k3 = func(x=x + dt / 2 * k2, t=t + dt / 2)
    k4 = func(x=x + dt * k3, t=t + dt)
    return x + dt / 6 * (k1 + 2 * k2 + 2 * k3 + k4)


def rk4(func: callable, x0: Tensor, T: Tensor) -> Tensor:
    """Integrate func starting from x0 along T."""
    x = x0
    for i in range(len(T) - 1):
        x = rk4_step(func=func, x=x, t=T[i], dt=T[i + 1] - T[i])
    return x


class VelocityNet(nn.Module):
    """Toy model that learns a input_dim-D velocity field."""

    def __init__(self, input_dim: int, inner_dim: int, n_steps: int):
        super().__init__()
        T = torch.linspace(0, 1, n_steps + 1)
        self.register_buffer("T", T)
        self.fc_in = nn.Linear(input_dim + 1, inner_dim)
        self.fc2 = nn.Linear(inner_dim, inner_dim)
        self.fc3 = nn.Linear(inner_dim, inner_dim)
        self.fc_out = nn.Linear(inner_dim, input_dim)
        self.act = F.relu

    def forward(self, x: Tensor, t: Tensor = None) -> Tensor:
        """Perform either a single step when training, or integrate during inference."""
        if self.training:
            return self.forward_single_pass(x, t=t)
        return rk4(func=self.forward_single_pass, x0=x, T=self.T)

    def forward_single_pass(self, x: Tensor, t: Tensor) -> Tensor:
        """Return NN output."""
        t = t.expand(x.size(0), 1)
        x = torch.cat([x, t], dim=1)
        x = self.act(self.fc_in(x))
        x = self.act(self.fc2(x))
        x = self.act(self.fc3(x))
        return self.fc_out(x)
