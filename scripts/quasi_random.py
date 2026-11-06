# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""Generate quasi-random sequences for parameter sampling."""

from dataclasses import dataclass

import numpy as np
import sigfig
from scipy.stats import qmc


@dataclass
class Parameter:
    """Sampled parameter in [vmin, vmax], optionally in log scale."""

    vmin: float
    vmax: float
    log: bool = False


def generate_quasi_random_sequence(
    parameters: list[Parameter], N: int, seed: int = 0
) -> np.ndarray:
    """Generate Sobol sequence scaled to parameter ranges."""
    if np.log2(N) != np.floor(np.log2(N)):
        print("[WARNING] Sobol sequence performs best when N is a power of 2.")

    dim = len(parameters)
    sampler = qmc.Sobol(d=dim, scramble=True, seed=seed)
    sample = sampler.random(n=N)

    scaled_sample = np.empty_like(sample)
    for i, param in enumerate(parameters):
        vmin, vmax = param.vmin, param.vmax
        if param.log:
            vmin, vmax = np.log10(vmin), np.log10(vmax)
        scaled_sample[:, i] = vmin + sample[:, i] * (vmax - vmin)
        if param.log:
            scaled_sample[:, i] = np.power(10, scaled_sample[:, i])

    return scaled_sample


def main():  # noqa: D103
    n_digits = 3
    parameters = [
        Parameter(vmin=1e-4, vmax=1e-2, log=True),  # entropy reg
    ]

    samples = generate_quasi_random_sequence(parameters, N=32)

    samples = samples.tolist()
    for i, sample in enumerate(samples):
        sample = [sigfig.round(x, sigfigs=n_digits, type=str) for x in sample]
        print(f"f {i} " + " ".join(sample))


if __name__ == "__main__":
    main()
