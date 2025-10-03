# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from dataclasses import dataclass

import numpy as np
from scipy.stats import qmc


def main():
    parameters = [
        Parameter(20, 500, log=True),
        Parameter(1e2, 1e4, log=True),
    ]

    samples = generate_quasi_random_sequence(parameters, N=32)

    samples = samples.tolist()
    for i, sample in enumerate(samples):
        sample = [str(int(e)) for e in sample]
        print(f"f {i} " + " ".join(sample))


@dataclass
class Parameter:
    vmin: float
    vmax: float
    log: bool = False


def generate_quasi_random_sequence(
    parameters: list[Parameter], N: int, seed: int = 0
) -> np.array:
    if np.log2(N) != np.floor(np.log2(N)):
        print("[WARNING] Sobol sequence performs best when N is a power of 2.")

    dim = len(parameters)
    sampler = qmc.Sobol(d=dim, scramble=True, seed=seed)
    sample = sampler.random(n=N)

    scaled_sample = np.empty_like(sample)
    for i, param in enumerate(parameters):
        if param.log:
            param.vmin = np.log10(param.vmin)
            param.vmax = np.log10(param.vmax)
        scaled_sample[:, i] = param.vmin + sample[:, i] * (param.vmax - param.vmin)
        if param.log:
            scaled_sample[:, i] = np.power(10, scaled_sample[:, i])

    return scaled_sample


if __name__ == "__main__":
    main()
