# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from pathlib import Path

from blueprint.utils import config, tensorboard
from blueprint.utils import plot as splt


def main(study_dir: str):
    """Plot a graph to analyze the results of a study."""
    study_dir = Path(study_dir)
    run_dirs = [
        d
        for d in study_dir.iterdir()
        if d.is_dir() and d.name.startswith("v") and d.name[1:].isdigit()
    ]

    for run_dir in run_dirs:
        cfg = config.load_config(run_dir / "config.yaml")
        df = tensorboard.load_events_file(tensorboard.find_events_file(run_dir))

        x = cfg["trainer"]["entropy_reg"]
        y = df["E_3D"].iloc[-1]
        splt.plt.scatter(x, y)

    splt.plt.title("Study entropy regularization")
    splt.plt.xscale("log")
    splt.plt.xlabel("entropy_reg")
    splt.plt.ylabel("E_3D")
    splt.savefig()


if __name__ == "__main__":
    main(study_dir="outputs/studies_selfsuper_nfluos")
