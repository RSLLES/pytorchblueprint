# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from blueprint.utils import progress_bar
from blueprint.utils.progress_bar import TrainingClock


def test_training_clock_excludes_the_first_step(monkeypatch):
    """The first step pays for compilation, so the ETA times only the later ones."""
    now = [0.0]
    monkeypatch.setattr(progress_bar.time, "monotonic", lambda: now[0])
    clock = TrainingClock(steps_left=11)

    now[0] = 100.0  # first step: compilation
    clock.count_step()
    assert clock.eta() is None

    now[0] = 102.0  # two more steps at 1s each
    clock.count_step(2)
    assert clock.eta() == "8s"
