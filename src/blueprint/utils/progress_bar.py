# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import shutil
import textwrap
import time
from dataclasses import dataclass

from tqdm import tqdm

from blueprint.utils import format


@dataclass
class _Timing:
    """Time spent on the timed steps of one kind, and how many steps are left."""

    left: int | None
    seconds: float = 0.0
    n_timed: int = 0
    started: bool = False


class TrainingClock:
    """Time the run; estimate the time left from steps and validation batches.

    Each wall-clock interval is charged to the kind of step that ends it. The run's
    first interval of each kind is not: it carries one-off setup and compile costs.
    """

    def __init__(self, steps_left: int | None, val_batches_left: int):
        """Start the clock; steps_left is None for a run of unbounded length."""
        self._t_start = self._t_last = time.monotonic()
        self._steps = _Timing(left=steps_left)
        self._val_batches = _Timing(left=val_batches_left)

    def count_step(self, n: int = 1):
        """Record n finished training steps."""
        self._count(self._steps, n)

    def count_val_batch(self):
        """Record a finished validation batch."""
        self._count(self._val_batches, 1)

    def _count(self, timing: _Timing, n: int):
        now = time.monotonic()
        if timing.left is not None:
            timing.left -= n
        if timing.started:
            timing.seconds += now - self._t_last
            timing.n_timed += n
        timing.started = True
        self._t_last = now

    def elapsed(self) -> str:
        """Time since the clock started."""
        return format.format_duration(time.monotonic() - self._t_start)

    def eta(self) -> str | None:
        """Time left for the whole run, or None if unbounded or no step is timed yet."""
        if self._steps.left is None or self._steps.n_timed == 0:
            return None
        seconds = sum(
            timing.seconds / timing.n_timed * timing.left
            for timing in (self._steps, self._val_batches)
            if timing.n_timed > 0 and timing.left is not None
        )
        return format.format_duration(seconds)


class WrappedTextProgressBar:
    """A tqdm progress bar with a wrapped-text footer.

    The progress bar sits on top and text below is wrapped to terminal width.
    Each visible row is its own single-line tqdm bar, so tqdm's
    one-line-per-bar cursor bookkeeping stays correct however the text wraps.
    """

    def __init__(
        self,
        total: int,
        clock: TrainingClock | None = None,
        disable: bool = False,
    ):
        """Initialize a progress bar, with an optional whole-run clock for the ETA."""
        self._disable = disable
        self._clock = clock
        self._pbar = tqdm(total=total, position=0, leave=False, disable=disable)
        self._desc_bars: list[tqdm] = []

    def _desc_bar(self, i: int) -> tqdm:
        """Get or create a tqdm bar for wrapping text at line i."""
        while i >= len(self._desc_bars):
            self._desc_bars.append(
                tqdm(
                    total=0,
                    bar_format="{desc}",
                    position=len(self._desc_bars) + 1,
                    leave=False,
                )
            )
        return self._desc_bars[i]

    def set_text(self, text: str):
        """Set and wrap text to display below the progress bar."""
        if self._disable:
            return
        width = max(shutil.get_terminal_size().columns - 1, 20)
        lines = textwrap.wrap(
            text, width=width, break_long_words=False, break_on_hyphens=False
        ) or [""]
        for i in range(max(len(lines), len(self._desc_bars))):
            self._desc_bar(i).set_description_str(lines[i] if i < len(lines) else "")

    def update(self, n: int = 1) -> bool:
        """Update progress by n steps; return whether the bar was redrawn."""
        if self._disable:
            return False
        redrawn = bool(self._pbar.update(n))
        if self._clock is None:
            return redrawn
        self._clock.count_step(n)
        eta = self._clock.eta()
        if eta is not None:
            # refreshing here would render on every step, bypassing tqdm's mininterval
            self._pbar.set_description_str(f"ETA {eta}", refresh=False)
        return redrawn

    def __enter__(self) -> "WrappedTextProgressBar":
        return self

    def __exit__(self, *exc: object):
        for bar in self._desc_bars:
            bar.close()
        self._pbar.close()
