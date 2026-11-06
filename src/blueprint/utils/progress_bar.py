# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import shutil
import textwrap
import time

from tqdm import tqdm

from blueprint.utils import format


class TrainingClock:
    """Measure the run's step rate, ignoring the first step's one-off compile cost."""

    def __init__(self, steps_left: int) -> None:
        self.steps_left = steps_left
        self._t_first_step: float | None = None
        self._steps_timed = 0

    def count_step(self, n: int = 1) -> None:
        """Record n finished steps; the run's first one only starts the clock."""
        self.steps_left -= n
        if self._t_first_step is None:
            self._t_first_step = time.monotonic()
            n -= 1
        self._steps_timed += n

    def eta(self) -> str | None:
        """Time left for the whole run, or None until a step has been timed."""
        if self._t_first_step is None or self._steps_timed == 0:
            return None
        elapsed = time.monotonic() - self._t_first_step
        return format.format_duration(elapsed / self._steps_timed * self.steps_left)


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
    ) -> None:
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

    def set_text(self, text: str) -> None:
        """Set and wrap text to display below the progress bar."""
        if self._disable:
            return
        width = max(shutil.get_terminal_size().columns - 1, 20)
        lines = textwrap.wrap(
            text, width=width, break_long_words=False, break_on_hyphens=False
        ) or [""]
        for i in range(max(len(lines), len(self._desc_bars))):
            self._desc_bar(i).set_description_str(lines[i] if i < len(lines) else "")

    def update(self, n: int = 1) -> None:
        """Update progress by n steps."""
        if self._disable:
            return
        self._pbar.update(n)
        if self._clock is None:
            return
        self._clock.count_step(n)
        eta = self._clock.eta()
        if eta is not None:
            # refreshing here would render on every step, bypassing tqdm's mininterval
            self._pbar.set_description_str(f"ETA {eta}", refresh=False)

    def __enter__(self) -> "WrappedTextProgressBar":
        return self

    def __exit__(self, *exc) -> None:
        for bar in self._desc_bars:
            bar.close()
        self._pbar.close()
