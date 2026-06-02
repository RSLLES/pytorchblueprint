# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import shutil
import textwrap

from tqdm import tqdm


class WrappedTextProgressBar:
    """A tqdm progress bar with a wrapped-text footer.

    The progress bar sits on top and text below is wrapped to terminal width.
    Each visible row is its own single-line tqdm bar, so tqdm's
    one-line-per-bar cursor bookkeeping stays correct however the text wraps.
    """

    def __init__(self, total: int, disable: bool = False) -> None:
        """Initialize a progress bar with optional disable flag."""
        self._disable = disable
        self._pbar = tqdm(total=total, position=0, leave=False)
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

    def __enter__(self) -> "WrappedTextProgressBar":
        return self

    def __exit__(self, *exc) -> None:
        for bar in self._desc_bars:
            bar.close()
        self._pbar.close()
