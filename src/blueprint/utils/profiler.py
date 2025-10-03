# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch


class MemoryProfiler:
    """Profile GPU memory usage and dump a snapshot."""

    def __init__(self, prefix="memory", disable=False, exit_on_done=True):
        """Set up profiler with a prefix and options."""
        self.prefix = prefix
        self.disable = disable
        self.exit_on_done = exit_on_done

    def __enter__(self):
        """Start recording GPU memory history."""
        if self.disable:
            return
        torch.cuda.memory._record_memory_history(max_entries=100_000)

    def __exit__(self, exc_type, exc_value, traceback):
        """Dump snapshot and stop recording; exit program if requested."""
        if self.disable:
            return
        torch.cuda.memory._dump_snapshot(f"{self.prefix}.pickle")
        torch.cuda.memory._record_memory_history(enabled=None)
        if self.exit_on_done:
            exit()
