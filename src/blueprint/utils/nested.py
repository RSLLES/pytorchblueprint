# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""Utility functions to deal with nested tensors (padding/expanding)."""

import torch
from torch import Tensor


def pad_sequence(
    x: list[Tensor],
    target_len: int | None = None,
    padding_value: float = 0.0,
    strict: bool = True,
    returns_lengths: bool = False,
) -> Tensor | tuple[Tensor, Tensor]:
    """Pad a list of tensors to a uniform length."""
    if not strict:
        x = [e[:target_len] for e in x]
    padded = torch.nn.utils.rnn.pad_sequence(
        x, batch_first=True, padding_value=padding_value
    )

    if target_len is not None:
        delta_len = target_len - padded.size(1)
        if delta_len < 0 and strict:
            raise ValueError(f"Sequence length exceeds max_len={target_len}")
        if delta_len > 0:
            pad_size = list(padded.shape)
            pad_size[1] = delta_len
            pad = padded.new_full(pad_size, padding_value)
            padded = torch.cat([padded, pad], dim=1)

    if not returns_lengths:
        return padded
    lengths = torch.tensor(
        [e.size(0) for e in x], device=padded.device, dtype=torch.int
    )
    return padded, lengths


def lengths_to_mask(lengths: Tensor, padded_len: int) -> Tensor:
    """Return a boolean mask of shape (B, padded_len) that marks valid positions."""
    return torch.arange(padded_len, device=lengths.device) < lengths.unsqueeze(1)


def expand_to_list(x: Tensor, lengths: Tensor) -> list[Tensor]:
    """Slice a padded tensor back into a list using the given lengths."""
    return [e[:i] for e, i in zip(x, lengths, strict=True)]
