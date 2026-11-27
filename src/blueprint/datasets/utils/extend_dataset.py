# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""Loop over a dataset to extend its size to a target length."""

from typing import Any

from torch.utils.data import Dataset


class ExtendDataset[T](Dataset[T]):
    """Extend a dataset to a target length."""

    def __init__(self, ds: Dataset[T], length: int):
        super().__init__()
        if len(ds) > length:  # ty:ignore[invalid-argument-type]
            raise ValueError(
                f"Cannot extend a dataset of length {len(ds)} to {length}."  # ty:ignore[invalid-argument-type]
            )
        self.dataset = ds
        self.length = length

    def __len__(self) -> int:
        return self.length

    def __getitem__(self, index: int) -> T:
        if index < 0 or index >= self.__len__():
            raise IndexError(f"Index {index} out of range for length {self.length}.")
        return self.dataset[index % len(self.dataset)]  # ty:ignore[invalid-argument-type]

    def increment_seed(self):
        """Advance internal seed."""
        self.dataset.increment_seed()  # ty:ignore[unresolved-attribute]

    def collate_fn(self, batch: list[T]) -> Any:
        """Provide the custom collate function of the base dataset."""
        return self.dataset.collate_fn(batch)  # ty:ignore[unresolved-attribute]
