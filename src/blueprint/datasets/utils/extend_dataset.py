# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""Loop over a dataset to extend its size to a target length."""

from torch.utils.data import Dataset


class ExtendDataset(Dataset):
    """Extend a dataset to a target length."""

    def __init__(self, ds: Dataset, length: int):
        super().__init__()
        self.dataset = ds
        self.length = length

    def __len__(self):
        return self.length

    def __getitem__(self, idx: int):
        if idx < 0 or idx >= self.__len__():
            raise StopIteration()
        idx = idx % len(self.dataset)
        return self.dataset[idx]

    def increment_seed(self):
        """Advance internal seed."""
        self.dataset.increment_seed()

    def collate_fn(self, batch):
        """Provide the custom collate function of the base dataset."""
        return self.dataset.collate_fn(batch)
