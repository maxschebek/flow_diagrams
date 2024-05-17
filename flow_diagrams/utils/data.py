import numpy as np
from torch.utils import data
import torch


# See https://jax.readthedocs.io/en/latest/notebooks/Neural_Network_and_Data_Loading.html
def numpy_collate(batch):
    if isinstance(batch[0], np.ndarray):
        return np.stack(batch)
    elif isinstance(batch[0], (tuple, list)):
        transposed = zip(*batch)
        return [numpy_collate(samples) for samples in transposed]
    else:
        return np.array(batch)


class NumpyLoader(data.DataLoader):
    def __init__(
        self,
        dataset,
        batch_size=1,
        shuffle=False,
        sampler=None,
        batch_sampler=None,
        num_workers=0,
        pin_memory=False,
        drop_last=False,
        timeout=0,
        worker_init_fn=None,
    ):
        super(self.__class__, self).__init__(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            sampler=sampler,
            batch_sampler=batch_sampler,
            num_workers=num_workers,
            collate_fn=numpy_collate,
            pin_memory=pin_memory,
            drop_last=drop_last,
            timeout=timeout,
            worker_init_fn=worker_init_fn,
        )


# define a custom dataset
class Dataset(torch.utils.data.Dataset):
    """
    Simple wrapper for the dataset.
    """

    def __init__(self, pos, energies, scale):
        assert pos.shape[0] == energies.shape[0]

        self.energies = energies
        self.pos = pos
        self.scale = scale

    def __len__(self):
        return self.pos.shape[0]

    def __getitem__(self, idx):
        return (self.pos[idx], self.energies[idx], self.scale[idx])


def split_data(train_fraction, positions, energies, scales):
    n_total = positions.shape[0]
    n_train = int(train_fraction * n_total)

    # get indices for training set
    ids_train = np.array(np.random.choice(n_total, n_train, replace=False))
    ids_test = np.setdiff1d(np.array(range(n_total)), ids_train)

    dataset_train = Dataset(
        positions[ids_train], energies[ids_train], scales[ids_train]
    )

    dataset_test = Dataset(positions[ids_test], energies[ids_test], scales[ids_test])

    return dataset_train, dataset_test
