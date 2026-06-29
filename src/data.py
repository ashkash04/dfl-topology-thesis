"""Fashion-MNIST loading and client partitioning."""

import numpy as np
from torchvision import datasets, transforms
from torch.utils.data import Subset

# Dataset-wide channel statistics, used to normalize inputs.
_MEAN = (0.2860,)
_STD = (0.3530,)


def load_datasets(data_root="./data"):
    """Return the Fashion-MNIST train and test sets, downloading if needed."""
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(_MEAN, _STD),
    ])
    train_set = datasets.FashionMNIST(
        root=data_root, train=True, download=True, transform=transform
    )
    test_set = datasets.FashionMNIST(
        root=data_root, train=False, download=True, transform=transform
    )
    return train_set, test_set


def iid_partition(train_set, num_clients, seed=0):
    """Split the training set IID across clients; return one Subset per client.
    
    Each client receives a random, representative slice, so all clients share
    the full dataset's class distribution.
    """
    rng = np.random.default_rng(seed)   # local RNG keeps the split reproducable
    indices = np.arange(len(train_set))
    rng.shuffle(indices)

    # array_split tolerates a non-divisible count (some clients get one extra).
    chunks = np.array_split(indices, num_clients)
    return [Subset(train_set, chunk.tolist()) for chunk in chunks]