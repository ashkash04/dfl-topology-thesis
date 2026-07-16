"""Fashion-MNIST loading and client partitioning."""

import numpy as np
from torchvision import datasets, transforms
from torch.utils.data import Subset

from config import NUM_CLASSES

# Dataset-wide mean and standard deviation, used to normalize inputs.
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
    rng = np.random.default_rng(seed)   # local RNG keeps the split reproducible
    indices = np.arange(len(train_set))
    rng.shuffle(indices)

    # np.split would crash if the total isn't evenly divisible, 
    # array_split tolerates a non-divisible count (some clients get one extra).
    chunks = np.array_split(indices, num_clients)
    return [Subset(train_set, chunk.tolist()) for chunk in chunks]


def dirichlet_partition(train_set, num_clients, alpha, seed=0):
    """Split the training set non-IID across clients using a Dirichlet distribution.
    
    For each class, the fraction of its samples assigned to each client is drawn
    from a symmetric Dirichlet(alpha) distribution. Small alpha (~0.1) produces
    highly skewed clients that see only a few classes; large alpha (~100)
    approaches a uniform, IID-like split. Alpha therefore acts as a single
    continuous knob controlling data heterogeneity.

    Returns one Subset per client.
    """
    rng = np.random.default_rng(seed)

    labels = np.array(train_set.targets)
    indices_by_class = [np.where(labels == c)[0] for c in range(NUM_CLASSES)]

    client_indices = [[] for _ in range(num_clients)]

    for c in range(NUM_CLASSES):
        class_indices = indices_by_class[c]
        rng.shuffle(class_indices)

        # Draw this class's per-client split proportions (sum to 1).
        proportions = rng.dirichlet(alpha=np.repeat(alpha, num_clients))

        # Convert proportions to integer counts; give any rounding remainder to
        # the largest-share client so no samples are dropped or duplicated.
        counts = (proportions * len(class_indices)).astype(int)
        remainder = len(class_indices) - counts.sum()
        counts[np.argmax(proportions)] += remainder

        # Deal that many of this class's samples to each client in sequence.
        start = 0
        for client_id in range(num_clients):
            end = start + counts[client_id]
            client_indices[client_id].extend(class_indices[start:end].tolist())
            start = end

    # Shuffle each client's combined indices so classes aren't grouped in order.
    for idx_list in client_indices:
        rng.shuffle(idx_list)
    
    return [Subset(train_set, idx_list) for idx_list in client_indices]