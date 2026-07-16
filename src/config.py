"""Central experiment configuration.

Every hyperparameter and experimental factor lives here, so a run is fully
specified by this file. Changing an experiment means editing values here, not
going through the codebase.
"""

import torch

# --- Hardware ---
# Use the GPU when available; all models and batches are moved to this device.
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# --- Federation setup ---
NUM_CLIENTS = 10        # number of clients the data is partitioned across
NUM_CLASSES = 10        # Fashion-MNIST has 10 classes
SEED = 0                # default seed for single-run demo scripts

# --- Local training (performed by each client every round) ---
LOCAL_EPOCHS = 1        # passes over local data before communication round
BATCH_SIZE = 64
LEARNING_RATE = 1e-3

# --- Communication ---
NUM_ROUNDS = 50         # number of local-train-then-mix rounds

# --- Experimental factors (swept by run_experiments.py) ---

# Communication topologies to compare. Each becomes a doubly-stochastic mixing
# matrix; they span the connectivity spectrum from poorly connected (line) to
# fully connected (mesh).
TOPOLOGIES = ["line", "ring", "star", "hybrid", "mesh"]

# Dirichlet concentration alpha controls data heterogeneity across clients.
# Low alpha => highly skewed (non-IID); high alpha => near-uniform (IID-like).
ALPHA_VALUES = [100.0, 0.5, 0.1]

# Multiple seeds so reported differences reflect signal, not initialisation
# noise. Each (topology, alpha) configuration is run once per seed.
SEEDS = [0, 1, 2]