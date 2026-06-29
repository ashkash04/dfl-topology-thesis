"""Experiment hyperparameters. A run is fully specified by these values."""

import torch

# --- Hardware ---
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# --- Data / federation setup ---
NUM_CLIENTS = 10
SEED = 0

# --- Local training (what each client does per round) ---
LOCAL_EPOCHS = 1
BATCH_SIZE = 64
LEARNING_RATE = 1e-3

# --- Federated / decentralized training ---
NUM_ROUNDS = 50

# --- Model ---
NUM_CLASSES = 10