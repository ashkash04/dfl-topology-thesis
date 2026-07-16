"""Decentralized training loop: neighbour-averaging via a mixing matrix.

Unlike FedAvg, there is no server and no single global model. Each client keeps
its own model, trains locally, then replaces its weights with a W-weighted
average of its neighbours' weights; the mixing step X_k+1 <- X_k W from D-PSGD
(Lian et al., 2017). Accuracy is therefore per-client; we report the average
and the worst client, since poorly-connected clients can be left behind
under heterogeneous data.

This module is imported by run_experiments.py (for mix_weights and 
evaluate_client) and can also be run standalone as a single-topology demo.
"""

import copy
import torch
from torch.utils.data import DataLoader

import config
from model import SmallCNN
from data import load_datasets, iid_partition
from client import local_train
from topology import make_topology, spectral_gap


def mix_weights(client_state_dicts, W):
    """Average each client's weights with its neighbours, weighted by W.
    
    Row i of W defines client i's new weights as a weighted sum over all
    clients; non-neighbours have weight 0 and contribute nothing.
    """
    num_clients = len(client_state_dicts)
    keys = client_state_dicts[0].keys()

    new_state_dicts = []
    for i in range(num_clients):
        mixed = copy.deepcopy(client_state_dicts[i])
        for key in keys:
            stacked = torch.stack([client_state_dicts[j][key].float() for j in range(num_clients)], dim=0)
            weights = torch.tensor(W[i], dtype=torch.float32, device=stacked.device)
            weights = weights.view([num_clients] + [1] * (stacked.dim() - 1))
            mixed[key] = (weights * stacked).sum(dim=0)
        new_state_dicts.append(mixed)
    
    return new_state_dicts


def evaluate_client(model, test_set):
    """Return test accuracy of a single client's model on the shared test set."""
    model.eval()
    loader = DataLoader(test_set, batch_size=256, shuffle=False)

    correct = total = 0
    with torch.no_grad():
        for images, labels in loader:
            images = images.to(config.DEVICE)
            labels = labels.to(config.DEVICE)
            predictions = model(images).argmax(dim=1)
            correct += (predictions == labels).sum().item()
            total += labels.size(0)
    return correct / total


def run_decentralized(topology_name):
    """Run decentralized training over a topology; print avg and worst accuracy."""
    torch.manual_seed(config.SEED)

    train_set, test_set = load_datasets()
    client_datasets = iid_partition(train_set, config.NUM_CLIENTS, seed=config.SEED)

    W = make_topology(topology_name, config.NUM_CLIENTS)
    print(f"Topology: {topology_name}   "
          f"spectral gap: {spectral_gap(W):.4f}")

    initial_model = SmallCNN(num_classes=config.NUM_CLASSES)
    client_models = [
        copy.deepcopy(initial_model).to(config.DEVICE)
        for _ in range(config.NUM_CLIENTS)
    ]

    for round_num in range(config.NUM_ROUNDS):
        client_weights = [
            local_train(client_models[i], client_datasets[i])
            for i in range(config.NUM_CLIENTS)
        ]
        mixed = mix_weights(client_weights, W)
        for i in range(config.NUM_CLIENTS):
            client_models[i].load_state_dict(mixed[i])
        
        accuracies = [
            evaluate_client(client_models[i], test_set)
            for i in range(config.NUM_CLIENTS)
        ]
        avg_acc = sum(accuracies) / len(accuracies)
        worst_acc = min(accuracies)
        print(f"Round {round_num + 1:3d}/{config.NUM_ROUNDS}    "
              f"avg: {avg_acc:.4f}  "
              f"worst: {worst_acc:.4f}")


if __name__ == "__main__":
    run_decentralized("ring")