"""Centralized FedAvg baseline (single-run demo).

Runs standard Federated Averaging (McMahan et al., 2017): clients train locally, a centralized server
averages their weights into one global model each round. This is the centralized reference the decentralized
topologies are compared against. The full experiment sweep is in run_experiments.py; this file is a standalone
demonstration of the baseline.
"""

import copy
import torch
from torch.utils.data import DataLoader

import config
from model import SmallCNN
from data import load_datasets, iid_partition
from client import local_train


def average_weights(state_dicts):
    """unweighted FedAvg aggregation: the element-wise mean of a list of state_dicts.
    
    The unweighted mean assumes equal-sized client datasets, which the IID
    split guarantees; uneven splits would require weighting by sample count.
    """
    avg = copy.deepcopy(state_dicts[0])
    for key in avg.keys():
        stacked = torch.stack([sd[key].float() for sd in state_dicts], dim=0)
        avg[key] = stacked.mean(dim=0)
    return avg

def weighted_average_weights(client_weights, client_sizes):
    """Average client model parameters in proportion to local dataset size."""
    if len(client_weights) != len(client_sizes):
        raise ValueError(
            "client_weights and client_sizes must have the same length."
        )

    total_samples = sum(client_sizes)

    if total_samples <= 0:
        raise ValueError("Total number of client samples must be positive.")

    averaged_weights = {}

    for parameter_name in client_weights[0]:
        first_tensor = client_weights[0][parameter_name]

        # Neural-network parameters are floating-point tensors
        if torch.is_floating_point(first_tensor):
            weighted_parameter = torch.zeros_like(first_tensor)

            for weights, client_size in zip(client_weights, client_sizes):
                client_weight = client_size / total_samples

                weighted_parameter.add_(
                    weights[parameter_name],
                    alpha=client_weight,
                )

            averaged_weights[parameter_name] = weighted_parameter

        else:
            # Included for state-dictionary entries that are not floating point
            averaged_weights[parameter_name] = first_tensor.clone()

    return averaged_weights

def evaluate(model, test_set):
    """Return test accuracy of `model` on `test_set`."""
    model.eval()
    loader = DataLoader(test_set, batch_size=config.EVAL_BATCH_SIZE, shuffle=False)

    correct = total = 0
    with torch.no_grad():
        for images, labels in loader:
            images = images.to(config.DEVICE)
            labels = labels.to(config.DEVICE)
            predictions = model(images).argmax(dim=1)
            correct += (predictions == labels).sum().item()
            total += labels.size(0)
    return correct / total


def run_fedavg():
    """Run FedAvg and print test accuracy per round."""
    torch.manual_seed(config.SEED)

    train_set, test_set = load_datasets()
    client_datasets = iid_partition(train_set, config.NUM_CLIENTS, seed=config.SEED)
    global_model = SmallCNN(num_classes=config.NUM_CLASSES).to(config.DEVICE)

    for round_num in range(config.NUM_ROUNDS):
        client_weights = [local_train(global_model, dataset) for dataset in client_datasets]
        global_model.load_state_dict(average_weights(client_weights))

        acc = evaluate(global_model, test_set)
        print(f"Round {round_num + 1:3d}/{config.NUM_ROUNDS}    test accuracy: {acc:.4f}")


if __name__ == "__main__":
    run_fedavg()