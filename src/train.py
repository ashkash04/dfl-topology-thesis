"""Centralized FedAvg baseline."""

import copy
import torch
from torch.utils.data import DataLoader

import config
from model import SmallCNN
from data import load_datasets, iid_partition
from client import local_train


def average_weights(state_dicts):
    """FedAvg aggregation: the element-wise mean of a list of state_dicts.
    
    The unweighted mean assumes equal-sized client datasets, which the IID
    split guarantees; uneven splits would require weighting by sample count.
    """
    avg = copy.deepcopy(state_dicts[0])
    for key in avg.keys():
        stacked = torch.stack([sd[key].float() for sd in state_dicts], dim=0)
        avg[key] = stacked.mean(dim=0)
    return avg


def evaluate(model, test_set):
    """Return test accuracy of `model` on `test_set`."""
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