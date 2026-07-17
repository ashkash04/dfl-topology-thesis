"""Experiment sweep for the centralized FedAvg baseline.

For each Dirichlet heterogeneity level alpha and random seed, this script runs
centralized Federated Averaging and records the global test accuracy after every
communication round.

The same model, data partitions, local training procedure, hyperparameters,
alpha values, and seeds are used as in the decentralized sweeps.

Output CSV columns:
    alpha           - Dirichlet concentration (data heterogeneity level)
    seed            - random seed for this run
    round           - communication round (1..NUM_ROUNDS)
    accuracy        - mean test accuracy across all clients this round
"""

import csv
import os

import torch

import config
from client import local_train
from data import dirichlet_partition, load_datasets
from fedavg_baseline import average_weights, evaluate
from model import SmallCNN


def run_one(alpha, seed, train_set, test_set):
    """Run one FedAvg configuration and return its per-round results."""
    torch.manual_seed(seed)

    client_datasets = dirichlet_partition(
        train_set,
        config.NUM_CLIENTS,
        alpha=alpha,
        seed=seed
    )

    global_model = SmallCNN(
        num_classes=config.NUM_CLASSES,
    ).to(config.DEVICE)

    rows = []

    for round_num in range(config.NUM_ROUNDS):
        client_weights = [
            local_train(global_model, client_dataset)
            for client_dataset in client_datasets
        ]

        global_weights = average_weights(client_weights)
        global_model.load_state_dict(global_weights)

        accuracy = evaluate(global_model, test_set)

        rows.append(
            {
                "alpha": alpha,
                "seed": seed,
                "round": round_num + 1,
                "accuracy": accuracy,
            }
        )

        print(
            f"Round {round_num + 1:3d}/{config.NUM_ROUNDS}  "
            f"test accuracy: {accuracy:.4f}"
        )
    
    return rows


def main():
    """Sweep all FedAvg configurations and write the results to CSV."""
    train_set, test_set = load_datasets()

    os.makedirs("../results", exist_ok=True)
    out_path = "../results/fedavg_results.csv"

    fieldnames = [
        "alpha",
        "seed",
        "round",
        "accuracy",
    ]

    total = len(config.ALPHA_VALUES) * len(config.SEEDS)
    done = 0

    with open(out_path, "w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()

        for alpha in config.ALPHA_VALUES:
            for seed in config.SEEDS:
                done += 1

                print(
                    f"\n[{done}/{total}]    "
                    f"FedAvg alpha={alpha} seed={seed}"
                )

                rows = run_one(
                    alpha,
                    seed,
                    train_set,
                    test_set,
                )

                writer.writerows(rows)
                file.flush()
    
    print(f"\nDone. Results written to {out_path}")


if __name__ == "__main__":
    main()