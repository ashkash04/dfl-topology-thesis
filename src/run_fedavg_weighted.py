"""Run sample-weighted FedAvg for every (alpha, seed) configuration.

This uses the same client partitions, model, optimizer, local training,
number of rounds, and seeds as the unweighted FedAvg baseline.

The only difference is that client models are aggregated in proportion
to the number of local training examples held by each client.
"""

import csv
import os

import torch

import config
from model import SmallCNN
from data import load_datasets, dirichlet_partition
from client import local_train
from fedavg_baseline import weighted_average_weights, evaluate


def load_completed_runs(path):
    """Return completed (alpha, seed) configurations."""
    completed = set()

    if not os.path.exists(path):
        return completed

    with open(path, newline="") as file:
        for row in csv.DictReader(file):
            if int(row["round"]) == config.NUM_ROUNDS:
                completed.add(
                    (
                        float(row["alpha"]),
                        int(row["seed"]),
                    )
                )

    return completed


def run_one(alpha, seed, train_set, test_set):
    """Run one sample-weighted FedAvg configuration."""
    torch.manual_seed(seed)

    client_datasets = dirichlet_partition(
        train_set,
        config.NUM_CLIENTS,
        alpha=alpha,
        seed=seed,
    )

    client_sizes = [
        len(client_dataset)
        for client_dataset in client_datasets
    ]

    total_samples = sum(client_sizes)

    print(f"  Client sizes: {client_sizes}")
    print(
        "  Aggregation weights: "
        + str([
            round(size / total_samples, 4)
            for size in client_sizes
        ])
    )

    global_model = SmallCNN(
        num_classes=config.NUM_CLASSES
    ).to(config.DEVICE)

    rows = []

    for round_num in range(config.NUM_ROUNDS):
        client_weights = [
            local_train(global_model, client_dataset)
            for client_dataset in client_datasets
        ]

        global_weights = weighted_average_weights(
            client_weights,
            client_sizes,
        )

        global_model.load_state_dict(global_weights)

        current_round = round_num + 1

        if (
            current_round % config.EVAL_EVERY == 0
            or current_round == config.NUM_ROUNDS
        ):
            accuracy = evaluate(global_model, test_set)

            rows.append({
                "alpha": alpha,
                "seed": seed,
                "round": current_round,
                "accuracy": accuracy,
            })

            print(
                f"  Round {current_round:3d}/{config.NUM_ROUNDS}  "
                f"accuracy={accuracy:.4f}"
            )

    return rows


def main():
    """Run every alpha and seed and append the results to CSV."""
    train_set, test_set = load_datasets()

    os.makedirs("../results", exist_ok=True)

    out_path = "../results/fedavg_weighted_results.csv"

    fieldnames = [
        "alpha",
        "seed",
        "round",
        "accuracy",
    ]

    completed = load_completed_runs(out_path)

    all_runs = [
        (float(alpha), int(seed))
        for alpha in config.ALPHA_VALUES
        for seed in config.SEEDS
    ]

    remaining_runs = [
        run
        for run in all_runs
        if run not in completed
    ]

    print(
        f"Completed: {len(all_runs) - len(remaining_runs)}/"
        f"{len(all_runs)}"
    )
    print(f"Remaining: {len(remaining_runs)}")

    needs_header = (
        not os.path.exists(out_path)
        or os.path.getsize(out_path) == 0
    )

    with open(out_path, "a", newline="") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=fieldnames,
        )

        if needs_header:
            writer.writeheader()

        for run_number, (alpha, seed) in enumerate(
            remaining_runs,
            start=1,
        ):
            print(
                f"\n[{run_number}/{len(remaining_runs)}] "
                f"weighted FedAvg: alpha={alpha}, seed={seed}"
            )

            rows = run_one(
                alpha,
                seed,
                train_set,
                test_set,
            )

            writer.writerows(rows)
            file.flush()

    print(
        "\nDone. Weighted FedAvg results written to "
        f"{out_path}"
    )


if __name__ == "__main__":
    main()