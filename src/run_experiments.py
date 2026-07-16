"""Experiment sweep: run every (topology, alpha, seed) combination and log to CSV.

For each communication topology, each Dirichlet heterogeneity level (alpha), and
each seed, this runs the full decentralized training loop and records per-round
metrics. All rows are written to a single CSV in results/.

Output CSV columns:
    topology        - name of the communication topology
    alpha           - Dirichlet concentration (data heterogeneity level)
    seed            - random seed for this run
    spectral_gap    - connectivity measure of the topology (constant per topology)
    round           - communication round (1..NUM_ROUNDS)
    avg_acc         - mean test accuracy across all clients this round
    worst_acc       - minimum test accuracy this round

Per-round rows (not just final) are logged so convergence speed can be analysed;
each configuration is repeated over multiple seeds so differences reflect signal
rather than initialisation noise.
"""

import copy
import csv
import os

import torch

import config
from model import SmallCNN
from data import load_datasets, dirichlet_partition
from client import local_train
from topology import make_topology, spectral_gap
from train_decentralized import mix_weights, evaluate_client


def run_one(topology_name, alpha, seed, train_set, test_set):
    """Run one configuration; return a list of per-round metric rows (dicts)."""
    torch.manual_seed(seed)

    client_datasets = dirichlet_partition(
        train_set, config.NUM_CLIENTS, alpha=alpha, seed=seed
    )

    W = make_topology(topology_name, config.NUM_CLIENTS)
    gap = spectral_gap(W)

    initial_model = SmallCNN(num_classes=config.NUM_CLASSES)
    client_models = [
        copy.deepcopy(initial_model).to(config.DEVICE)
        for _ in range(config.NUM_CLIENTS)
    ]

    rows = []
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
        rows.append({
            "topology": topology_name,
            "alpha": alpha,
            "seed": seed,
            "spectral_gap": gap,
            "round": round_num + 1,
            "avg_acc": sum(accuracies) / len(accuracies),
            "worst_acc": min(accuracies),
        })
    
    return rows


def main():
    """Sweep all configurations and write every per-round row to one CSV."""
    train_set, test_set = load_datasets()

    os.makedirs("../results", exist_ok=True)
    out_path = "../results/decentralized_results.csv"

    fieldnames = [
        "topology", "alpha", "seed", "spectral_gap",
        "round", "avg_acc", "worst_acc",
    ]

    total = len(config.TOPOLOGIES) * len(config.ALPHA_VALUES) * len(config.SEEDS)
    done = 0

    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for topology_name in config.TOPOLOGIES:
            for alpha in config.ALPHA_VALUES:
                for seed in config.SEEDS:
                    done += 1
                    print(f"[{done}/{total}] "
                          f"topology={topology_name} alpha={alpha} seed={seed}")
                    
                    rows = run_one(topology_name, alpha, seed, train_set, test_set)
                    writer.writerows(rows)
                    f.flush()
    
    print(f"\nDone. Results written to {out_path}")


if __name__ == "__main__":
    main()