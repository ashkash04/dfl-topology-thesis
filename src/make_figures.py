"""Generate all figures from the completed experiment results.

The script reads the decentralized, unweighted FedAvg, and sample-weighted
FedAvg CSV files in ``results/`` and writes PNG and PDF versions of every
figure to ``figures/``.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd

import config
from data import dirichlet_partition, load_datasets
from topology import make_topology, spectral_gap


TOPOLOGIES = ["line", "ring", "star", "hybrid", "mesh"]
METHODS = [
    "line",
    "ring",
    "star",
    "hybrid",
    "mesh",
    "fedavg_unweighted",
    "fedavg_weighted",
]
ALPHAS = [100.0, 0.5, 0.1]

LABELS = {
    "line": "Line",
    "ring": "Ring",
    "star": "Star",
    "hybrid": "Hybrid",
    "mesh": "Mesh",
    "fedavg_unweighted": "FedAvg (equal-client)",
    "fedavg_weighted": "FedAvg (sample-weighted)",
}

ALPHA_LABELS = {
    100.0: r"$\alpha=100$ (near-IID)",
    0.5: r"$\alpha=0.5$",
    0.1: r"$\alpha=0.1$ (highly non-IID)",
}

CLASS_NAMES = [
    "T-shirt/Top",
    "Trouser",
    "Pullover",
    "Dress",
    "Coat",
    "Sandal",
    "Shirt",
    "Sneaker",
    "Bag",
    "Ankle boots",
]


def save_figure(figure, output_dir, filename):
    """Save a figure in both raster and vector formats."""
    figure.savefig(
        output_dir / f"{filename}.png",
        dpi=300,
        bbox_inches="tight",
    )
    figure.savefig(
        output_dir / f"{filename}.pdf",
        bbox_inches="tight",
    )
    plt.close(figure)


def read_results_csv(path, required_columns):
    """Read one result CSV and verify that its required columns exist."""
    if not path.exists():
        raise FileNotFoundError(
            f"Missing result file: {path}\n"
            "Run the corresponding experiment sweep before generating figures."
        )

    frame = pd.read_csv(path)
    missing_columns = set(required_columns) - set(frame.columns)

    if missing_columns:
        raise ValueError(
            f"{path.name} is missing required columns: "
            f"{sorted(missing_columns)}"
        )

    return frame


def keep_common_evaluation_rounds(frame):
    """Keep only rounds evaluated by every current experiment script.

    Earlier seeds may contain measurements for all 50 rounds, while later seeds
    contain measurements every ``config.EVAL_EVERY`` rounds. Filtering here
    prevents a convergence point from being averaged over a different number of
    seeds than the other points.
    """
    if config.EVAL_EVERY <= 0:
        raise ValueError("config.EVAL_EVERY must be a positive integer.")

    rounds = frame["round"].astype(int)
    mask = (
        (rounds % config.EVAL_EVERY == 0)
        | (rounds == config.NUM_ROUNDS)
    )
    return frame.loc[mask].copy()


def load_results(repository_root):
    """Load and align decentralized and both FedAvg result files."""
    results_dir = repository_root / "results"

    decentralized = read_results_csv(
        results_dir / "decentralized_results.csv",
        {
            "topology",
            "alpha",
            "seed",
            "spectral_gap",
            "round",
            "avg_acc",
            "worst_acc",
        },
    )
    fedavg_unweighted = read_results_csv(
        results_dir / "fedavg_results.csv",
        {"alpha", "seed", "round", "accuracy"},
    )
    fedavg_weighted = read_results_csv(
        results_dir / "fedavg_weighted_results.csv",
        {"alpha", "seed", "round", "accuracy"},
    )

    decentralized = keep_common_evaluation_rounds(decentralized)
    fedavg_unweighted = keep_common_evaluation_rounds(fedavg_unweighted)
    fedavg_weighted = keep_common_evaluation_rounds(fedavg_weighted)

    return decentralized, fedavg_unweighted, fedavg_weighted


def make_graph(topology_name, num_clients):
    """Create a NetworkX graph matching one experiment topology."""
    if topology_name == "line":
        return nx.path_graph(num_clients)
    if topology_name == "ring":
        return nx.cycle_graph(num_clients)
    if topology_name == "star":
        return nx.star_graph(num_clients - 1)
    if topology_name == "mesh":
        return nx.complete_graph(num_clients)
    if topology_name == "hybrid":
        half = num_clients // 2
        graph = nx.Graph()
        graph.add_nodes_from(range(num_clients))

        # First ring: nodes 0..half-1.
        for i in range(half):
            ring_1 = (i + 1) % half
            graph.add_edge(i, ring_1)

        # Second ring: nodes half..num_clients-1.
        for i in range(half, num_clients):
            ring_2 = half + ((i - half + 1) % (num_clients - half))
            graph.add_edge(i, ring_2)

        # Bridge joining the two rings.
        graph.add_edge(0, half)
        return graph

    raise ValueError(f"Unknown topology: {topology_name}")


def make_topology_figure(output_dir):
    """Plot the five communication graphs."""
    figure, axes = plt.subplots(
        1,
        len(TOPOLOGIES),
        figsize=(15, 3.2),
        layout="constrained",
    )

    for axis, topology_name in zip(axes, TOPOLOGIES):
        graph = make_graph(topology_name, config.NUM_CLIENTS)

        if topology_name in {"ring", "mesh"}:
            positions = nx.circular_layout(graph)
        elif topology_name == "line":
            positions = {node: (node, 0) for node in graph.nodes()}
        else:
            positions = nx.spring_layout(graph, seed=0)

        nx.draw_networkx(
            graph,
            pos=positions,
            ax=axis,
            node_size=260,
            font_size=7,
            width=0.9,
        )

        mixing_matrix = make_topology(
            topology_name,
            config.NUM_CLIENTS,
        )
        gap = spectral_gap(mixing_matrix)

        axis.set_title(
            f"{LABELS[topology_name]}\n"
            f"{graph.number_of_edges()} edges, gap={gap:.3f}"
        )
        axis.set_axis_off()

    figure.suptitle(
        "Communication topologies used in the experiments",
        fontsize=14,
    )
    save_figure(figure, output_dir, "figure_1_topologies")


def client_class_proportions(train_set, client_datasets):
    """Return class proportions and total sample counts for each client."""
    labels = np.asarray(train_set.targets)
    proportions = np.zeros(
        (len(client_datasets), config.NUM_CLASSES),
        dtype=float,
    )
    totals = []

    for client_id, client_dataset in enumerate(client_datasets):
        client_indices = np.asarray(client_dataset.indices)
        client_labels = labels[client_indices]
        counts = np.bincount(
            client_labels,
            minlength=config.NUM_CLASSES,
        )

        totals.append(len(client_indices))

        if counts.sum() > 0:
            proportions[client_id] = counts / counts.sum()

    return proportions, totals


def make_partition_figure(output_dir, repository_root):
    """Show example Dirichlet client partitions for the three alpha values."""
    data_root = repository_root / "src" / "data"
    train_set, _ = load_datasets(data_root=data_root)

    figure, axes = plt.subplots(
        1,
        len(ALPHAS),
        figsize=(16, 5.2),
        layout="constrained",
    )

    image = None

    for axis, alpha in zip(axes, ALPHAS):
        client_datasets = dirichlet_partition(
            train_set,
            config.NUM_CLIENTS,
            alpha=alpha,
            seed=config.SEEDS[0],
        )
        proportions, totals = client_class_proportions(
            train_set,
            client_datasets,
        )

        image = axis.imshow(
            proportions,
            aspect="auto",
            vmin=0.0,
            vmax=1.0,
        )

        axis.set_title(ALPHA_LABELS[alpha])
        axis.set_xlabel("Fashion-MNIST class")
        axis.set_xticks(
            np.arange(config.NUM_CLASSES),
            CLASS_NAMES,
            rotation=45,
            ha="right",
        )
        axis.set_yticks(
            np.arange(config.NUM_CLIENTS),
            [
                f"Client {client_id} (n={total})"
                for client_id, total in enumerate(totals)
            ],
        )

    axes[0].set_ylabel("Client")
    figure.colorbar(
        image,
        ax=axes,
        label="Proportion of the client's local dataset",
        shrink=0.85,
    )
    figure.suptitle(
        "Example Dirichlet client partitions using seed 0",
        fontsize=14,
    )
    save_figure(figure, output_dir, "figure_2_dirichlet_partitions")


def summarize_fedavg(frame):
    """Return per-alpha, per-round mean and standard deviation for FedAvg."""
    return (
        frame
        .groupby(["alpha", "round"], as_index=False)
        .agg(
            mean_accuracy=("accuracy", "mean"),
            std_accuracy=("accuracy", "std"),
        )
    )


def plot_accuracy_curve(
    axis,
    summary,
    alpha,
    method_name,
    method_colors,
    linestyle,
    linewidth,
):
    """Plot one summarized accuracy curve with a one-standard-deviation band."""
    subset = summary[
        summary["alpha"] == alpha
    ].sort_values("round")

    rounds = subset["round"].to_numpy()
    mean_accuracy = 100.0 * subset["mean_accuracy"].to_numpy()
    std_accuracy = (
        100.0
        * subset["std_accuracy"].fillna(0.0).to_numpy()
    )

    axis.plot(
        rounds,
        mean_accuracy,
        label=LABELS[method_name],
        color=method_colors[method_name],
        linestyle=linestyle,
        linewidth=linewidth,
    )
    axis.fill_between(
        rounds,
        mean_accuracy - std_accuracy,
        mean_accuracy + std_accuracy,
        color=method_colors[method_name],
        alpha=0.10,
    )


def make_convergence_figure(
    decentralized,
    fedavg_unweighted,
    fedavg_weighted,
    output_dir,
):
    """Plot mean accuracy over communication rounds for all methods."""
    decentralized_summary = (
        decentralized
        .groupby(["alpha", "topology", "round"], as_index=False)
        .agg(
            mean_accuracy=("avg_acc", "mean"),
            std_accuracy=("avg_acc", "std"),
        )
    )

    fedavg_unweighted_summary = summarize_fedavg(fedavg_unweighted)
    fedavg_weighted_summary = summarize_fedavg(fedavg_weighted)

    default_colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    method_colors = {
        method: default_colors[index % len(default_colors)]
        for index, method in enumerate(METHODS)
    }

    figure, axes = plt.subplots(
        1,
        len(ALPHAS),
        figsize=(16.5, 4.8),
        sharex=True,
        sharey=True,
        layout="constrained",
    )

    for axis, alpha in zip(axes, ALPHAS):
        for topology_name in TOPOLOGIES:
            subset = decentralized_summary[
                (decentralized_summary["alpha"] == alpha)
                & (decentralized_summary["topology"] == topology_name)
            ].sort_values("round")

            rounds = subset["round"].to_numpy()
            mean_accuracy = 100.0 * subset["mean_accuracy"].to_numpy()
            std_accuracy = (
                100.0
                * subset["std_accuracy"].fillna(0.0).to_numpy()
            )

            axis.plot(
                rounds,
                mean_accuracy,
                label=LABELS[topology_name],
                color=method_colors[topology_name],
                linewidth=1.6,
            )
            axis.fill_between(
                rounds,
                mean_accuracy - std_accuracy,
                mean_accuracy + std_accuracy,
                color=method_colors[topology_name],
                alpha=0.10,
            )

        plot_accuracy_curve(
            axis,
            fedavg_unweighted_summary,
            alpha,
            "fedavg_unweighted",
            method_colors,
            linestyle="--",
            linewidth=2.0,
        )
        plot_accuracy_curve(
            axis,
            fedavg_weighted_summary,
            alpha,
            "fedavg_weighted",
            method_colors,
            linestyle=":",
            linewidth=2.2,
        )

        axis.set_title(ALPHA_LABELS[alpha])
        axis.set_xlabel("Communication round")
        axis.grid(alpha=0.25)

    axes[0].set_ylabel("Mean test accuracy (%)")
    handles, labels = axes[-1].get_legend_handles_labels()
    figure.legend(
        handles,
        labels,
        loc="lower center",
        ncol=4,
        frameon=False,
        bbox_to_anchor=(0.5, -0.10),
    )
    figure.suptitle(
        "Convergence across communication topologies and FedAvg baselines",
        fontsize=14,
    )
    save_figure(figure, output_dir, "figure_3_convergence")


def final_fedavg_summary(frame, method_name):
    """Summarize one FedAvg method at the configured final round."""
    final_rows = frame[frame["round"] == config.NUM_ROUNDS]

    summary = (
        final_rows
        .groupby("alpha", as_index=False)
        .agg(
            mean_accuracy=("accuracy", "mean"),
            std_accuracy=("accuracy", "std"),
        )
    )
    summary["method"] = method_name
    return summary


def make_final_accuracy_figure(
    decentralized,
    fedavg_unweighted,
    fedavg_weighted,
    output_dir,
):
    """Compare final-round accuracy across all methods and alpha values."""
    decentralized_final = decentralized[
        decentralized["round"] == config.NUM_ROUNDS
    ]

    decentralized_summary = (
        decentralized_final
        .groupby(["alpha", "topology"], as_index=False)
        .agg(
            mean_accuracy=("avg_acc", "mean"),
            std_accuracy=("avg_acc", "std"),
        )
        .rename(columns={"topology": "method"})
    )

    fedavg_unweighted_summary = final_fedavg_summary(
        fedavg_unweighted,
        "fedavg_unweighted",
    )
    fedavg_weighted_summary = final_fedavg_summary(
        fedavg_weighted,
        "fedavg_weighted",
    )

    combined = pd.concat(
        [
            decentralized_summary,
            fedavg_unweighted_summary,
            fedavg_weighted_summary,
        ],
        ignore_index=True,
    )

    figure, axis = plt.subplots(
        figsize=(10.5, 5.2),
        layout="constrained",
    )

    x_positions = np.arange(len(METHODS))
    offsets = np.linspace(-0.22, 0.22, len(ALPHAS))

    for offset, alpha in zip(offsets, ALPHAS):
        subset = (
            combined[combined["alpha"] == alpha]
            .set_index("method")
            .reindex(METHODS)
        )

        axis.errorbar(
            x_positions + offset,
            100.0 * subset["mean_accuracy"].to_numpy(),
            yerr=(
                100.0
                * subset["std_accuracy"].fillna(0.0).to_numpy()
            ),
            marker="o",
            linestyle="none",
            capsize=3,
            label=ALPHA_LABELS[alpha],
        )

    axis.set_xticks(
        x_positions,
        [LABELS[method] for method in METHODS],
        rotation=15,
        ha="right",
    )
    axis.set_xlabel("Method")
    axis.set_ylabel("Final test accuracy (%)")
    axis.set_title(
        "Final-round accuracy across methods and heterogeneity levels"
    )
    axis.grid(axis="y", alpha=0.25)
    axis.legend(frameon=False)

    save_figure(figure, output_dir, "figure_4_final_accuracy")


def make_average_minimum_figure(decentralized, output_dir):
    """Compare average and minimum client-model accuracy at round 50."""
    final_round = decentralized[
        decentralized["round"] == config.NUM_ROUNDS
    ]

    summary = (
        final_round
        .groupby(["alpha", "topology"], as_index=False)
        .agg(
            average_accuracy=("avg_acc", "mean"),
            minimum_accuracy=("worst_acc", "mean"),
        )
    )

    default_colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    average_color = default_colors[0]
    minimum_color = default_colors[1]

    figure, axes = plt.subplots(
        1,
        len(ALPHAS),
        figsize=(15, 4.5),
        sharex=True,
        layout="constrained",
    )

    for axis, alpha in zip(axes, ALPHAS):
        subset = (
            summary[summary["alpha"] == alpha]
            .set_index("topology")
            .reindex(TOPOLOGIES)
        )

        y_positions = np.arange(len(TOPOLOGIES))
        average_values = 100.0 * subset["average_accuracy"].to_numpy()
        minimum_values = 100.0 * subset["minimum_accuracy"].to_numpy()

        for y_position, minimum, average in zip(
            y_positions,
            minimum_values,
            average_values,
        ):
            axis.plot(
                [minimum, average],
                [y_position, y_position],
                linewidth=2,
                color="0.7",
            )

        axis.scatter(
            minimum_values,
            y_positions,
            color=minimum_color,
            label="Minimum model",
            zorder=3,
        )
        axis.scatter(
            average_values,
            y_positions,
            color=average_color,
            label="Average model",
            zorder=3,
        )

        axis.set_title(ALPHA_LABELS[alpha])
        axis.set_xlabel("Final test accuracy (%)")
        axis.set_yticks(
            y_positions,
            [LABELS[name] for name in TOPOLOGIES],
        )
        axis.grid(axis="x", alpha=0.25)
        axis.invert_yaxis()

    handles, labels = axes[-1].get_legend_handles_labels()
    figure.legend(
        handles,
        labels,
        loc="lower center",
        ncol=2,
        frameon=False,
        bbox_to_anchor=(0.5, -0.05),
    )
    figure.suptitle(
        "Average and minimum decentralized client-model accuracy",
        fontsize=14,
    )
    save_figure(
        figure,
        output_dir,
        "figure_5_average_vs_minimum",
    )


def make_spectral_gap_figure(decentralized, output_dir):
    """Plot spectral gap against final mean accuracy."""
    final_round = decentralized[
        decentralized["round"] == config.NUM_ROUNDS
    ]

    summary = (
        final_round
        .groupby(
            ["alpha", "topology", "spectral_gap"],
            as_index=False,
        )
        .agg(
            mean_accuracy=("avg_acc", "mean"),
            std_accuracy=("avg_acc", "std"),
        )
    )

    default_colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    topology_colors = {
        topology_name: default_colors[index % len(default_colors)]
        for index, topology_name in enumerate(TOPOLOGIES)
    }

    figure, axes = plt.subplots(
        1,
        len(ALPHAS),
        figsize=(15, 4.5),
        sharey=True,
        layout="constrained",
    )

    for axis, alpha in zip(axes, ALPHAS):
        subset = summary[summary["alpha"] == alpha]

        for topology_name in TOPOLOGIES:
            row = subset[
                subset["topology"] == topology_name
            ].iloc[0]

            x_value = row["spectral_gap"]
            y_value = 100.0 * row["mean_accuracy"]
            error = 100.0 * row["std_accuracy"]

            axis.errorbar(
                x_value,
                y_value,
                yerr=error,
                marker="o",
                linestyle="none",
                capsize=3,
                color=topology_colors[topology_name],
            )
            axis.annotate(
                LABELS[topology_name],
                (x_value, y_value),
                xytext=(5, 5),
                textcoords="offset points",
                fontsize=8,
            )

        axis.set_xscale("log")
        axis.set_title(ALPHA_LABELS[alpha])
        axis.set_xlabel("Spectral gap (log scale)")
        axis.grid(alpha=0.25)

    axes[0].set_ylabel("Final mean test accuracy (%)")
    figure.suptitle(
        "Spectral gap and accuracy levels",
        fontsize=14,
    )
    save_figure(
        figure,
        output_dir,
        "figure_6_spectral_gap_vs_accuracy",
    )


def main():
    """Generate every thesis figure."""
    source_dir = Path(__file__).resolve().parent
    repository_root = source_dir.parent
    output_dir = repository_root / "figures"
    output_dir.mkdir(parents=True, exist_ok=True)

    (
        decentralized,
        fedavg_unweighted,
        fedavg_weighted,
    ) = load_results(repository_root)

    make_topology_figure(output_dir)
    make_partition_figure(output_dir, repository_root)
    make_convergence_figure(
        decentralized,
        fedavg_unweighted,
        fedavg_weighted,
        output_dir,
    )
    make_final_accuracy_figure(
        decentralized,
        fedavg_unweighted,
        fedavg_weighted,
        output_dir,
    )
    make_average_minimum_figure(decentralized, output_dir)
    make_spectral_gap_figure(decentralized, output_dir)

    print(f"Done. Figures written to {output_dir}")


if __name__ == "__main__":
    main()
