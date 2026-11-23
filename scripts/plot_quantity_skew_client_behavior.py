import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
METRICS = ROOT / "results/metrics"
RUNS = ROOT / "results/runs"
CLIENT_REPORTS = ROOT / "results/clients/sensors_only"
OUTPUT = ROOT / "results/figures/sensors_only_quantity_skew"
SEEDS = (7, 21, 42)
INCLUDE_STRONG = False

SCENARIOS = (
    ("IID", "iid", "sensors_only_quantity_skew"),
    ("Moderate quantity-skew", "quantity_skew", "sensors_only_quantity_skew"),
)
STRONG_SCENARIO = (
    "Strong quantity-skew",
    "quantity_skew_strong",
    "sensors_only_quantity_skew_strong",
)
SEED_COLORS = {7: "#4C72B0", 21: "#DD8452", 42: "#55A868"}


def load_local_metrics(metric_root: str, scenario: str, seed: int) -> pd.DataFrame:
    path = METRICS / metric_root / scenario / "local_normal_only" / f"seed_{seed}" / "views.csv"
    if not path.is_file():
        raise FileNotFoundError(f"Missing local evaluation: {path}")
    frame = pd.read_csv(path)
    if frame["view"].nunique() != 6:
        raise ValueError(f"Expected six local client views in {path}")
    return frame.assign(seed=seed)


def load_client_sizes(scenario: str) -> dict[str, int]:
    path = CLIENT_REPORTS / scenario / "client_summary.csv"
    if not path.is_file():
        raise FileNotFoundError(f"Missing client summary: {path}")
    frame = pd.read_csv(path)
    return dict(zip(frame["client"], frame["n_normal_real"].astype(int)))


def load_fedavg_rounds(scenario: str, run_root: str, seed: int) -> list[dict]:
    path = RUNS / run_root / scenario / "fedavg_normal_only" / f"seed_{seed}" / "history.json"
    if not path.is_file():
        raise FileNotFoundError(f"Missing FedAvg history: {path}")
    with path.open(encoding="utf-8") as handle:
        history = json.load(handle)
    return history["rounds"]


def plot_local_client_pr_auc() -> Path:
    fig, axes = plt.subplots(1, len(SCENARIOS), figsize=(13, 6), sharey=True, squeeze=False)

    for ax, (scenario_label, scenario, metric_root) in zip(axes[0], SCENARIOS):
        sizes = load_client_sizes(scenario)
        client_names = [f"client_{position}" for position in range(1, 7)]
        for seed in SEEDS:
            metrics = load_local_metrics(metric_root, scenario, seed).set_index("view")
            for position, client in enumerate(client_names):
                if client not in metrics.index:
                    raise ValueError(f"{client} missing from local metrics for {scenario}, seed {seed}")
                ax.scatter(
                    position,
                    metrics.loc[client, "test_pr_auc"],
                    color=SEED_COLORS[seed],
                    s=45,
                    alpha=0.85,
                    label=f"seed {seed}" if position == 0 else None,
                    zorder=3,
                )

        scenario_fedavg_aps = []
        for seed in SEEDS:
            path = METRICS / metric_root / scenario / "fedavg_normal_only" / f"seed_{seed}" / "views.csv"
            if not path.is_file():
                raise FileNotFoundError(f"Missing FedAvg evaluation: {path}")
            frame = pd.read_csv(path)
            scenario_fedavg_aps.append(float(frame.loc[frame["view"] == "global", "test_pr_auc"].iloc[0]))
        fedavg_mean = float(np.mean(scenario_fedavg_aps))
        ax.axhline(
            fedavg_mean,
            color="#222222",
            linestyle="--",
            linewidth=1.5,
            label=f"FedAvg global mean AP={fedavg_mean:.3f}",
        )
        ax.set_title(scenario_label)
        ax.set_xticks(range(len(client_names)))
        ax.set_xticklabels([f"{c}\n(n={sizes[c]})" for c in client_names], fontsize=8)
        ax.set_xlabel("Local client (normal-real training rows)")
        ax.grid(axis="y", alpha=0.25)
        ax.set_ylim(0.55, 0.75)

    axes[0, 0].set_ylabel("Local model test PR-AUC")
    handles, labels = axes[0, -1].get_legend_handles_labels()
    seed_handles, seed_labels = axes[0, 0].get_legend_handles_labels()
    unique = dict(zip(seed_labels + labels, seed_handles + handles))
    fig.legend(unique.values(), unique.keys(), loc="lower center", ncol=4, bbox_to_anchor=(0.5, 0.02))
    fig.suptitle(
        "Client-level local-model test PR-AUC across seeds\n"
        "Each local model is evaluated on the same held-out test set; dashed lines are global FedAvg reference"
    )
    fig.tight_layout(rect=(0, 0.13, 1, 0.92))
    suffix = "iid_moderate_strong" if INCLUDE_STRONG else "iid_moderate"
    path = OUTPUT / f"local_client_test_pr_auc_{suffix}.png"
    OUTPUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_fedavg_client_training_losses() -> Path:
    fig, axes = plt.subplots(6, len(SCENARIOS), figsize=(13, 20), sharex=True, squeeze=False)

    for column, (scenario_label, scenario, run_root) in enumerate(SCENARIOS):
        client_sizes = {}
        seed_rounds = {}
        for seed in SEEDS:
            rounds = load_fedavg_rounds(scenario, run_root, seed)
            seed_rounds[seed] = rounds
            summary_path = RUNS / run_root / scenario / "fedavg_normal_only" / f"seed_{seed}" / "summary.json"
            if not summary_path.is_file():
                raise FileNotFoundError(f"Missing FedAvg summary: {summary_path}")
            with summary_path.open(encoding="utf-8") as handle:
                summary = json.load(handle)
            for client, details in summary["clients"].items():
                client_sizes[client] = int(details["n_train"])

        clients = sorted(client_sizes, key=lambda value: int(value.split("_")[1]))
        for client_position, client in enumerate(clients):
            ax = axes[client_position, column]
            seed_losses: list[np.ndarray] = []
            max_round = max(len(rounds) for rounds in seed_rounds.values())
            for seed in SEEDS:
                rounds = seed_rounds[seed]
                values = np.asarray(
                    [
                        float(round_record["clients"][client]["train_loss"])
                        for round_record in rounds
                        if client in round_record["clients"]
                    ]
                )
                seed_losses.append(values)
                ax.plot(
                    np.arange(1, len(values) + 1),
                    values,
                    color=SEED_COLORS[seed],
                    alpha=0.38,
                    linewidth=1.0,
                    label=f"seed {seed}" if client_position == 0 and column == 0 else None,
                )

            round_grid = np.arange(1, max_round + 1)
            padded = np.full((len(seed_losses), max_round), np.nan)
            for seed_position, values in enumerate(seed_losses):
                padded[seed_position, : len(values)] = values
            mean_loss = np.nanmean(padded, axis=0)
            std_loss = np.nanstd(padded, axis=0, ddof=0)
            valid = np.isfinite(mean_loss)
            ax.plot(round_grid[valid], mean_loss[valid], color="#222222", linewidth=2.2, label="Mean")
            ax.fill_between(
                round_grid[valid],
                mean_loss[valid] - std_loss[valid],
                mean_loss[valid] + std_loss[valid],
                color="#222222",
                alpha=0.16,
                linewidth=0,
            )
            ax.set_title(f"{client} (n={client_sizes[client]})")
            if client_position == len(clients) - 1:
                ax.set_xlabel("Federated round")
            if column == 0:
                ax.set_ylabel("Client train reconstruction loss")
            ax.grid(alpha=0.25)

    for column, (scenario_label, _, _) in enumerate(SCENARIOS):
        client_title = axes[0, column].get_title()
        axes[0, column].set_title(f"{scenario_label}\n{client_title}")

    handles, labels = axes[0, 0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc="upper center", ncol=4, bbox_to_anchor=(0.5, 0.985))
    fig.suptitle(
        "FedAvg client-level training loss by round\n"
        "Thin colored lines: seeds 7, 21, 42; thick black line and band: mean ± SD",
        y=0.999,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.98))
    suffix = "iid_moderate_strong" if INCLUDE_STRONG else "iid_moderate"
    path = OUTPUT / f"fedavg_client_train_loss_{suffix}.png"
    OUTPUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Plot client-level sensor-only experiment results.")
    parser.add_argument(
        "--include-strong",
        action="store_true",
        help="Also include strong quantity-skew results (requires completed runs and evaluations).",
    )
    args = parser.parse_args()
    if args.include_strong:
        SCENARIOS = (*SCENARIOS, STRONG_SCENARIO)
        OUTPUT = ROOT / "results/figures/sensors_only_quantity_skew_strong"
        INCLUDE_STRONG = True
    for figure in (plot_local_client_pr_auc(), plot_fedavg_client_training_losses()):
        print(f"Saved client-behavior plot: {figure.relative_to(ROOT)}")
