import argparse
import json
import re
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.config import load_yaml, resolve_path

SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_SECONDARY = "#52514e"
MUTED = "#898781"
GRIDLINE = "#e1e0d9"
BASELINE = "#c3c2b7"
METHOD_COLORS = {"centralized": "#2a78d6", "local": "#eb6834", "fedavg": "#1baf7a", "fedprox": "#eda100"}
METHOD_ORDER = ("centralized", "local", "fedavg", "fedprox")
SCENARIOS = ("iid", "non_iid")
CENTRALIZED_SCENARIO = "all"
HEADLINE_METRICS = {"test_pr_auc": "PR-AUC", "test_roc_auc": "ROC-AUC", "test_f1": "F1"}
CLIENT_METRICS = {"test_pr_auc": "PR-AUC", "test_roc_auc": "ROC-AUC"}
GROUP_ORDER = (("SPLIT_REAL", 0), ("SPLIT_GRAY", 0), ("SPLIT_GRAY", 1), ("SPLIT_CAT", 1))
FEDPROX_PATTERN = re.compile(r"fedprox_mu_(.+)")


def _read_json(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


class ResultStore:
    """Reads training runs (results/runs) and their evaluations (results/metrics) for one seed."""

    def __init__(self, runs_dir: Path, metrics_dir: Path, seed: int) -> None:
        self.runs_dir = runs_dir
        self.metrics_dir = metrics_dir
        self.seed = seed

    def run_dir(self, scenario: str, method: str) -> Path:
        return self.runs_dir / scenario / method / f"seed_{self.seed}"

    def metrics_path(self, scenario: str, method: str) -> Path:
        return self.metrics_dir / scenario / method / f"seed_{self.seed}"

    def methods(self, scenario: str) -> list[str]:
        directory = self.runs_dir / scenario
        if not directory.exists():
            return []
        return sorted(p.name for p in directory.iterdir() if (p / f"seed_{self.seed}" / "summary.json").exists())

    def run_summary(self, scenario: str, method: str) -> dict[str, Any]:
        return _read_json(self.run_dir(scenario, method) / "summary.json")

    def history(self, scenario: str, method: str) -> dict[str, Any]:
        return _read_json(self.run_dir(scenario, method) / "history.json")

    def metrics_summary(self, scenario: str, method: str) -> dict[str, Any]:
        return _read_json(self.metrics_path(scenario, method) / "summary.json")

    def views(self, scenario: str, method: str) -> pd.DataFrame:
        return pd.read_csv(self.metrics_path(scenario, method) / "views.csv")

    def scores(self, scenario: str, method: str) -> pd.DataFrame:
        return pd.read_csv(self.metrics_path(scenario, method) / "scores.csv")


def select_fedprox_mu(store: ResultStore) -> pd.DataFrame:
    """Per scenario, the FedProx mu with the lowest best validation loss (weighted over client scalers)."""
    rows = []
    for scenario in SCENARIOS:
        for method in store.methods(scenario):
            if method.endswith("_normal_only"):
                continue
            match = FEDPROX_PATTERN.fullmatch(method)
            if match:
                summary = store.run_summary(scenario, method)
                rows.append(
                    {
                        "scenario": scenario,
                        "method": method,
                        "proximal_mu": float(summary["proximal_mu"]),
                        "best_validation_loss": float(summary["best_validation_loss"]),
                        "best_round": summary["best_round"],
                        "rounds_run": summary["rounds_run"],
                    }
                )
    table = pd.DataFrame(rows, columns=["scenario", "method", "proximal_mu", "best_validation_loss", "best_round", "rounds_run"])
    if table.empty:
        return table.assign(selected=pd.Series(dtype=bool))
    best = table.groupby("scenario")["best_validation_loss"].transform("min")
    return table.assign(selected=table["best_validation_loss"] == best).sort_values(["scenario", "proximal_mu"])


def compared_methods(store: ResultStore, selection: pd.DataFrame) -> dict[str, dict[str, tuple[str, str]]]:
    """For each scenario: family -> (scenario directory, method directory) of the runs being compared."""
    selected = selection[selection["selected"]].set_index("scenario")["method"] if not selection.empty else pd.Series(dtype=str)
    compared: dict[str, dict[str, tuple[str, str]]] = {}
    for scenario in SCENARIOS:
        available = store.methods(scenario)
        entries: dict[str, tuple[str, str]] = {}
        if "centralized" in store.methods(CENTRALIZED_SCENARIO):
            entries["centralized"] = (CENTRALIZED_SCENARIO, "centralized")
        for family in ("local", "fedavg"):
            if family in available:
                entries[family] = (scenario, family)
        if scenario in selected.index:
            entries["fedprox"] = (scenario, selected[scenario])
        compared[scenario] = entries
    return compared


def comparison_table(store: ResultStore, compared: dict[str, dict[str, tuple[str, str]]]) -> pd.DataFrame:
    rows = []
    for scenario, entries in compared.items():
        for family, (run_scenario, method) in entries.items():
            if not store.metrics_path(run_scenario, method).joinpath("summary.json").exists():
                continue
            metrics = store.metrics_summary(run_scenario, method)
            run = store.run_summary(run_scenario, method)
            row: dict[str, Any] = {"scenario": scenario, "family": family, "method": method, "n_views": len(metrics["views"])}
            for name, values in metrics["test"].items():
                row[f"{name}_mean"] = values["mean"]
                row[f"{name}_std"] = values["std"]
            row.update(_training_columns(run, family))
            rows.append(row)
    return pd.DataFrame(rows)


def _training_columns(run: dict[str, Any], family: str) -> dict[str, Any]:
    if family in ("fedavg", "fedprox"):
        communication = run["communication"]
        return {
            "best_step": run["best_round"],
            "steps_run": run["rounds_run"],
            "step_unit": "round",
            "stopped_early": run["stopped_early"],
            "total_seconds": run["total_seconds"],
            "parameter_bytes_down": communication["parameter_bytes_down"],
            "parameter_bytes_up": communication["parameter_bytes_up"],
            "scaler_bytes_up": communication["scaler_bytes_up"],
        }
    if family == "local":
        clients = run["clients"].values()
        return {
            "best_step": float(np.mean([c["best_epoch"] for c in clients])),
            "steps_run": float(np.mean([c["epochs_run"] for c in clients])),
            "step_unit": "epoch (mean over clients)",
            "stopped_early": all(c["stopped_early"] for c in clients),
            "total_seconds": float(sum(c["total_seconds"] for c in clients)),
        }
    return {
        "best_step": run["best_epoch"],
        "steps_run": run["epochs_run"],
        "step_unit": "epoch",
        "stopped_early": run["stopped_early"],
        "total_seconds": run["total_seconds"],
    }


def per_client_table(store: ResultStore, compared: dict[str, dict[str, tuple[str, str]]]) -> pd.DataFrame:
    frames = []
    for scenario, entries in compared.items():
        for family, (run_scenario, method) in entries.items():
            if family == "centralized" or not store.metrics_path(run_scenario, method).joinpath("views.csv").exists():
                continue
            views = store.views(run_scenario, method)
            columns = ["view", "threshold_threshold", *[c for c in views.columns if c.startswith("test_")]]
            frames.append(views[columns].assign(scenario=scenario, family=family, method=method))
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


class FigureWriter:
    """Static PNG figures for the report (light surface, fixed method colors)."""

    def __init__(self, directory: Path) -> None:
        self.directory = directory
        self.directory.mkdir(parents=True, exist_ok=True)

    def validation_curves(self, store: ResultStore, compared: dict[str, dict[str, tuple[str, str]]], local_epochs: int) -> None:
        fig, axes = self._figure(len(SCENARIOS), sharey=False)
        for ax, scenario in zip(axes, SCENARIOS):
            for family in METHOD_ORDER:
                if family not in compared[scenario]:
                    continue
                run_scenario, method = compared[scenario][family]
                x, y = self._validation_curve(store, run_scenario, method, family, local_epochs)
                ax.plot(x, y, color=METHOD_COLORS[family], linewidth=2, label=self._label(family, method))
            self._style(ax, title=self._scenario_title(scenario, compared), xlabel="Training epochs (federated: round x local epochs)", ylabel="Validation MSE")
            ax.set_yscale("log")
        self._legend(fig, axes)
        fig.text(0.01, 0.005, "Centralized uses the global scaler; local and federated losses use the client scalers (weighted by training size).",
                 color=MUTED, fontsize=8)
        self._save(fig, "validation_curves.png")

    def metrics_by_method(self, table: pd.DataFrame, compared: dict[str, dict[str, tuple[str, str]]]) -> None:
        fig, axes = self._figure(len(SCENARIOS), sharey=True)
        metrics = list(HEADLINE_METRICS)
        for ax, scenario in zip(axes, SCENARIOS):
            rows = table[table["scenario"] == scenario].set_index("family")
            families = [f for f in METHOD_ORDER if f in rows.index]
            width = 0.8 / max(len(families), 1)
            for position, family in enumerate(families):
                offsets = np.arange(len(metrics)) + (position - (len(families) - 1) / 2) * width
                means = [rows.loc[family, f"{m}_mean"] for m in metrics]
                stds = [rows.loc[family, f"{m}_std"] for m in metrics]
                ax.bar(offsets, means, width=width * 0.9, color=METHOD_COLORS[family], label=self._label(family, rows.loc[family, "method"]),
                       yerr=stds, error_kw={"ecolor": INK_SECONDARY, "elinewidth": 1, "capsize": 2}, edgecolor=SURFACE, linewidth=1)
            ax.axhline(0.5, color=MUTED, linewidth=1, linestyle=":")
            ax.set_xticks(np.arange(len(metrics)), [HEADLINE_METRICS[m] for m in metrics])
            ax.set_ylim(0, 1)
            self._style(ax, title=self._scenario_title(scenario, compared), ylabel="Test value (mean ± std over views)")
        self._legend(fig, axes)
        fig.text(0.01, 0.005, "Dotted line: 0.5 (ROC-AUC of a random score).", color=MUTED, fontsize=8)
        self._save(fig, "metrics_by_method.png")

    def per_client(self, clients: pd.DataFrame, compared: dict[str, dict[str, tuple[str, str]]]) -> None:
        if clients.empty:
            return
        fig, axes = plt.subplots(len(CLIENT_METRICS), len(SCENARIOS), figsize=(11, 6.5), sharey="row", facecolor=SURFACE, squeeze=False)
        for row, (metric, label) in enumerate(CLIENT_METRICS.items()):
            for column, scenario in enumerate(SCENARIOS):
                ax = axes[row][column]
                subset = clients[clients["scenario"] == scenario]
                families = [f for f in METHOD_ORDER if f in set(subset["family"])]
                for position, family in enumerate(families):
                    data = subset[subset["family"] == family].sort_values("view")
                    x = np.arange(len(data)) + (position - (len(families) - 1) / 2) * 0.18
                    ax.plot(x, data[metric], linestyle="none", marker="o", markersize=7, color=METHOD_COLORS[family],
                            markeredgecolor=SURFACE, markeredgewidth=1.5, label=self._label(family, data["method"].iloc[0]))
                    ax.set_xticks(np.arange(len(data)), [v.replace("client_", "C") for v in data["view"]])
                ax.axhline(0.5, color=MUTED, linewidth=1, linestyle=":")
                title = self._scenario_title(scenario, compared) if row == 0 else None
                self._style(ax, title=title, ylabel=f"Test {label}" if column == 0 else None)
        self._legend(fig, [ax for row_axes in axes for ax in row_axes])
        self._save(fig, "per_client.png")

    def score_distributions(self, store: ResultStore, compared: dict[str, dict[str, tuple[str, str]]], categories: pd.Series) -> None:
        panels = [(scenario, family) for scenario in SCENARIOS for family in METHOD_ORDER if family in compared[scenario]]
        if not panels:
            return
        n_columns = len(METHOD_ORDER)
        fig, axes = plt.subplots(len(SCENARIOS), n_columns, figsize=(13, 6), sharex=True, facecolor=SURFACE, squeeze=False)
        group_labels = [f"{category.replace('SPLIT_', '')} {'anomaly' if label else 'normal'}" for category, label in GROUP_ORDER]
        for row, scenario in enumerate(SCENARIOS):
            for column, family in enumerate(METHOD_ORDER):
                ax = axes[row][column]
                if family not in compared[scenario]:
                    ax.set_visible(False)
                    continue
                run_scenario, method = compared[scenario][family]
                path = store.metrics_path(run_scenario, method) / "scores.csv"
                if not path.exists():
                    ax.set_visible(False)
                    continue
                scores = pd.read_csv(path)
                scores = scores[scores["partition"] == "test"].assign(category=lambda f: categories.loc[f["row_id"]].to_numpy())
                data = [scores[(scores["category"] == c) & (scores["label"] == l)]["score"].to_numpy() for c, l in GROUP_ORDER]
                boxes = ax.boxplot(data, orientation="horizontal", widths=0.6, showfliers=False, patch_artist=True)
                for box in boxes["boxes"]:
                    box.set(facecolor=METHOD_COLORS[family], edgecolor=INK_SECONDARY, linewidth=1)
                for element in ("whiskers", "caps"):
                    for line in boxes[element]:
                        line.set(color=INK_SECONDARY, linewidth=1)
                for line in boxes["medians"]:
                    line.set(color=INK, linewidth=1.5)
                ax.set_yticks(np.arange(1, len(GROUP_ORDER) + 1), group_labels if column == 0 else [""] * len(GROUP_ORDER))
                ax.set_xscale("log")
                title = self._label(family, method) if family == "centralized" else f"{self._label(family, method)} - {self._scenario_title(scenario, compared)}"
                self._style(ax, title=title, xlabel="Test reconstruction error" if row == len(SCENARIOS) - 1 else None)
        fig.text(0.01, 0.005, "Scores pooled over the views of each method (local and federated: one view per client scaler). Outliers hidden.",
                 color=MUTED, fontsize=8)
        self._save(fig, "score_distributions.png")

    @staticmethod
    def _validation_curve(store: ResultStore, scenario: str, method: str, family: str, local_epochs: int) -> tuple[np.ndarray, np.ndarray]:
        history = store.history(scenario, method)
        if family in ("fedavg", "fedprox"):
            rounds = history["rounds"]
            return np.array([r["round"] * local_epochs for r in rounds]), np.array([r["validation_loss"] for r in rounds])
        if family == "centralized":
            epochs = history["epochs"]
            return np.array([e["epoch"] for e in epochs]), np.array([e["validation_loss"] for e in epochs])
        summary = store.run_summary(scenario, method)
        curves, weights = [], []
        for client, info in summary["clients"].items():
            client_history = _read_json(store.run_dir(scenario, method) / client / "history.json")["epochs"]
            curves.append(pd.Series([e["validation_loss"] for e in client_history], index=[e["epoch"] for e in client_history]))
            weights.append(info["n_train"])
        frame = pd.concat(curves, axis=1).ffill()
        weighted = (frame * np.array(weights)).sum(axis=1) / float(sum(weights))
        return weighted.index.to_numpy(), weighted.to_numpy()

    @staticmethod
    def _label(family: str, method: str) -> str:
        return {"centralized": "Centralized", "local": "Local", "fedavg": "FedAvg", "fedprox": "FedProx"}[family]

    @staticmethod
    def _scenario_title(scenario: str, compared: dict[str, dict[str, tuple[str, str]]]) -> str:
        title = scenario.upper().replace("_", "-")
        if "fedprox" in compared[scenario]:
            match = FEDPROX_PATTERN.fullmatch(compared[scenario]["fedprox"][1])
            title += f" (FedProx mu = {match.group(1)})"
        return title

    @staticmethod
    def _figure(n_columns: int, sharey: bool):
        fig, axes = plt.subplots(1, n_columns, figsize=(11, 4.5), sharey=sharey, facecolor=SURFACE, squeeze=False)
        return fig, list(axes[0])

    @staticmethod
    def _style(ax, title: str | None = None, xlabel: str | None = None, ylabel: str | None = None) -> None:
        ax.set_facecolor(SURFACE)
        ax.grid(True, axis="y", color=GRIDLINE, linewidth=0.6)
        ax.set_axisbelow(True)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        for side in ("left", "bottom"):
            ax.spines[side].set_color(BASELINE)
        ax.tick_params(colors=MUTED, labelcolor=INK_SECONDARY, labelsize=9)
        if title:
            ax.set_title(title, color=INK, fontsize=11, loc="left")
        if xlabel:
            ax.set_xlabel(xlabel, color=INK_SECONDARY, fontsize=9)
        if ylabel:
            ax.set_ylabel(ylabel, color=INK_SECONDARY, fontsize=9)

    @staticmethod
    def _legend(fig, axes) -> None:
        handles, labels = {}, {}
        for ax in axes:
            for handle, label in zip(*ax.get_legend_handles_labels()):
                if label not in handles:
                    handles[label], labels[label] = handle, label
        if len(handles) >= 2:
            fig.legend(list(handles.values()), list(labels.values()), loc="upper center", ncol=len(handles), frameon=False,
                       labelcolor=INK_SECONDARY, fontsize=9)

    def _save(self, fig, name: str) -> None:
        fig.tight_layout(rect=(0, 0.03, 1, 0.92))
        fig.savefig(self.directory / name, dpi=150, facecolor=SURFACE)
        plt.close(fig)


def run_consolidation(data_config: str, experiments_config: str, seed: int | None) -> dict[str, Any]:
    data = load_yaml(data_config)
    experiments = load_yaml(experiments_config)
    outputs = experiments["outputs"]
    seed = int(seed if seed is not None else experiments["federated"]["fedprox_selection_seed"])
    store = ResultStore(resolve_path(outputs["runs_dir"]), resolve_path(outputs["metrics_dir"]), seed)

    selection = select_fedprox_mu(store)
    compared = compared_methods(store, selection)
    comparison = comparison_table(store, compared)
    clients = per_client_table(store, compared)

    metrics_dir = resolve_path(outputs["metrics_dir"])
    metrics_dir.mkdir(parents=True, exist_ok=True)
    selection.to_csv(metrics_dir / "fedprox_mu_selection.csv", index=False)
    comparison.to_csv(metrics_dir / "comparison.csv", index=False)
    clients.to_csv(metrics_dir / "per_client.csv", index=False)

    index = pd.read_csv(resolve_path(data["outputs"]["splits_dir"]) / "partition_index.csv").set_index("row_id")["category"]
    figures = FigureWriter(resolve_path(outputs["figures_dir"]))
    figures.validation_curves(store, compared, int(experiments["federated"]["local_epochs"]))
    if not comparison.empty:
        figures.metrics_by_method(comparison, compared)
    figures.per_client(clients, compared)
    figures.score_distributions(store, compared, index)
    return {"seed": seed, "compared": compared}


def main() -> None:
    parser = argparse.ArgumentParser(description="Select the FedProx mu and consolidate tables and figures.")
    parser.add_argument("--data-config", default="configs/data.yaml")
    parser.add_argument("--experiments-config", default="configs/experiments.yaml")
    parser.add_argument("--seed", type=int, default=None)
    args = parser.parse_args()
    run_consolidation(args.data_config, args.experiments_config, args.seed)


if __name__ == "__main__":
    main()
