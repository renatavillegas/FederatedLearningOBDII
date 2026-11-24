#!/usr/bin/env python
"""Generate official, synchronized ROC and Precision-Recall curves for client-holdout study."""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import (
    average_precision_score,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)

from src.config import load_yaml, resolve_path
from src.evaluation.report import RunSpec, RunEvaluator
from src.experiments.common import ExperimentData

RECALL_GRID = np.linspace(0.0, 1.0, 501)
FPR_GRID = np.linspace(0.0, 1.0, 501)
SEEDS = (7, 21, 42)


def interpolate_precision(labels: np.ndarray, scores: np.ndarray) -> tuple[np.ndarray, float]:
    precision, recall, _ = precision_recall_curve(labels, scores)
    ap = float(average_precision_score(labels, scores))
    order = np.argsort(recall, kind="stable")
    recall_asc = recall[order]
    prec_asc = precision[order]
    uniq_recall, inv = np.unique(recall_asc, return_inverse=True)
    max_prec = np.zeros_like(uniq_recall)
    np.maximum.at(max_prec, inv, prec_asc)
    interpolated = np.interp(RECALL_GRID, uniq_recall, max_prec)
    return interpolated, ap


def interpolate_tpr(labels: np.ndarray, scores: np.ndarray) -> tuple[np.ndarray, float]:
    fpr, tpr, _ = roc_curve(labels, scores)
    roc_auc = float(roc_auc_score(labels, scores))
    interpolated = np.interp(FPR_GRID, fpr, tpr)
    interpolated[0] = 0.0
    interpolated[-1] = 1.0
    return interpolated, roc_auc


def evaluate_method_curves(
    scenario: str,
    method_name: str,
    base_dir: Path,
    data: ExperimentData,
    evaluator: RunEvaluator,
    test_clients: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, float, float, float, float]:
    test_frame = data.partition("test")
    test_labels = data.labels(test_frame)
    kind = "centralized" if "centralized" in method_name else ("local" if "local" in method_name else "federated")
    run_scenario = "all" if kind == "centralized" else scenario

    seed_pr_curves, seed_roc_curves = [], []
    seed_aps, seed_aucs = [], []

    for s in SEEDS:
        run_dir = base_dir / f"seed_{s}"
        run = RunSpec(run_scenario, method_name, s, run_dir)
        views = evaluator._views(run)
        client_pr_curves, client_roc_curves = [], []
        client_aps, client_aucs = [], []

        for view, model, scaler in views:
            feats = data.features(test_frame, scaler)
            with torch.no_grad():
                preds = model(torch.as_tensor(feats, dtype=torch.float32)).numpy()
            errors = np.mean(np.sort((feats - preds) ** 2, axis=1)[:, -6:], axis=1)

            for c in sorted(np.unique(test_clients)):
                if kind == "local" and view != c:
                    continue
                mask = test_clients == c
                pr_c, ap_c = interpolate_precision(test_labels[mask], errors[mask])
                roc_c, auc_c = interpolate_tpr(test_labels[mask], errors[mask])
                client_pr_curves.append(pr_c)
                client_roc_curves.append(roc_c)
                client_aps.append(ap_c)
                client_aucs.append(auc_c)

        seed_pr_curves.append(np.mean(client_pr_curves, axis=0))
        seed_roc_curves.append(np.mean(client_roc_curves, axis=0))
        seed_aps.append(np.mean(client_aps))
        seed_aucs.append(np.mean(client_aucs))

    mean_pr = np.mean(seed_pr_curves, axis=0)
    std_pr = np.std(seed_pr_curves, axis=0, ddof=1)
    mean_roc = np.mean(seed_roc_curves, axis=0)
    std_roc = np.std(seed_roc_curves, axis=0, ddof=1)
    mean_ap = float(np.mean(seed_aps))
    std_ap = float(np.std(seed_aps, ddof=1))
    mean_auc = float(np.mean(seed_aucs))
    std_auc = float(np.std(seed_aucs, ddof=1))

    return mean_pr, std_pr, mean_roc, std_roc, mean_ap, std_ap, mean_auc, std_auc


def plot_official_curves() -> Path:
    model_config = load_yaml(ROOT / "configs/model.yaml")

    data_qty = ExperimentData(ROOT / "configs/data_sensors_client_holdout_quantity.yaml")
    eval_qty = RunEvaluator(data_qty, model_config, score_aggregation="top_5_percent")
    test_clients_qty = (
        data_qty.partition("test")
        .index.map(data_qty.clients.load("quantity_skew_test").set_index("row_id")["client"])
        .to_numpy()
    )

    data_iid = ExperimentData(ROOT / "configs/data_sensors_client_holdout_iid.yaml")
    eval_iid = RunEvaluator(data_iid, model_config, score_aggregation="top_5_percent")
    test_clients_iid = (
        data_iid.partition("test")
        .index.map(data_iid.clients.load("iid_test").set_index("row_id")["client"])
        .to_numpy()
    )

    # Series definitions: (label, scenario, method_name, base_dir, data, evaluator, test_clients, color, linestyle)
    series_defs = [
        (
            "Centralized (Quantity-skew)",
            "quantity_skew",
            "centralized_normal_only_sensors105_client_quantity_skew_latent8",
            ROOT / "results/runs/sensors_only_client_holdout/all/centralized_normal_only_sensors105_client_quantity_skew_latent8",
            data_qty,
            eval_qty,
            test_clients_qty,
            "#222222",
            "-",
        ),
        (
            "FedAvg - IID",
            "iid",
            "fedavg_normal_only",
            ROOT / "results/runs/sensors_only_client_holdout/iid/fedavg_normal_only",
            data_iid,
            eval_iid,
            test_clients_iid,
            "#2a78d6",
            "-",
        ),
        (
            "Local - IID",
            "iid",
            "local_normal_only",
            ROOT / "results/runs/sensors_only_client_holdout/iid/local_normal_only",
            data_iid,
            eval_iid,
            test_clients_iid,
            "#4C72B0",
            ":",
        ),
        (
            "FedAvg - Quantity-skew",
            "quantity_skew",
            "fedavg_normal_only",
            ROOT / "results/runs/sensors_only_client_holdout/quantity_skew/fedavg_normal_only",
            data_qty,
            eval_qty,
            test_clients_qty,
            "#1baf7a",
            "-",
        ),
        (
            "FedProx ($\\mu=0.1$) - Quantity-skew",
            "quantity_skew",
            "fedprox_mu_0.1_normal_only",
            ROOT / "results/runs/sensors_only_client_holdout/quantity_skew/fedprox_mu_0.1_normal_only",
            data_qty,
            eval_qty,
            test_clients_qty,
            "#8f63b8",
            "-.",
        ),
        (
            "Local - Quantity-skew",
            "quantity_skew",
            "local_normal_only",
            ROOT / "results/runs/sensors_only_client_holdout/quantity_skew/local_normal_only",
            data_qty,
            eval_qty,
            test_clients_qty,
            "#eb6834",
            "--",
        ),
    ]

    fig, (roc_ax, pr_ax) = plt.subplots(1, 2, figsize=(14, 6))

    for label, scen, meth, path, data, evaluator, clients, color, ls in series_defs:
        print(f"Evaluating {label}...")
        mean_pr, std_pr, mean_roc, std_roc, m_ap, s_ap, m_auc, s_auc = evaluate_method_curves(
            scen, meth, path, data, evaluator, clients
        )

        roc_label = f"{label}: ROC-AUC {m_auc:.3f} ± {s_auc:.3f}"
        roc_ax.plot(FPR_GRID, mean_roc, label=roc_label, color=color, linestyle=ls, linewidth=1.8)
        roc_ax.fill_between(
            FPR_GRID,
            np.clip(mean_roc - std_roc, 0.0, 1.0),
            np.clip(mean_roc + std_roc, 0.0, 1.0),
            color=color,
            alpha=0.12,
        )

        pr_label = f"{label}: PR-AUC {m_ap:.3f} ± {s_ap:.3f}"
        pr_ax.plot(RECALL_GRID, mean_pr, label=pr_label, color=color, linestyle=ls, linewidth=1.8)
        pr_ax.fill_between(
            RECALL_GRID,
            np.clip(mean_pr - std_pr, 0.0, 1.0),
            np.clip(mean_pr + std_pr, 0.0, 1.0),
            color=color,
            alpha=0.12,
        )

    # Reference lines
    roc_ax.plot([0, 1], [0, 1], linestyle=":", color="#777777", label="No-skill reference")
    roc_ax.set(
        xlim=(0.0, 1.0),
        ylim=(0.0, 1.02),
        xlabel="False Positive Rate (FPR)",
        ylabel="True Positive Rate (Recall)",
    )
    roc_ax.grid(alpha=0.25)
    roc_ax.legend(loc="lower right", fontsize=8.5, frameon=True, facecolor="white", framealpha=0.9)

    pr_ax.axhline(0.5, linestyle=":", color="#777777", label="No-skill prevalence (0.500)")
    pr_ax.set(
        xlim=(0.0, 1.0),
        ylim=(0.0, 1.02),
        xlabel="Recall",
        ylabel="Precision",
    )
    pr_ax.grid(alpha=0.25)
    pr_ax.legend(loc="lower left", fontsize=8.5, frameon=True, facecolor="white", framealpha=0.9)

    fig.tight_layout()

    output_dir = ROOT / "results/figures/sensors_only_client_holdout"
    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / "roc_precision_recall_client_holdout_seeds_7_21_42.png"
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Official figure saved at: {out_path}")
    return out_path


if __name__ == "__main__":
    plot_official_curves()
