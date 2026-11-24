#!/usr/bin/env python
"""Generate clean, English-language UMAP error and decision visualizations without top titles."""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import umap

from src.config import load_yaml
from src.data.preprocess import Standardizer
from src.experiments.common import ExperimentData
from src.models.model import build_model


def generate_umap_visualizations():
    data = ExperimentData(ROOT / "configs/data_sensors_client_holdout_iid.yaml")
    model_config = load_yaml(ROOT / "configs/model.yaml")
    run_dir = ROOT / "results/runs/sensors_only_client_holdout/iid/fedavg_normal_only/seed_42"

    scaler = Standardizer.load(run_dir / "scaler.json")
    state = torch.load(run_dir / "model.pt", map_location="cpu")

    model = build_model(model_config, len(scaler.feature_columns), seed=42)
    model.load_state_dict(state)
    model.eval()

    test_frame = data.partition("test")
    labels = data.labels(test_frame)
    feats = data.features(test_frame, scaler)

    with torch.no_grad():
        x_t = torch.as_tensor(feats, dtype=torch.float32)
        latent = model.encoder(x_t).numpy()
        preds = model.decoder(torch.as_tensor(latent, dtype=torch.float32)).numpy()

    errors = np.mean(np.sort((feats - preds) ** 2, axis=1)[:, -6:], axis=1)

    val_norm = data.normal_real_partition("validation")
    val_feats = data.features(val_norm, scaler)
    with torch.no_grad():
        val_preds = model(torch.as_tensor(val_feats, dtype=torch.float32)).numpy()
    val_errors = np.mean(np.sort((val_feats - val_preds) ** 2, axis=1)[:, -6:], axis=1)
    thresh_p95 = float(np.percentile(val_errors, 95))

    tp = int(np.sum((labels == 1) & (errors >= thresh_p95)))
    fp = int(np.sum((labels == 0) & (errors >= thresh_p95)))
    fn = int(np.sum((labels == 1) & (errors < thresh_p95)))
    tn = int(np.sum((labels == 0) & (errors < thresh_p95)))
    total = len(labels)

    prec = tp / (tp + fp) * 100
    rec = tp / (tp + fn) * 100
    f1 = 2 * prec * rec / (prec + rec)
    fpr = fp / (fp + tn) * 100

    print(f"Fitting UMAP on {len(latent)} test latent vectors...")
    reducer = umap.UMAP(n_neighbors=15, min_dist=0.1, random_state=42)
    embedding = reducer.fit_transform(latent)

    # Output directory
    out_dir = ROOT / "ArtigoMO809/figures"
    out_dir.mkdir(parents=True, exist_ok=True)

    # -------------------------------------------------------------
    # 1. Continuous Reconstruction Error Plot
    # -------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(7.5, 6), dpi=200)
    sc = ax.scatter(
        embedding[:, 0],
        embedding[:, 1],
        c=errors,
        cmap="inferno",
        s=14,
        alpha=0.85,
        edgecolors="none",
        rasterized=True,
    )
    ax.axis("off")

    cbar = fig.colorbar(sc, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Reconstruction Error (Top 5% MSE)", fontsize=10)
    cbar.ax.axhline(thresh_p95, color="#0ea5e9", linestyle="--", linewidth=2)
    cbar.ax.text(
        1.15,
        thresh_p95,
        f"p95 Threshold\n({thresh_p95:.2f})",
        color="#0284c7",
        fontweight="bold",
        va="center",
        fontsize=8.5,
        transform=cbar.ax.get_yaxis_transform(),
    )

    info_box = (
        f"Seed: 42 | Metric: Top 5% MSE\n"
        f"Threshold (p95): {thresh_p95:.2f}\n"
        f"Test Confusion Matrix:\n"
        f"  TP: {tp} | FP: {fp}\n"
        f"  FN: {fn} | TN: {tn}\n"
        f"Recall: {rec:.1f}% | Precision: {prec:.1f}%\n"
        f"FPR: {fpr:.1f}%   | F1-score: {f1:.1f}%"
    )
    ax.text(
        0.03,
        0.97,
        info_box,
        transform=ax.transAxes,
        fontsize=8,
        va="top",
        bbox=dict(boxstyle="round,pad=0.5", facecolor="white", edgecolor="#cbd5e1", alpha=0.92),
    )

    fig.tight_layout()
    p95_path = out_dir / "fedavg_umap_errors_p95.png"
    fig.savefig(p95_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {p95_path}")

    # -------------------------------------------------------------
    # 2. Operational Decision (TP, FP, TN, FN) Plot
    # -------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(7.5, 6), dpi=200)

    mask_fn = (labels == 1) & (errors < thresh_p95)
    mask_fp = (labels == 0) & (errors >= thresh_p95)
    mask_tn = (labels == 0) & (errors < thresh_p95)
    mask_tp = (labels == 1) & (errors >= thresh_p95)

    ax.scatter(
        embedding[mask_tn, 0],
        embedding[mask_tn, 1],
        c="#3b82f6",
        label=f"True Negative (TN): {tn} ({tn/total*100:.1f}%)",
        s=14,
        alpha=0.6,
        edgecolors="none",
        rasterized=True,
    )
    ax.scatter(
        embedding[mask_fn, 0],
        embedding[mask_fn, 1],
        c="#94a3b8",
        label=f"False Negative (FN): {fn} ({fn/total*100:.1f}%)",
        s=14,
        alpha=0.6,
        edgecolors="none",
        rasterized=True,
    )
    ax.scatter(
        embedding[mask_tp, 0],
        embedding[mask_tp, 1],
        c="#ef4444",
        label=f"True Positive (TP): {tp} ({tp/total*100:.1f}%)",
        s=16,
        alpha=0.85,
        edgecolors="none",
        rasterized=True,
    )
    ax.scatter(
        embedding[mask_fp, 0],
        embedding[mask_fp, 1],
        c="#f59e0b",
        label=f"False Positive (FP): {fp} ({fp/total*100:.1f}%)",
        s=18,
        alpha=0.9,
        edgecolors="none",
        rasterized=True,
    )

    ax.axis("off")
    ax.legend(
        loc="upper right",
        fontsize=8.5,
        frameon=True,
        facecolor="white",
        edgecolor="#cbd5e1",
        framealpha=0.92,
    )

    fig.tight_layout()
    decision_path = out_dir / "fedavg_umap_p95_decision.png"
    fig.savefig(decision_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {decision_path}")


if __name__ == "__main__":
    generate_umap_visualizations()
