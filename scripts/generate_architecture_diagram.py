#!/usr/bin/env python
"""Generate high-resolution English architecture diagram for the SBC paper."""

from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, ArrowStyle, FancyArrowPatch

ROOT = Path(__file__).resolve().parents[1]
OUTPUT_PATH = ROOT / "ArtigoMO809/figures/federated_architecture.png"


def create_architecture_diagram():
    fig, ax = plt.subplots(figsize=(15, 9.2), dpi=250)
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)
    ax.axis("off")

    # Colors
    c_server_bg = "#F8FAFC"
    c_server_border = "#3B82F6"
    c_superlink = "#DBEAFE"
    c_superlink_border = "#2563EB"
    c_serverapp = "#FFFFFF"
    c_serverapp_border = "#64748B"
    c_agg = "#DCFCE7"
    c_agg_border = "#16A34A"
    c_val = "#FEF3C7"
    c_val_border = "#D97706"
    c_chkpt = "#FFEDD5"
    c_chkpt_border = "#EA580C"
    c_eval = "#F3E8FF"
    c_eval_border = "#9333EA"

    c_sim_client = "#F1F5F9"
    c_sim_border = "#64748B"
    c_edge_client = "#FEF2F2"
    c_edge_border = "#DC2626"

    # -------------------------------------------------------------
    # 1. Central Server Container
    # -------------------------------------------------------------
    server_box = FancyBboxPatch(
        (33, 15), 34, 76,
        boxstyle="round,pad=1.2,rounding_size=2",
        facecolor=c_server_bg, edgecolor=c_server_border, linewidth=2.2
    )
    ax.add_patch(server_box)
    ax.text(50, 88.5, "FLOWER FEDERATED SERVER", ha="center", va="center",
            fontsize=13, fontweight="bold", color="#1E3A8A")

    # 1.1 SuperLink
    p_link = FancyBboxPatch(
        (35.5, 78), 29, 7.5,
        boxstyle="round,pad=0.5,rounding_size=1",
        facecolor=c_superlink, edgecolor=c_superlink_border, linewidth=1.5
    )
    ax.add_patch(p_link)
    ax.text(50, 82.5, "Flower SuperLink", ha="center", va="center",
            fontsize=10.5, fontweight="bold", color="#1E40AF")
    ax.text(50, 79.5, "Central communication broker & node discovery (gRPC)", ha="center", va="center",
            fontsize=8, color="#334155")

    # 1.2 ServerApp
    p_app = FancyBboxPatch(
        (35.5, 65.5), 29, 9,
        boxstyle="round,pad=0.5,rounding_size=1",
        facecolor=c_serverapp, edgecolor=c_serverapp_border, linewidth=1.4
    )
    ax.add_patch(p_app)
    ax.text(50, 71.5, "Flower ServerApp", ha="center", va="center",
            fontsize=10.5, fontweight="bold", color="#0F172A")
    ax.text(50, 68.5, "Global model initialization & round orchestration\n(5 local epochs, learning rate 0.001, proximal $\\mu$)",
            ha="center", va="center", fontsize=7.5, color="#475569")

    # 1.3 Parameter Aggregation
    p_agg = FancyBboxPatch(
        (35.5, 49), 29, 12.5,
        boxstyle="round,pad=0.5,rounding_size=1",
        facecolor=c_agg, edgecolor=c_agg_border, linewidth=1.6
    )
    ax.add_patch(p_agg)
    ax.text(50, 58, "Federated Parameter Aggregation", ha="center", va="center",
            fontsize=10.5, fontweight="bold", color="#14532D")
    ax.text(50, 54, "FedAvg: sample-weighted model averaging\nFedProx: proximal regularization against client drift\nModel size: 7,409 parameters (29.64 KB)",
            ha="center", va="center", fontsize=7.5, color="#166534")

    # 1.4 Global Validation
    p_val = FancyBboxPatch(
        (35.5, 33.5), 29, 11,
        boxstyle="round,pad=0.5,rounding_size=1",
        facecolor=c_val, edgecolor=c_val_border, linewidth=1.5
    )
    ax.add_patch(p_val)
    ax.text(50, 41, "Global Server Validation & Early Stopping", ha="center", va="center",
            fontsize=10, fontweight="bold", color="#92400E")
    ax.text(50, 36.5, "Reconstruction MSE evaluated on pooled normal records\nPatience = 10 rounds without validation improvement\nCalibrates operational decision threshold (P-99 / P-95)",
            ha="center", va="center", fontsize=7.5, color="#78350F")

    # 1.5 Best Checkpoint
    p_chkpt = FancyBboxPatch(
        (35.5, 18), 29, 11,
        boxstyle="round,pad=0.5,rounding_size=1",
        facecolor=c_chkpt, edgecolor=c_chkpt_border, linewidth=1.5
    )
    ax.add_patch(p_chkpt)
    ax.text(50, 25.5, "Optimal Global Checkpoint ($w^*$)", ha="center", va="center",
            fontsize=10, fontweight="bold", color="#9A3412")
    ax.text(50, 21.5, "Preserves lowest validation loss weights across 200 rounds\nExports model parameters for edge deployment",
            ha="center", va="center", fontsize=7.5, color="#7C2D12")

    # 1.6 Offline Final Evaluation
    p_eval = FancyBboxPatch(
        (35.5, 3), 29, 8.5,
        boxstyle="round,pad=0.5,rounding_size=1",
        facecolor=c_eval, edgecolor=c_eval_border, linewidth=1.6
    )
    ax.add_patch(p_eval)
    ax.text(50, 8.5, "OFFLINE CLIENT-HOLDOUT TEST EVALUATION", ha="center", va="center",
            fontsize=9.5, fontweight="bold", color="#581C87")
    ax.text(50, 5.2, "Evaluated on held-out test cohort of each client (50% normal / 50% anomaly)\nROC-AUC, PR-AUC, Recall, FPR, F1-score",
            ha="center", va="center", fontsize=7.5, color="#6B21A8")

    # Vertical internal arrows in Server
    arrow_args = dict(arrowstyle="-|>", mutation_scale=12, color="#475569", lw=1.5)
    ax.add_patch(FancyArrowPatch((50, 78), (50, 74.5), **arrow_args))
    ax.add_patch(FancyArrowPatch((50, 65.5), (50, 61.5), **arrow_args))
    ax.add_patch(FancyArrowPatch((50, 49), (50, 44.5), **arrow_args))
    ax.add_patch(FancyArrowPatch((50, 33.5), (50, 29), **arrow_args))
    ax.add_patch(FancyArrowPatch((50, 18), (50, 11.5), **arrow_args))

    # -------------------------------------------------------------
    # 2. Clients (Left: 1, 2, 3 | Right: 4, 5, 6)
    # -------------------------------------------------------------
    left_clients = [
        ("CLIENT 1", "Simulated Node\nNormal-only real training\nAutoencoder ($105\\rightarrow 32\\rightarrow 8$)", 72, False),
        ("CLIENT 2", "Simulated Node (or Edge)\nLow volume: 253 train samples\nHeld-out test vehicle live9", 50, False),
        ("CLIENT 3", "Simulated Node\nMedium volume: 643 train samples\nNormal-only local optimization", 28, False),
    ]

    right_clients = [
        ("CLIENT 4", "PHYSICAL EDGE NODE\nRaspberry Pi 5 (Cortex-A76)\nMedium volume: 612 train samples\nLocal Flower SuperNode", 70, True),
        ("CLIENT 5", "Simulated Node\nHigh volume: 971 train samples\nNormal-only local optimization", 48, False),
        ("CLIENT 6", "Simulated Node (or Edge)\nHigh volume: 993 train samples\nHeld-out test cohort (702 samples)", 26, False),
    ]

    # Draw left clients
    for title, desc, y_pos, is_edge in left_clients:
        bg = c_edge_client if is_edge else c_sim_client
        bd = c_edge_border if is_edge else c_sim_border
        p_c = FancyBboxPatch(
            (2, y_pos), 22, 16.5,
            boxstyle="round,pad=0.5,rounding_size=1.5",
            facecolor=bg, edgecolor=bd, linewidth=1.8 if is_edge else 1.3
        )
        ax.add_patch(p_c)
        ax.text(13, y_pos + 13.5, title, ha="center", va="center",
                fontsize=10.5, fontweight="bold", color="#991B1B" if is_edge else "#0F172A")
        ax.text(13, y_pos + 7.5, desc, ha="center", va="center",
                fontsize=7.5, color="#334155")

    # Draw right clients
    for title, desc, y_pos, is_edge in right_clients:
        bg = c_edge_client if is_edge else c_sim_client
        bd = c_edge_border if is_edge else c_sim_border
        p_c = FancyBboxPatch(
            (76, y_pos), 22, 18.5 if is_edge else 16.5,
            boxstyle="round,pad=0.5,rounding_size=1.5",
            facecolor=bg, edgecolor=bd, linewidth=2.0 if is_edge else 1.3
        )
        ax.add_patch(p_c)
        ax.text(87, y_pos + (15.5 if is_edge else 13.5), title, ha="center", va="center",
                fontsize=10.5, fontweight="bold", color="#991B1B" if is_edge else "#0F172A")
        ax.text(87, y_pos + (8.5 if is_edge else 7.5), desc, ha="center", va="center",
                fontsize=7.5, color="#334155")

    # -------------------------------------------------------------
    # 3. Communication Channels & Interconnects
    # -------------------------------------------------------------
    # Downlink: Blue arrows (Server -> Clients)
    # Uplink: Green arrows (Clients -> Aggregator)
    down_args = dict(arrowstyle="-|>", mutation_scale=10, color="#2563EB", lw=1.3, linestyle="--")
    up_args = dict(arrowstyle="-|>", mutation_scale=10, color="#16A34A", lw=1.3)

    # Connections to Left Clients
    for _, _, y_pos, _ in left_clients:
        client_y = y_pos + 8.5
        # Downlink from ServerApp (50, 67) to Client
        ax.add_patch(FancyArrowPatch((35.5, 70), (24, client_y + 2), **down_args))
        # Uplink from Client to Aggregator (35.5, 55)
        ax.add_patch(FancyArrowPatch((24, client_y - 2), (35.5, 55), **up_args))

    # Connections to Right Clients
    for _, _, y_pos, is_edge in right_clients:
        client_y = y_pos + (9.5 if is_edge else 8.5)
        # Downlink from ServerApp (64.5, 70) to Client
        ax.add_patch(FancyArrowPatch((64.5, 70), (76, client_y + 2), **down_args))
        # Uplink from Client to Aggregator (64.5, 55)
        ax.add_patch(FancyArrowPatch((76, client_y - 2), (64.5, 55), **up_args))

    # Legend for Communication Flow
    p_leg = FancyBboxPatch(
        (2, 2.5), 30, 8,
        boxstyle="round,pad=0.4,rounding_size=1",
        facecolor="#FFFFFF", edgecolor="#CBD5E1", linewidth=1.2
    )
    ax.add_patch(p_leg)
    ax.text(3.5, 7.5, "COMMUNICATION FLOW PER ROUND:", fontsize=8, fontweight="bold", color="#0F172A")
    ax.plot([3.5, 8.5], [5.2, 5.2], color="#2563EB", lw=1.5, linestyle="--")
    ax.text(9.5, 5.2, "Downlink: global weights ($w^t$, 29.64 KB)", fontsize=7.5, va="center", color="#1E40AF")
    ax.plot([3.5, 8.5], [3.6, 3.6], color="#16A34A", lw=1.5)
    ax.text(9.5, 3.6, "Uplink: local weights update ($w_k^{t+1}$, 29.64 KB)", fontsize=7.5, va="center", color="#15803D")

    # Legend for Node Types
    p_node_leg = FancyBboxPatch(
        (68, 2.5), 30, 8,
        boxstyle="round,pad=0.4,rounding_size=1",
        facecolor="#FFFFFF", edgecolor="#CBD5E1", linewidth=1.2
    )
    ax.add_patch(p_node_leg)
    ax.text(69.5, 7.5, "NODE DEPLOYMENT SCHEME:", fontsize=8, fontweight="bold", color="#0F172A")
    ax.scatter([71], [5.2], c="#DC2626", s=60, marker="s")
    ax.text(73, 5.2, "Physical Edge Device (Raspberry Pi 5)", fontsize=7.5, va="center", color="#991B1B", fontweight="bold")
    ax.scatter([71], [3.6], c="#64748B", s=60, marker="s")
    ax.text(73, 3.6, "Simulated Federated Nodes (Local Host)", fontsize=7.5, va="center", color="#334155")

    fig.tight_layout()
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTPUT_PATH, dpi=250, bbox_inches="tight")
    plt.close(fig)
    print(f"English architecture diagram successfully saved at: {OUTPUT_PATH}")


if __name__ == "__main__":
    create_architecture_diagram()
