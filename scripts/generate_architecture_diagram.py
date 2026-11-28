#!/usr/bin/env python
"""Generate clean, compact English architecture diagram with 5 server blocks and generic clients."""

from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

ROOT = Path(__file__).resolve().parents[1]
OUTPUT_PATH = ROOT / "ArtigoMO809/figures/federated_architecture.png"


def create_architecture_diagram():
    fig, ax = plt.subplots(figsize=(11.5, 6.2), dpi=250)
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)
    ax.axis("off")

    # Unified Colors
    c_server_bg = "#F8FAFC"
    c_server_border = "#1E40AF"

    # All server inner blocks have the same color
    c_server_block = "#EFF6FF"
    c_server_block_border = "#3B82F6"
    c_server_text = "#1E3A8A"

    # All client blocks have the same color
    c_client_block = "#F1F5F9"
    c_client_border = "#475569"
    c_client_text = "#0F172A"

    # -------------------------------------------------------------
    # 1. Central Server Container
    # -------------------------------------------------------------
    server_box = FancyBboxPatch(
        (32, 2), 36, 96,
        boxstyle="round,pad=1.0,rounding_size=2",
        facecolor=c_server_bg, edgecolor=c_server_border, linewidth=2.0
    )
    ax.add_patch(server_box)
    ax.text(50, 94.5, "FLOWER FEDERATED SERVER", ha="center", va="center",
            fontsize=11.5, fontweight="bold", color="#1E3A8A")

    # 1.1 SuperLink
    p_link = FancyBboxPatch(
        (35, 78), 30, 10,
        boxstyle="round,pad=0.4,rounding_size=1",
        facecolor=c_server_block, edgecolor=c_server_block_border, linewidth=1.4
    )
    ax.add_patch(p_link)
    ax.text(50, 83, "Flower SuperLink", ha="center", va="center",
            fontsize=10.5, fontweight="bold", color=c_server_text)

    # 1.2 ServerApp
    p_app = FancyBboxPatch(
        (35, 60), 30, 10,
        boxstyle="round,pad=0.4,rounding_size=1",
        facecolor=c_server_block, edgecolor=c_server_block_border, linewidth=1.4
    )
    ax.add_patch(p_app)
    ax.text(50, 65, "Flower ServerApp", ha="center", va="center",
            fontsize=10.5, fontweight="bold", color=c_server_text)

    # 1.3 Parameter Aggregation
    p_agg = FancyBboxPatch(
        (35, 42), 30, 10,
        boxstyle="round,pad=0.4,rounding_size=1",
        facecolor=c_server_block, edgecolor=c_server_block_border, linewidth=1.4
    )
    ax.add_patch(p_agg)
    ax.text(50, 47, "FedAvg / FedProx Aggregator", ha="center", va="center",
            fontsize=10, fontweight="bold", color=c_server_text)

    # 1.4 Global Validation
    p_val = FancyBboxPatch(
        (35, 24), 30, 10,
        boxstyle="round,pad=0.4,rounding_size=1",
        facecolor=c_server_block, edgecolor=c_server_block_border, linewidth=1.4
    )
    ax.add_patch(p_val)
    ax.text(50, 29, "Global Validation & Early Stopping", ha="center", va="center",
            fontsize=9.5, fontweight="bold", color=c_server_text)

    # 1.5 Offline Evaluation
    p_eval = FancyBboxPatch(
        (35, 6), 30, 10,
        boxstyle="round,pad=0.4,rounding_size=1",
        facecolor=c_server_block, edgecolor=c_server_block_border, linewidth=1.4
    )
    ax.add_patch(p_eval)
    ax.text(50, 11, "Offline Evaluation", ha="center", va="center",
            fontsize=10, fontweight="bold", color=c_server_text)

    # Vertical internal arrows in Server
    arrow_args = dict(arrowstyle="-|>", mutation_scale=11, color="#475569", lw=1.4)
    ax.add_patch(FancyArrowPatch((50, 78), (50, 70), **arrow_args))
    ax.add_patch(FancyArrowPatch((50, 60), (50, 52), **arrow_args))
    ax.add_patch(FancyArrowPatch((50, 42), (50, 34), **arrow_args))
    ax.add_patch(FancyArrowPatch((50, 24), (50, 16), **arrow_args))

    # -------------------------------------------------------------
    # 2. Clients (Left: 1, 2, 3 | Right: 4, 5, 6) - Purely generic
    # -------------------------------------------------------------
    left_clients = [
        ("Client 1", 74),
        ("Client 2", 49),
        ("Client 3", 24),
    ]

    right_clients = [
        ("Client 4", 74),
        ("Client 5", 49),
        ("Client 6", 24),
    ]

    # Draw left clients
    for title, y_pos in left_clients:
        p_c = FancyBboxPatch(
            (2, y_pos), 23, 11,
            boxstyle="round,pad=0.4,rounding_size=1.2",
            facecolor=c_client_block, edgecolor=c_client_border, linewidth=1.4
        )
        ax.add_patch(p_c)
        ax.text(13.5, y_pos + 5.5, title, ha="center", va="center",
                fontsize=10, fontweight="bold", color=c_client_text)

    # Draw right clients
    for title, y_pos in right_clients:
        p_c = FancyBboxPatch(
            (75, y_pos), 23, 11,
            boxstyle="round,pad=0.4,rounding_size=1.2",
            facecolor=c_client_block, edgecolor=c_client_border, linewidth=1.4
        )
        ax.add_patch(p_c)
        ax.text(86.5, y_pos + 5.5, title, ha="center", va="center",
                fontsize=10, fontweight="bold", color=c_client_text)

    # -------------------------------------------------------------
    # 3. Communication Channels & Interconnects
    # -------------------------------------------------------------
    down_args = dict(arrowstyle="-|>", mutation_scale=9, color="#2563EB", lw=1.3, linestyle="--")
    up_args = dict(arrowstyle="-|>", mutation_scale=9, color="#16A34A", lw=1.3)

    # Connections to Left Clients
    for _, y_pos in left_clients:
        client_y = y_pos + 5.5
        ax.add_patch(FancyArrowPatch((35, 65), (25, client_y + 1.8), **down_args))
        ax.add_patch(FancyArrowPatch((25, client_y - 1.8), (35, 47), **up_args))

    # Connections to Right Clients
    for _, y_pos in right_clients:
        client_y = y_pos + 5.5
        ax.add_patch(FancyArrowPatch((65, 65), (75, client_y + 1.8), **down_args))
        ax.add_patch(FancyArrowPatch((75, client_y - 1.8), (65, 47), **up_args))

    fig.tight_layout()
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTPUT_PATH, dpi=250, bbox_inches="tight")
    plt.close(fig)
    print(f"Clean compact architecture diagram saved at: {OUTPUT_PATH}")


if __name__ == "__main__":
    create_architecture_diagram()
