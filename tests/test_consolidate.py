import json
from pathlib import Path

import pandas as pd
import pytest

from src.evaluation.consolidate import (
    ResultStore,
    comparison_table,
    compared_methods,
    per_client_table,
    select_fedprox_mu,
)

SEED = 42


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload))


def federated_summary(mu: float | None, loss: float) -> dict:
    return {
        "proximal_mu": mu,
        "best_validation_loss": loss,
        "best_round": 10,
        "rounds_run": 20,
        "stopped_early": True,
        "total_seconds": 100.0,
        "communication": {"parameter_bytes_down": 1, "parameter_bytes_up": 2, "scaler_bytes_up": 3},
    }


def metrics_summary(views: list[str], f1: float) -> dict:
    return {"views": views, "test": {"test_f1": {"mean": f1, "std": 0.0, "n_views": len(views)}}}


@pytest.fixture
def store(tmp_path) -> ResultStore:
    runs, metrics = tmp_path / "runs", tmp_path / "metrics"
    losses = {"iid": {0.01: 0.30, 0.1: 0.20}, "non_iid": {0.01: 0.10, 0.1: 0.40}}
    for scenario, grid in losses.items():
        for mu, loss in grid.items():
            method = f"fedprox_mu_{mu:g}"
            write_json(runs / scenario / method / f"seed_{SEED}" / "summary.json", federated_summary(mu, loss))
            write_json(metrics / scenario / method / f"seed_{SEED}" / "summary.json", metrics_summary(["client_1"], mu))
        write_json(
            runs / scenario / "fedprox_mu_0.001_normal_only" / f"seed_{SEED}" / "summary.json",
            federated_summary(0.001, 0.001),
        )
        write_json(runs / scenario / "fedavg" / f"seed_{SEED}" / "summary.json", federated_summary(None, 0.5))
        write_json(metrics / scenario / "fedavg" / f"seed_{SEED}" / "summary.json", metrics_summary(["client_1"], 0.6))
        views = pd.DataFrame({"view": ["client_1", "client_2"], "threshold_threshold": [0.1, 0.2], "test_f1": [0.5, 0.7]})
        (metrics / scenario / "fedavg" / f"seed_{SEED}").mkdir(parents=True, exist_ok=True)
        views.to_csv(metrics / scenario / "fedavg" / f"seed_{SEED}" / "views.csv", index=False)
    write_json(
        runs / "all" / "centralized" / f"seed_{SEED}" / "summary.json",
        {"best_epoch": 5, "epochs_run": 10, "stopped_early": True, "total_seconds": 50.0},
    )
    write_json(metrics / "all" / "centralized" / f"seed_{SEED}" / "summary.json", metrics_summary(["global"], 0.7))
    return ResultStore(runs, metrics, SEED)


def test_select_fedprox_mu(store):
    selection = select_fedprox_mu(store)
    selected = selection[selection["selected"]].set_index("scenario")["proximal_mu"].to_dict()
    assert selected == {"iid": 0.1, "non_iid": 0.01}
    assert len(selection) == 4
    assert not selection["method"].str.endswith("_normal_only").any()


def test_compared_methods(store):
    compared = compared_methods(store, select_fedprox_mu(store))
    assert compared["iid"] == {
        "centralized": ("all", "centralized"),
        "fedavg": ("iid", "fedavg"),
        "fedprox": ("iid", "fedprox_mu_0.1"),
    }
    assert compared["non_iid"]["fedprox"] == ("non_iid", "fedprox_mu_0.01")


def test_comparison_table(store):
    table = comparison_table(store, compared_methods(store, select_fedprox_mu(store)))
    iid = table[table["scenario"] == "iid"].set_index("family")
    assert iid.loc["fedprox", "test_f1_mean"] == pytest.approx(0.1)
    assert iid.loc["centralized", "step_unit"] == "epoch"
    assert iid.loc["fedavg", "parameter_bytes_up"] == 2
    assert iid.loc["fedavg", "n_views"] == 1


def test_per_client_table_skips_centralized(store):
    clients = per_client_table(store, compared_methods(store, select_fedprox_mu(store)))
    assert set(clients["family"]) == {"fedavg"}
    assert len(clients) == 4


def test_empty_selection(tmp_path):
    empty = ResultStore(tmp_path / "runs", tmp_path / "metrics", SEED)
    assert select_fedprox_mu(empty).empty
