import pandas as pd

from scripts.plot_client_holdout_results import (
    _load_matched_method_metrics,
    _method_family,
    plot_data_distribution,
)


def test_method_family_recognizes_fedprox():
    assert _method_family("fedprox_mu_0.1_normal_only") == "fedprox"


def test_method_family_preserves_existing_methods():
    assert (
        _method_family("centralized_normal_only_sensors105_client_quantity_skew_latent8")
        == "centralized"
    )
    assert _method_family("fedavg_normal_only") == "fedavg"
    assert _method_family("local_normal_only") == "local"


def test_fedprox_comparison_loads_matching_seeds_from_per_run_metrics(tmp_path):
    scenario_dir = tmp_path / "quantity_skew"
    scenario_dir.mkdir()
    methods = (
        "centralized_normal_only_sensors105_client_quantity_skew_latent8",
        "fedavg_normal_only",
        "local_normal_only",
        "fedprox_mu_0.1_normal_only",
    )
    for method in methods:
        for seed in (7, 21, 42):
            if method.startswith("fedprox") and seed == 21:
                continue
            run_dir = scenario_dir / method / f"seed_{seed}"
            run_dir.mkdir(parents=True)
            pd.DataFrame(
                {
                    "method": [method],
                    "seed": [seed],
                    "client": ["client_1"],
                    "test_pr_auc": [0.6],
                    "test_roc_auc": [0.6],
                    "test_recall": [0.1],
                    "test_false_positive_rate": [0.01],
                }
            ).to_csv(run_dir / "client_test_views.csv", index=False)
    summary_path = scenario_dir / "summary.csv"
    pd.DataFrame().to_csv(summary_path, index=False)

    metrics = _load_matched_method_metrics(summary_path, "fedprox_mu_0.1_normal_only")

    assert set(metrics["seed"]) == {7, 42}
    assert set(metrics["method"].map(_method_family)) == {
        "centralized",
        "fedavg",
        "local",
        "fedprox",
    }


def test_data_distribution_plot_shows_train_and_test_composition(tmp_path):
    report_dir = tmp_path / "report"
    report_dir.mkdir()
    pd.DataFrame(
        {
            "client": ["client_1", "client_2"],
            "n_normal": [80, 60],
            "n_anomaly": [20, 40],
            "n_normal_real": [70, 55],
            "n_vehicles": [3, 2],
        }
    ).to_csv(report_dir / "client_summary.csv", index=False)
    pd.DataFrame(
        {
            "client": ["client_1", "client_2"],
            "n_normal": [10, 15],
            "n_anomaly": [10, 15],
            "n_test_vehicles": [1, 1],
        }
    ).to_csv(report_dir / "test_client_summary.csv", index=False)

    output = plot_data_distribution(report_dir, tmp_path / "figures", "iid")

    assert output.is_file()
    assert output.stat().st_size > 0
