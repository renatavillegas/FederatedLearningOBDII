from src.evaluation.client_test import (
    _is_default_client_test_method,
    _is_client_holdout_federated_method,
    _matches_method_filter,
)


def test_client_test_method_filter_accepts_exact_method():
    method = "fedprox_mu_0.01_normal_only"
    assert _matches_method_filter(method, method)
    assert not _matches_method_filter("fedprox_mu_0.1_normal_only", method)
    assert _matches_method_filter(method, None)


def test_client_holdout_includes_normal_only_fedprox_methods():
    assert _is_client_holdout_federated_method("fedprox_mu_0.001_normal_only")
    assert _is_client_holdout_federated_method("fedprox_mu_1_normal_only")


def test_client_holdout_excludes_other_federated_methods():
    assert _is_client_holdout_federated_method("local_normal_only")
    assert _is_client_holdout_federated_method("fedavg_normal_only")
    assert not _is_client_holdout_federated_method("fedprox_mu_0.01")
    assert not _is_client_holdout_federated_method("fedprox_mu_0.01_anomalies")


def test_default_client_test_methods_exclude_fedprox_grid_candidates():
    assert _is_default_client_test_method("local_normal_only")
    assert _is_default_client_test_method("fedavg_normal_only")
    assert not _is_default_client_test_method("fedprox_mu_0.1_normal_only")
