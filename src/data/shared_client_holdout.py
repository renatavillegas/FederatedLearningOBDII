import argparse
import json
import zlib
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.config import load_yaml, resolve_path
from src.data.client_holdout import _client_partition_summary
from src.data.clients import ClientDistributionReporter, ClientStore, ScenarioSpec
from src.data.preprocess import FeatureSelector, build_scaler
from src.data.schema import DataSchema
from src.data.split import (
    GroupPartitioner,
    PartitionRatios,
    PartitionStore,
    PartitionValidator,
    allocate_counts,
    build_group_manifest,
)


def _allocate_vehicle_groups(
    group_sizes: pd.Series,
    clients: list[str],
    weights: np.ndarray,
    rng: np.random.Generator,
) -> dict[str, str]:
    if len(group_sizes) < 2 * len(clients):
        raise ValueError(
            "Shared client holdout requires at least two training vehicles per client"
        )
    shares = weights / weights.sum()
    groups = group_sizes.index.to_numpy(dtype=object).copy()
    rng.shuffle(groups)
    remaining = sorted(groups, key=lambda group: int(group_sizes[group]))
    assigned: dict[str, str] = {}
    totals = dict.fromkeys(clients, 0)
    targets = {
        client: float(share) * float(group_sizes.sum())
        for client, share in zip(clients, shares)
    }

    for client in sorted(clients, key=lambda name: shares[clients.index(name)]):
        for _ in range(2):
            group = remaining.pop(0)
            assigned[str(group)] = client
            totals[client] += int(group_sizes[group])

    for group in reversed(remaining):
        client = max(clients, key=lambda name: targets[name] - totals[name])
        assigned[str(group)] = client
        totals[client] += int(group_sizes[group])
    return assigned


def _assign_groups_evenly(
    groups: list[str],
    clients: list[str],
    rng: np.random.Generator,
) -> dict[str, str]:
    if len(groups) < len(clients):
        raise ValueError(
            f"Need at least {len(clients)} vehicle groups for held-out clients; "
            f"found {len(groups)}"
        )
    shuffled = np.asarray(sorted(groups), dtype=object)
    rng.shuffle(shuffled)
    counts = _allocate_proportionally(len(shuffled), [1] * len(clients))
    result: dict[str, str] = {}
    start = 0
    for client, count in zip(clients, counts):
        result.update({str(group): client for group in shuffled[start : start + count]})
        start += count
    return result


def _rng(seed: int, name: str) -> np.random.Generator:
    return np.random.default_rng([seed, zlib.crc32(name.encode())])


def _allocate_proportionally(total: int, weights: list[int] | np.ndarray) -> list[int]:
    values = np.asarray(weights, dtype=float)
    if not np.isfinite(values).all() or (values < 0).any() or values.sum() <= 0:
        raise ValueError(f"Allocation weights must be finite, non-negative and nonzero: {weights}")
    return allocate_counts(total, values / values.sum())


def build_shared_client_holdout_splits(
    frame: pd.DataFrame,
    schema: DataSchema,
    scenario_weights: dict[str, list[float]],
    ratios: PartitionRatios,
    n_clients: int,
    seed: int,
) -> tuple[pd.DataFrame, dict[str, dict[str, pd.DataFrame]]]:
    """Build one vehicle-level split and scenario-specific training assignments."""
    if not scenario_weights:
        raise ValueError("At least one client scenario is required")
    clients = [f"client_{position + 1}" for position in range(n_clients)]
    for scenario, weights in scenario_weights.items():
        if len(weights) != n_clients or any(
            not np.isfinite(weight) or weight <= 0 for weight in weights
        ):
            raise ValueError(f"{scenario} must have {n_clients} finite positive weights")

    categories = schema.assign_categories(frame[schema.source_column])
    partition_index = GroupPartitioner(schema, ratios, seed).assign(frame)
    partition_index["category"] = categories.loc[
        partition_index["row_id"]
    ].to_numpy()
    live_categories = {
        category.name for category in schema.categories if category.family == "live_real"
    }
    labels = frame[schema.label_column]
    real_rows = partition_index["category"].isin(live_categories)
    if not labels.loc[partition_index.loc[real_rows, "row_id"]].eq(
        schema.normal_label
    ).all():
        raise ValueError("The shared holdout protocol expects real vehicles to be normal")

    vehicle_partition = (
        partition_index.loc[real_rows]
        .drop_duplicates("group_id")
        .set_index("group_id")["partition"]
    )
    vehicle_sizes = (
        partition_index.loc[real_rows].groupby("group_id")["row_id"].size()
    )
    fixed_assignments: dict[str, str] = {}
    for partition in ("validation", "test"):
        groups = vehicle_partition[vehicle_partition == partition].index.tolist()
        fixed_assignments.update(
            _assign_groups_evenly(
                groups,
                clients,
                _rng(seed, f"shared-{partition}-vehicles"),
            )
        )

    scenario_vehicle_assignments: dict[str, dict[str, str]] = {}
    train_groups = vehicle_partition[vehicle_partition == "train"].index
    train_sizes = vehicle_sizes.loc[train_groups]
    for scenario, weights in scenario_weights.items():
        scenario_rng = _rng(seed, f"shared-{scenario}-train-vehicles")
        scenario_vehicle_assignments[scenario] = {
            **fixed_assignments,
            **_allocate_vehicle_groups(
                train_sizes,
                clients,
                np.asarray(weights, dtype=float),
                scenario_rng,
            ),
        }

    real_normal_counts: dict[str, dict[str, int]] = {}
    for scenario, group_to_client in scenario_vehicle_assignments.items():
        counts = {partition: dict.fromkeys(clients, 0) for partition in ("train", "validation", "test")}
        for group, group_partition in vehicle_partition.items():
            client = group_to_client[str(group)]
            counts[group_partition][client] += int(vehicle_sizes[group])
        real_normal_counts[scenario] = counts

    partition_names = ("train", "validation", "test")
    baseline_counts = real_normal_counts[next(iter(scenario_weights))]
    for scenario in scenario_weights:
        for partition in ("validation", "test"):
            if real_normal_counts[scenario][partition] != baseline_counts[partition]:
                raise RuntimeError("Held-out real-vehicle counts differ between scenarios")

    partition_rows: dict[str, dict[int, str]] = {
        scenario: {} for scenario in scenario_weights
    }
    mapping_rows: dict[str, dict[str, dict[int, str]]] = {
        scenario: {partition: {} for partition in partition_names}
        for scenario in scenario_weights
    }
    for scenario, group_to_client in scenario_vehicle_assignments.items():
        for row in partition_index.loc[real_rows].itertuples(index=False):
            partition_rows[scenario][int(row.row_id)] = str(row.partition)
            mapping_rows[scenario][str(row.partition)][int(row.row_id)] = group_to_client[
                str(row.group_id)
            ]

    synthetic = partition_index.loc[~real_rows]
    normal_rows = synthetic.loc[
        labels.loc[synthetic["row_id"]].to_numpy() == schema.normal_label, "row_id"
    ].to_numpy(copy=True)
    anomaly_rows = synthetic.loc[
        labels.loc[synthetic["row_id"]].to_numpy() == schema.anomaly_label, "row_id"
    ].to_numpy(copy=True)
    total_real_by_partition = {
        partition: sum(baseline_counts[partition].values())
        for partition in partition_names
    }
    normal_rng = _rng(seed, "shared-synthetic-normal-partitions")
    normal_rng.shuffle(normal_rows)
    normal_partition_sizes = _allocate_proportionally(
        len(normal_rows), [total_real_by_partition[p] for p in partition_names]
    )
    normal_by_partition: dict[str, np.ndarray] = {}
    start = 0
    for partition, count in zip(partition_names, normal_partition_sizes):
        normal_by_partition[partition] = normal_rows[start : start + count]
        start += count

    target_anomalies_by_partition = {
        partition: total_real_by_partition[partition] + len(normal_by_partition[partition])
        for partition in partition_names
    }
    if sum(target_anomalies_by_partition.values()) != len(anomaly_rows):
        raise ValueError(
            "Cannot create balanced shared validation/test cohorts: total synthetic "
            "anomaly count does not equal total normal count"
        )
    anomaly_rng = _rng(seed, "shared-synthetic-anomaly-partitions")
    anomaly_rng.shuffle(anomaly_rows)
    anomaly_partition_sizes = [
        target_anomalies_by_partition[partition] for partition in partition_names
    ]
    anomaly_by_partition: dict[str, np.ndarray] = {}
    start = 0
    for partition, count in zip(partition_names, anomaly_partition_sizes):
        anomaly_by_partition[partition] = anomaly_rows[start : start + count]
        start += count

    for scenario in scenario_weights:
        for partition in partition_names:
            normal_ids = normal_by_partition[partition].copy()
            synthetic_rng = _rng(seed, f"shared-{partition}-synthetic-normal")
            synthetic_rng.shuffle(normal_ids)
            normal_client_counts = real_normal_counts[scenario][partition]
            normal_allocations = _allocate_proportionally(
                len(normal_ids), [normal_client_counts[client] for client in clients]
            )
            synthetic_normal_by_client: dict[str, int] = {}
            start = 0
            for client, count in zip(clients, normal_allocations):
                selected = normal_ids[start : start + count]
                start += count
                synthetic_normal_by_client[client] = count
                for row_id in selected:
                    mapping_rows[scenario][partition][int(row_id)] = client
                    partition_rows[scenario][int(row_id)] = partition

            anomaly_ids = anomaly_by_partition[partition].copy()
            anomaly_rng = _rng(seed, f"shared-{partition}-synthetic-anomaly")
            anomaly_rng.shuffle(anomaly_ids)
            target_by_client = [
                normal_client_counts[client] + synthetic_normal_by_client[client]
                for client in clients
            ]
            anomaly_allocations = _allocate_proportionally(
                len(anomaly_ids), target_by_client
            )
            start = 0
            for client, count in zip(clients, anomaly_allocations):
                selected = anomaly_ids[start : start + count]
                start += count
                for row_id in selected:
                    mapping_rows[scenario][partition][int(row_id)] = client
                    partition_rows[scenario][int(row_id)] = partition

    expected_rows = set(frame.index)
    split_index = partition_index.copy()
    split_index["partition"] = split_index["row_id"].map(
        partition_rows[next(iter(scenario_weights))]
    )
    if split_index["partition"].isna().any() or set(split_index["row_id"]) != expected_rows:
        raise RuntimeError("Shared split failed to assign every data row to one partition")

    assignments: dict[str, dict[str, pd.DataFrame]] = {}
    for scenario in scenario_weights:
        assignments[scenario] = {}
        for partition in partition_names:
            mapping = mapping_rows[scenario][partition]
            expected_partition_rows = set(
                split_index.loc[split_index["partition"] == partition, "row_id"]
            )
            if set(mapping) != expected_partition_rows:
                raise RuntimeError(
                    f"{scenario}/{partition} assignment does not cover its partition"
                )
            assignments[scenario][partition] = pd.DataFrame(
                {
                    "row_id": list(mapping),
                    "client": list(mapping.values()),
                    "partition": partition,
                }
            ).sort_values("row_id").reset_index(drop=True)

    reference = assignments[next(iter(scenario_weights))]
    for scenario in list(scenario_weights)[1:]:
        for partition in ("validation", "test"):
            left = reference[partition].set_index("row_id")["client"].sort_index()
            right = assignments[scenario][partition].set_index("row_id")["client"].sort_index()
            if not left.equals(right):
                raise RuntimeError(
                    f"Shared {partition} row-to-client assignment differs in {scenario}"
                )

    return split_index.sort_values("row_id").reset_index(drop=True), assignments


def run_shared_client_holdout(
    data_config_path: str | Path,
    experiments_config_path: str | Path,
    overwrite: bool = False,
) -> dict[str, Any]:
    data_config = load_yaml(data_config_path)
    experiments = load_yaml(experiments_config_path)
    schema = DataSchema.from_config(data_config)
    frame = pd.read_csv(resolve_path(data_config["raw_path"]))
    scenarios = experiments["scenarios"]
    weights = {
        name: [float(value) for value in config["normal_count_weights"]]
        if "normal_count_weights" in config
        else [1.0] * int(experiments["n_clients"])
        for name, config in scenarios.items()
    }
    ratios = PartitionRatios(**data_config["split"]["ratios"])
    split_index, assignments = build_shared_client_holdout_splits(
        frame,
        schema,
        weights,
        ratios,
        int(experiments["n_clients"]),
        int(data_config["seed"]),
    )

    outputs = data_config["outputs"]
    split_dir = resolve_path(outputs["splits_dir"])
    expected_paths = [
        split_dir / "partition_index.csv",
        *[
            split_dir / f"clients_{scenario}{suffix}.csv"
            for scenario in scenarios
            for suffix in ("", "_validation", "_test")
        ],
    ]
    existing = [path for path in expected_paths if path.exists()]
    if existing and not overwrite:
        raise FileExistsError(
            f"Shared split outputs already exist under {split_dir}; "
            "inspect them or pass --overwrite to replace this dedicated split"
        )

    validator = PartitionValidator(schema, data_config["split"]["required_conditions"])
    report = validator.report(
        split_index, frame, schema.candidate_feature_columns(frame)
    )
    validator.validate(report)
    PartitionStore(outputs["splits_dir"]).save(split_index)
    client_store = ClientStore(outputs["splits_dir"])
    for scenario, partition_assignments in assignments.items():
        for partition, suffix in (
            ("train", ""),
            ("validation", "_validation"),
            ("test", "_test"),
        ):
            client_store.save(
                f"{scenario}{suffix}",
                partition_assignments[partition][["row_id", "client"]],
            )

    manifests_dir = resolve_path(outputs["manifests_dir"])
    manifests_dir.mkdir(parents=True, exist_ok=True)
    groups = build_group_manifest(split_index, frame, schema)
    for scenario, partition_assignments in assignments.items():
        row_clients = pd.concat(
            partition_assignments.values(), ignore_index=True
        ).set_index("row_id")["client"]
        group_clients = (
            split_index[["row_id", "group_id"]]
            .assign(client=lambda table: table["row_id"].map(row_clients))
            .drop_duplicates("group_id")
            .set_index("group_id")["client"]
        )
        scenario_manifest = groups.copy()
        scenario_manifest["client"] = scenario_manifest["group_id"].map(group_clients)
        scenario_manifest.to_csv(
            manifests_dir / f"groups_{scenario}.csv", index=False
        )

    feature_columns = FeatureSelector.load_columns(
        resolve_path(data_config["preprocessing"]["feature_columns_path"])
    )
    reporter = ClientDistributionReporter(schema, feature_columns)
    client_report_root = resolve_path(outputs["client_report_dir"])
    summary: dict[str, Any] = {"scenarios": {}, "shared_test_assignment": True}
    for scenario, partition_assignments in assignments.items():
        scenario_dir = client_report_root / scenario
        scenario_dir.mkdir(parents=True, exist_ok=True)
        train_summary = reporter.client_summary(
            partition_assignments["train"],
            split_index.loc[split_index["partition"] == "train"],
            frame,
            ScenarioSpec.from_config(scenario, scenarios[scenario]),
        )
        test_summary = _client_partition_summary(
            partition_assignments["test"],
            split_index.loc[split_index["partition"] == "test"],
            frame,
            schema,
            partition="test",
        )
        validation_summary = _client_partition_summary(
            partition_assignments["validation"],
            split_index.loc[split_index["partition"] == "validation"],
            frame,
            schema,
            partition="validation",
        )
        train_summary.to_csv(scenario_dir / "client_summary.csv", index=False)
        test_summary.to_csv(scenario_dir / "test_client_summary.csv", index=False)
        validation_summary.to_csv(
            scenario_dir / "validation_client_summary.csv", index=False
        )

        for partition, table in (
            ("validation", validation_summary),
            ("test", test_summary),
        ):
            if (table["n_test_vehicles"] < 1).any():
                raise ValueError(f"{scenario}: every {partition} client needs a real vehicle")
            if (table["n_normal"] == 0).any() or (table["n_anomaly"] == 0).any():
                raise ValueError(f"{scenario}: every {partition} client needs both classes")
            fractions = table["n_anomaly"] / (table["n_normal"] + table["n_anomaly"])
            if not fractions.between(0.45, 0.55).all():
                raise ValueError(f"{scenario}: {partition} cohorts are not balanced")

        report = {
            "scenario": scenario,
            "client_train_summary": train_summary.to_dict(orient="records"),
            "client_validation_summary": validation_summary.to_dict(orient="records"),
            "client_test_summary": test_summary.to_dict(orient="records"),
            "partition_checks": validator.report(
                split_index, frame, schema.candidate_feature_columns(frame)
            )["checks"],
        }
        with (scenario_dir / "client_holdout_report.json").open(
            "w", encoding="utf-8"
        ) as handle:
            json.dump(report, handle, indent=2, ensure_ascii=False)
        summary["scenarios"][scenario] = report

    scalers_dir = resolve_path(outputs["scalers_dir"])
    real_categories = {
        category.name for category in schema.categories if category.family == "live_real"
    }
    train_ids = split_index.loc[split_index["partition"] == "train", "row_id"]
    category_by_row = schema.assign_categories(frame[schema.source_column])
    normal_real_ids = train_ids[
        category_by_row.loc[train_ids].isin(real_categories).to_numpy()
        & frame.loc[train_ids, schema.label_column].eq(schema.normal_label).to_numpy()
    ]
    scaler = build_scaler(feature_columns, data_config["preprocessing"]).fit(
        frame.loc[normal_real_ids]
    )
    scaler.save(scalers_dir / "global_train.json")

    split_report_dir = resolve_path(outputs["split_report_dir"])
    split_report_dir.mkdir(parents=True, exist_ok=True)
    with (split_report_dir / "shared_split_report.json").open(
        "w", encoding="utf-8"
    ) as handle:
        json.dump(
            {
                "seed": int(data_config["seed"]),
                "split_report": validator.report(
                    split_index, frame, schema.candidate_feature_columns(frame)
                ),
                "scenario_train_real_normal_counts": {
                    scenario: {
                        client: int(count)
                        for client, count in rows.groupby("client")["row_id"].size().items()
                    }
                    for scenario, rows in (
                        (
                            name,
                            partition_assignments["train"]
                            .merge(
                                split_index[["row_id", "category"]],
                                on="row_id",
                                validate="one_to_one",
                            )
                            .merge(
                                frame[[schema.label_column]].rename_axis("row_id").reset_index(),
                                on="row_id",
                                validate="one_to_one",
                            )
                            .loc[
                                lambda table: table["category"].isin(real_categories)
                                & table[schema.label_column].eq(schema.normal_label),
                                ["row_id", "client"],
                            ],
                        )
                        for name, partition_assignments in assignments.items()
                    )
                },
                "test_assignment_sha256": _assignment_fingerprint(
                    assignments[next(iter(scenarios))]["test"]
                ),
            },
            handle,
            indent=2,
            ensure_ascii=False,
        )
    return summary


def _assignment_fingerprint(assignment: pd.DataFrame) -> str:
    import hashlib

    canonical = assignment.sort_values("row_id")[["row_id", "client"]].to_csv(
        index=False, lineterminator="\n"
    )
    return hashlib.sha256(canonical.encode()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Create common validation/test cohorts for client scenarios."
    )
    parser.add_argument(
        "--data-config",
        default="configs/data_sensors_client_holdout_shared.yaml",
    )
    parser.add_argument(
        "--experiments-config",
        default="configs/experiments_sensors_client_holdout_shared.yaml",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace outputs under the dedicated shared-split paths",
    )
    args = parser.parse_args()
    report = run_shared_client_holdout(
        args.data_config, args.experiments_config, overwrite=args.overwrite
    )
    for scenario, details in report["scenarios"].items():
        test = pd.DataFrame(details["client_test_summary"])
        print(
            f"{scenario}: test vehicles/client="
            f"{test['n_test_vehicles'].tolist()}, test rows/client="
            f"{test['n_test'].tolist()}"
        )
    print("Shared validation/test row-to-client assignments verified.")


if __name__ == "__main__":
    main()
