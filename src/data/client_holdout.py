import argparse
import json
import zlib
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.config import load_yaml, resolve_path
from src.data.clients import (
    ClientDistributionReporter,
    ClientStore,
    ScenarioSpec,
    fit_client_scalers,
)
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


def build_client_holdout_split(
    frame: pd.DataFrame,
    schema: DataSchema,
    scenario: str,
    scenario_config: dict[str, Any],
    n_clients: int,
    ratios: PartitionRatios,
    seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Assign vehicle groups to clients before splitting each client's groups."""
    categories = schema.assign_categories(frame[schema.source_column])
    index = GroupPartitioner(schema, ratios, seed).assign(frame)
    index["category"] = categories.loc[index["row_id"]].to_numpy()
    units = {category.name: category.assignment_unit for category in schema.categories}
    index["client"] = pd.Series(index=index.index, dtype="object")
    client_names = [f"client_{position + 1}" for position in range(n_clients)]

    weights = np.asarray(scenario_config.get("normal_count_weights", [1.0] * n_clients), dtype=float)
    if len(weights) != n_clients or not np.isfinite(weights).all() or (weights <= 0).any():
        raise ValueError(f"Scenario {scenario} must define {n_clients} finite positive client weights")
    shares = weights / weights.sum()
    rng = np.random.default_rng([seed, zlib.crc32(f"client-holdout:{scenario}".encode())])

    source_rows = index[index["category"].map(units) == "source"]
    source_groups = source_rows.groupby("group_id", sort=True).size()
    if len(source_groups) < 2 * n_clients:
        raise ValueError(
            f"Client-aware holdout needs at least {2 * n_clients} real source groups; found {len(source_groups)}"
        )
    vehicle_client = _assign_vehicle_groups(source_groups, client_names, shares, rng)
    vehicle_partitions: dict[str, str] = {}
    for client in client_names:
        groups = [group for group, owner in vehicle_client.items() if owner == client]
        vehicle_partitions.update(_split_vehicle_groups(groups, ratios, rng))
    for group_id, client in vehicle_client.items():
        index.loc[index["group_id"] == group_id, "client"] = client
        index.loc[index["group_id"] == group_id, "partition"] = vehicle_partitions[group_id]

    record_rows = index[index["category"].map(units) == "record"]
    label_values = frame.loc[record_rows["row_id"], schema.label_column].to_numpy()
    record_rows = record_rows.assign(label=label_values)
    for (category, label), rows in record_rows.groupby(["category", "label"], sort=True):
        if label != schema.normal_label:
            continue
        row_ids = rows["row_id"].to_numpy().copy()
        rng.shuffle(row_ids)
        client_counts = allocate_counts(len(row_ids), shares)
        start = 0
        for client, count in zip(client_names, client_counts):
            client_rows = row_ids[start : start + count]
            start += count
            partition_counts = allocate_counts(int(count), ratios.as_list())
            partition_start = 0
            for partition, partition_count in zip(("train", "validation", "test"), partition_counts):
                selected = client_rows[partition_start : partition_start + partition_count]
                mask = index["row_id"].isin(selected)
                index.loc[mask, "client"] = client
                index.loc[mask, "partition"] = partition
                partition_start += partition_count

    labeled_index = index.assign(
        label=frame.loc[index["row_id"], schema.label_column].to_numpy()
    )
    normal_counts = (
        labeled_index[labeled_index["label"] == schema.normal_label]
        .groupby(["partition", "client"])
        .size()
        .to_dict()
    )
    anomaly_fractions = {
        client: float(fraction)
        for client, fraction in zip(client_names, scenario_config["anomaly_fraction"])
    }
    partition_names = ("train", "validation", "test")
    desired_anomalies = np.asarray(
        [
            [
                normal_counts.get((partition, client), 0)
                * anomaly_fractions[client]
                / (1.0 - anomaly_fractions[client])
                for client in client_names
            ]
            for partition in partition_names
        ],
        dtype=float,
    )
    if desired_anomalies.sum() <= 0:
        raise ValueError("No normal records are available to balance synthetic anomalies")
    anomaly_weights = desired_anomalies.ravel() / desired_anomalies.sum()
    anomaly_rows = record_rows[record_rows["label"] == schema.anomaly_label]
    for _, rows in anomaly_rows.groupby("category", sort=True):
        row_ids = rows["row_id"].to_numpy().copy()
        rng.shuffle(row_ids)
        bin_counts = allocate_counts(len(row_ids), anomaly_weights)
        start = 0
        for bin_position, count in enumerate(bin_counts):
            partition_position, client_position = divmod(bin_position, len(client_names))
            selected = row_ids[start : start + count]
            start += count
            mask = index["row_id"].isin(selected)
            index.loc[mask, "client"] = client_names[client_position]
            index.loc[mask, "partition"] = partition_names[partition_position]

    if index["client"].isna().any() or index["partition"].isna().any():
        raise ValueError("Client-aware split left rows without a client or partition")
    mapping = index[["row_id", "client", "partition"]].copy()
    index = index[["row_id", "group_id", "category", "partition"]]
    return index.sort_values("row_id").reset_index(drop=True), mapping.sort_values("row_id").reset_index(drop=True)


def _assign_vehicle_groups(
    sizes: pd.Series,
    clients: list[str],
    shares: np.ndarray,
    rng: np.random.Generator,
) -> dict[str, str]:
    groups = sizes.index.to_numpy(dtype=object).copy()
    rng.shuffle(groups)
    ordered = sorted(groups, key=lambda group: int(sizes[group]), reverse=True)
    if len(ordered) < 2 * len(clients):
        raise ValueError("Not enough vehicle groups to give every client separate train and test vehicles")

    assigned: dict[str, str] = {}
    counts = dict.fromkeys(clients, 0)
    totals = dict.fromkeys(clients, 0)
    target_total = float(sizes.sum())
    target = {client: target_total * float(share) for client, share in zip(clients, shares)}
    remaining = list(ordered)
    for client in sorted(clients, key=lambda name: shares[clients.index(name)]):
        for _ in range(2):
            group = min(remaining, key=lambda candidate: int(sizes[candidate]))
            remaining.remove(group)
            assigned[str(group)] = client
            counts[client] += 1
            totals[client] += int(sizes[group])
    for group in remaining:
        client = max(clients, key=lambda candidate: target[candidate] - totals[candidate])
        assigned[str(group)] = client
        counts[client] += 1
        totals[client] += int(sizes[group])
    return assigned


def _split_vehicle_groups(
    groups: list[str], ratios: PartitionRatios, rng: np.random.Generator
) -> dict[str, str]:
    if len(groups) < 2:
        raise ValueError("Every client needs at least one train vehicle and one test vehicle")
    groups = list(groups)
    rng.shuffle(groups)
    test_count = max(1, int(round(len(groups) * ratios.test)))
    validation_count = max(1, int(round(len(groups) * ratios.validation))) if len(groups) >= 4 else 0
    while test_count + validation_count >= len(groups):
        if validation_count > 0:
            validation_count -= 1
        elif test_count > 1:
            test_count -= 1
        else:
            raise ValueError("Unable to keep at least one training vehicle per client")
    partitions = (
        ["train"] * (len(groups) - test_count - validation_count)
        + ["validation"] * validation_count
        + ["test"] * test_count
    )
    return dict(zip(groups, partitions))


def run_client_holdout(
    data_config_path: str | Path,
    experiments_config_path: str | Path,
    scenario: str,
) -> dict[str, Any]:
    data_config = load_yaml(data_config_path)
    experiments = load_yaml(experiments_config_path)
    scenario_config = experiments["scenarios"][scenario]
    schema = DataSchema.from_config(data_config)
    frame = pd.read_csv(resolve_path(data_config["raw_path"]))
    ratios = PartitionRatios(**data_config["split"]["ratios"])
    index, mapping = build_client_holdout_split(
        frame,
        schema,
        scenario,
        scenario_config,
        int(experiments["n_clients"]),
        ratios,
        int(data_config["seed"]),
    )
    validator = PartitionValidator(schema, data_config["split"]["required_conditions"])
    report = validator.report(
        index,
        frame,
        schema.candidate_feature_columns(frame),
    )
    validator.validate(report)

    outputs = data_config["outputs"]
    split_store = PartitionStore(outputs["splits_dir"])
    split_store.save(index)
    report_dir = resolve_path(outputs["split_report_dir"])
    report_dir.mkdir(parents=True, exist_ok=True)
    with (report_dir / "split_report.json").open("w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2, ensure_ascii=False)
    manifests_dir = resolve_path(outputs["manifests_dir"])
    manifests_dir.mkdir(parents=True, exist_ok=True)
    groups = build_group_manifest(index, frame, schema)
    groups.to_csv(manifests_dir / "groups.csv", index=False)

    client_store = ClientStore(outputs["splits_dir"])
    for partition, suffix in (("train", ""), ("test", "_test")):
        assignment = mapping.loc[mapping["partition"] == partition, ["row_id", "client"]]
        client_store.save(f"{scenario}{suffix}", assignment)

    feature_columns = FeatureSelector.load_columns(resolve_path(data_config["preprocessing"]["feature_columns_path"]))
    reporter = ClientDistributionReporter(schema, feature_columns)
    train_index = index[index["partition"] == "train"]
    train_assignment = mapping.loc[mapping["partition"] == "train", ["row_id", "client"]]
    spec = ScenarioSpec.from_config(scenario, scenario_config)
    client_summary = reporter.client_summary(train_assignment, train_index, frame, spec)
    client_report_dir = resolve_path(outputs["client_report_dir"]) / scenario
    client_report_dir.mkdir(parents=True, exist_ok=True)
    client_summary.to_csv(client_report_dir / "client_summary.csv", index=False)

    test_index = index[index["partition"] == "test"]
    test_summary = _client_partition_summary(mapping, test_index, frame, schema)
    test_summary.to_csv(client_report_dir / "test_client_summary.csv", index=False)
    if (test_summary["n_test_vehicles"] < 1).any():
        raise ValueError("Every client must have at least one held-out real vehicle")
    if (test_summary["n_normal"] == 0).any() or (test_summary["n_anomaly"] == 0).any():
        raise ValueError("Every client test partition must contain both normal and anomalous records")

    scalers_dir = resolve_path(outputs["scalers_dir"])
    live_real_categories = {
        category.name for category in schema.categories if category.family == "live_real"
    }
    all_categories = schema.assign_categories(frame[schema.source_column])
    train_real_ids = train_index.loc[
        all_categories.loc[train_index["row_id"]].isin(live_real_categories).to_numpy()
        & frame.loc[train_index["row_id"], schema.label_column].eq(schema.normal_label).to_numpy(),
        "row_id",
    ]
    global_scaler = build_scaler(feature_columns, data_config["preprocessing"]).fit(frame.loc[train_real_ids])
    global_scaler.save(scalers_dir / "global_train.json")
    for client, rows in train_assignment.groupby("client")["row_id"]:
        client_scaler = build_scaler(feature_columns, data_config["preprocessing"]).fit(frame.loc[rows])
        client_scaler.save(scalers_dir / scenario / f"{client}.json")

    report_summary = {
        "scenario": scenario,
        "client_train_summary": client_summary.to_dict(orient="records"),
        "client_test_summary": test_summary.to_dict(orient="records"),
        "test_source_groups_per_client": test_summary.set_index("client")["n_test_vehicles"].to_dict(),
        "partition_checks": report["checks"],
        "split_ratios": data_config["split"]["ratios"],
    }
    with (client_report_dir / "client_holdout_report.json").open("w", encoding="utf-8") as handle:
        json.dump(report_summary, handle, indent=2, ensure_ascii=False)
    return report_summary


def _client_partition_summary(
    mapping: pd.DataFrame,
    partition_index: pd.DataFrame,
    frame: pd.DataFrame,
    schema: DataSchema,
    partition: str = "test",
) -> pd.DataFrame:
    selected = mapping[mapping["partition"] == partition].merge(
        partition_index[["row_id", "group_id", "category"]],
        on="row_id",
        validate="one_to_one",
    )
    selected["label"] = frame.loc[selected["row_id"], schema.label_column].to_numpy()
    summary = selected.groupby(["client", "label"]).size().unstack(fill_value=0)
    summary = summary.rename(
        columns={schema.normal_label: "n_normal", schema.anomaly_label: "n_anomaly"}
    )
    for column in ("n_normal", "n_anomaly"):
        if column not in summary:
            summary[column] = 0
    live_real = {category.name for category in schema.categories if category.family == "live_real"}
    vehicles = selected[selected["category"].isin(live_real)]
    summary["n_test_vehicles"] = vehicles.groupby("client")["group_id"].nunique()
    summary["n_test_vehicles"] = summary["n_test_vehicles"].fillna(0).astype(int)
    summary["n_test"] = summary["n_normal"] + summary["n_anomaly"]
    return summary.reset_index().sort_values("client").reset_index(drop=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Create a source-grouped client train/test split.")
    parser.add_argument("--data-config", required=True)
    parser.add_argument("--experiments-config", required=True)
    parser.add_argument("--scenario", required=True)
    args = parser.parse_args()
    run_client_holdout(args.data_config, args.experiments_config, args.scenario)


if __name__ == "__main__":
    main()
