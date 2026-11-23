import argparse
import json
import zlib
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import ks_2samp

from src.config import load_yaml, resolve_path
from src.data.preprocess import FeatureSelector, Standardizer, build_scaler, scaler_compatibility_report
from src.data.schema import DataSchema
from src.data.split import PartitionStore, allocate_counts


def client_name(position: int) -> str:
    return f"client_{position + 1}"


@dataclass(frozen=True)
class ScenarioSpec:
    name: str
    anomaly_fraction: tuple[float, ...]
    reference_scenario: str | None = None
    required_band: tuple[float, float] | None = None
    normal_count_weights: tuple[float, ...] | None = None

    def __post_init__(self) -> None:
        if any(not 0.0 < value < 1.0 for value in self.anomaly_fraction):
            raise ValueError(f"Scenario {self.name}: anomaly fractions must be in (0, 1)")
        if self.required_band is not None and not self.required_band[0] <= self.required_band[1]:
            raise ValueError(f"Scenario {self.name}: invalid required_band")
        if self.normal_count_weights is not None:
            if self.reference_scenario is not None:
                raise ValueError(f"Scenario {self.name}: use normal_count_weights or reference_scenario, not both")
            if any(not np.isfinite(value) or value <= 0.0 for value in self.normal_count_weights):
                raise ValueError(f"Scenario {self.name}: normal_count_weights must be finite and positive")

    @classmethod
    def from_config(cls, name: str, config: Mapping[str, Any]) -> "ScenarioSpec":
        band = config.get("required_band")
        return cls(
            name=name,
            anomaly_fraction=tuple(float(value) for value in config["anomaly_fraction"]),
            reference_scenario=config.get("reference_scenario"),
            required_band=tuple(float(value) for value in band) if band is not None else None,
            normal_count_weights=(
                tuple(float(value) for value in config["normal_count_weights"])
                if "normal_count_weights" in config
                else None
            ),
        )


@dataclass
class TrainPools:
    """Training records grouped into allocation units.

    Vehicles (assignment_unit "source") are allocated whole and must contain only normal records.
    Record-level categories form one pool per (category, label).
    """

    vehicle_rows: dict[str, np.ndarray]
    normal_pools: dict[str, np.ndarray] = field(default_factory=dict)
    anomaly_pools: dict[str, np.ndarray] = field(default_factory=dict)

    @classmethod
    def from_index(cls, train_index: pd.DataFrame, labels: pd.Series, schema: DataSchema) -> "TrainPools":
        units = {spec.name: spec.assignment_unit for spec in schema.categories}
        labeled = train_index.assign(label=labels.loc[train_index["row_id"]].to_numpy())
        by_source = labeled[labeled["category"].map(units) == "source"]
        not_normal = by_source.loc[by_source["label"] != schema.normal_label, "group_id"].unique()
        if len(not_normal):
            raise ValueError(f"Vehicles with non-normal records cannot be allocated: {sorted(not_normal)}")
        vehicle_rows = {
            group: rows.to_numpy() for group, rows in by_source.groupby("group_id", sort=True)["row_id"]
        }
        pools = cls(vehicle_rows=vehicle_rows)
        by_record = labeled[labeled["category"].map(units) == "record"]
        for (category, label), rows in by_record.groupby(["category", "label"], sort=True)["row_id"]:
            target = pools.normal_pools if label == schema.normal_label else pools.anomaly_pools
            target[f"{category}/{label}"] = np.sort(rows.to_numpy())
        return pools

    @property
    def n_normal(self) -> int:
        return sum(len(rows) for rows in self.vehicle_rows.values()) + self.n_normal_records

    @property
    def n_normal_records(self) -> int:
        return sum(len(rows) for rows in self.normal_pools.values())

    @property
    def n_anomaly(self) -> int:
        return sum(len(rows) for rows in self.anomaly_pools.values())


class ClientAllocator:
    """Allocates training records to clients given a target number of normal records and anomaly fractions."""

    def __init__(self, n_clients: int) -> None:
        if n_clients < 1:
            raise ValueError("n_clients must be positive")
        self.n_clients = n_clients

    def allocate(
        self,
        pools: TrainPools,
        desired_normals: Sequence[float],
        anomaly_fraction: Sequence[float],
        rng: np.random.Generator,
        normal_count_weights: Sequence[float] | None = None,
    ) -> pd.DataFrame:
        desired = np.asarray(desired_normals, dtype=float)
        fractions = np.asarray(anomaly_fraction, dtype=float)
        if len(desired) != self.n_clients or len(fractions) != self.n_clients:
            raise ValueError("desired_normals and anomaly_fraction must have one value per client")

        vehicle_client, vehicle_records = self._allocate_vehicles(pools, desired, rng)
        if normal_count_weights is None:
            normal_records = self._fill_deficit(desired - vehicle_records, pools.n_normal_records)
        else:
            weights = np.asarray(normal_count_weights, dtype=float)
            if len(weights) != self.n_clients or not np.isfinite(weights).all() or (weights <= 0.0).any():
                raise ValueError("normal_count_weights must contain one finite positive weight per client")
            normal_records = np.asarray(allocate_counts(pools.n_normal_records, weights / weights.sum()))
        normals = vehicle_records + normal_records

        weights = fractions / (1.0 - fractions) * normals
        if weights.sum() <= 0:
            raise ValueError("No normal records allocated; anomaly weights are undefined")
        anomalies = np.asarray(allocate_counts(pools.n_anomaly, weights / weights.sum()))

        parts = [
            pd.DataFrame({"row_id": pools.vehicle_rows[vehicle], "client": client_name(client)})
            for vehicle, client in vehicle_client.items()
        ]
        parts += self._draw_from_pools(pools.normal_pools, normal_records, rng)
        parts += self._draw_from_pools(pools.anomaly_pools, anomalies, rng)
        assignment = pd.concat(parts, ignore_index=True)
        return assignment.sort_values("row_id").reset_index(drop=True)

    def _allocate_vehicles(
        self, pools: TrainPools, desired: np.ndarray, rng: np.random.Generator
    ) -> tuple[dict[str, int], np.ndarray]:
        """Largest vehicles first, each to the client with the largest remaining deficit."""
        vehicles = np.array(sorted(pools.vehicle_rows))
        rng.shuffle(vehicles)
        sizes = np.array([len(pools.vehicle_rows[vehicle]) for vehicle in vehicles])
        order = np.argsort(-sizes, kind="stable")
        assigned = np.zeros(self.n_clients)
        vehicle_client = {}
        for position in order:
            client = int(np.argmax(desired - assigned))
            vehicle_client[str(vehicles[position])] = client
            assigned[client] += sizes[position]
        return vehicle_client, assigned

    def _fill_deficit(self, deficit: np.ndarray, total: int) -> np.ndarray:
        """Distribute `total` unit records one at a time to the client with the largest remaining deficit."""
        filler = np.zeros(self.n_clients, dtype=int)
        remaining = deficit.astype(float).copy()
        for _ in range(total):
            client = int(np.argmax(remaining))
            filler[client] += 1
            remaining[client] -= 1
        return filler

    def _draw_from_pools(
        self, pools: Mapping[str, np.ndarray], per_client: np.ndarray, rng: np.random.Generator
    ) -> list[pd.DataFrame]:
        """Split per-client counts across pools proportionally to pool size, then draw rows at random."""
        if not pools:
            if per_client.sum():
                raise ValueError("Records requested from an empty pool set")
            return []
        names = list(pools)
        total = int(per_client.sum())
        if total != sum(len(pools[name]) for name in names):
            raise ValueError("Per-client counts must use every record of the pools")
        share = per_client / total if total else np.full(self.n_clients, 1.0 / self.n_clients)
        counts_by_pool = {}
        used = np.zeros(self.n_clients, dtype=int)
        for name in names[:-1]:
            counts_by_pool[name] = np.asarray(allocate_counts(len(pools[name]), share))
            used += counts_by_pool[name]
        counts_by_pool[names[-1]] = per_client - used
        if (counts_by_pool[names[-1]] < 0).any():
            raise ValueError("Negative allocation while splitting counts across pools")

        parts = []
        for name in names:
            rows = pools[name].copy()
            rng.shuffle(rows)
            start = 0
            for client, count in enumerate(counts_by_pool[name]):
                parts.append(pd.DataFrame({"row_id": rows[start:start + count], "client": client_name(client)}))
                start += int(count)
        return parts


class ScenarioBuilder:
    def __init__(self, allocator: ClientAllocator, seed: int) -> None:
        self.allocator = allocator
        self.seed = seed

    def build(self, specs: Sequence[ScenarioSpec], pools: TrainPools) -> dict[str, pd.DataFrame]:
        assignments: dict[str, pd.DataFrame] = {}
        for spec in self._ordered(specs):
            rng = np.random.default_rng([self.seed, zlib.crc32(spec.name.encode())])
            desired = self._desired_normals(spec, pools, assignments)
            assignments[spec.name] = self.allocator.allocate(
                pools,
                desired,
                spec.anomaly_fraction,
                rng,
                normal_count_weights=spec.normal_count_weights,
            )
        return {spec.name: assignments[spec.name] for spec in specs}

    def _desired_normals(
        self, spec: ScenarioSpec, pools: TrainPools, built: Mapping[str, pd.DataFrame]
    ) -> np.ndarray:
        n_clients = self.allocator.n_clients
        if spec.normal_count_weights is not None:
            weights = np.asarray(spec.normal_count_weights, dtype=float)
            if len(weights) != n_clients:
                raise ValueError(f"Scenario {spec.name}: expected {n_clients} normal_count_weights")
            n_real_normals = sum(len(rows) for rows in pools.vehicle_rows.values())
            return n_real_normals * weights / weights.sum()
        if spec.reference_scenario is None:
            return np.full(n_clients, pools.n_normal / n_clients)
        sizes = built[spec.reference_scenario]["client"].value_counts()
        sizes = np.array([sizes.get(client_name(k), 0) for k in range(n_clients)], dtype=float)
        weights = np.asarray(spec.anomaly_fraction) * sizes
        anomalies = np.asarray(allocate_counts(pools.n_anomaly, weights / weights.sum()))
        return sizes - anomalies

    @staticmethod
    def _ordered(specs: Sequence[ScenarioSpec]) -> list[ScenarioSpec]:
        by_name = {spec.name: spec for spec in specs}
        ordered: list[ScenarioSpec] = []
        visiting: set[str] = set()

        def visit(spec: ScenarioSpec) -> None:
            if spec in ordered:
                return
            if spec.name in visiting:
                raise ValueError(f"Circular reference_scenario involving {spec.name}")
            visiting.add(spec.name)
            if spec.reference_scenario is not None:
                if spec.reference_scenario not in by_name:
                    raise ValueError(f"Unknown reference_scenario: {spec.reference_scenario}")
                visit(by_name[spec.reference_scenario])
            ordered.append(spec)

        for spec in specs:
            visit(spec)
        return ordered


class ClientStore:
    """Reads and writes client assignments in data/splits/clients_<scenario>.csv."""

    def __init__(self, splits_dir: str | Path) -> None:
        self.directory = resolve_path(splits_dir)

    def path(self, scenario: str) -> Path:
        return self.directory / f"clients_{scenario}.csv"

    def save(self, scenario: str, assignment: pd.DataFrame) -> None:
        self.directory.mkdir(parents=True, exist_ok=True)
        assignment[["row_id", "client"]].to_csv(self.path(scenario), index=False)

    def load(self, scenario: str) -> pd.DataFrame:
        return pd.read_csv(self.path(scenario))

    def rows(self, frame: pd.DataFrame, scenario: str, client: str) -> pd.DataFrame:
        assignment = self.load(scenario)
        row_ids = assignment.loc[assignment["client"] == client, "row_id"]
        if row_ids.empty:
            raise KeyError(f"Client {client} not found in scenario {scenario}")
        return frame.loc[row_ids]


class ClientDistributionReporter:
    def __init__(self, schema: DataSchema, feature_columns: Sequence[str]) -> None:
        self.schema = schema
        self.feature_columns = list(feature_columns)

    def client_summary(
        self, assignment: pd.DataFrame, train_index: pd.DataFrame, frame: pd.DataFrame, spec: ScenarioSpec
    ) -> pd.DataFrame:
        labeled = self._labeled(assignment, train_index, frame)
        units = {s.name: s.assignment_unit for s in self.schema.categories}
        summary = labeled.groupby("client").agg(
            n_records=("row_id", "size"),
            n_anomaly=("label", lambda values: int((values == self.schema.anomaly_label).sum())),
        )
        summary["n_normal"] = summary["n_records"] - summary["n_anomaly"]
        summary["anomaly_fraction"] = summary["n_anomaly"] / summary["n_records"]
        vehicles = labeled[labeled["category"].map(units) == "source"]
        summary["n_vehicles"] = vehicles.groupby("client")["group_id"].nunique()
        summary["n_normal_real"] = vehicles.groupby("client").size()
        composition = (
            labeled.groupby(["client", "category", "label"]).size().unstack(["category", "label"], fill_value=0)
        )
        composition.columns = [f"n_{category}_{label}" for category, label in composition.columns]
        summary = summary.join(composition).fillna({"n_vehicles": 0})
        order = [client_name(k) for k in range(len(spec.anomaly_fraction))]
        summary = summary.reindex(pd.Index(order, name="client"))
        summary.insert(0, "target_anomaly_fraction", list(spec.anomaly_fraction))
        if spec.normal_count_weights is not None:
            summary.insert(1, "target_normal_count_weight", list(spec.normal_count_weights))
        summary["n_vehicles"] = summary["n_vehicles"].astype(int)
        summary["n_normal_real"] = summary["n_normal_real"].fillna(0).astype(int)
        return summary.reset_index()

    def ks_table(self, assignment: pd.DataFrame, frame: pd.DataFrame) -> pd.DataFrame:
        pooled = frame.loc[assignment["row_id"], self.feature_columns]
        rows = []
        for client, client_rows in assignment.groupby("client", sort=True)["row_id"]:
            local = frame.loc[client_rows, self.feature_columns]
            for feature in self.feature_columns:
                result = ks_2samp(local[feature].to_numpy(), pooled[feature].to_numpy())
                rows.append(
                    {"client": client, "feature": feature, "ks_statistic": result.statistic, "p_value": result.pvalue}
                )
        return pd.DataFrame(rows)

    @staticmethod
    def ks_summary(ks: pd.DataFrame) -> dict[str, dict[str, float]]:
        grouped = ks.groupby("client")["ks_statistic"]
        return {
            client: {"median": float(values.median()), "max": float(values.max())}
            for client, values in grouped
        }

    def check(
        self, assignment: pd.DataFrame, train_index: pd.DataFrame, summary: pd.DataFrame, spec: ScenarioSpec
    ) -> dict[str, bool]:
        train_rows = set(train_index["row_id"])
        assigned = assignment["row_id"]
        units = {s.name: s.assignment_unit for s in self.schema.categories}
        merged = assignment.merge(train_index[["row_id", "group_id", "category"]], on="row_id", how="left")
        vehicles = merged[merged["category"].map(units) == "source"]
        checks = {
            "rows_only_from_train": bool(set(assigned) <= train_rows),
            "every_train_row_assigned_once": bool(assigned.is_unique and set(assigned) == train_rows),
            "vehicles_in_single_client": bool((vehicles.groupby("group_id")["client"].nunique() == 1).all()),
            "every_client_has_records": bool((summary["n_records"] > 0).all()),
        }
        if spec.required_band is not None:
            low, high = spec.required_band
            checks["anomaly_fraction_in_required_band"] = bool(summary["anomaly_fraction"].between(low, high).all())
        return checks

    def _labeled(self, assignment: pd.DataFrame, train_index: pd.DataFrame, frame: pd.DataFrame) -> pd.DataFrame:
        merged = assignment.merge(train_index[["row_id", "group_id", "category"]], on="row_id", how="left")
        return merged.assign(label=frame.loc[merged["row_id"], self.schema.label_column].to_numpy())


def scenario_comparison(summaries: Mapping[str, pd.DataFrame]) -> pd.DataFrame:
    columns = ["client", "n_records", "n_normal", "n_anomaly", "anomaly_fraction", "n_vehicles"]
    tables = [
        summary[columns].set_index("client").add_prefix(f"{scenario}_") for scenario, summary in summaries.items()
    ]
    return pd.concat(tables, axis=1).reset_index()


def fit_client_scalers(
    assignment: pd.DataFrame, frame: pd.DataFrame, feature_columns: Sequence[str], preprocessing: Mapping[str, Any]
) -> dict[str, Standardizer]:
    return {
        client: build_scaler(feature_columns, preprocessing).fit(frame.loc[rows])
        for client, rows in assignment.groupby("client", sort=True)["row_id"]
    }


def client_group_manifest(groups: pd.DataFrame, assignment: pd.DataFrame, train_index: pd.DataFrame) -> pd.DataFrame:
    group_client = (
        assignment.merge(train_index[["row_id", "group_id"]], on="row_id")
        .drop_duplicates("group_id")
        .set_index("group_id")["client"]
    )
    manifest = groups[groups["partition"] == "train"].copy()
    manifest["client"] = manifest["group_id"].map(group_client)
    return manifest.reset_index(drop=True)


def run_clients(data_config_path: str | Path, experiments_config_path: str | Path) -> dict[str, Any]:
    data_config = load_yaml(data_config_path)
    experiments = load_yaml(experiments_config_path)
    schema = DataSchema.from_config(data_config)
    outputs = data_config["outputs"]
    preprocessing = data_config["preprocessing"]

    frame = pd.read_csv(resolve_path(data_config["raw_path"]))
    index = PartitionStore(outputs["splits_dir"]).load()
    train_index = index[index["partition"] == "train"].reset_index(drop=True)
    feature_columns = FeatureSelector.load_columns(resolve_path(preprocessing["feature_columns_path"]))

    specs = [ScenarioSpec.from_config(name, spec) for name, spec in experiments["scenarios"].items()]
    n_clients = int(experiments["n_clients"])
    for spec in specs:
        if len(spec.anomaly_fraction) != n_clients:
            raise ValueError(f"Scenario {spec.name}: expected {n_clients} anomaly fractions")
        if spec.normal_count_weights is not None and len(spec.normal_count_weights) != n_clients:
            raise ValueError(f"Scenario {spec.name}: expected {n_clients} normal_count_weights")

    pools = TrainPools.from_index(train_index, frame[schema.label_column], schema)
    assignments = ScenarioBuilder(ClientAllocator(n_clients), int(data_config["seed"])).build(specs, pools)

    reporter = ClientDistributionReporter(schema, feature_columns)
    store = ClientStore(outputs["splits_dir"])
    report_dir = resolve_path(outputs["client_report_dir"])
    scalers_dir = resolve_path(outputs["scalers_dir"])
    manifests_dir = resolve_path(outputs["manifests_dir"])
    groups = pd.read_csv(manifests_dir / "groups.csv")

    global_scaler = build_scaler(feature_columns, preprocessing).fit(frame.loc[train_index["row_id"]])

    report: dict[str, Any] = {"seed": int(data_config["seed"]), "n_clients": n_clients, "scenarios": {}}
    summaries = {}
    client_scalers = {}
    for spec in specs:
        assignment = assignments[spec.name]
        scenario_dir = report_dir / spec.name
        scenario_dir.mkdir(parents=True, exist_ok=True)

        summary = reporter.client_summary(assignment, train_index, frame, spec)
        summary.to_csv(scenario_dir / "client_summary.csv", index=False)
        summaries[spec.name] = summary

        ks = reporter.ks_table(assignment, frame)
        ks.to_csv(scenario_dir / "ks_features.csv", index=False)

        scalers = fit_client_scalers(assignment, frame, feature_columns, preprocessing)
        client_scalers[spec.name] = scalers
        scaler_compatibility_report(scalers, global_scaler).to_csv(
            scenario_dir / "scaler_compatibility.csv", index=False
        )

        report["scenarios"][spec.name] = {
            "anomaly_fraction_target": list(spec.anomaly_fraction),
            "normal_count_weights": list(spec.normal_count_weights) if spec.normal_count_weights else None,
            "reference_scenario": spec.reference_scenario,
            "required_band": list(spec.required_band) if spec.required_band else None,
            "checks": reporter.check(assignment, train_index, summary, spec),
            "anomaly_fraction_achieved": summary.set_index("client")["anomaly_fraction"].round(4).to_dict(),
            "max_abs_deviation_from_target": float(
                (summary["anomaly_fraction"] - summary["target_anomaly_fraction"]).abs().max()
            ),
            "ks_statistic": reporter.ks_summary(ks),
        }

    scenario_comparison(summaries).to_csv(report_dir / "scenario_comparison.csv", index=False)
    with (report_dir / "clients_report.json").open("w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2, ensure_ascii=False)

    failed = [
        f"{scenario}:{name}"
        for scenario, entry in report["scenarios"].items()
        for name, passed in entry["checks"].items()
        if not passed
    ]
    if failed:
        raise ValueError(f"Client checks failed: {failed}")

    global_scaler.save(scalers_dir / "global_train.json")
    for spec in specs:
        store.save(spec.name, assignments[spec.name])
        client_group_manifest(groups, assignments[spec.name], train_index).to_csv(
            manifests_dir / f"groups_{spec.name}.csv", index=False
        )
        for client, scaler in client_scalers[spec.name].items():
            scaler.save(scalers_dir / spec.name / f"{client}.json")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Distribute the training partition among federated clients.")
    parser.add_argument("--data-config", default="configs/data.yaml")
    parser.add_argument("--experiments-config", default="configs/experiments.yaml")
    args = parser.parse_args()
    run_clients(args.data_config, args.experiments_config)


if __name__ == "__main__":
    main()
