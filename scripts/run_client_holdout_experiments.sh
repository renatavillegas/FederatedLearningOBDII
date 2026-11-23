#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT}"

PYTHON="${PYTHON:-python}"
MODEL_CONFIG="${MODEL_CONFIG:-configs/model.yaml}"
EXPERIMENTS_CONFIG="configs/experiments_sensors_client_holdout.yaml"
SEEDS=("$@")
if ((${#SEEDS[@]} == 0)); then
    SEEDS=(7 21 42)
fi

for seed in "${SEEDS[@]}"; do
    if [[ ! "${seed}" =~ ^[0-9]+$ ]]; then
        echo "Seeds must be non-negative integers; received '${seed}'" >&2
        exit 2
    fi
done

for scenario in iid quantity_skew; do
    if [[ "${scenario}" == "iid" ]]; then
        DATA_CONFIG="configs/data_sensors_client_holdout_iid.yaml"
        split_dir="data/splits/sensors_only_client_holdout_iid"
    else
        DATA_CONFIG="configs/data_sensors_client_holdout_quantity.yaml"
        split_dir="data/splits/sensors_only_client_holdout_quantity"
    fi

    for required_file in \
        "${split_dir}/partition_index.csv" \
        "${split_dir}/clients_${scenario}.csv" \
        "${split_dir}/clients_${scenario}_test.csv"; do
        if [[ ! -f "${required_file}" ]]; then
            echo "Missing ${required_file}; generate the client-aware split before training." >&2
            exit 1
        fi
    done
done

for scenario in iid quantity_skew; do
    if [[ "${scenario}" == "iid" ]]; then
        DATA_CONFIG="configs/data_sensors_client_holdout_iid.yaml"
    else
        DATA_CONFIG="configs/data_sensors_client_holdout_quantity.yaml"
    fi
    if [[ ! -f "data/processed/sensors_only/feature_columns.json" ]]; then
        echo "Missing sensors-only feature list; run preprocessing before training." >&2
        exit 1
    fi

    for seed in "${SEEDS[@]}"; do
        central_tag="sensors105_client_${scenario}_latent8"
        central_dir="results/runs/sensors_only_client_holdout/all/centralized_normal_only_${central_tag}/seed_${seed}"
        local_dir="results/runs/sensors_only_client_holdout/${scenario}/local_normal_only/seed_${seed}"
        fedavg_dir="results/runs/sensors_only_client_holdout/${scenario}/fedavg_normal_only/seed_${seed}"

        if [[ ! -f "${central_dir}/summary.json" ]]; then
            "${PYTHON}" -m src.experiments.centralized \
                --normal-only-real --run-tag "${central_tag}" \
                --data-config "${DATA_CONFIG}" \
                --experiments-config "${EXPERIMENTS_CONFIG}" \
                --model-config "${MODEL_CONFIG}" --seed "${seed}"
        fi
        if [[ ! -f "${local_dir}/summary.json" ]]; then
            "${PYTHON}" -m src.experiments.local --scenario "${scenario}" \
                --normal-only-real --data-config "${DATA_CONFIG}" \
                --experiments-config "${EXPERIMENTS_CONFIG}" \
                --model-config "${MODEL_CONFIG}" --seed "${seed}"
        fi
        if [[ ! -f "${fedavg_dir}/summary.json" ]]; then
            scripts/run_federated.sh "${scenario}" fedavg 0.0 "${seed}" true \
                "${MODEL_CONFIG}" "${DATA_CONFIG}" "${EXPERIMENTS_CONFIG}"
        fi
    done
done

selection_seed="${SEEDS[0]}"
for scenario in iid quantity_skew; do
    if [[ "${scenario}" == "iid" ]]; then
        DATA_CONFIG="configs/data_sensors_client_holdout_iid.yaml"
    else
        DATA_CONFIG="configs/data_sensors_client_holdout_quantity.yaml"
    fi
    central_tag="sensors105_client_${scenario}_latent8"
    central_dir="results/runs/sensors_only_client_holdout/all/centralized_normal_only_${central_tag}/seed_${selection_seed}"
    "${PYTHON}" -m src.evaluation.report \
        --data-config "${DATA_CONFIG}" \
        --experiments-config "${EXPERIMENTS_CONFIG}" \
        --model-config "${MODEL_CONFIG}" \
        --run-dir "${central_dir}" \
        --score-sweep-validation
done

score_aggregation="$("${PYTHON}" - \
    "results/metrics/sensors_only_client_holdout/all/centralized_normal_only_sensors105_client_iid_latent8/seed_${selection_seed}/score_sweep_validation.csv" \
    "results/metrics/sensors_only_client_holdout/all/centralized_normal_only_sensors105_client_quantity_skew_latent8/seed_${selection_seed}/score_sweep_validation.csv" <<'PY'
import csv
import sys
from collections import defaultdict

scores_by_scenario = {}
for path in sys.argv[1:]:
    scores = defaultdict(list)
    with open(path, newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            scores[row["score_aggregation"]].append(float(row["validation_pr_auc"]))
    if not scores:
        raise SystemExit(f"No validation PR-AUC values found in {path}")
    scores_by_scenario[path] = {
        name: sum(values) / len(values) for name, values in scores.items()
    }

common = set.intersection(*(set(scores) for scores in scores_by_scenario.values()))
if not common:
    raise SystemExit("Validation sweeps do not share score-aggregation candidates")
averages = {
    name: sum(scores[name] for scores in scores_by_scenario.values()) / len(scores_by_scenario)
    for name in common
}
print(max(averages, key=averages.get))
PY
)"
echo "Selected ${score_aggregation} by mean validation PR-AUC across IID and quantity-skew (seed ${selection_seed})"

for scenario in iid quantity_skew; do
    if [[ "${scenario}" == "iid" ]]; then
        DATA_CONFIG="configs/data_sensors_client_holdout_iid.yaml"
    else
        DATA_CONFIG="configs/data_sensors_client_holdout_quantity.yaml"
    fi
    eval_args=()
    for seed in "${SEEDS[@]}"; do
        eval_args+=(--seed "${seed}")
    done
    "${PYTHON}" -m src.evaluation.client_test \
        --data-config "${DATA_CONFIG}" \
        --experiments-config "${EXPERIMENTS_CONFIG}" \
        --model-config "${MODEL_CONFIG}" \
        --scenario "${scenario}" \
        --score-aggregation "${score_aggregation}" \
        "${eval_args[@]}"

    "${PYTHON}" -m scripts.plot_client_holdout_results \
        --scenario "${scenario}" \
        --data-config "${DATA_CONFIG}" \
        --experiments-config "${EXPERIMENTS_CONFIG}"
done
