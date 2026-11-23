#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT}"

if [[ -z "${VIRTUAL_ENV:-}" ]]; then
    echo "Activate the project virtual environment before running this script." >&2
    exit 2
fi

PYTHON="${VIRTUAL_ENV}/bin/python"
if [[ ! -x "${PYTHON}" ]]; then
    echo "Python executable not found in the active environment: ${PYTHON}" >&2
    exit 2
fi
if ! command -v flwr >/dev/null 2>&1; then
    echo "Flower CLI 'flwr' was not found in the active environment." >&2
    exit 2
fi

DATA_CONFIG="configs/data_sensors.yaml"
EXPERIMENTS_CONFIG="configs/experiments_sensors_quantity.yaml"
MODEL_CONFIG="configs/model.yaml"
RUNS_DIR="results/runs/sensors_only_quantity_skew"
SCORE_AGGREGATION="top_5_percent"
SEEDS=(42 7 21)

for required_file in \
    data/splits/partition_index.csv \
    data/splits/clients_iid.csv \
    data/splits/clients_quantity_skew.csv \
    data/manifests/sensors_only/groups.csv \
    data/processed/sensors_only/feature_columns.json
do
    if [[ ! -f "${required_file}" ]]; then
        echo "Required data artifact is missing: ${required_file}" >&2
        echo "Run the data split, sensor preprocessing, and client-allocation phases first." >&2
        exit 1
    fi
done

run_centralized() {
    local seed="$1"
    local run_dir="${RUNS_DIR}/all/centralized_normal_only_sensors105_quantity_latent8/seed_${seed}"
    if [[ -f "${run_dir}/summary.json" ]]; then
        echo "Using existing centralized run: ${run_dir}"
        return
    fi

    echo "Training centralized sensor-only latent-8 (seed ${seed})"
    "${PYTHON}" -m src.experiments.centralized \
        --normal-only-real \
        --run-tag sensors105_quantity_latent8 \
        --data-config "${DATA_CONFIG}" \
        --experiments-config "${EXPERIMENTS_CONFIG}" \
        --model-config "${MODEL_CONFIG}" \
        --seed "${seed}"
}

run_local() {
    local scenario="$1"
    local seed="$2"
    local run_dir="${RUNS_DIR}/${scenario}/local_normal_only/seed_${seed}"
    if [[ -f "${run_dir}/summary.json" ]]; then
        echo "Using existing local run: ${run_dir}"
        return
    fi

    echo "Training local sensor-only latent-8 (${scenario}, seed ${seed})"
    "${PYTHON}" -m src.experiments.local \
        --scenario "${scenario}" \
        --normal-only-real \
        --data-config "${DATA_CONFIG}" \
        --experiments-config "${EXPERIMENTS_CONFIG}" \
        --model-config "${MODEL_CONFIG}" \
        --seed "${seed}"
}

run_fedavg() {
    local scenario="$1"
    local seed="$2"
    local run_dir="${RUNS_DIR}/${scenario}/fedavg_normal_only/seed_${seed}"
    if [[ -f "${run_dir}/summary.json" ]]; then
        echo "Using existing FedAvg run: ${run_dir}"
        return
    fi

    echo "Training FedAvg sensor-only latent-8 (${scenario}, seed ${seed})"
    bash scripts/run_federated.sh \
        "${scenario}" fedavg 0.0 "${seed}" true \
        "${MODEL_CONFIG}" "${DATA_CONFIG}" "${EXPERIMENTS_CONFIG}"
}

for seed in "${SEEDS[@]}"; do
    run_centralized "${seed}"
    for scenario in iid quantity_skew; do
        run_local "${scenario}" "${seed}"
        run_fedavg "${scenario}" "${seed}"
    done
done

for seed in "${SEEDS[@]}"; do
    run_dirs=(
        --run-dir "${RUNS_DIR}/all/centralized_normal_only_sensors105_quantity_latent8/seed_${seed}"
        --run-dir "${RUNS_DIR}/iid/local_normal_only/seed_${seed}"
        --run-dir "${RUNS_DIR}/iid/fedavg_normal_only/seed_${seed}"
        --run-dir "${RUNS_DIR}/quantity_skew/local_normal_only/seed_${seed}"
        --run-dir "${RUNS_DIR}/quantity_skew/fedavg_normal_only/seed_${seed}"
    )

    echo "Evaluating seed ${seed} with ${SCORE_AGGREGATION}"
    "${PYTHON}" -m src.evaluation.report \
        --data-config "${DATA_CONFIG}" \
        --experiments-config "${EXPERIMENTS_CONFIG}" \
        --model-config "${MODEL_CONFIG}" \
        --score-aggregation "${SCORE_AGGREGATION}" \
        "${run_dirs[@]}"
done

echo
echo "Completed centralized/local/FedAvg sensor-only comparisons for seeds: ${SEEDS[*]}"
echo "Metrics: results/metrics/sensors_only_quantity_skew/summary.csv"
echo "Each scenario and method is evaluated with the fixed ${SCORE_AGGREGATION} score aggregation."
