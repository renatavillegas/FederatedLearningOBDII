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
EXPERIMENTS_CONFIG="configs/experiments_sensors_quantity_strong.yaml"
MODEL_CONFIG="configs/model.yaml"
RUNS_DIR="results/runs/sensors_only_quantity_skew_strong"
SEEDS=(42 7 21)
SCENARIO="quantity_skew_strong"

for required_file in \
    data/splits/partition_index.csv \
    data/manifests/sensors_only/groups.csv \
    data/processed/sensors_only/feature_columns.json
do
    if [[ ! -f "${required_file}" ]]; then
        echo "Required data artifact is missing: ${required_file}" >&2
        echo "Run the sensor split and preprocessing phases first." >&2
        exit 1
    fi
done

client_assignment="data/splits/clients_${SCENARIO}.csv"
client_summary="results/clients/sensors_only/${SCENARIO}/client_summary.csv"
if [[ ! -f "${client_assignment}" || ! -f "${client_summary}" ]]; then
    echo "Generating strong quantity-skew client assignment."
    "${PYTHON}" -m src.data.clients \
        --data-config "${DATA_CONFIG}" \
        --experiments-config "${EXPERIMENTS_CONFIG}"
fi

for required_file in "${client_assignment}" "${client_summary}"; do
    if [[ ! -f "${required_file}" ]]; then
        echo "Client allocation did not create ${required_file}" >&2
        exit 1
    fi
done

run_centralized() {
    local seed="$1"
    local run_dir="${RUNS_DIR}/all/centralized_normal_only_sensors105_quantity_strong_latent8/seed_${seed}"
    if [[ -f "${run_dir}/summary.json" ]]; then
        echo "Using existing centralized run: ${run_dir}"
        return
    fi
    echo "Training centralized sensor-only latent-8 (seed ${seed})"
    "${PYTHON}" -m src.experiments.centralized \
        --normal-only-real \
        --run-tag sensors105_quantity_strong_latent8 \
        --data-config "${DATA_CONFIG}" \
        --experiments-config "${EXPERIMENTS_CONFIG}" \
        --model-config "${MODEL_CONFIG}" \
        --seed "${seed}"
}

run_local() {
    local seed="$1"
    local run_dir="${RUNS_DIR}/${SCENARIO}/local_normal_only/seed_${seed}"
    if [[ -f "${run_dir}/summary.json" ]]; then
        echo "Using existing local run: ${run_dir}"
        return
    fi
    echo "Training local sensor-only latent-8 (${SCENARIO}, seed ${seed})"
    "${PYTHON}" -m src.experiments.local \
        --scenario "${SCENARIO}" \
        --normal-only-real \
        --data-config "${DATA_CONFIG}" \
        --experiments-config "${EXPERIMENTS_CONFIG}" \
        --model-config "${MODEL_CONFIG}" \
        --seed "${seed}"
}

run_fedavg() {
    local seed="$1"
    local run_dir="${RUNS_DIR}/${SCENARIO}/fedavg_normal_only/seed_${seed}"
    if [[ -f "${run_dir}/summary.json" ]]; then
        echo "Using existing FedAvg run: ${run_dir}"
        return
    fi
    echo "Training FedAvg sensor-only latent-8 (${SCENARIO}, seed ${seed})"
    bash scripts/run_federated.sh \
        "${SCENARIO}" fedavg 0.0 "${seed}" true \
        "${MODEL_CONFIG}" "${DATA_CONFIG}" "${EXPERIMENTS_CONFIG}"
}

for seed in "${SEEDS[@]}"; do
    run_centralized "${seed}"
    run_local "${seed}"
    run_fedavg "${seed}"
done

for seed in "${SEEDS[@]}"; do
    "${PYTHON}" -m src.evaluation.report \
        --data-config "${DATA_CONFIG}" \
        --experiments-config "${EXPERIMENTS_CONFIG}" \
        --model-config "${MODEL_CONFIG}" \
        --score-aggregation top_5_percent \
        --run-dir "${RUNS_DIR}/all/centralized_normal_only_sensors105_quantity_strong_latent8/seed_${seed}" \
        --run-dir "${RUNS_DIR}/${SCENARIO}/local_normal_only/seed_${seed}" \
        --run-dir "${RUNS_DIR}/${SCENARIO}/fedavg_normal_only/seed_${seed}"
done

echo
echo "Strong quantity-skew training and evaluation completed for seeds: ${SEEDS[*]}."
echo "Client distribution: ${client_summary}"
echo "Metrics: results/metrics/sensors_only_quantity_skew_strong/summary.csv"
