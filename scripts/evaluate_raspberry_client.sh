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
    echo "Python executable not found in ${PYTHON}" >&2
    exit 2
fi

SCENARIO="${1:-iid}"
SEED="${2:-42}"
METHOD="${3:-fedavg_normal_only}"
EDGE_CLIENT_ID="${4:-${EDGE_CLIENT_ID:-client_4}}"
MODEL_CONFIG="${MODEL_CONFIG:-configs/model.yaml}"
EXPERIMENTS_CONFIG="configs/experiments_sensors_client_holdout_raspberry_iid.yaml"

case "${SCENARIO}" in
    iid)
        DATA_CONFIG="configs/data_sensors_client_holdout_iid.yaml"
        ;;
    quantity_skew)
        DATA_CONFIG="configs/data_sensors_client_holdout_quantity.yaml"
        ;;
    *)
        echo "Unknown scenario '${SCENARIO}'. Must be iid or quantity_skew." >&2
        exit 2
        ;;
esac

RUN_DIR="results/runs/sensors_only_client_holdout_raspberry_${EDGE_CLIENT_ID}/${SCENARIO}/${METHOD}/seed_${SEED}"
if [[ ! -f "${RUN_DIR}/summary.json" ]]; then
    echo "Run summary not found: ${RUN_DIR}/summary.json" >&2
    exit 1
fi

for quantile in 0.99 0.95; do
    "${PYTHON}" -m src.evaluation.client_test \
        --data-config "${DATA_CONFIG}" \
        --experiments-config "${EXPERIMENTS_CONFIG}" \
        --model-config "${MODEL_CONFIG}" \
        --scenario "${SCENARIO}" \
        --seed "${SEED}" \
        --method "${METHOD}" \
        --score-aggregation top_5_percent \
        --threshold-quantile "${quantile}"
done

"${PYTHON}" -m src.evaluation.update_raspberry_summary \
    --scenario "${SCENARIO}" \
    --method "${METHOD}" \
    --seed "${SEED}" \
    --metrics-dir "results/metrics/sensors_only_client_holdout_raspberry_${EDGE_CLIENT_ID}" \
    --runs-dir "results/runs/sensors_only_client_holdout_raspberry_${EDGE_CLIENT_ID}"
