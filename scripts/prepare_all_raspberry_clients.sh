#!/usr/bin/env bash
# Prepare edge client bundles for client_2, client_4 and client_6 in both scenarios.
# Usage:
#   ./scripts/prepare_all_raspberry_clients.sh [--seed 42] [--records 128]
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT}"

if [[ -z "${VIRTUAL_ENV:-}" ]]; then
    echo "Activate the project virtual environment before running this script." >&2
    exit 2
fi
PYTHON="${VIRTUAL_ENV}/bin/python"

SEED=42
RECORDS=128
while [[ $# -gt 0 ]]; do
    case "$1" in
        --seed)    SEED="$2";    shift 2 ;;
        --records) RECORDS="$2"; shift 2 ;;
        *) echo "Unknown argument: $1" >&2; exit 2 ;;
    esac
done

CLIENTS=("client_2" "client_4" "client_6")
SCENARIOS_AND_CONFIGS=(
    "iid:configs/data_sensors_client_holdout_iid.yaml"
    "quantity_skew:configs/data_sensors_client_holdout_quantity.yaml"
)

for entry in "${SCENARIOS_AND_CONFIGS[@]}"; do
    SCENARIO="${entry%%:*}"
    DATA_CONFIG="${entry##*:}"
    for CLIENT in "${CLIENTS[@]}"; do
        OUTPUT_DIR="results/raspberry_flower_client/${SCENARIO}/${CLIENT}"
        if [[ -f "${OUTPUT_DIR}/edge_client_data.npz" ]]; then
            echo "[skip] Bundle already exists: ${OUTPUT_DIR}"
            continue
        fi
        echo "[prepare] scenario=${SCENARIO} client=${CLIENT} seed=${SEED} records=${RECORDS}"
        "${PYTHON}" scripts/prepare_flower_raspberry_client.py \
            --scenario "${SCENARIO}" \
            --client "${CLIENT}" \
            --seed "${SEED}" \
            --records "${RECORDS}" \
            --data-config "${DATA_CONFIG}" \
            --output-dir "${OUTPUT_DIR}"
        echo "[done] ${OUTPUT_DIR}"
    done
done

echo ""
echo "All bundles ready. Copy each client directory to the Raspberry Pi with:"
echo "  scp -r results/raspberry_flower_client/<scenario>/<client> pi@<pi-ip>:~/fl_bundle/"
