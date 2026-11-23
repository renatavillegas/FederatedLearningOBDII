#!/usr/bin/env bash
set -euo pipefail

SERVER_IP="${1:?Usage: run_raspberry_flower_client.sh <server-ip> [client-bundle-dir]}"
CLIENT_ID="${CLIENT_ID:-client_4}"
CLIENT_ROOT="${2:-${HOME}/Documents/Estudo/federatedLearning/client_4}"
CLIENT_DATA_DIR="${CLIENT_DATA_DIR:-${CLIENT_ROOT}}"
PORT="${FLOWER_FLEET_PORT:-8000}"

if [[ ! "${CLIENT_ID}" =~ ^client_[1-6]$ ]]; then
    echo "CLIENT_ID must be one of client_1 through client_6." >&2
    exit 2
fi
if [[ ! -f "${CLIENT_ROOT}/configs/model.yaml" ]]; then
    echo "Missing ${CLIENT_ROOT}/configs/model.yaml; copy the prepared client bundle first." >&2
    exit 1
fi
if [[ ! -f "${CLIENT_DATA_DIR}/edge_client_data.npz" || ! -f "${CLIENT_DATA_DIR}/bundle_metadata.json" ]]; then
    echo "Missing edge client data bundle in ${CLIENT_DATA_DIR}." >&2
    exit 1
fi
if ! command -v flower-supernode >/dev/null 2>&1; then
    echo "flower-supernode is not installed in the active Python environment." >&2
    exit 1
fi

export FLWR_DISABLE_RUNTIME_DEPENDENCY_INSTALLATION=1
export FLOWER_CLIENT_PROJECT_ROOT="${CLIENT_ROOT}"
export FLOWER_CLIENT_DATA_DIR="${CLIENT_DATA_DIR}"
export OMP_NUM_THREADS="${CPU_THREADS_PER_WORKER:-1}"
export MKL_NUM_THREADS="${CPU_THREADS_PER_WORKER:-1}"
export OPENBLAS_NUM_THREADS="${CPU_THREADS_PER_WORKER:-1}"
export NUMEXPR_NUM_THREADS="${CPU_THREADS_PER_WORKER:-1}"

echo "Connecting ${CLIENT_ID} to ${SERVER_IP}:${PORT}; client data stays on this device."
exec flower-supernode --insecure \
    --superlink "${SERVER_IP}:${PORT}" \
    --max-retries 1000000 \
    --max-wait-time 86400 \
    --node-config "client-id='${CLIENT_ID}' project-root='${CLIENT_ROOT}' client-data-dir='${CLIENT_DATA_DIR}'"
