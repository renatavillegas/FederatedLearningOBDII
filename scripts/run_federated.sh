#!/usr/bin/env bash
# Starts a local Flower 1.42 deployment (1 SuperLink + 6 SuperNodes, separate processes),
# submits one federated run and stops every process when the run finishes.
# The SuperLink serves every API over HTTP on SUPERLINK_PORT; each SuperNode serves its runtime API on its own port.
#
# Usage: scripts/run_federated.sh <scenario> <fedavg|fedprox> [proximal_mu] [seed] [normal-only-real] [model-config] [data-config] [experiments-config]
set -euo pipefail

SCENARIO="${1:?scenario required (for example iid, non_iid, or quantity_skew)}"
METHOD="${2:?method required (fedavg or fedprox)}"
MU="${3:-0.0}"
SEED="${4:--1}"
NORMAL_ONLY_REAL="${5:-false}"
MODEL_CONFIG="${6:-configs/model.yaml}"
DATA_CONFIG="${7:-configs/data.yaml}"
EXPERIMENTS_CONFIG="${8:-configs/experiments.yaml}"
N_CLIENTS=6
SUPERLINK_PORT=8000
SUPERNODE_BASE_PORT=9100
CPU_THREADS_PER_WORKER="${CPU_THREADS_PER_WORKER:-${OMP_NUM_THREADS:-1}}"

if [[ ! "${CPU_THREADS_PER_WORKER}" =~ ^[1-9][0-9]*$ ]]; then
    echo "CPU_THREADS_PER_WORKER must be a positive integer" >&2
    exit 2
fi

export OMP_NUM_THREADS="${CPU_THREADS_PER_WORKER}"
export MKL_NUM_THREADS="${CPU_THREADS_PER_WORKER}"
export OPENBLAS_NUM_THREADS="${CPU_THREADS_PER_WORKER}"
export NUMEXPR_NUM_THREADS="${CPU_THREADS_PER_WORKER}"
# Run Flower Apps with the dependencies in the active Python environment.
export FLWR_DISABLE_RUNTIME_DEPENDENCY_INSTALLATION=1

case "${NORMAL_ONLY_REAL}" in
    true|false) ;;
    *)
        echo "normal-only-real must be true or false" >&2
        exit 2
        ;;
esac

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export FLWR_HOME="${FLWR_HOME:-${ROOT}/.flwr}"
LOG_ROOT="${LOG_ROOT:-${ROOT}/results/logs}"
LOG_DIR="${LOG_ROOT}/$(date +%Y%m%d_%H%M%S)_${SCENARIO}_${METHOD}_mu${MU}"
mkdir -p "${FLWR_HOME}" "${LOG_DIR}"
RUN_STARTED_AT="$(date --iso-8601=seconds)"
SECONDS=0

cat > "${FLWR_HOME}/config.toml" <<EOF
[superlink]
default = "local-deployment"

[superlink.local-deployment]
address = "127.0.0.1:${SUPERLINK_PORT}"
insecure = true
EOF

PIDS=()
cleanup() {
    local exit_status=$?
    trap - EXIT
    for pid in "${PIDS[@]}"; do
        kill "${pid}" 2>/dev/null || true
    done
    wait 2>/dev/null || true
    printf 'started_at=%s\nfinished_at=%s\nelapsed_seconds=%s\nexit_status=%s\n' \
        "${RUN_STARTED_AT}" "$(date --iso-8601=seconds)" "${SECONDS}" "${exit_status}" \
        > "${LOG_DIR}/runtime.txt"
    return "${exit_status}"
}
trap cleanup EXIT

check_alive() {
    for pid in "${PIDS[@]}"; do
        if ! kill -0 "${pid}" 2>/dev/null; then
            echo "A Flower process exited early; see ${LOG_DIR}" >&2
            exit 1
        fi
    done
}

flower-superlink --insecure --port "${SUPERLINK_PORT}" > "${LOG_DIR}/superlink.log" 2>&1 &
PIDS+=($!)
sleep 3
check_alive

for k in $(seq 1 "${N_CLIENTS}"); do
    flower-supernode --insecure \
        --superlink "127.0.0.1:${SUPERLINK_PORT}" \
        --port "$((SUPERNODE_BASE_PORT + k))" \
        --node-config "client-id='client_${k}'" \
        > "${LOG_DIR}/supernode_client_${k}.log" 2>&1 &
    PIDS+=($!)
done
sleep 3
check_alive

cd "${ROOT}"
flwr run . local-deployment --stream \
    --run-config "project-root='${ROOT}' scenario='${SCENARIO}' method='${METHOD}' proximal-mu=${MU} seed=${SEED} normal-only-real=${NORMAL_ONLY_REAL} model-config='${MODEL_CONFIG}' data-config='${DATA_CONFIG}' experiments-config='${EXPERIMENTS_CONFIG}'" \
    2>&1 | tee "${LOG_DIR}/run.log"
