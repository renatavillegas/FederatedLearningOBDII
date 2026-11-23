#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT}"

if [[ -z "${VIRTUAL_ENV:-}" ]]; then
    echo "Activate the project virtual environment before running this script." >&2
    exit 2
fi
PYTHON="${VIRTUAL_ENV}/bin/python"
if [[ ! -x "${PYTHON}" ]] || ! command -v flwr >/dev/null 2>&1; then
    echo "Python and Flower CLI must be available in the active virtual environment." >&2
    exit 2
fi

SCENARIO="${1:-iid}"
METHOD="${2:-fedavg}"
MU="${3:-0.0}"
SEED="${4:-42}"
EDGE_CLIENT_ID="${EDGE_CLIENT_ID:-client_4}"
SUPERLINK_HOST="${SUPERLINK_HOST:?Set SUPERLINK_HOST to this machine private LAN IPv4 address.}"
FLEET_PORT="${FLOWER_FLEET_PORT:-8000}"
MAX_ROUNDS="${MAX_ROUNDS:-0}"
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

case "${METHOD}" in
    fedavg)
        MU="0.0"
        METHOD_DIR="fedavg_normal_only"
        ;;
    fedprox)
        METHOD_DIR="fedprox_mu_${MU}_normal_only"
        ;;
    *)
        echo "Unknown method '${METHOD}'. Must be fedavg or fedprox." >&2
        exit 2
        ;;
esac

RUN_DIR="results/runs/sensors_only_client_holdout_raspberry_client4/${SCENARIO}/${METHOD_DIR}/seed_${SEED}"
CLIENT_BUNDLE="results/raspberry_flower_client/${SCENARIO}/${EDGE_CLIENT_ID}"

if [[ ! -f "${CLIENT_BUNDLE}/edge_client_data.npz" ]]; then
    echo "Missing client bundle: ${CLIENT_BUNDLE}/edge_client_data.npz" >&2
    echo "Run: python scripts/prepare_flower_raspberry_client.py --scenario ${SCENARIO} --data-config ${DATA_CONFIG} --client ${EDGE_CLIENT_ID} --seed ${SEED}" >&2
    exit 1
fi

if [[ -f "${RUN_DIR}/summary.json" ]]; then
    echo "Run output already exists: ${RUN_DIR}" >&2
    echo "Choose a new SEED or remove/archive the existing result before rerunning." >&2
    exit 1
fi

export FLWR_DISABLE_RUNTIME_DEPENDENCY_INSTALLATION=1
export OMP_NUM_THREADS="${CPU_THREADS_PER_WORKER:-1}"
export MKL_NUM_THREADS="${CPU_THREADS_PER_WORKER:-1}"
export OPENBLAS_NUM_THREADS="${CPU_THREADS_PER_WORKER:-1}"
export NUMEXPR_NUM_THREADS="${CPU_THREADS_PER_WORKER:-1}"
export FLWR_HOME="${ROOT}/.flwr_raspberry_${SCENARIO}"
LOG_DIR="${ROOT}/results/logs/raspberry_${SCENARIO}"
mkdir -p "${FLWR_HOME}" "${LOG_DIR}"

cat > "${FLWR_HOME}/config.toml" <<EOF
[superlink]
default = "local-deployment"

[superlink.local-deployment]
address = "${SUPERLINK_HOST}:${FLEET_PORT}"
insecure = true
EOF

LOG_FILE="${LOG_DIR}/${METHOD_DIR}_seed_${SEED}_$(date +%Y%m%d_%H%M%S).log"
PIDS=()
cleanup() {
    local exit_status=$?
    trap - EXIT
    for pid in "${PIDS[@]}"; do
        kill "${pid}" 2>/dev/null || true
    done
    wait 2>/dev/null || true
    return "${exit_status}"
}
trap cleanup EXIT

flower-superlink --insecure --disable-runtime-dependency-installation \
    --host "${SUPERLINK_HOST}" --port "${FLEET_PORT}" > "${LOG_FILE}.superlink" 2>&1 &
PIDS+=($!)
sleep 3
if ! kill -0 "${PIDS[0]}" 2>/dev/null; then
    echo "Flower SuperLink failed to start; inspect ${LOG_FILE}.superlink" >&2
    exit 1
fi

for client_number in 1 2 3 4 5 6; do
    client="client_${client_number}"
    if [[ "${client}" == "${EDGE_CLIENT_ID}" ]]; then
        continue
    fi
    flower-supernode --insecure \
        --superlink "${SUPERLINK_HOST}:${FLEET_PORT}" \
        --port "$((9100 + client_number))" \
        --node-config "client-id='${client}'" \
        > "${LOG_FILE}.${client}" 2>&1 &
    PIDS+=($!)
done
sleep 3
for pid in "${PIDS[@]}"; do
    if ! kill -0 "${pid}" 2>/dev/null; then
        echo "A local Flower process failed; inspect ${LOG_FILE}.*" >&2
        exit 1
    fi
done

echo "SuperLink is listening on ${SUPERLINK_HOST}:${FLEET_PORT}."
echo "5 simulated clients running locally. Edge client ${EDGE_CLIENT_ID} must connect from the Raspberry Pi."
run_config_parts=(
    "project-root='${ROOT}'"
    "scenario='${SCENARIO}'"
    "method='${METHOD}'"
    "proximal-mu=${MU}"
    "seed=${SEED}"
    "normal-only-real=true"
    "edge-client-id='${EDGE_CLIENT_ID}'"
    "max-rounds=${MAX_ROUNDS}"
    "model-config='${MODEL_CONFIG}'"
    "data-config='${DATA_CONFIG}'"
    "experiments-config='${EXPERIMENTS_CONFIG}'"
)
printf -v run_config '%s ' "${run_config_parts[@]}"
flwr run . local-deployment --stream --run-config "${run_config}" \
    2>&1 | tee "${LOG_FILE}"

if [[ ! -f "${RUN_DIR}/summary.json" ]]; then
    echo "Flower exited without a completed run summary. Inspect ${LOG_FILE} and ${LOG_FILE}.*." >&2
    exit 1
fi
if [[ ! -s "${RUN_DIR}/edge_inference_metrics.csv" ]]; then
    echo "The run completed without Raspberry inference metrics." >&2
    exit 1
fi
echo "Run summary: ${RUN_DIR}/summary.json"
echo "Edge inference metrics: ${RUN_DIR}/edge_inference_metrics.csv"
