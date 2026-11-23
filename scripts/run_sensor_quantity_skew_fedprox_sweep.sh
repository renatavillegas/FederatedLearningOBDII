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
SEED="$("${PYTHON}" -c 'from src.config import load_yaml; print(load_yaml("configs/experiments_sensors_quantity.yaml")["federated"]["fedprox_selection_seed"])')"
SCENARIOS=(iid quantity_skew)

for required_file in \
    data/splits/partition_index.csv \
    data/splits/clients_iid.csv \
    data/splits/clients_quantity_skew.csv \
    data/manifests/sensors_only/groups.csv \
    data/processed/sensors_only/feature_columns.json
do
    if [[ ! -f "${required_file}" ]]; then
        echo "Required data artifact is missing: ${required_file}" >&2
        echo "Run the sensor split, preprocessing, and client-allocation phases first." >&2
        exit 1
    fi
done

mapfile -t MU_VALUES < <(
    "${PYTHON}" -c '
from src.config import load_yaml
config = load_yaml("configs/experiments_sensors_quantity.yaml")
for value in config["federated"]["fedprox_mu_grid"]:
    print(f"{float(value):g}")
'
)

if [[ "${#MU_VALUES[@]}" -eq 0 ]]; then
    echo "FedProx mu grid is empty in ${EXPERIMENTS_CONFIG}" >&2
    exit 1
fi

for scenario in "${SCENARIOS[@]}"; do
    for mu in "${MU_VALUES[@]}"; do
        run_dir="results/runs/sensors_only_quantity_skew/${scenario}/fedprox_mu_${mu}_normal_only/seed_${SEED}"
        if [[ -f "${run_dir}/summary.json" ]]; then
            echo "Using existing FedProx run: ${run_dir}"
            continue
        fi

        echo "Running sensors-only FedProx (${scenario}, mu=${mu}, seed=${SEED})"
        bash scripts/run_federated.sh \
            "${scenario}" fedprox "${mu}" "${SEED}" true \
            "${MODEL_CONFIG}" "${DATA_CONFIG}" "${EXPERIMENTS_CONFIG}"
    done
done

SELECTION_FILE="$(mktemp)"
trap 'rm -f "${SELECTION_FILE}"' EXIT

"${PYTHON}" - "${SEED}" "${MU_VALUES[@]}" <<'PY' | tee "${SELECTION_FILE}"
import json
import sys
from pathlib import Path

seed = int(sys.argv[1])
mu_values = sys.argv[2:]
scenarios = ("iid", "quantity_skew")
selected = {}

print("\nFedProx model selection by normal-real validation loss:")
print("| Scenario | mu | Validation loss | Best round | Rounds |")
print("|---|---:|---:|---:|---:|")
for scenario in scenarios:
    candidates = []
    for mu in mu_values:
        path = Path(
            f"results/runs/sensors_only_quantity_skew/{scenario}/"
            f"fedprox_mu_{mu}_normal_only/seed_{seed}/summary.json"
        )
        if not path.is_file():
            raise FileNotFoundError(f"Expected completed FedProx run summary: {path}")
        with path.open(encoding="utf-8") as handle:
            summary = json.load(handle)
        if (
            summary["scenario"] != scenario
            or summary["seed"] != seed
            or not summary["normal_label_used_for_training_selection"]
            or summary["method"] != f"fedprox_mu_{mu}_normal_only"
        ):
            raise ValueError(f"Run metadata does not match the sensors-only protocol: {path}")
        candidates.append((float(summary["best_validation_loss"]), mu, summary))

    loss, mu, summary = min(candidates, key=lambda candidate: candidate[0])
    selected[scenario] = mu
    print(
        f"| {scenario} | {mu} | {loss:.6f} "
        f"| {summary['best_round']} | {summary['rounds_run']} |"
    )

print("\nSelected mu per scenario:")
for scenario, mu in selected.items():
    print(f"{scenario}={mu}")
PY

IID_MU="$(sed -n 's/^iid=//p' "${SELECTION_FILE}")"
QUANTITY_SKEW_MU="$(sed -n 's/^quantity_skew=//p' "${SELECTION_FILE}")"
if [[ -z "${IID_MU}" || -z "${QUANTITY_SKEW_MU}" ]]; then
    echo "Could not read the selected FedProx mu values." >&2
    exit 1
fi

echo
echo "Evaluating only the validation-selected FedProx models on the test partition."
"${PYTHON}" -m src.evaluation.report \
    --data-config "${DATA_CONFIG}" \
    --experiments-config "${EXPERIMENTS_CONFIG}" \
    --model-config "${MODEL_CONFIG}" \
    --score-aggregation top_5_percent \
    --run-dir "results/runs/sensors_only_quantity_skew/iid/fedprox_mu_${IID_MU}_normal_only/seed_${SEED}" \
    --run-dir "results/runs/sensors_only_quantity_skew/quantity_skew/fedprox_mu_${QUANTITY_SKEW_MU}_normal_only/seed_${SEED}"

echo
echo "FedProx sweep and selected-model evaluation completed for seed ${SEED}."
echo "Selected runs and metrics are under results/{runs,metrics}/sensors_only_quantity_skew/."
