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

SEED="${1:-42}"
if [[ ! "${SEED}" =~ ^[0-9]+$ ]]; then
    echo "Seed must be a non-negative integer; received '${SEED}'" >&2
    exit 2
fi

DATA_CONFIG="configs/data_sensors_client_holdout_quantity.yaml"
EXPERIMENTS_CONFIG="configs/experiments_sensors_client_holdout.yaml"
MODEL_CONFIG="configs/model.yaml"

mapfile -t MU_VALUES < <(
    "${PYTHON}" -c '
from src.config import load_yaml
config = load_yaml("configs/experiments_sensors_client_holdout.yaml")
for value in config["federated"]["fedprox_mu_grid"]:
    print(f"{float(value):g}")
'
)
if [[ "${#MU_VALUES[@]}" -eq 0 ]]; then
    echo "FedProx mu grid is empty in ${EXPERIMENTS_CONFIG}" >&2
    exit 1
fi

SELECTION_CSV="results/metrics/sensors_only_client_holdout/fedprox_selection/quantity_skew_seed_${SEED}.csv"
SELECTED_MU="$("${PYTHON}" - "${SEED}" "${SELECTION_CSV}" "${MU_VALUES[@]}" <<'PY'
import csv
import json
import math
import sys
from pathlib import Path

seed = int(sys.argv[1])
output = Path(sys.argv[2])
mu_values = sys.argv[3:]
candidates = []

for mu in mu_values:
    run_dir = Path(
        f"results/runs/sensors_only_client_holdout/quantity_skew/"
        f"fedprox_mu_{mu}_normal_only/seed_{seed}"
    )
    summary_path = run_dir / "summary.json"
    if not summary_path.is_file():
        raise FileNotFoundError(
            f"Missing completed FedProx run: {summary_path}. "
            "Train all configured mu candidates before evaluating."
        )
    with summary_path.open(encoding="utf-8") as handle:
        summary = json.load(handle)
    expected_method = f"fedprox_mu_{mu}_normal_only"
    if (
        summary.get("scenario") != "quantity_skew"
        or summary.get("seed") != seed
        or summary.get("method") != expected_method
        or not summary.get("normal_label_used_for_training_selection")
    ):
        raise ValueError(f"Run metadata does not match the client-aware protocol: {summary_path}")
    validation_loss = float(summary["best_validation_loss"])
    if not math.isfinite(validation_loss) or validation_loss < 0:
        raise ValueError(f"Invalid best_validation_loss in {summary_path}")
    candidates.append(
        {
            "scenario": "quantity_skew",
            "seed": seed,
            "mu": mu,
            "method": expected_method,
            "best_validation_loss": validation_loss,
            "best_round": summary["best_round"],
            "rounds_run": summary["rounds_run"],
            "run_dir": str(run_dir),
        }
    )

selected = min(candidates, key=lambda row: row["best_validation_loss"])
output.parent.mkdir(parents=True, exist_ok=True)
with output.open("w", newline="", encoding="utf-8") as handle:
    writer = csv.DictWriter(handle, fieldnames=selected.keys())
    writer.writeheader()
    writer.writerows(sorted(candidates, key=lambda row: row["best_validation_loss"]))
print(selected["mu"])
PY
)"

METHOD="fedprox_mu_${SELECTED_MU}_normal_only"
echo "Validation-based FedProx selection (seed ${SEED}): ${METHOD}"
echo "Candidate comparison saved to ${SELECTION_CSV}"

for quantile in 0.99 0.95; do
    if [[ "${quantile}" == "0.99" ]]; then
        percentile="p99"
    else
        percentile="p95"
    fi
    echo "Evaluating selected run on client validation/test cohorts (threshold ${percentile})"
    "${PYTHON}" -m src.evaluation.client_test \
        --data-config "${DATA_CONFIG}" \
        --experiments-config "${EXPERIMENTS_CONFIG}" \
        --model-config "${MODEL_CONFIG}" \
        --scenario quantity_skew \
        --seed "${SEED}" \
        --method "${METHOD}" \
        --score-aggregation top_5_percent \
        --threshold-quantile "${quantile}"
done

echo "Evaluation complete. Results are under:"
echo "  results/metrics/sensors_only_client_holdout/client_test/{quantity_skew,p95/quantity_skew}/${METHOD}/"
