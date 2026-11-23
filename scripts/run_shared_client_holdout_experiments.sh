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

MODEL_CONFIG="${MODEL_CONFIG:-configs/model.yaml}"
DATA_CONFIG="configs/data_sensors_client_holdout_shared.yaml"
EXPERIMENTS_CONFIG="configs/experiments_sensors_client_holdout_shared.yaml"
LATENT_DIM="$("${PYTHON}" - "${MODEL_CONFIG}" <<'PY'
import sys
from src.config import load_yaml

print(int(load_yaml(sys.argv[1])["architecture"]["latent_dim"]))
PY
)"
SPLIT_DIR="data/splits/sensors_only_client_holdout_shared"
RUN_FEDPROX=true
SEEDS=()
for argument in "$@"; do
    if [[ "${argument}" == "--skip-fedprox" ]]; then
        RUN_FEDPROX=false
    else
        SEEDS+=("${argument}")
    fi
done
if ((${#SEEDS[@]} == 0)); then
    SEEDS=(7 21 42)
fi
for seed in "${SEEDS[@]}"; do
    if [[ ! "${seed}" =~ ^[0-9]+$ ]]; then
        echo "Seeds must be non-negative integers; received '${seed}'" >&2
        exit 2
    fi
done

if [[ ! -f "data/processed/sensors_only/feature_columns.json" ]]; then
    echo "Missing sensors-only feature list; run preprocessing before training." >&2
    exit 1
fi

if [[ ! -f "${SPLIT_DIR}/partition_index.csv" ]]; then
    "${PYTHON}" -m src.data.shared_client_holdout \
        --data-config "${DATA_CONFIG}" \
        --experiments-config "${EXPERIMENTS_CONFIG}"
fi
for scenario in iid quantity_skew; do
    for suffix in "" "_validation" "_test"; do
        if [[ ! -f "${SPLIT_DIR}/clients_${scenario}${suffix}.csv" ]]; then
            echo "Missing ${SPLIT_DIR}/clients_${scenario}${suffix}.csv" >&2
            exit 1
        fi
    done
done

SCENARIOS=(iid quantity_skew)
run_shared_federated() {
    FLWR_HOME="${ROOT}/.flwr_sensors_only_client_holdout_shared" \
        LOG_ROOT="${ROOT}/results/logs/sensors_only_client_holdout_shared" \
        scripts/run_federated.sh "$@"
}

for scenario in "${SCENARIOS[@]}"; do
    for seed in "${SEEDS[@]}"; do
        central_tag="sensors105_client_${scenario}_latent${LATENT_DIM}"
        central_dir="results/runs/sensors_only_client_holdout_shared/all/centralized_normal_only_${central_tag}/seed_${seed}"
        local_dir="results/runs/sensors_only_client_holdout_shared/${scenario}/local_normal_only/seed_${seed}"
        fedavg_dir="results/runs/sensors_only_client_holdout_shared/${scenario}/fedavg_normal_only/seed_${seed}"

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
            run_shared_federated "${scenario}" fedavg 0.0 "${seed}" true \
                "${MODEL_CONFIG}" "${DATA_CONFIG}" "${EXPERIMENTS_CONFIG}"
        fi
    done
done

selection_seed="${SEEDS[0]}"
for scenario in "${SCENARIOS[@]}"; do
    central_tag="sensors105_client_${scenario}_latent${LATENT_DIM}"
    central_dir="results/runs/sensors_only_client_holdout_shared/all/centralized_normal_only_${central_tag}/seed_${selection_seed}"
    "${PYTHON}" -m src.evaluation.report \
        --data-config "${DATA_CONFIG}" \
        --experiments-config "${EXPERIMENTS_CONFIG}" \
        --model-config "${MODEL_CONFIG}" \
        --run-dir "${central_dir}" \
        --score-sweep-validation
done

score_aggregation="$("${PYTHON}" - "${selection_seed}" "${LATENT_DIM}" <<'PY'
import csv
import sys
from collections import defaultdict
from pathlib import Path

seed = sys.argv[1]
latent_dim = sys.argv[2]
scores_by_scenario = {}
for scenario in ("iid", "quantity_skew"):
    path = Path(
        "results/metrics/sensors_only_client_holdout_shared/all/"
        f"centralized_normal_only_sensors105_client_{scenario}_latent{latent_dim}/"
        f"seed_{seed}/score_sweep_validation.csv"
    )
    scores = defaultdict(list)
    with path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            scores[row["score_aggregation"]].append(float(row["validation_pr_auc"]))
    if not scores:
        raise SystemExit(f"No validation PR-AUC values found in {path}")
    scores_by_scenario[scenario] = {
        name: sum(values) / len(values) for name, values in scores.items()
    }

common = set.intersection(*(set(scores) for scores in scores_by_scenario.values()))
if not common:
    raise SystemExit("Validation sweeps do not share score-aggregation candidates")
average = {
    name: sum(scores[name] for scores in scores_by_scenario.values()) / 2
    for name in common
}
print(max(average, key=average.get))
PY
)"
echo "Selected ${score_aggregation} by mean validation PR-AUC across scenarios (seed ${selection_seed})."

for scenario in "${SCENARIOS[@]}"; do
    for quantile in 0.99 0.95; do
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
            --threshold-quantile "${quantile}" \
            "${eval_args[@]}"
    done
    "${PYTHON}" -m scripts.plot_client_holdout_results \
        --scenario "${scenario}" \
        --data-config "${DATA_CONFIG}" \
        --experiments-config "${EXPERIMENTS_CONFIG}" --data-only
    "${PYTHON}" -m scripts.plot_client_holdout_results \
        --scenario "${scenario}" \
        --data-config "${DATA_CONFIG}" \
        --experiments-config "${EXPERIMENTS_CONFIG}"
    "${PYTHON}" -m scripts.plot_client_holdout_results \
        --scenario "${scenario}" \
        --data-config "${DATA_CONFIG}" \
        --experiments-config "${EXPERIMENTS_CONFIG}" \
        --threshold-quantile 0.95
done

if [[ "${RUN_FEDPROX}" == false ]]; then
    echo "Baseline-only run completed; FedProx grid and evaluation were skipped."
    exit 0
fi

fedprox_selection_seed="$("${PYTHON}" -c \
    'from src.config import load_yaml; print(load_yaml("configs/experiments_sensors_client_holdout_shared.yaml")["federated"]["fedprox_selection_seed"])')"
mapfile -t MU_VALUES < <(
    "${PYTHON}" -c '
from src.config import load_yaml
config = load_yaml("configs/experiments_sensors_client_holdout_shared.yaml")
for value in config["federated"]["fedprox_mu_grid"]:
    print(f"{float(value):g}")
'
)
if (("${#MU_VALUES[@]}" == 0)); then
    echo "FedProx mu grid is empty in ${EXPERIMENTS_CONFIG}" >&2
    exit 1
fi

for scenario in "${SCENARIOS[@]}"; do
    for mu in "${MU_VALUES[@]}"; do
        run_dir="results/runs/sensors_only_client_holdout_shared/${scenario}/fedprox_mu_${mu}_normal_only/seed_${fedprox_selection_seed}"
        if [[ -f "${run_dir}/summary.json" ]]; then
            echo "Using existing FedProx selection run: ${run_dir}"
            continue
        fi
        run_shared_federated "${scenario}" fedprox "${mu}" \
            "${fedprox_selection_seed}" true "${MODEL_CONFIG}" \
            "${DATA_CONFIG}" "${EXPERIMENTS_CONFIG}"
    done
done

selection_file="results/metrics/sensors_only_client_holdout_shared/fedprox_selection/shared_split_selection.json"
"${PYTHON}" - "${fedprox_selection_seed}" "${selection_file}" "${MU_VALUES[@]}" <<'PY'
import json
import sys
from pathlib import Path

seed = int(sys.argv[1])
destination = Path(sys.argv[2])
mu_values = sys.argv[3:]
selected = {}
for scenario in ("iid", "quantity_skew"):
    candidates = []
    for mu in mu_values:
        path = Path(
            f"results/runs/sensors_only_client_holdout_shared/{scenario}/"
            f"fedprox_mu_{mu}_normal_only/seed_{seed}/summary.json"
        )
        with path.open(encoding="utf-8") as handle:
            summary = json.load(handle)
        expected_method = f"fedprox_mu_{mu}_normal_only"
        if (
            summary["scenario"] != scenario
            or summary["seed"] != seed
            or summary["method"] != expected_method
            or not summary["normal_label_used_for_training_selection"]
        ):
            raise ValueError(f"Run metadata does not match the shared protocol: {path}")
        candidates.append(
            (float(summary["best_validation_loss"]), mu, summary["best_round"])
        )
    loss, mu, best_round = min(candidates, key=lambda row: row[0])
    selected[scenario] = {
        "mu": mu,
        "validation_loss": loss,
        "best_round": best_round,
        "selection_seed": seed,
    }
destination.parent.mkdir(parents=True, exist_ok=True)
destination.write_text(json.dumps(selected, indent=2) + "\n", encoding="utf-8")
for scenario, selection in selected.items():
    print(
        f"{scenario}: selected mu={selection['mu']} "
        f"(validation loss {selection['validation_loss']:.6f}, "
        f"round {selection['best_round']})"
    )
PY

for scenario in "${SCENARIOS[@]}"; do
    selected_mu="$("${PYTHON}" -c \
        'import json,sys; print(json.load(open(sys.argv[1], encoding="utf-8"))[sys.argv[2]]["mu"])' \
        "${selection_file}" "${scenario}")"
    selected_method="fedprox_mu_${selected_mu}_normal_only"
    for seed in "${SEEDS[@]}"; do
        run_dir="results/runs/sensors_only_client_holdout_shared/${scenario}/${selected_method}/seed_${seed}"
        if [[ ! -f "${run_dir}/summary.json" ]]; then
            run_shared_federated "${scenario}" fedprox "${selected_mu}" \
                "${seed}" true "${MODEL_CONFIG}" "${DATA_CONFIG}" \
                "${EXPERIMENTS_CONFIG}"
        fi
    done
    for quantile in 0.99 0.95; do
        eval_args=()
        for seed in "${SEEDS[@]}"; do
            eval_args+=(--seed "${seed}")
        done
        "${PYTHON}" -m src.evaluation.client_test \
            --data-config "${DATA_CONFIG}" \
            --experiments-config "${EXPERIMENTS_CONFIG}" \
            --model-config "${MODEL_CONFIG}" \
            --scenario "${scenario}" \
            --method "${selected_method}" \
            --score-aggregation "${score_aggregation}" \
            --threshold-quantile "${quantile}" \
            "${eval_args[@]}"

        plot_args=()
        if [[ "${quantile}" != "0.99" ]]; then
            plot_args+=(--threshold-quantile "${quantile}")
        fi
        "${PYTHON}" -m scripts.plot_client_holdout_results \
            --scenario "${scenario}" \
            --data-config "${DATA_CONFIG}" \
            --experiments-config "${EXPERIMENTS_CONFIG}" \
            --include-method "${selected_method}" \
            --output-tag fedprox_shared_selected \
            "${plot_args[@]}"
    done
done

echo "Shared-cohort training and evaluation completed."
echo "Split artifacts: ${SPLIT_DIR}"
echo "Runs, metrics and figures: results/{runs,metrics,figures}/sensors_only_client_holdout_shared/"
