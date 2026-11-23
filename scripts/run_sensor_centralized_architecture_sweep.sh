#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT}"

if [[ -z "${VIRTUAL_ENV:-}" ]]; then
    echo "Activate the project virtual environment before running this script." >&2
    exit 2
fi

run_dir_for() {
    printf 'results/runs/sensors_only/all/centralized_normal_only_sensors105_latent%s/seed_42' "$1"
}

run_and_sweep() {
    local latent_dim="$1"
    local model_config="$2"
    local run_dir
    run_dir="$(run_dir_for "${latent_dim}")"

    if [[ ! -f "${run_dir}/summary.json" ]]; then
        echo "Training centralized sensor-only normal-only model: latent-${latent_dim}"
        python -m src.experiments.centralized \
            --normal-only-real \
            --run-tag "sensors105_latent${latent_dim}" \
            --data-config configs/data_sensors.yaml \
            --experiments-config configs/experiments_sensors.yaml \
            --model-config "${model_config}" \
            --seed 42
    else
        echo "Using existing centralized run: ${run_dir}"
    fi

    echo "Running validation-only score sweep: latent-${latent_dim}"
    python -m src.evaluation.report \
        --data-config configs/data_sensors.yaml \
        --model-config "${model_config}" \
        --experiments-config configs/experiments_sensors.yaml \
        --run-dir "${run_dir}" \
        --score-sweep-validation
}

run_and_sweep 4 configs/model_latent4.yaml
run_and_sweep 8 configs/model.yaml

latent16_run="results/runs/sensors_only/all/centralized_normal_only_sensors105/seed_42"
if [[ ! -f "${latent16_run}/summary.json" ]]; then
    echo "Expected existing latent-16 baseline was not found: ${latent16_run}" >&2
    exit 1
fi

echo "Running validation-only score sweep: latent-16"
python -m src.evaluation.report \
    --data-config configs/data_sensors.yaml \
    --model-config configs/model_latent16.yaml \
    --experiments-config configs/experiments_sensors.yaml \
    --run-dir "${latent16_run}" \
    --score-sweep-validation

echo
echo "Completed centralized sensor-only architecture sweeps."
echo "Compare validation PR-AUC in:"
echo "  results/metrics/sensors_only/all/centralized_normal_only_sensors105/seed_42/score_sweep_validation.csv"
echo "  results/metrics/sensors_only/all/centralized_normal_only_sensors105_latent4/seed_42/score_sweep_validation.csv"
echo "  results/metrics/sensors_only/all/centralized_normal_only_sensors105_latent8/seed_42/score_sweep_validation.csv"
echo "Select architecture and score aggregation using validation only before final test evaluation."
