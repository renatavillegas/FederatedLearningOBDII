#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT}"

if [[ -z "${VIRTUAL_ENV:-}" ]]; then
    echo "Activate the project virtual environment before running this script." >&2
    exit 2
fi

if ! command -v flwr >/dev/null 2>&1; then
    echo "Flower CLI 'flwr' was not found in the active environment." >&2
    exit 2
fi

for mu in 0.001 0.01 1; do
    run_dir="results/runs/iid/fedprox_mu_${mu}_normal_only/seed_42"
    if [[ -f "${run_dir}/summary.json" ]]; then
        echo "Skipping existing run: ${run_dir}"
        continue
    fi

    echo "Starting IID normal-only FedProx with mu=${mu}"
    bash scripts/run_federated.sh iid fedprox "${mu}" 42 true configs/model_latent16.yaml
done

python - <<'PY'
import json
from pathlib import Path

methods = ("0.001", "0.01", "0.1", "1")
print("\nIID FedProx validation comparison (choose mu by validation loss):")
print("| mu | Best validation loss | Best round | Rounds run |")
print("|---:|---:|---:|---:|")
for mu in methods:
    path = Path(f"results/runs/iid/fedprox_mu_{mu}_normal_only/seed_42/summary.json")
    if not path.is_file():
        raise FileNotFoundError(f"Expected completed run summary: {path}")
    with path.open(encoding="utf-8") as handle:
        summary = json.load(handle)
    if summary["scenario"] != "iid" or summary["seed"] != 42 or not summary["normal_label_used_for_training_selection"]:
        raise ValueError(f"Run metadata does not match the IID normal-only seed-42 protocol: {path}")
    print(
        f"| {mu} | {summary['best_validation_loss']:.6f} "
        f"| {summary['best_round']} | {summary['rounds_run']} |"
    )
PY

echo
echo "Select mu using only the validation loss above. Evaluate the selected run on test only after selection."
