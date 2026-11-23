# Current experiment protocol and commands

This document describes the **current sensors-only, normal-real-only,
client-aware** study. Earlier commands for global partitioning and label-skew
are not the current protocol; see [dataset.md](./dataset.md) for the naming
distinction.

## Locked configuration

| Setting | Current choice |
|---|---|
| Features | 105, after excluding 15 columns from three context/counting signals |
| Model | Dense autoencoder 105-32-8-32-105 (`configs/model.yaml`) |
| Training rows | Normal records from real-vehicle sources only |
| Clients | Six |
| Scaler | Shared z-score fitted on pooled normal-real training records |
| Partitioning | Assign whole real-vehicle groups to clients, then split within client |
| Scenarios | Approximately balanced `iid`; `quantity_skew` weights `[1,1,2,2,4,4]` |
| Federated budget | Up to 200 rounds, five local epochs, all six clients, patience 10 |
| Seeds | 7, 21, 42 for the Centralized/Local/FedAvg comparison |
| Score aggregation | `top_5_percent`, selected on validation and fixed |
| Operating points | p99 primary; p95 sensitivity analysis |

Current configuration files:

- `configs/data_sensors_client_holdout_iid.yaml`
- `configs/data_sensors_client_holdout_quantity.yaml`
- `configs/experiments_sensors_client_holdout.yaml`
- `configs/model.yaml`

The normal-only filter uses source and label metadata to select normal real
records; these columns are not model inputs. Test cohorts are balanced with
synthetic anomalies and should not be interpreted as natural-prevalence data.

## Prepare client-aware data

The split seed is 42 and is separate from the training seeds. Do not regenerate
the splits between method/seed runs if a matched comparison is intended.

```bash
python -m src.data.preprocess --config configs/data_sensors.yaml

python -m src.data.client_holdout \
  --data-config configs/data_sensors_client_holdout_iid.yaml \
  --experiments-config configs/experiments_sensors_client_holdout.yaml \
  --scenario iid

python -m src.data.client_holdout \
  --data-config configs/data_sensors_client_holdout_quantity.yaml \
  --experiments-config configs/experiments_sensors_client_holdout.yaml \
  --scenario quantity_skew
```

Inspect the generated `client_summary.csv` and `test_client_summary.csv` under
`results/clients/sensors_only_client_holdout_{iid,quantity}/{iid,quantity_skew}/`.
Each distribution figure combines stacked normal/anomalous counts for the
client's training partition and held-out test partition; the training panel also
marks the real-normal subset that actually enters model optimization. Generate
the two scenario figures before training:

```bash
python -m scripts.plot_client_holdout_results --scenario iid \
  --data-config configs/data_sensors_client_holdout_iid.yaml --data-only
python -m scripts.plot_client_holdout_results --scenario quantity_skew \
  --data-config configs/data_sensors_client_holdout_quantity.yaml --data-only
```

The generated figures are
[IID](../results/figures/sensors_only_client_holdout/iid/client_data_distribution.png)
and
[quantity-skew](../results/figures/sensors_only_client_holdout/quantity_skew/client_data_distribution.png).

## Centralized, Local, and FedAvg

Run the resumable battery for seeds 7, 21, and 42:

```bash
bash scripts/run_client_holdout_experiments.sh
```

For one seed only:

```bash
bash scripts/run_client_holdout_experiments.sh 42
```

The script trains missing Centralized, Local, and FedAvg runs, selects the fixed
score aggregation from validation only, evaluates client test cohorts at p99,
and creates plots. Outputs are under
`results/{runs,metrics,figures}/sensors_only_client_holdout/`.

## FedProx

The initial quantity-skew screening trained `mu` values `[0.001, 0.01, 0.1,
1.0]` on seed 42. Minimum normal-real validation loss selected `mu=0.1`.
Selection results are in
`results/metrics/sensors_only_client_holdout/fedprox_selection/`.

The selected fixed `mu=0.1` has been trained on seeds 7, 21, and 42. Seed 42
was used for the initial `mu` selection; the other two seeds reuse that value
without re-running the sweep. The command below reproduces seed 21 if needed
(activate the project virtual environment first):

```bash
scripts/run_federated.sh quantity_skew fedprox 0.1 21 true \
  configs/model.yaml \
  configs/data_sensors_client_holdout_quantity.yaml \
  configs/experiments_sensors_client_holdout.yaml
```

Seed 21 has already been evaluated at p99 and p95. To reproduce that evaluation:

```bash
for quantile in 0.99 0.95; do
  python -m src.evaluation.client_test \
    --data-config configs/data_sensors_client_holdout_quantity.yaml \
    --experiments-config configs/experiments_sensors_client_holdout.yaml \
    --model-config configs/model.yaml \
    --scenario quantity_skew --seed 21 \
    --method fedprox_mu_0.1_normal_only \
    --score-aggregation top_5_percent \
    --threshold-quantile "$quantile"
done
```

Generate comparison plots with Centralized, Local and FedAvg. The output tag
prevents overwriting the existing three-method plots:

```bash
python -m scripts.plot_client_holdout_results \
  --scenario quantity_skew \
  --data-config configs/data_sensors_client_holdout_quantity.yaml \
  --include-method fedprox_mu_0.1_normal_only \
  --output-tag fedprox_mu_0.1

python -m scripts.plot_client_holdout_results \
  --scenario quantity_skew \
  --data-config configs/data_sensors_client_holdout_quantity.yaml \
  --threshold-quantile 0.95 \
  --include-method fedprox_mu_0.1_normal_only \
  --output-tag fedprox_mu_0.1
```

The refreshed comparisons use all three matched seeds. Images are saved in
`results/figures/sensors_only_client_holdout/quantity_skew/fedprox_mu_0.1/`
and its `p95/` subdirectory.

## Raspberry Pi inference pilot (optional)

This first edge check measures CPU inference only; it does not run Flower,
capture OBD-II data, or benchmark preprocessing. From the project root, export
the default quantity-skew FedAvg seed-42 model and 128 held-out records:

```bash
python scripts/raspberry_pi_inference_pilot.py prepare
```

Copy `results/raspberry_pi_pilot/obd_model_state.pt`,
`results/raspberry_pi_pilot/obd_model_config.json`,
`results/raspberry_pi_pilot/obd_inputs.npz`, and
`scripts/raspberry_pi_inference_pilot.py` to the Pi. Install compatible
PyTorch and NumPy on the Pi, then run:

```bash
python3 raspberry_pi_inference_pilot.py benchmark
```

The script reports median and p95 per-record latency, throughput, peak process
RSS, and maximum score difference from the project-machine reference. It saves
the measurements to `raspberry_pi_benchmark.json`. Temperature and throttling
should be recorded separately with the Pi's available system tools. Since input
features are already scaled on the project machine, these numbers measure model
inference rather than end-to-end sensor processing.

## IID versus non-IID terminology

The current study compares approximate quantity-balanced `iid` against
`quantity_skew`; the latter is the current non-IID-by-quantity/vehicle scenario.
It maintains a common 0.50 target anomaly fraction and changes target normal-real
client counts. It also changes vehicle composition because real vehicles are
kept intact.

The older `non_iid` config varies client anomaly fractions from 0.20 through
0.80. Since current normal-only training filters anomalous rows before
optimization, that older label-skew setting is not the current core study and
must not be conflated with `quantity_skew`.

## UMAP visualization of client training data

Generate separate exploratory projections for the IID and quantity-skew
training partitions:

```bash
python scripts/plot_client_holdout_umap.py
```

Each figure colors its training partition by source/class and shows the normal-real
records actually used for optimization colored by client. UMAP is fit only on
normal-real training features with that scenario's shared training scaler; the
remaining training records are projected onto that embedding. Client-aware
partition construction can produce different train rows between scenarios, so
each UMAP is fitted separately and the coordinates should not be compared across
the two figures. These plots are descriptive, not model-quality metrics: UMAP
preserves local neighborhoods approximately, and its axes and global distances
should not be given physical meaning. The plots are not used for training or
model selection. Method reference: McInnes, Healy, and Melville, “UMAP:
Uniform Manifold Approximation and Projection for Dimension Reduction”
(2018), https://doi.org/10.48550/arXiv.1802.03426.

For a representation learned by the autoencoder, plot the selected FedAvg
seed-42 checkpoint's latent codes:

```bash
python scripts/plot_fedavg_latent_umap.py
```

This writes separate IID and quantity-skew images under
`results/figures/sensors_only_client_holdout/umap/latent_fedavg_seed42/`.
The reducer is fit on latent codes of normal-real training records, then
projects all training records. Anomalies are shown for inspection but were not
used to optimize the normal-only autoencoder. The latent plots use the existing
scenario-specific training splits and checkpoints, not the matched-cohort
follow-up models. Coordinates from different scenarios or from the input-space
and latent-space projections are not directly comparable.

## Matched-cohort follow-up experiment

The original IID and quantity-skew runs were split independently by scenario.
For a cleaner scenario comparison, the follow-up uses a single vehicle-level
train/validation/test split and a fixed row-to-client assignment for validation
and test. Only training-client assignments change between IID and
quantity-skew. Whole real vehicles remain indivisible, so quantity-skew also
changes training vehicle composition; it is not an isolated intervention on
sample count.

With the current data and seed 42, the common split places 27 real vehicles in
training, six in validation, and six in test. The six held-out vehicles are
assigned one per client, identically in both scenarios. The synthetic rows
assigned to each validation/test client are also fixed across scenarios. Each
partition is balanced by construction (50% normal and 50% anomalous); this is a
controlled evaluation cohort, not a natural-prevalence estimate. The current
test cohort has one real vehicle per client, so per-client results remain
sensitive to that vehicle's characteristics.

Prepare the common split once, then run the resumable training/evaluation
battery:

```bash
python -m src.data.shared_client_holdout
bash scripts/run_shared_client_holdout_experiments.sh
```

The runner defaults to seeds 7, 21, and 42. A smaller pilot can use one seed:

```bash
bash scripts/run_shared_client_holdout_experiments.sh --skip-fedprox 42
```

The `--skip-fedprox` pilot trains Centralized, Local, and FedAvg only. The full
runner trains those methods for both scenarios, selects the score aggregation
using validation only, evaluates p99 and p95 on the fixed test cohorts, and runs
a FedProx `mu` grid separately for each scenario. The
validation-selected FedProx `mu` is then trained/evaluated on the requested
seeds. FedProx screening uses the configured selection seed 42. Activate the
project virtual environment before running the battery; it includes federated
Flower runs and is computationally intensive.

All new artifacts are isolated from the original experiment:

- Shared partition and assignments:
  `data/splits/sensors_only_client_holdout_shared/`
- Runs, metrics, and figures:
  `results/{runs,metrics,figures}/sensors_only_client_holdout_shared/`
- Split counts, checks, and test-assignment fingerprint:
  `results/splits/sensors_only_client_holdout_shared/shared_split_report.json`
- Per-client data summaries:
  `results/clients/sensors_only_client_holdout_shared/`

The preparation command refuses to overwrite an existing shared split. Use
`--overwrite` only when intentionally regenerating this dedicated split; doing
so invalidates all results trained against its previous assignments.

## Result interpretation

See [client_holdout_results.md](./client_holdout_results.md) for the
three-seed Centralized/Local/FedAvg results and
[fedprox_client_holdout_results.md](./fedprox_client_holdout_results.md) for
the FedProx screening and fixed-`mu` results on seeds 7, 21, and 42. These are
descriptive results on one fixed partition, not statistical evidence across
vehicle populations. Test anomalies are synthetic, and each held-out client has
only one or two real vehicles.
