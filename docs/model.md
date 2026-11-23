# Model and training setup

## Current sensors-only model

The current model is an undercomplete dense autoencoder. Its input dimension is
the 105-feature list in `data/processed/sensors_only/feature_columns.json`.

| Item | Current value |
|---|---|
| Architecture | 105 → 32 → 8 → 32 → 105 |
| Hidden activation | ReLU |
| Output | Linear |
| Trainable parameters | 7,409 |
| Training objective | Mean squared reconstruction error |
| Optimizer | Adam, learning rate 0.001 |
| Batch size | 32 |
| Device | CPU |
| Training data | Normal records from real vehicles only |
| Config | `configs/model.yaml` |

The dimensions are determined by `configs/model.yaml` and the loaded feature
count. The output layer reconstructs all input features; labels and source
metadata are used only to select eligible training/validation rows, never as
model inputs or targets.

## Centralized, local, FedAvg and FedProx

- **Centralized:** one model is trained on the pooled normal-real training
  records.
- **Local:** one independent model is trained per client on that client's
  normal-real training records.
- **FedAvg:** all six clients train locally for five epochs per round and the
  server averages model parameters weighted by local example count.
- **FedProx:** the same communication and weighted aggregation scheme as FedAvg,
  with proximal penalty `(mu / 2) * ||w - w_global||^2` during each client's
  local optimization.

Federated training runs for up to 200 rounds, with all six clients participating
each round and patience 10 based on normal-real validation reconstruction loss.
The best validation-loss checkpoint is kept. The current model has a 105-feature input, so its parameter count differs from
earlier full-feature or latent-16 experiments.

The optimizer state is retained between a client's rounds in Flower
`context.state`; global model weights are refreshed each round. The current
Flower processes run on one host to study the FL algorithm, not on six physical
vehicles.

## Anomaly score

The fixed score used for the current client-aware reports is the mean of the
largest 5% of per-feature squared reconstruction errors (`top_5_percent`).
For 105 features, the score averages the largest six feature errors for each
record. This aggregation was selected using validation PR-AUC and then held
fixed across methods and client scenarios.

Higher scores indicate larger reconstruction errors and are treated as more
anomalous. Selection of the score aggregation and operating threshold is
described in [evaluation.md](./evaluation.md).
