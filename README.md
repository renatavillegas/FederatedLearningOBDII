# Federated anomaly detection on OBD-II data

This project studies federated learning (FL) behavior for a sensors-only OBD-II
anomaly-detection experiment. It is a research/seminar prototype, not a production
fault detector.

## Current study in one page

- **Inputs:** 120 candidate numeric sensor statistics; 15 columns from three
  context/counting signals are excluded, leaving **105 model features**.
- **Training objective:** a 105-32-8-32-105 autoencoder trained only on normal
  records from real vehicles (`normal-real-only`).
- **Federated setup:** one Flower server and six logical clients. Raw records stay
  on their assigned client; model parameters and training metrics are exchanged.
  The current Flower deployment runs as processes on one host, not six physical
  devices.
- **Current client-aware scenarios:** approximate quantity-IID and `quantity_skew`
  (non-IID by normal-real sample count and vehicle composition). Both target a
  50% anomaly fraction in each client/partition for balanced evaluation. The
  quantity-skew weights are `[1, 1, 2, 2, 4, 4]`.
- **Methods:** centralized and independent local models, FedAvg, and FedProx.
  `mu=0.1` was selected for quantity-skew on seed 42 by normal-real validation
  loss. The fixed `mu` has now been run on seeds 7, 21, and 42.
- **Evaluation:** fixed `top_5_percent` score aggregation; p99 is the primary
  operating point and p95 is a sensitivity analysis. Thresholds use normal-real
  validation scores only. Test anomalies are synthetic, so recall is not evidence
  of real-fault detection.

## Start here

- [docs/dataset.md](docs/dataset.md): data, feature selection, and client assignment.
- [docs/model.md](docs/model.md): current model and training setup.
- [docs/architecture.md](docs/architecture.md): Flower components and editable
  [architecture diagram](docs/architecture.drawio).
- [docs/evaluation.md](docs/evaluation.md): score, threshold, metrics, and caveats.
- [docs/experiments.md](docs/experiments.md): current commands and experiment order.
- [docs/client_holdout_results.md](docs/client_holdout_results.md): Centralized,
  Local, and FedAvg results for the client-aware battery.
- [docs/fedprox_client_holdout_results.md](docs/fedprox_client_holdout_results.md):
  FedProx quantity-skew screening and fixed-`mu` results.

Use the experiment configs under `configs/data_sensors_client_holdout_*.yaml` and
`configs/experiments_sensors_client_holdout.yaml` for the current client-aware
comparison. Earlier `iid`/`non_iid` experiments use a different assignment
protocol; see the explicit distinction in the experiment documentation.
