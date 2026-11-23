# Evaluation protocol

The current implementation is in `src/evaluation/metrics.py`,
`src/evaluation/report.py`, and `src/evaluation/client_test.py`.

## Scores and fixed aggregation

For each record, the autoencoder produces per-feature squared reconstruction
errors. The fixed current anomaly score is the mean of the largest 5% of those
errors (`top_5_percent`; six features for a 105-feature input). The aggregation
was selected by validation PR-AUC using the centralized normal-only model and
then frozen across methods and scenarios. It is not selected from test results.

Higher scores are more anomalous. PR-AUC (average precision) and ROC-AUC are
computed from continuous scores and do not depend on the classification
threshold.

## Thresholds

For the current normal-only evaluation, each model's threshold is the requested
quantile (p99 by default) of its scores on normal-real validation records:

- **p99:** primary, conservative operating point.
- **p95:** sensitivity analysis, with a higher expected detection rate and
  potentially more false positives.

Thresholds are computed from validation records and applied to each held-out
test record using `score >= threshold`. A pooled normal-real calibration target
does not guarantee the same false-positive rate for every held-out client or
vehicle. Changing p99 to p95 changes evaluation only and does not require
retraining.

The labeled synthetic anomalies in validation are not used to calibrate these
normal-only thresholds. The training/validation selection does use source and
label metadata to identify normal real records, so the procedure is not entirely
label-free.

## Client-level test protocol

The current protocol creates a separate held-out cohort for each of six clients.
The global centralized/FedAvg/FedProx model is evaluated on all six cohorts.
Each local model is evaluated on its own client's cohort. For a global model,
the score aggregation and threshold are common across clients for a given run;
local models have thresholds calibrated for their own model scores on the
normal-real validation data.

The test cohorts are balanced 50/50 by construction. Thus 0.5 is the random
ranking reference for PR-AUC on these cohorts, not a claim about anomaly
prevalence in a real fleet. Test anomalies are synthetic; report recall as
synthetic-anomaly recall, not real-fault detection.

Per-client outputs include `validation_*` and `test_*` metrics, sample counts,
confusion counts, score means and threshold information. The client test
summary is descriptive: each client has only one or two real held-out vehicles.

## Reporting and model selection

- Choose `top_5_percent` using validation PR-AUC and hold it fixed.
- Choose FedProx `mu` using normal-real validation loss only. The initial
  quantity-skew screening selected `mu=0.1` on seed 42 from `[0.001, 0.01,
  0.1, 1.0]`. The selected fixed `mu` has been run on seeds 7, 21, and 42.
- Keep test metrics out of score aggregation, `mu`, checkpoint, and threshold
  selection.
- Report client-level PR-AUC first, ROC-AUC as a secondary ranking metric, and
  p99 operating point as primary; show p95 as sensitivity analysis.
- Show centralized, local, FedAvg, and FedProx together when available.
- Means and standard deviations over seeds describe variation over initialization
  and optimization on the same split; three seeds are not confidence intervals
  and do not represent new vehicle populations.

## Important limitations

- Real records are normal only, while anomalous records are synthetic; source and
  class are confounded. These results do not validate real fault detection.
- The client-aware quantity-skew setting changes both number and identity of
  vehicles per client, so it is not a pure sample-count intervention.
- A global threshold can transfer poorly to an individual held-out vehicle. The
  quantity-skew client-2 p95 case is a particularly visible example.
- Keeping raw training records local is not a formal privacy guarantee; model
  updates and metrics are exchanged.

Detailed current results are in
[client_holdout_results.md](./client_holdout_results.md) and
[fedprox_client_holdout_results.md](./fedprox_client_holdout_results.md).
