# FedProx client-aware results — sensors-only quantity-skew

## Protocol and selection

This report covers the client-aware sensors-only `quantity_skew` FedProx
screening and the fixed-`mu` runs on seeds 7, 21, and 42. The normal-only
autoencoder, six client assignments, shared normal-real training scaler, 105
features, and `top_5_percent` score aggregation are unchanged from the FedAvg
comparison.

The 105 inputs are obtained by excluding 15 columns from three context/counting
signals out of 120 candidate sensor statistics. Training uses normal records from
real vehicles only. The client-aware `quantity_skew` setup targets normal-real
training weights `[1, 1, 2, 2, 4, 4]`; it is non-IID by quantity and vehicle
composition, while the target anomaly fraction remains 0.50 in all client
partitions.

The four `mu` candidates were selected using the predeclared criterion:
minimum `best_validation_loss` on the normal-real validation data. No test metric
was used to select `mu`.

| `mu` | Best validation loss | Best round | Rounds run | Selection |
|---:|---:|---:|---:|---|
| 0.001 | 0.217785 | 162 | 172 |  |
| 0.01 | 0.229893 | 99 | 109 |  |
| **0.1** | **0.207706** | **195** | **200** | **Selected** |
| 1.0 | 0.393747 | 200 | 200 |  |

The candidates and their validation selection values are in
[`quantity_skew_seed_42.csv`](../results/metrics/sensors_only_client_holdout/fedprox_selection/quantity_skew_seed_42.csv).
The selected run is
[`fedprox_mu_0.1_normal_only/seed_42`](../results/runs/sensors_only_client_holdout/quantity_skew/fedprox_mu_0.1_normal_only/seed_42/summary.json).

## Matched-seed macro comparison

The table reports unweighted means over the six clients, first averaged within
each seed, followed by mean ± standard deviation across seeds 7, 21, and 42.
These are descriptive results over a fixed client partition, not a significance
test or variation across vehicle populations. Ranking metrics do not depend on
the operating threshold; p99 and p95 change only the decision threshold. Test
sets are balanced 50/50 by construction, so 0.5 is the random-ranking reference
for PR-AUC in these test cohorts.

| Method | Test PR-AUC | Test ROC-AUC | Recall p99 | FPR p99 | Recall p95 | FPR p95 |
|---|---:|---:|---:|---:|---:|---:|
| Centralized | 0.7123 ± 0.0104 | 0.6241 ± 0.0151 | 16.08% ± 1.99% | 0.71% ± 0.11% | 40.83% ± 0.66% | 7.86% ± 1.35% |
| FedAvg | 0.6822 ± 0.0147 | 0.5760 ± 0.0096 | 12.71% ± 1.56% | 0.95% ± 0.34% | 37.79% ± 1.14% | 19.77% ± 1.35% |
| **FedProx, `mu=0.1`** | **0.6894 ± 0.0044** | **0.5854 ± 0.0067** | **13.96% ± 1.52%** | **0.92% ± 0.29%** | **37.33% ± 0.97%** | **18.18% ± 1.75%** |
| Local | 0.5859 ± 0.0039 | 0.5320 ± 0.0085 | 4.35% ± 0.87% | 2.83% ± 0.03% | 24.65% ± 0.10% | 14.20% ± 0.44% |

### Interpretation

- Across three seeds, FedProx `mu=0.1` has slightly higher macro ranking metrics
  than FedAvg: PR-AUC by 0.0072 and ROC-AUC by 0.0094. Seed-level variation and
  the small number of seeds mean this is descriptive, not evidence of a
  statistically reliable improvement.
- At p99, FedProx's macro recall is 1.25 percentage points higher than FedAvg's,
  while macro FPR is nearly the same (0.92% vs. 0.95%). By seed, however, the
  p99 FPR difference changes direction, so the aggregate should not be read as a
  guaranteed per-run benefit.
- At p95, FedProx lowers macro FPR from 19.77% to 18.18% (1.59 percentage
  points), while macro recall is 0.46 points lower. This suggests a modest
  operating-point trade-off, not a general resolution of threshold transfer.
- The centralized model remains ahead of FedProx in PR-AUC and ROC-AUC. FedProx
  remains ahead of the independent local baseline on those metrics.
- The test data contain synthetic anomalies; recall here is not evidence of
  detection of real vehicle failures.

## Client-level comparison: FedAvg vs selected FedProx

PR-AUC is threshold-independent. Operating-point entries are `recall / FPR`.
FPR is reported as a percentage. Each value is the mean across seeds 7, 21, and
42 on the same client test cohorts.

| Client | PR-AUC FedAvg | PR-AUC FedProx | p99 FedAvg | p99 FedProx | p95 FedAvg | p95 FedProx |
|---|---:|---:|---:|---:|---:|---:|
| client 1 | 0.779 | 0.785 | 13.9% / 1.2% | 15.8% / 1.2% | 39.2% / 2.8% | 38.6% / 4.0% |
| client 2 | 0.579 | 0.599 | 12.2% / 1.9% | 13.1% / 1.7% | 43.3% / 84.7% | 43.3% / 74.7% |
| client 3 | 0.628 | 0.631 | 14.1% / 0.0% | 16.7% / 0.0% | 34.1% / 12.1% | 34.1% / 12.1% |
| client 4 | 0.658 | 0.665 | 12.4% / 1.1% | 13.4% / 1.1% | 36.7% / 4.0% | 36.1% / 3.6% |
| client 5 | 0.781 | 0.779 | 12.5% / 0.5% | 12.4% / 0.6% | 35.8% / 4.4% | 35.4% / 5.2% |
| client 6 | 0.668 | 0.677 | 11.1% / 1.0% | 12.4% / 1.0% | 37.6% / 10.6% | 36.5% / 9.5% |

Client 2 remains the important p95 failure case: its mean FPR across the three
seeds is 74.7% for FedProx, compared with 84.7% for FedAvg. Although lower,
74.7% is still a very high false-positive rate. In seed 21, both methods have
82.5% FPR; the other two seeds account for most of the average improvement. The
cohort has 117 normal
records from one real vehicle (`live9`) and only three synthetic normal records,
so this points to poor threshold transfer to that held-out vehicle, not a stable
estimate for the client population. At p99, client 2's mean FPR is 1.7% for
FedProx and 1.9% for FedAvg.

Client effects remain mixed: FedProx has higher mean PR-AUC for clients 1–4 and
6, but slightly lower for client 5. It is not a uniform per-client improvement.

## Training cost and limitations

The selected FedProx runs used all 200 configured rounds and did not stop early.
Best validation losses were 0.219943 (seed 7, round 199), 0.240328 (seed 21,
round 200), and 0.207706 (seed 42, round 195). Runtime was 1,806.3, 1,828.6,
and 1,792.9 seconds, respectively (mean about 30.2 minutes). Each transferred
35.6 MB down and 35.6 MB up in model parameters. The model has 7,409 parameters.
These are host measurements, not Raspberry Pi measurements.

The `mu` screening itself was performed on seed 42; seeds 7 and 21 repeated the
selected fixed value without re-running the sweep. The six client cohorts share
the same underlying partition and do not count as independent repetitions.
Avoid claims that FedProx is statistically better or generally solves quantity
skew. Keep p99 as the primary operating point and p95 as sensitivity analysis.

## Result files

- Validation selection: [`fedprox_selection/quantity_skew_seed_42.csv`](../results/metrics/sensors_only_client_holdout/fedprox_selection/quantity_skew_seed_42.csv)
- Selected p99 client metrics: [`client_test_views.csv`](../results/metrics/sensors_only_client_holdout/client_test/quantity_skew/fedprox_mu_0.1_normal_only/seed_42/client_test_views.csv)
- Seed-7 p99 client metrics: [`client_test_views.csv`](../results/metrics/sensors_only_client_holdout/client_test/quantity_skew/fedprox_mu_0.1_normal_only/seed_7/client_test_views.csv)
- Seed-21 p99 client metrics: [`client_test_views.csv`](../results/metrics/sensors_only_client_holdout/client_test/quantity_skew/fedprox_mu_0.1_normal_only/seed_21/client_test_views.csv)
- Selected p95 client metrics: [`client_test_views.csv`](../results/metrics/sensors_only_client_holdout/client_test/p95/quantity_skew/fedprox_mu_0.1_normal_only/seed_42/client_test_views.csv)
- Seed-7 p95 client metrics: [`client_test_views.csv`](../results/metrics/sensors_only_client_holdout/client_test/p95/quantity_skew/fedprox_mu_0.1_normal_only/seed_7/client_test_views.csv)
- Seed-21 p95 client metrics: [`client_test_views.csv`](../results/metrics/sensors_only_client_holdout/client_test/p95/quantity_skew/fedprox_mu_0.1_normal_only/seed_21/client_test_views.csv)
- p99 plots: [`FedProx comparison figures`](../results/figures/sensors_only_client_holdout/quantity_skew/fedprox_mu_0.1/)
- p95 plots: [`FedProx p95 comparison figures`](../results/figures/sensors_only_client_holdout/quantity_skew/p95/fedprox_mu_0.1/)
- Existing three-seed FedAvg/client-aware analysis: [`client_holdout_results.md`](./client_holdout_results.md)
