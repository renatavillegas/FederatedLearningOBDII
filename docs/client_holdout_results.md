# Client-aware sensor-only results

## Study question and scope

This experiment asks how centralized, independent local, and FedAvg training compare
when normal-real OBD-II training records are distributed across six clients. It has
two client distributions: approximately IID and quantity-skew with target normal-real
training weights `[1, 1, 2, 2, 4, 4]`. The purpose is to study federated learning
behavior, not to produce a production detector.

The current sensors-only feature list starts with 120 candidate statistics and excludes
15 columns from three context/counting signals, leaving 105 inputs. Training uses only
normal records from real vehicles. In this client-aware study, `iid` denotes approximately
balanced quantity and a shared 0.50 anomaly-fraction target; `quantity_skew` is the
non-IID-by-quantity scenario, with whole-vehicle composition also changing. It is not the
older `non_iid` label-skew configuration, which varies anomaly fractions across clients.

The client-aware protocol assigns whole real vehicles to clients and partitions their
records by vehicle into train, validation, and test. Synthetic records are allocated
separately. The resulting test cohorts are balanced to 50% normal and 50% anomaly by
construction. Each cohort contains one or two held-out real vehicles.

The realized vehicle partitions also differ between scenarios: IID has 27/6/6
real vehicles in train/validation/test, while quantity-skew has 24/7/8, with no
real test vehicles shared across the scenarios. Therefore, the cross-scenario
contrast is descriptive and is not an isolated estimate of the quantity-skew
effect; each method comparison within a scenario uses that scenario's matched
split.

All runs use the same 105 sensor features, a normal-only autoencoder with latent
dimension 8, and three seeds (7, 21, 42). The selected score is the mean of the largest
5% of feature-wise squared reconstruction errors (`top_5_percent`), selected using
validation PR-AUC. The p99 and p95 operating points below are post-training threshold
evaluations on the same saved models; they are not separate training configurations.

## Main results

Values below are macro means across clients and seeds. FPR and recall depend on the
threshold; PR-AUC and ROC-AUC do not. The test label prevalence is 50%, so 0.5 is the
random-ranking reference for PR-AUC in this constructed test set.

### IID

| Method | PR-AUC | ROC-AUC | Recall p99 | FPR p99 | Recall p95 | FPR p95 |
|---|---:|---:|---:|---:|---:|---:|
| Centralized | 0.729 | 0.653 | 16.5% | 0.79% | 35.6% | 3.37% |
| FedAvg | 0.728 | 0.645 | 13.6% | 0.81% | 32.5% | 2.77% |
| Local | 0.679 | 0.601 | 4.9% | 1.02% | 20.3% | 4.57% |

FedAvg's PR-AUC is effectively the same as centralized in this IID split (difference
`-0.001`); its recall at p99 is 2.9 percentage points lower. Local models have lower
macro ranking metrics and much lower p99 recall. This supports the limited observation
that aggregation can help relative to training on each small client alone, while this
FedAvg run is close to centralized ranking performance in the IID scenario.

### Quantity-skew

| Method | PR-AUC | ROC-AUC | Recall p99 | FPR p99 | Recall p95 | FPR p95 |
|---|---:|---:|---:|---:|---:|---:|
| Centralized | 0.712 | 0.624 | 16.1% | 0.71% | 40.8% | 7.86% |
| FedAvg | 0.682 | 0.576 | 12.7% | 0.95% | 37.8% | 19.77% |
| Local | 0.586 | 0.532 | 4.3% | 2.83% | 24.6% | 14.20% |

Compared with centralized, FedAvg is lower by 0.030 PR-AUC and 0.048 ROC-AUC; its
p99 recall is 3.4 percentage points lower. Local models are lower still on macro
ranking and recall metrics. Thus, these results do not show FedAvg outperforming the
centralized model under this quantity-skew protocol. They do show that the global
federated model generally has better macro performance than independent local models.

The quantity-skew intervention also changes vehicle composition: whole vehicles are
kept together, so clients differ in both number and identity of vehicles. It should be
described as quantity-skew with vehicle heterogeneity, not as a pure sample-count-only
effect.

## Reading the client-level results

- The quantity-skew clients have 216, 253, 643, 612, 971, and 993 normal-real training
  records. The largest client has about 4.6 times the normal-real records of the smallest.
  IID clients have 602–765 such records.
- In the IID scenario, FedAvg's client-level PR-AUC is above centralized for clients 1,
  3, 4, and 6, but below it for clients 2 and 5. This is a heterogeneous response, not
  a uniform gain from federation.
- Under quantity-skew, FedAvg's PR-AUC is close to centralized for clients 1 and 5,
  and lower for clients 2, 3, 4, and 6. The small clients 2 and 3 are among the weakest
  FedAvg cohorts by ROC-AUC. This pattern is consistent with limited local observations
  and client/vehicle distribution differences, but the experiment cannot establish
  which factor caused the drop.
- Local models are trained on still less information per client and are less stable
  across clients. This makes limited local data a plausible explanation for weaker local
  baselines, not a proof that data scarcity is the only cause.
- At p99, the average FPR is low, but recall is also low (about 4–17% depending on
  method/scenario). At p95, recall rises, but the cost is not uniform: quantity-skew
  FedAvg has macro FPR 19.77%. For client 2, its p95 threshold flags 99–105 of 120
  normal test records across the three seeds. The test cohort's 120 normals include
  117 records from one real vehicle (`live9`); this is an important case study in
  threshold transfer to a held-out vehicle, not an estimate for all vehicles assigned
  to that client.

The local models use their own threshold calibrated on the same pooled normal-real
validation population, scored by that local model. Centralized and FedAvg each have
one model and one threshold per run, applied to all client test cohorts. Therefore p95
and p99 are comparable operating policies across methods, but not client-specific
calibration.

## Benefits and limitations of federated learning shown here

### Benefits evidenced by the experiment

- FedAvg combines local optimization without centralizing the raw training records in
  the federated training loop.
- In IID, FedAvg's macro PR-AUC is nearly the centralized value.
- In both scenarios, FedAvg's macro PR-AUC and recall exceed the local baseline, showing
  how aggregation can benefit clients whose individual training sets are small.
- Client-level results expose variation hidden by a single global average and identify
  where an operating threshold transfers poorly.

### Limitations and claims to avoid

- FedAvg does not match centralized performance on every client, and it falls further
  behind centralized in the quantity-skew scenario.
- A low pooled-validation FPR target is not a guarantee of low FPR on every held-out
  client/vehicle. p95 makes this distribution shift particularly visible.
- Real records are normal-only; the test anomalies are synthetic. Recall is not evidence
  of detection of real vehicle failures.
- Source and label are strongly confounded in the dataset. A model may exploit real vs.
  synthetic differences; this protocol does not remove that limitation.
- Each test client has only one or two real held-out vehicles. Do not claim population-
  wide generalization, fairness, or a reliable per-client failure rate from these
  cohorts.
- Three seeds quantify variation from initialization/training randomness on the same
  split; they do not represent new vehicle populations or confidence intervals.
- FL keeps raw data local in this experiment but does not by itself provide formal
  privacy guarantees. Model updates and communication are still exchanged.

## Would FedProx help?

It is a reasonable next comparison, especially for quantity-skew, but it is a hypothesis
to test rather than a guaranteed fix. FedProx adds a proximal penalty during local
optimization to limit client models drifting too far from the current global model. This
can help when local optimization diverges under heterogeneous data.

FedProx does not create missing examples, add vehicles to the small clients, correct a
held-out vehicle's score-distribution shift, remove synthetic-label confounding, or
guarantee that p95/p99 transfers to each client. The current quantity-skew gap could be
caused by sample volume, vehicle composition, or both. A positive FedProx result would
therefore be evidence about optimization under this protocol, not proof that data scarcity
was the cause.

Recommended bounded experiment:

1. Use FedAvg as the fixed baseline and keep the client-aware splits, model, features,
   score aggregation, local epochs, and seeds unchanged.
2. Select `mu` using normal-real validation loss only, with the predeclared grid
   `[0.001, 0.01, 0.1, 1.0]`. Do not use test PR-AUC/FPR to select `mu`.
3. Prioritize quantity-skew. Include IID as a reference if time permits.
4. Repeat the selected `mu` for seeds 7, 21, and 42, then report per-client PR-AUC,
   ROC-AUC, recall/FPR at both p99 and p95, and training/communication cost.
5. A useful outcome may be no improvement: it would show the limits of this
   proximal-optimization intervention for the observed heterogeneity.

Use the p99 results as the primary, conservative operating point and p95 as sensitivity
analysis. Show p95/p99 for centralized and FedAvg together, not just for the federated
model, so the threshold trade-off is not presented as a FedAvg-only effect. Add local
models in the per-client view as a reference. Keep PR-AUC/ROC-AUC as the threshold-
independent ranking results.

## Raspberry Pi work in parallel

The existing model is small: 7,409 parameters and a serialized state dictionary of
about 32.9 KB. On the current machine, client-aware FedAvg runs took about 13.5–29.6
minutes for 93–200 rounds and exchanged about 33.1–71.1 MB in each run in the recorded
parameter directions combined. These are host measurements, not Raspberry Pi
measurements, and do not predict Pi latency.

The Pi work can proceed in parallel with FedProx if it is framed as an **edge feasibility
pilot**, not as a production deployment:

1. **Smallest useful pilot (inference):** load the saved model on the Pi; run a
   representative sensor record/batch; measure model file size, per-record latency,
   throughput, peak RAM, CPU utilization, and temperature/throttling. Compare results
   with the host using the same inputs and preprocessing.
2. **More FL-relevant pilot (local client training):** run one Flower client on the Pi
   while the server and other simulated clients run on the host. Measure per-round local
   fit time, RAM, temperature/throttling, and communication bytes. State clearly that
   only one client is on physical edge hardware.
3. Do not claim a six-device deployment from one Pi. A real multi-device or connectivity
   study would be additional work.

Indicative effort, assuming the Raspberry Pi is available and the current CPU/PyTorch
environment can be installed: roughly half to one day for an inference-only pilot; around
two to four focused days for a reproducible Flower client-training pilot, including
environment setup and repeated measurement. Hardware/OS dependency issues could extend
this. Training on the Pi is not required to answer the main FL algorithm question; the
inference-only benchmark is the lower-risk parallel task.

## Suggested seminar narrative and figures

1. Introduce the privacy/data-locality motivation and state that the exercise studies FL,
   not a production-ready fault detector.
2. Show the client training-volume and held-out cohort distribution. Explain that
   quantity-skew also changes vehicle composition.
3. Compare centralized, FedAvg, and local macro/client-level PR-AUC and ROC-AUC for IID
   and quantity-skew. Use `client_test_auc.png`.
4. Show p99 as the primary operating point and p95 as a sensitivity analysis for all
   methods. Use `client_test_operating_point.png`; call out the quantity-skew client-2
   FedAvg p95 result and its single-vehicle test cohort.
5. Explain the trade-off: pooled validation calibration does not guarantee per-client
   operating performance under distribution shift.
6. Conclude with the observed FL benefit over local training, the IID closeness to
   centralized, the quantity-skew gap, and the limitations imposed by the small
   vehicle cohorts and synthetic anomalies.
7. Present the three-seed FedProx comparison as descriptive follow-up evidence;
   the fixed `mu=0.1` now has results for seeds 7, 21, and 42. Raspberry Pi
   measurements remain pending.

The initial quantity-skew FedProx results, including the validation-only `mu`
selection, same-seed FedAvg comparison, and p99/p95 client analysis, are documented
in [`fedprox_client_holdout_results.md`](./fedprox_client_holdout_results.md).

Figures are under
`results/figures/sensors_only_client_holdout/{iid,quantity_skew}/` and the p95 variants
are in each scenario's `p95/` subdirectory. Per-client tables are in
`results/metrics/sensors_only_client_holdout/client_test/` and
`results/metrics/sensors_only_client_holdout/client_test/p95/`.

## References for framing

- Robalino-Díaz et al. (2026), “Structural impact of non-IID heterogeneity on federated
  behavioral anomaly detection in IoT and IoMT systems,” *Frontiers in Artificial
  Intelligence*. https://doi.org/10.3389/frai.2026.1825067
- Xu et al. (2024), “PeFAD: A Parameter-Efficient Federated Framework for Time Series
  Anomaly Detection,” *KDD 2024*. https://doi.org/10.1145/3637528.3671753
- Lindskog-Münzing and Prehofer (2025), “Federated learning for automotive applications,”
  *Journal of Highway and Transportation Research and Development (English Edition)*.
  https://doi.org/10.26599/HTRD.2025.9480055

The IoT/IoMT and time-series papers support the FL/anomaly-detection framing; the
automotive review supports domain motivation. None is a direct replication of this
OBD-II dataset or a validation of real-fault detection here.
