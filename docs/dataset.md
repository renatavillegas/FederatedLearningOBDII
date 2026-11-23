# Dataset and client-aware partitions

## Source and labels

- Input: `data/raw/Vehicle_Record.csv`, copied from
  `datasetKagglehub/2/Vehicle_Record.csv`. Its SHA-256 is recorded in the audit
  output.
- Reference dataset:OBD-II Sensor Anomaly and Failure Prediction Dataset (Kaggle, Badrnezhad, s.d.).
- `source` identifies the real-vehicle sources (`live1` ... `live39`) or synthetic
  pools (`synthetic_cat`, `synthetic_gray`).
- `anomaly_label` is 0 for normal and 1 for anomaly. Real-vehicle records are
  normal; synthetic records provide the anomalous test examples.
- `failure_type`, severity and time-to-failure columns are metadata, not model
  inputs.

The source and label are strongly associated: real rows are normal,
`synthetic_cat` rows are anomalies, and `synthetic_gray` is predominantly
anomalous. The experiment therefore cannot establish detection of real vehicle
faults.

## Feature selection: sensors-only

The current configs are `configs/data_sensors_client_holdout_iid.yaml` and
`configs/data_sensors_client_holdout_quantity.yaml`. They begin with **120
candidate numeric sensor features**, represented as five statistics
(`mean`, `std`, `max`, `min`, `roc`) for each signal. They exclude three
context/counting signals:

- `TIME_RUN_WITH_MIL_ON`
- `TIME_SINCE_TROUBLE_CODES_CLEARED`
- `DISTANCE_TRAVELED_WITH_MIL_ON`

These are 15 feature columns in total (three signals × five statistics).
The resulting `data/processed/sensors_only/feature_columns.json` contains
**105 features**. No additional feature was removed as constant in this current
feature list. Source, label, and other metadata columns are also excluded.

The feature list is shared across the client-aware scenarios and was determined
without using validation or test data to choose model inputs.

## Train/validation/test protocol

The client-aware protocol is implemented in `src/data/client_holdout.py`.
Unlike the earlier global partition-then-client-assignment workflow, it first
assigns whole real-vehicle groups to one of six clients, then splits each
client's vehicle groups into train, validation, and test. A vehicle cannot
appear in more than one partition or client. The target partition ratios are
70/15/15; indivisible vehicles make the realized ratios approximate.

Synthetic records do not have vehicle IDs, so they are assigned at the record
level. Synthetic normal and anomalous records are allocated separately. The
anomaly allocation targets the configured anomaly fraction for each
client-partition; the current scenarios target 50%. This produces balanced
evaluation cohorts by construction, not a natural fleet prevalence estimate.

The held-out test cohorts contain:

| Scenario | Test vehicles by client | Test normals and anomalies |
|---|---|---|
| IID | 1 per client | Equal counts per client: 178, 100, 168, 190, 137, 157 of each class |
| Quantity-skew | 1, 1, 1, 1, 2, 2 | Equal counts per client: 165, 120, 130, 159, 365, 351 of each class |

Real test vehicles contain only normal records. Synthetic records are the source
of test anomalies and some synthetic normal rows are included to meet the
configured cohort balance. The client-level test files and reports are saved
under `data/splits/sensors_only_client_holdout_{iid,quantity}/` and
`results/clients/sensors_only_client_holdout_{iid,quantity}/`.

The scenario-specific client-aware split builders assign vehicles to clients
before splitting each client's vehicle groups. Consequently, the realized
train/validation/test vehicle sets differ between `iid` and `quantity_skew`
despite sharing the configured seed. In the current outputs, IID has 27/6/6
real vehicles in train/validation/test, while quantity-skew has 24/7/8; the
two scenarios have no real test vehicles in common. Thus, cross-scenario
performance differences are also affected by different train and held-out
vehicles, not only by the client quantity allocation. Method comparisons within
each scenario remain matched to that scenario's split, but the IID-versus-
quantity-skew contrast is descriptive rather than a controlled quantity-only
comparison.

## What IID and non-IID mean in this study

The current client-aware comparison uses `iid` and `quantity_skew`; it does not
use a current scenario named `non_iid`.

### `iid`: approximately balanced client allocation

In `configs/experiments_sensors_client_holdout.yaml`, clients have equal
normal-real count weights and a 0.50 target anomaly fraction. Whole vehicles are
kept intact, so equal weights cannot yield identical record counts. The realized
normal-real training counts are 602, 765, 746, 616, 631, and 642, with 4–5 real
training vehicles per client.

Here, “IID” means the *allocation target is approximately balanced* in sample
volume and class ratio. It is not a claim that vehicle records are statistically
independent or that feature distributions are identical across clients.

### `quantity_skew`: non-IID by quantity and vehicle composition

The configured normal-real training-count weights are `[1, 1, 2, 2, 4, 4]`.
Since vehicle groups are indivisible and differ in size, the realized normal-real
training counts are 216, 253, 643, 612, 971, and 993, with 2, 2, 4, 4, 6, and 6
real training vehicles respectively. The largest client has about 4.6 times the
normal-real training records of the smallest.

All clients still target the same 0.50 anomaly fraction within each partition.
This isolates quantity/class-ratio targets better than label-skew, but it does
**not** isolate sample count from vehicle identity: assigning whole vehicles
also changes each client's vehicle profiles. Describe this as
**quantity-skew with vehicle heterogeneity**, not a pure sample-count effect.

### Older `non_iid` label-skew experiment

`configs/experiments_sensors.yaml` and the older `src/data/clients.py`
allocation use a different definition: the target anomaly fractions are
`[0.20, 0.30, 0.40, 0.60, 0.70, 0.80]`, with client sizes referenced to the
older IID allocation. That is label/class-proportion skew. It is not the
non-IID intervention used for the current normal-only client-aware results:
the current training filters anomalies out, so varying client anomaly fractions
would not vary the local training label mix.

## Scaling and preprocessing

For the current normal-only runs, a shared z-score scaler is fitted on the union
of normal-real training records only. The same saved scaler is used by all
clients, centralized/local/federated models, and validation/test evaluation.
No validation or test rows are used to fit it. Features are converted to
`float32`; non-finite feature values are imputed using training-set medians.

This centralized scaler is a controlled comparison choice, not a fully
decentralized preprocessing method. Scaler artifacts are saved under
`data/processed/sensors_only_client_holdout_{iid,quantity}/scalers/`.
