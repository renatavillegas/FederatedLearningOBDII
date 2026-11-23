# Current federated architecture

**Editable diagram:** [architecture.drawio](./architecture.drawio), viewable and
exportable in [diagrams.net](https://app.diagrams.net/).

## Components

| Component | Implementation | Role |
|---|---|---|
| Flower SuperLink | `flower-superlink` | Coordinates the local Flower deployment |
| Six SuperNodes | `flower-supernode`, one `client-id` per node | Host six logical client apps |
| ServerApp | `src/federated/server.py` | Initializes global weights, orchestrates rounds, validates, stops and saves best checkpoint |
| ClientApp | `src/federated/client.py` | Trains on one client's normal-real records and returns updated weights and metrics |
| Aggregation | `src/federated/fedavg.py`, `src/federated/fedprox.py` | FedAvg or FedProx strategy; weighted by local example count |
| Payload definitions | `src/federated/protocol.py` | Shared model, config, metric, scaler and optimizer-state records |

`scripts/run_federated.sh` starts the SuperLink and six SuperNodes as separate
processes on localhost, submits a Flower run, then stops the processes. This is
a **single-host simulation**: clients are logical partitions and do not represent
six physical vehicles or devices.

## Federated round

1. The server sends current global model weights and round configuration to all
   six clients. FedProx also sends `proximal-mu`.
2. Each client trains for five local epochs on its own normal-real training
   records using the common scaler. Raw records are not sent.
3. Each client returns its updated model weights and summary metrics
   (`num-examples`, training loss and time, client index).
4. The server aggregates model weights weighted by `num-examples` (FedAvg and
   FedProx both use weighted aggregation; FedProx's proximal term is applied
   locally).
5. The server evaluates reconstruction loss on normal-real validation records.
   The lowest-loss round is checkpointed; training stops after 10 rounds without
   improvement or at 200 rounds.
6. The saved global checkpoint is evaluated offline on each held-out client test
   cohort. This evaluation does not send test records through Flower.

## Preprocessing and evaluation boundary

The current normal-only protocol fits one shared scaler using pooled
normal-real training rows. Local, centralized and federated runs use this scaler.
The server stores/reads the experiment's validation data for checkpoint
selection; the test cohorts remain for final reporting only.

This keeps the comparison controlled but does not model decentralized scaler
fitting. FL prevents raw training records from being transmitted in this
implementation; it does not provide differential privacy or other formal
privacy guarantees for updates.

## Historical setup distinction

Older `iid`/`non_iid` experiment commands used `src/data/clients.py` and
`configs/experiments_sensors.yaml`. Their `non_iid` setting varies client
anomaly fractions. The current client-aware normal-only study instead uses
`iid` and `quantity_skew`; see [dataset.md](./dataset.md) for the exact
definitions.
