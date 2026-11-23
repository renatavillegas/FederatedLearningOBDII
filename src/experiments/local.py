import argparse
import time
from typing import Any

from src.config import load_yaml
from src.experiments.common import (
    ExperimentData,
    RunRecorder,
    StoppingConfig,
    fit_normal_real_preprocessing,
    fit_with_validation,
    run_directory,
    select_normal_real_records,
)
from src.models.model import build_model, model_size
from src.models.trainer import LocalTrainer, TrainingConfig, set_seed

METHOD = "local"
NORMAL_ONLY_METHOD = "local_normal_only"


class LocalExperiment:
    """Trains independent local autoencoders, optionally on normal-real records with a shared scaler."""

    def __init__(self, data: ExperimentData, model_config: dict[str, Any], experiments_config: dict[str, Any]) -> None:
        self.data = data
        self.model_config = model_config
        self.experiments_config = experiments_config

    def run(self, scenario: str, seed: int, normal_only_real: bool = False) -> dict[str, Any]:
        baselines = self.experiments_config["baselines"]
        stopping = StoppingConfig(int(baselines["max_epochs"]), int(baselines["patience"]), float(baselines["min_delta"]))
        method = NORMAL_ONLY_METHOD if normal_only_real else METHOD
        directory = run_directory(self.experiments_config["outputs"]["runs_dir"], scenario, method, seed)
        validation = (
            self.data.normal_real_partition("validation")
            if normal_only_real
            else self.data.partition("validation")
        )
        if normal_only_real:
            feature_columns, shared_scaler, feature_metadata = fit_normal_real_preprocessing(self.data)
        else:
            feature_columns = self.data.feature_columns
            shared_scaler = None
            feature_metadata = None

        clients = {}
        for position, client in enumerate(self.data.client_names(scenario)):
            client_seed = seed + position
            set_seed(client_seed)
            client_frame = self.data.client_rows(scenario, client)
            if normal_only_real:
                train_frame = select_normal_real_records(client_frame, self.data.schema)
                if shared_scaler is None:
                    raise RuntimeError("Normal-real preprocessing was not initialized")
                scaler = shared_scaler
            else:
                train_frame = client_frame
                scaler = self.data.client_scaler(scenario, client)
            train_features = self.data.features(train_frame, scaler)
            validation_features = self.data.features(validation, scaler)

            model = build_model(self.model_config, len(feature_columns), seed)
            trainer = LocalTrainer(model, TrainingConfig.from_config(self.model_config["training"]), client_seed)
            start = time.perf_counter()
            fit = fit_with_validation(trainer, train_features, validation_features, stopping)

            client_summary = {
                "n_train": len(train_features),
                "best_epoch": fit.best_epoch,
                "best_validation_loss": fit.best_validation_loss,
                "epochs_run": len(fit.history),
                "stopped_early": fit.stopped_early,
                "training_seconds": fit.training_seconds,
                "total_seconds": time.perf_counter() - start,
            }
            if normal_only_real:
                client_summary["n_vehicles"] = int(train_frame[self.data.schema.source_column].nunique())
            clients[client] = client_summary
            recorder = RunRecorder(directory / client)
            recorder.save_json("history.json", {"epochs": fit.history})
            recorder.save_model(fit.best_state)

        summary = {
            "scenario": scenario,
            "method": method,
            "seed": seed,
            "initial_weights_seed": seed,
            "training_population": "normal_real_only" if normal_only_real else "all_training_records",
            "validation_population": "normal_real_only" if normal_only_real else "all_validation_records",
            "normal_label_used_for_training_selection": normal_only_real,
            "scaler": "normal_real_train_global" if normal_only_real else "client",
            "n_features": len(feature_columns),
            "stopping": vars(stopping),
            "n_validation": len(validation),
            "clients": clients,
            **model_size(model),
        }
        recorder = RunRecorder(directory)
        if normal_only_real:
            if shared_scaler is None or feature_metadata is None:
                raise RuntimeError("Normal-real preprocessing was not initialized")
            shared_scaler.save(directory / "scaler.json")
            recorder.save_json("features.json", feature_metadata)
        recorder.save_json("model_config.json", self.model_config)
        recorder.save_json("summary.json", summary)
        return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Independent local autoencoders, one per client.")
    parser.add_argument("--scenario", required=True)
    parser.add_argument("--data-config", default="configs/data.yaml")
    parser.add_argument("--model-config", default="configs/model.yaml")
    parser.add_argument("--experiments-config", default="configs/experiments.yaml")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument(
        "--normal-only-real",
        action="store_true",
        help="Train local models only on normal real records, keeping each vehicle assigned to one client",
    )
    args = parser.parse_args()
    data = ExperimentData(args.data_config)
    experiment = LocalExperiment(data, load_yaml(args.model_config), load_yaml(args.experiments_config))
    experiment.run(
        args.scenario,
        args.seed if args.seed is not None else data.seed,
        normal_only_real=args.normal_only_real,
    )


if __name__ == "__main__":
    main()
