import argparse
import re
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
)
from src.models.model import build_model, model_size
from src.models.trainer import LocalTrainer, TrainingConfig, set_seed

SCENARIO = "all"
METHOD = "centralized"
NORMAL_ONLY_METHOD = "centralized_normal_only"


class CentralizedExperiment:
    """Runs the mixed-data centralized baseline or a separate normal-real-only baseline."""

    def __init__(self, data: ExperimentData, model_config: dict[str, Any], experiments_config: dict[str, Any]) -> None:
        self.data = data
        self.model_config = model_config
        self.experiments_config = experiments_config

    def run(
        self, seed: int, normal_only_real: bool = False, run_tag: str | None = None
    ) -> dict[str, Any]:
        if run_tag is not None and not re.fullmatch(r"[A-Za-z0-9_-]+", run_tag):
            raise ValueError("run_tag may contain only letters, numbers, underscores, and hyphens")

        set_seed(seed)
        if normal_only_real:
            train_frame = self.data.normal_real_partition("train")
            validation_frame = self.data.normal_real_partition("validation")
            feature_columns, scaler, feature_metadata = fit_normal_real_preprocessing(self.data)
            method = f"{NORMAL_ONLY_METHOD}_{run_tag}" if run_tag else NORMAL_ONLY_METHOD
        else:
            train_frame = self.data.partition("train")
            validation_frame = self.data.partition("validation")
            scaler = self.data.global_scaler()
            feature_columns = self.data.feature_columns
            method = f"{METHOD}_{run_tag}" if run_tag else METHOD
        train_features = self.data.features(train_frame, scaler)
        validation_features = self.data.features(validation_frame, scaler)

        model = build_model(self.model_config, len(feature_columns), seed)
        trainer = LocalTrainer(model, TrainingConfig.from_config(self.model_config["training"]), seed)
        baselines = self.experiments_config["baselines"]
        stopping = StoppingConfig(int(baselines["max_epochs"]), int(baselines["patience"]), float(baselines["min_delta"]))

        start = time.perf_counter()
        fit = fit_with_validation(trainer, train_features, validation_features, stopping)
        total_seconds = time.perf_counter() - start

        summary = {
            "scenario": SCENARIO,
            "method": method,
            "seed": seed,
            "n_train": len(train_features),
            "n_validation": len(validation_features),
            "training_population": "normal_real_only" if normal_only_real else "all_training_records",
            "validation_population": "normal_real_only" if normal_only_real else "all_validation_records",
            "normal_label_used_for_training_selection": normal_only_real,
            "scaler": "normal_real_train_global" if normal_only_real else "global_train",
            "n_features": len(feature_columns),
            "stopping": vars(stopping),
            "best_epoch": fit.best_epoch,
            "best_validation_loss": fit.best_validation_loss,
            "epochs_run": len(fit.history),
            "stopped_early": fit.stopped_early,
            "training_seconds": fit.training_seconds,
            "total_seconds": total_seconds,
            **model_size(model),
        }
        recorder = RunRecorder(run_directory(self.experiments_config["outputs"]["runs_dir"], SCENARIO, method, seed))
        recorder.save_json("model_config.json", self.model_config)
        if normal_only_real:
            scaler.save(recorder.directory / "scaler.json")
            recorder.save_json("features.json", feature_metadata)
        recorder.save_json("history.json", {"epochs": fit.history})
        recorder.save_json("summary.json", summary)
        recorder.save_model(fit.best_state)
        return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Centralized autoencoder baseline.")
    parser.add_argument("--data-config", default="configs/data.yaml")
    parser.add_argument("--model-config", default="configs/model.yaml")
    parser.add_argument("--experiments-config", default="configs/experiments.yaml")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument(
        "--normal-only-real",
        action="store_true",
        help="Train a separate model using only normal records from real vehicles",
    )
    parser.add_argument(
        "--run-tag",
        help="Save a mixed-data centralized run under a distinct method directory",
    )
    args = parser.parse_args()
    data = ExperimentData(args.data_config)
    experiment = CentralizedExperiment(data, load_yaml(args.model_config), load_yaml(args.experiments_config))
    experiment.run(
        args.seed if args.seed is not None else data.seed,
        normal_only_real=args.normal_only_real,
        run_tag=args.run_tag,
    )


if __name__ == "__main__":
    main()
