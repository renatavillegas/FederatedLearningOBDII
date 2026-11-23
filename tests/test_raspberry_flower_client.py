import json

import numpy as np
import pytest

from src.config import set_project_root
from src.federated.client import FederatedClient


def _client_bundle(tmp_path):
    root = tmp_path / "pi"
    data_dir = root / "data"
    config_dir = root / "configs"
    data_dir.mkdir(parents=True)
    config_dir.mkdir()
    (config_dir / "model.yaml").write_text(
        "architecture:\n"
        "  encoder_dims: [4]\n"
        "  latent_dim: 2\n"
        "  activation: relu\n"
        "training:\n"
        "  loss: mse\n"
        "  optimizer: adam\n"
        "  learning_rate: 0.001\n"
        "  batch_size: 4\n"
        "  device: cpu\n",
        encoding="utf-8",
    )
    (data_dir / "bundle_metadata.json").write_text(
        json.dumps(
            {
                "scenario": "iid",
                "client": "client_4",
                "normal_only_training": True,
                "n_features": 6,
            }
        ),
        encoding="utf-8",
    )
    np.savez_compressed(
        data_dir / "edge_client_data.npz",
        train_x=np.ones((5, 6), dtype=np.float32),
        inference_x=np.zeros((3, 6), dtype=np.float32),
    )
    return root, data_dir


def test_edge_client_loads_only_its_prepared_shard(tmp_path):
    root, data_dir = _client_bundle(tmp_path)
    set_project_root(root)
    try:
        client = FederatedClient(
            "client_4",
            "iid",
            "unused-data-config.yaml",
            "configs/model.yaml",
            normal_only_real=True,
            client_data_dir=str(data_dir),
        )
    finally:
        set_project_root(None)

    assert client.position == 3
    assert client.n_features == 6
    assert tuple(client.features.shape) == (5, 6)
    assert client.inference_features is not None
    assert tuple(client.inference_features.shape) == (3, 6)


def test_edge_client_rejects_bundle_for_different_client(tmp_path):
    root, data_dir = _client_bundle(tmp_path)
    metadata_path = data_dir / "bundle_metadata.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    metadata["client"] = "client_3"
    metadata_path.write_text(json.dumps(metadata), encoding="utf-8")
    set_project_root(root)
    try:
        with pytest.raises(ValueError, match="does not match"):
            FederatedClient(
                "client_4",
                "iid",
                "unused-data-config.yaml",
                "configs/model.yaml",
                normal_only_real=True,
                client_data_dir=str(data_dir),
            )
    finally:
        set_project_root(None)
