from pathlib import Path
from typing import Any

import yaml

_DEFAULT_ROOT = Path(__file__).resolve().parents[1]
_project_root = _DEFAULT_ROOT


def project_root() -> Path:
    return _project_root


def set_project_root(path: str | Path | None) -> None:
    """Override the project root, used when the code runs from an installed Flower App bundle."""
    global _project_root
    _project_root = Path(path).resolve() if path else _DEFAULT_ROOT


def resolve_path(path: str | Path) -> Path:
    path = Path(path)
    return path if path.is_absolute() else _project_root / path


def load_yaml(path: str | Path) -> dict[str, Any]:
    with resolve_path(path).open(encoding="utf-8") as handle:
        return yaml.safe_load(handle)
