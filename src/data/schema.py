import re
from dataclasses import dataclass
from typing import Any

import pandas as pd

ASSIGNMENT_UNITS = ("source", "record")


@dataclass(frozen=True)
class CategorySpec:
    name: str
    family: str
    source_pattern: str
    assignment_unit: str

    def __post_init__(self) -> None:
        if self.assignment_unit not in ASSIGNMENT_UNITS:
            raise ValueError(
                f"Category {self.name}: assignment_unit must be one of {ASSIGNMENT_UNITS}"
            )

    def matches(self, sources: pd.Series) -> pd.Series:
        return sources.astype(str).str.fullmatch(self.source_pattern)


@dataclass(frozen=True)
class DataSchema:
    source_column: str
    label_column: str
    metadata_columns: tuple[str, ...]
    normal_label: int
    anomaly_label: int
    categories: tuple[CategorySpec, ...]

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> "DataSchema":
        columns = config["columns"]
        return cls(
            source_column=columns["source"],
            label_column=columns["label"],
            metadata_columns=tuple(columns["metadata"]),
            normal_label=int(config["labels"]["normal"]),
            anomaly_label=int(config["labels"]["anomaly"]),
            categories=tuple(CategorySpec(**spec) for spec in config["categories"]),
        )

    @property
    def category_names(self) -> list[str]:
        return [category.name for category in self.categories]

    def category(self, name: str) -> CategorySpec:
        for category in self.categories:
            if category.name == name:
                return category
        raise KeyError(name)

    def validate_columns(self, frame: pd.DataFrame) -> None:
        required = {self.source_column, self.label_column}
        missing = sorted(required - set(frame.columns))
        if missing:
            raise ValueError(f"Missing required columns: {missing}")

    def candidate_feature_columns(self, frame: pd.DataFrame) -> list[str]:
        metadata = set(self.metadata_columns)
        return [
            column
            for column in frame.columns
            if column not in metadata and pd.api.types.is_numeric_dtype(frame[column])
            and not pd.api.types.is_bool_dtype(frame[column])
        ]

    def non_numeric_non_metadata_columns(self, frame: pd.DataFrame) -> list[str]:
        metadata = set(self.metadata_columns)
        features = set(self.candidate_feature_columns(frame))
        return [
            column for column in frame.columns if column not in metadata and column not in features
        ]

    def assign_categories(self, sources: pd.Series) -> pd.Series:
        """Map each source value to a category name; unmatched sources are NA."""
        matches = pd.DataFrame(
            {category.name: category.matches(sources) for category in self.categories},
            index=sources.index,
        )
        ambiguous = matches.sum(axis=1) > 1
        if ambiguous.any():
            values = sorted(sources[ambiguous].astype(str).unique())
            raise ValueError(f"Sources matching more than one category: {values}")
        assigned = matches.idxmax(axis=1).where(matches.any(axis=1))
        return assigned.astype("string")


def natural_sort_key(value: str) -> tuple:
    return tuple(int(part) if part.isdigit() else part for part in re.split(r"(\d+)", value))
