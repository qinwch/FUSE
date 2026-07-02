from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class OrbitPaths:
    data_dir: Path
    models_dir: Path
    outputs_dir: Path
    references_dir: Path

    def ensure_directories(self) -> None:
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.models_dir.mkdir(parents=True, exist_ok=True)
        self.outputs_dir.mkdir(parents=True, exist_ok=True)
        self.references_dir.mkdir(parents=True, exist_ok=True)

    def train_dataset(self, dataset_name: str) -> Path:
        return self.data_dir / f"{dataset_name}-train.h5"

    def val_dataset(self, dataset_name: str) -> Path:
        return self.data_dir / f"{dataset_name}-val.h5"

    def test_dataset(self, dataset_name: str) -> Path:
        return self.data_dir / f"{dataset_name}-test.h5"

    def model_path(self, model_name: str) -> Path:
        return self.models_dir / model_name

    def output_path(self, output_name: str) -> Path:
        return self.outputs_dir / output_name

    def reference_path(self, reference_name: str) -> Path:
        return self.references_dir / reference_name


DEFAULT_ORBIT_PATHS = OrbitPaths(
    data_dir=Path("artifacts/orbit/datasets"),
    models_dir=Path("artifacts/orbit/models"),
    outputs_dir=Path("artifacts/orbit/outputs"),
    references_dir=Path("artifacts/orbit/references"),
)
