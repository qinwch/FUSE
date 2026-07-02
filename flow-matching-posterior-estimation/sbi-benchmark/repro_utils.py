from __future__ import annotations

import random
from pathlib import Path
from typing import Iterable

import numpy as np
import torch


def observation_numbers(num_observations: int) -> Iterable[int]:
    if num_observations < 1:
        raise ValueError("--num_observations must be at least 1")
    return range(1, num_observations + 1)


def resolve_optional_path(value: str | None) -> Path | None:
    if value is None or value == "":
        return None
    return Path(value)


def dataset_files_exist(dataset_dir: Path | None) -> bool:
    if dataset_dir is None:
        return False
    return (dataset_dir / "x.npy").exists() and (dataset_dir / "theta.npy").exists()


def resolve_generation_batch_size(value: int | None, fallback: int) -> int:
    if value is None:
        return fallback
    return value


def should_set_seed(seed: int | None) -> bool:
    return seed is not None


def ensure_directory(path: Path | None) -> None:
    if path is not None:
        path.mkdir(parents=True, exist_ok=True)


def set_global_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
