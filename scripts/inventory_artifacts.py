#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import hashlib
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence


ARTIFACT_EXTENSIONS = {
    ".ckpt",
    ".err",
    ".h5",
    ".hdf5",
    ".log",
    ".npy",
    ".npz",
    ".out",
    ".pth",
    ".pt",
}

SKIP_DIRS = {
    ".git",
    ".mypy_cache",
    ".pytest_cache",
    "__pycache__",
    "cache",
    "wandb",
}


@dataclass(frozen=True)
class ArtifactRecord:
    logical_name: str
    current_path: str
    intended_path: str
    size_bytes: int
    size_human: str
    sha256: str
    present: str
    required_for: str
    zenodo_url: str
    notes: str


def is_artifact_path(path: Path) -> bool:
    return path.suffix.lower() in ARTIFACT_EXTENSIONS


def format_size(size_bytes: int) -> str:
    if size_bytes < 1024:
        return f"{size_bytes} B"
    size = float(size_bytes)
    for unit in ("KiB", "MiB", "GiB", "TiB"):
        size /= 1024.0
        if size < 1024.0:
            return f"{size:.2f} {unit}"
    return f"{size:.2f} PiB"


def sha256_or_status(path: Path, max_hash_bytes: int) -> str:
    size = path.stat().st_size
    if size > max_hash_bytes:
        return f"not-computed-file-exceeds-{max_hash_bytes}-bytes"

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def should_skip(path: Path) -> bool:
    return any(part in SKIP_DIRS for part in path.parts)


def collect_artifacts(root: Path, max_hash_bytes: int) -> list[ArtifactRecord]:
    records: list[ArtifactRecord] = []
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = sorted(dirname for dirname in dirnames if dirname not in SKIP_DIRS)
        for filename in sorted(filenames):
            path = Path(dirpath) / filename
            relative = path.relative_to(root)
            if path.is_symlink() or not path.is_file() or not is_artifact_path(path):
                continue
            current_path = relative.as_posix()
            records.append(
                ArtifactRecord(
                    logical_name=relative.name,
                    current_path=current_path,
                    intended_path=f"artifacts/{current_path}",
                    size_bytes=path.stat().st_size,
                    size_human=format_size(path.stat().st_size),
                    sha256=sha256_or_status(path, max_hash_bytes=max_hash_bytes),
                    present="yes",
                    required_for="unassigned",
                    zenodo_url="",
                    notes="discovered in workspace",
                )
            )
    return records


def write_csv(records: Sequence[ArtifactRecord], output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(ArtifactRecord.__dataclass_fields__))
        writer.writeheader()
        for record in records:
            writer.writerow(record.__dict__)


def markdown_table(records: Iterable[ArtifactRecord]) -> str:
    lines = [
        "| Logical name | Current path | Intended path | Size | SHA256/status | Present | Required for | Zenodo URL | Notes |",
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for record in records:
        lines.append(
            "| "
            + " | ".join(
                [
                    record.logical_name,
                    f"`{record.current_path}`",
                    f"`{record.intended_path}`",
                    record.size_human,
                    f"`{record.sha256}`",
                    record.present,
                    record.required_for,
                    record.zenodo_url,
                    record.notes,
                ]
            )
            + " |"
        )
    return "\n".join(lines) + "\n"


def write_markdown(records: Sequence[ArtifactRecord], output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    body = """# Artifact Registry

This file records datasets, checkpoints, generated samples, logs, and other non-source artifacts discovered in the working tree. Large artifacts are not committed to Git. The intended release target is Zenodo; fill the Zenodo URL and final SHA256 fields after upload.

"""
    body += markdown_table(records)
    output.write_text(body, encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Inventory non-source artifacts for release documentation.")
    parser.add_argument("--root", default=".", help="Repository root to scan.")
    parser.add_argument("--output", required=True, help="Output file path.")
    parser.add_argument("--format", choices=("markdown", "csv"), default="markdown")
    parser.add_argument(
        "--max-hash-bytes",
        type=int,
        default=512 * 1024 * 1024,
        help="Skip SHA256 computation for files larger than this many bytes.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root = Path(args.root).resolve()
    output = Path(args.output)
    records = collect_artifacts(root, max_hash_bytes=args.max_hash_bytes)
    if args.format == "csv":
        write_csv(records, output)
    else:
        write_markdown(records, output)


if __name__ == "__main__":
    main()
