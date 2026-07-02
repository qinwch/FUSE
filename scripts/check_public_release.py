#!/usr/bin/env python3
from __future__ import annotations

import subprocess
import sys
from pathlib import Path


SKIP_TRACKED_PATHS = {
    ".gitignore",
    "scripts/check_public_release.py",
    "tests/test_public_release_guard.py",
}
PRIVATE_PATH_PREFIXES = ("docs/superpowers/",)
PRIVATE_NOTE_REFERENCES = ("OPEN_QUESTIONS.md",)
WITHHELD_SAMPLE_TOKENS = ("sample_soft_new_fk", "orbitize_training_2")
SECRET_TOKENS = ("example-private-token", "ssh private-host", "密码")
MACHINE_PATH_TOKENS = ("/private/cluster", "/home/private-user", "/Users/private-user")
MACHINE_PATH_SCOPES = (
    ".vscode/",
    "README.md",
    "ARTIFACTS.md",
    "REPRODUCING.md",
    "FUSE_checkpoint/README.md",
    "docs/",
    "run.sh",
)


def _is_machine_path_scope(path: str) -> bool:
    return any(path == scope or path.startswith(scope) for scope in MACHINE_PATH_SCOPES)


def find_violations(files: dict[str, str], tracked_paths: list[str] | None = None) -> list[str]:
    violations: list[str] = []
    for path in sorted(tracked_paths or []):
        for prefix in PRIVATE_PATH_PREFIXES:
            if path.startswith(prefix):
                violations.append(f"{path}: private internal planning path {prefix}")

    for path, text in sorted(files.items()):
        if path.startswith(PRIVATE_PATH_PREFIXES):
            continue
        for token in PRIVATE_NOTE_REFERENCES:
            if token in text:
                violations.append(f"{path}: private note reference {token}")
        for token in WITHHELD_SAMPLE_TOKENS:
            if token not in text:
                continue
            if token == "orbitize_training_2":
                violations.append(f"{path}: private orbit experiment name {token}")
            else:
                violations.append(f"{path}: withheld FK sample filename {token}")
        if _is_machine_path_scope(path):
            for token in MACHINE_PATH_TOKENS:
                if token in text:
                    violations.append(f"{path}: machine-local path {token}")
        for token in SECRET_TOKENS:
            if token not in text:
                continue
            if token == "ssh private-host":
                violations.append(f"{path}: private host command {token}")
            else:
                violations.append(f"{path}: private credential string {token}")
    return violations


def _tracked_paths(root: Path) -> list[str]:
    result = subprocess.run(
        ["git", "ls-files", "-z"],
        cwd=root,
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    return [raw_path.decode("utf-8") for raw_path in result.stdout.split(b"\0") if raw_path]


def _tracked_text_files(root: Path, tracked_paths: list[str] | None = None) -> dict[str, str]:
    files: dict[str, str] = {}
    for relative in tracked_paths if tracked_paths is not None else _tracked_paths(root):
        if relative in SKIP_TRACKED_PATHS or relative.startswith(PRIVATE_PATH_PREFIXES):
            continue
        path = root / relative
        try:
            files[relative] = path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            continue
    return files


def main() -> int:
    root = Path(__file__).resolve().parents[1]
    tracked_paths = _tracked_paths(root)
    violations = find_violations(_tracked_text_files(root, tracked_paths), tracked_paths)
    for violation in violations:
        print(violation)
    if violations:
        return 1
    print("public release guard passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
