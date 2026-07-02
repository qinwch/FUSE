import tempfile
import unittest
from pathlib import Path

from scripts.inventory_artifacts import (
    ArtifactRecord,
    collect_artifacts,
    format_size,
    is_artifact_path,
    sha256_or_status,
)


class ArtifactInventoryTests(unittest.TestCase):
    def test_is_artifact_path_matches_large_result_extensions(self):
        self.assertTrue(is_artifact_path(Path("model.pt")))
        self.assertTrue(is_artifact_path(Path("samples.hdf5")))
        self.assertTrue(is_artifact_path(Path("posterior.npy")))
        self.assertFalse(is_artifact_path(Path("README.md")))
        self.assertFalse(is_artifact_path(Path("script.py")))

    def test_format_size_uses_binary_units(self):
        self.assertEqual(format_size(0), "0 B")
        self.assertEqual(format_size(1023), "1023 B")
        self.assertEqual(format_size(1024), "1.00 KiB")
        self.assertEqual(format_size(1024 * 1024), "1.00 MiB")

    def test_sha256_or_status_hashes_small_files(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "small.pt"
            path.write_bytes(b"abc")
            self.assertEqual(
                sha256_or_status(path, max_hash_bytes=10),
                "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad",
            )

    def test_sha256_or_status_skips_large_files(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "large.hdf5"
            path.write_bytes(b"abcdef")
            self.assertEqual(
                sha256_or_status(path, max_hash_bytes=3),
                "not-computed-file-exceeds-3-bytes",
            )

    def test_collect_artifacts_ignores_git_and_python_cache(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            (root / "model.pt").write_bytes(b"abc")
            (root / "README.md").write_text("docs", encoding="utf-8")
            (root / ".git").mkdir()
            (root / ".git" / "hidden.pt").write_bytes(b"abc")
            (root / "__pycache__").mkdir()
            (root / "__pycache__" / "cache.pyc").write_bytes(b"abc")

            records = collect_artifacts(root, max_hash_bytes=10)

        self.assertEqual(
            records,
            [
                ArtifactRecord(
                    logical_name="model.pt",
                    current_path="model.pt",
                    intended_path="artifacts/model.pt",
                    size_bytes=3,
                    size_human="3 B",
                    sha256="ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad",
                    present="yes",
                    required_for="unassigned",
                    zenodo_url="",
                    notes="discovered in workspace",
                )
            ],
        )

    def test_collect_artifacts_ignores_nested_skipped_directories(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            (root / "keep").mkdir()
            (root / "keep" / "model.pt").write_bytes(b"abc")
            (root / "experiments").mkdir()
            (root / "experiments" / "wandb").mkdir()
            (root / "experiments" / "wandb" / "run.pt").write_bytes(b"abc")
            (root / "cache").mkdir()
            (root / "cache" / "cached.pt").write_bytes(b"abc")

            records = collect_artifacts(root, max_hash_bytes=10)

        self.assertEqual([record.current_path for record in records], ["keep/model.pt"])

    def test_collect_artifacts_skips_symlinked_artifact_files(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            target = root / "outside.pt"
            target.write_bytes(b"abc")
            link = root / "linked.pt"
            try:
                link.symlink_to(target)
            except OSError as exc:
                raise unittest.SkipTest(f"symlink creation unavailable: {exc}") from exc

            records = collect_artifacts(root, max_hash_bytes=10)

        self.assertEqual([record.current_path for record in records], ["outside.pt"])


if __name__ == "__main__":
    unittest.main()
