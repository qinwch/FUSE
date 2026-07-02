import tempfile
import unittest
import importlib.util
from pathlib import Path


def load_repro_utils():
    root = Path(__file__).resolve().parents[1]
    path = root / "flow-matching-posterior-estimation" / "sbi-benchmark" / "repro_utils.py"
    spec = importlib.util.spec_from_file_location("sbibm_repro_utils", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load SBIBM repro utilities from {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


repro_utils = load_repro_utils()


class SbibmReproUtilsTests(unittest.TestCase):
    def test_observation_numbers_are_one_indexed_and_inclusive(self):
        self.assertEqual(list(repro_utils.observation_numbers(3)), [1, 2, 3])

    def test_observation_numbers_rejects_zero(self):
        with self.assertRaises(ValueError):
            list(repro_utils.observation_numbers(0))

    def test_dataset_files_exist_requires_x_and_theta(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir)
            self.assertFalse(repro_utils.dataset_files_exist(path))
            (path / "x.npy").write_bytes(b"x")
            self.assertFalse(repro_utils.dataset_files_exist(path))
            (path / "theta.npy").write_bytes(b"theta")
            self.assertTrue(repro_utils.dataset_files_exist(path))

    def test_resolve_optional_path_returns_none_for_empty_value(self):
        self.assertIsNone(repro_utils.resolve_optional_path(None))
        self.assertIsNone(repro_utils.resolve_optional_path(""))

    def test_resolve_generation_batch_size_uses_fallback_when_unset(self):
        self.assertEqual(repro_utils.resolve_generation_batch_size(None, 17), 17)

    def test_resolve_generation_batch_size_uses_explicit_override(self):
        self.assertEqual(repro_utils.resolve_generation_batch_size(32, 17), 32)

    def test_should_set_seed_is_false_when_unset(self):
        self.assertFalse(repro_utils.should_set_seed(None))

    def test_should_set_seed_is_true_for_zero(self):
        self.assertTrue(repro_utils.should_set_seed(0))

    def test_should_set_seed_is_true_for_one(self):
        self.assertTrue(repro_utils.should_set_seed(1))


if __name__ == "__main__":
    unittest.main()
