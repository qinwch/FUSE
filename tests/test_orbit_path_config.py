import tempfile
import unittest
from pathlib import Path

from orbit_train.path_config import DEFAULT_ORBIT_PATHS, OrbitPaths


class OrbitPathConfigTests(unittest.TestCase):
    def test_dataset_paths_use_name_and_data_dir(self):
        paths = OrbitPaths(
            data_dir=Path("artifacts/orbit/datasets"),
            models_dir=Path("artifacts/orbit/models"),
            outputs_dir=Path("artifacts/orbit/outputs"),
            references_dir=Path("artifacts/orbit/references"),
        )
        self.assertEqual(paths.train_dataset("orbit"), Path("artifacts/orbit/datasets/orbit-train.h5"))
        self.assertEqual(paths.val_dataset("orbit"), Path("artifacts/orbit/datasets/orbit-val.h5"))
        self.assertEqual(paths.test_dataset("orbit"), Path("artifacts/orbit/datasets/orbit-test.h5"))
        self.assertEqual(paths.model_path("orbit_mmdit.pth"), Path("artifacts/orbit/models/orbit_mmdit.pth"))
        self.assertEqual(paths.reference_path("mcmc_betapic.hdf5"), Path("artifacts/orbit/references/mcmc_betapic.hdf5"))

    def test_ensure_directories_creates_all_roots(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            paths = OrbitPaths(
                data_dir=root / "datasets",
                models_dir=root / "models",
                outputs_dir=root / "outputs",
                references_dir=root / "references",
            )
            paths.ensure_directories()
            self.assertTrue(paths.data_dir.is_dir())
            self.assertTrue(paths.models_dir.is_dir())
            self.assertTrue(paths.outputs_dir.is_dir())
            self.assertTrue(paths.references_dir.is_dir())

    def test_default_paths_match_release_layout(self):
        self.assertEqual(DEFAULT_ORBIT_PATHS.train_dataset("orbit"), Path("artifacts/orbit/datasets/orbit-train.h5"))
        self.assertEqual(DEFAULT_ORBIT_PATHS.model_path("orbit_mmdit.pth"), Path("artifacts/orbit/models/orbit_mmdit.pth"))
        self.assertEqual(DEFAULT_ORBIT_PATHS.reference_path("mcmc_betapic.hdf5"), Path("artifacts/orbit/references/mcmc_betapic.hdf5"))


if __name__ == "__main__":
    unittest.main()
