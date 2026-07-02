import unittest
from pathlib import Path
from unittest.mock import patch

from scripts import check_public_release
from scripts.check_public_release import find_violations


class PublicReleaseGuardTest(unittest.TestCase):
    def test_flags_private_note_reference(self):
        files = {"README.md": "See OPEN_QUESTIONS.md for missing files.\n"}
        violations = find_violations(files)
        self.assertIn("README.md: private note reference OPEN_QUESTIONS.md", violations)

    def test_flags_withheld_fk_sample_name(self):
        files = {"docs/orbit_flow_matching_record.md": "sample_soft_new_fk_1_8_new_03_5.npy\n"}
        violations = find_violations(files)
        self.assertIn(
            "docs/orbit_flow_matching_record.md: withheld FK sample filename sample_soft_new_fk",
            violations,
        )

    def test_flags_machine_path_in_public_docs(self):
        files = {"ARTIFACTS.md": "Current path: /private/cluster/flow_matching_record/orbitize_training\n"}
        violations = find_violations(files)
        self.assertIn("ARTIFACTS.md: machine-local path /private/cluster", violations)

    def test_allows_artifact_relative_paths(self):
        files = {
            "ARTIFACTS.md": "artifact-source/orbitize_training/model_latest.pt -> artifacts/orbit/models/orbit_mmdit.pth\n",
            "README.md": "Figure 5 FK samples are not included in the first public artifact release.\n",
        }
        self.assertEqual(find_violations(files), [])

    def test_flags_known_secret_strings(self):
        files = {"notes.txt": "ssh private-host with password example-private-token\n"}
        violations = find_violations(files)
        self.assertIn("notes.txt: private credential string example-private-token", violations)
        self.assertIn("notes.txt: private host command ssh private-host", violations)

    def test_flags_additional_machine_paths(self):
        files = {
            "run.sh": "cd /home/private-user/dingo-mmdit\n",
            "docs/setup.md": "Local mirror: /Users/private-user/Documents/Codex\n",
        }
        violations = find_violations(files)
        self.assertIn("docs/setup.md: machine-local path /Users/private-user", violations)
        self.assertIn("run.sh: machine-local path /home/private-user", violations)

    def test_allows_machine_paths_outside_machine_path_scopes(self):
        files = {"notes.txt": "Local context: /private/cluster/dingo-mmdit\n"}
        self.assertEqual(find_violations(files), [])

    def test_flags_machine_paths_inside_machine_path_scopes(self):
        files = {
            "README.md": "Local context: /private/cluster/dingo-mmdit\n",
            "docs/setup.md": "Local context: /private/cluster/dingo-mmdit\n",
            "run.sh": "cd /private/cluster/dingo-mmdit\n",
        }
        violations = find_violations(files)
        self.assertIn("README.md: machine-local path /private/cluster", violations)
        self.assertIn("docs/setup.md: machine-local path /private/cluster", violations)
        self.assertIn("run.sh: machine-local path /private/cluster", violations)

    def test_flags_private_orbit_experiment_name(self):
        files = {"ARTIFACTS.md": "orbitize_training_2/model_latest.pt\n"}
        violations = find_violations(files)
        self.assertIn("ARTIFACTS.md: private orbit experiment name orbitize_training_2", violations)

    def test_flags_tracked_private_planning_path_without_scanning_content(self):
        files = {
            "docs/superpowers/plans/private.md": "See OPEN_QUESTIONS.md on ssh private-host at /private/cluster\n"
        }
        violations = find_violations(
            files,
            tracked_paths=["docs/superpowers/plans/private.md"],
        )
        self.assertEqual(
            violations,
            [
                "docs/superpowers/plans/private.md: private internal planning path docs/superpowers/"
            ],
        )

    def test_flags_private_chinese_password_token(self):
        files = {"notes.txt": "登录密码 should not be public.\n"}
        violations = find_violations(files)
        self.assertIn("notes.txt: private credential string 密码", violations)

    def test_main_scans_repo_root_from_script_location(self):
        expected_root = Path(check_public_release.__file__).resolve().parents[1]
        with patch.object(check_public_release, "_tracked_paths", return_value=[]) as tracked_paths:
            with patch.object(check_public_release, "_tracked_text_files", return_value={}) as tracked_text_files:
                with patch("builtins.print") as mock_print:
                    self.assertEqual(check_public_release.main(), 0)

        tracked_paths.assert_called_once_with(expected_root)
        tracked_text_files.assert_called_once_with(expected_root, [])
        mock_print.assert_called_once_with("public release guard passed")

    def test_main_reports_tracked_private_planning_paths(self):
        with patch.object(
            check_public_release,
            "_tracked_paths",
            return_value=["docs/superpowers/plans/private.md"],
        ):
            with patch.object(check_public_release, "_tracked_text_files", return_value={}) as tracked_text_files:
                with patch("builtins.print") as mock_print:
                    self.assertEqual(check_public_release.main(), 1)

        tracked_text_files.assert_called_once()
        mock_print.assert_called_once_with(
            "docs/superpowers/plans/private.md: private internal planning path docs/superpowers/"
        )


if __name__ == "__main__":
    unittest.main()
