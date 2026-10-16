"""Configuration loading, startup and entry-point tests."""

import tempfile
import unittest
from pathlib import Path
from unittest import mock

from pydantic import ValidationError

import quicken_prices as qp
from settings import AppSettings


class ConfigurationTests(unittest.TestCase):
    def test_load_configuration_applies_model_defaults_without_writing(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            config_path = Path(temp_dir) / "configuration.yaml"
            config_path.write_text("tickers:\n  - TEST\npaths: {}\n", encoding="utf-8")
            with (
                mock.patch.object(qp, "validate_quicken_path", return_value=None),
                self.assertLogs(level="WARNING"),
            ):
                config = qp.load_configuration(str(config_path))

            self.assertEqual(config.tickers, ["TEST"])
            self.assertEqual(config.retries.max_retries, 3)
            self.assertEqual(Path(config.paths.base), Path(qp.__file__).resolve().parent)
            self.assertEqual(config.paths.quicken, "")
            self.assertTrue(config_path.is_file())

    def test_main_configures_logging_before_it_can_fail(self):
        """A failure before setup_logging must still reach a log handler."""
        order = []

        def record_logging():
            order.append("logging")

        def record_interpreter_check():
            order.append("interpreter")
            raise RuntimeError("wrong interpreter")

        with (
            mock.patch.object(qp, "_bootstrap_logging", side_effect=record_logging),
            mock.patch.object(
                qp, "ensure_project_python", side_effect=record_interpreter_check
            ),
            mock.patch.object(qp, "pause_before_exit"),
        ):
            with self.assertRaisesRegex(RuntimeError, "wrong interpreter"):
                qp.main()

        self.assertEqual(order, ["logging", "interpreter"])

    def test_load_configuration_rejects_unknown_and_coerced_values(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            config_path = Path(temp_dir) / "configuration.yaml"
            config_path.write_text(
                "tickers:\n  - TEST\npaths: {}\ncollection_period_years: '0.2'\n",
                encoding="utf-8",
            )

            with self.assertRaises(ValidationError):
                qp.load_configuration(str(config_path))

            config_path.write_text(
                "tickers:\n  - TEST\npaths: {}\nlegacy_option: true\n",
                encoding="utf-8",
            )
            with self.assertRaises(ValidationError):
                qp.load_configuration(str(config_path))

            config_path.write_text(
                "tickers:\n  - TEST\npaths: {}\n"
                "retries:\n  max_retries: '3'\n",
                encoding="utf-8",
            )
            with self.assertRaises(ValidationError):
                qp.load_configuration(str(config_path))

            config_path.write_text(
                "tickers:\n  - TEST\n  - test\npaths: {}\n",
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValidationError, "unique"):
                qp.load_configuration(str(config_path))

    def test_load_configuration_rejects_invalid_explicit_quicken_path(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            config_path = Path(temp_dir) / "configuration.yaml"
            config_path.write_text(
                "tickers:\n  - TEST\npaths:\n  quicken: missing.exe\n",
                encoding="utf-8",
            )

            with (
                mock.patch.object(qp, "validate_quicken_path") as detect_path,
                self.assertRaisesRegex(FileNotFoundError, "Configured Quicken executable"),
            ):
                qp.load_configuration(str(config_path))

            detect_path.assert_not_called()

    def test_settings_reject_ticker_path_separators(self):
        with self.assertRaises(ValidationError):
            AppSettings(paths={"base": "."}, tickers=["../outside"])

    def test_missing_configuration_is_not_created(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            config_path = Path(temp_dir) / "missing.yaml"

            with self.assertRaises(FileNotFoundError):
                qp.load_configuration(str(config_path))

            self.assertFalse(config_path.exists())

    def test_main_propagates_failures(self):
        # The interpreter check runs before configuration loading and needs a
        # checkout venv to exist. Neutralize it: this test is about a failing
        # configuration, not about the environment it runs in.
        with (
            mock.patch.object(qp, "ensure_project_python"),
            mock.patch.object(qp, "load_configuration", side_effect=RuntimeError("bad config")),
            mock.patch.object(qp, "pause_before_exit"),
        ):
            with self.assertLogs(level="ERROR") as captured:
                with self.assertRaisesRegex(RuntimeError, "bad config"):
                    qp.main()
        self.assertIn("An unexpected error occurred", captured.output[0])

    def test_startup_self_check_does_not_touch_user_cache(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            config = AppSettings(
                paths={"base": temp_dir},
                collection_period_years=0.1,
            )

            qp.run_startup_self_check(config)
            self.assertFalse((Path(temp_dir) / "cache").exists())


if __name__ == "__main__":
    unittest.main()
