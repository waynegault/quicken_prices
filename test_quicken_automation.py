"""Quicken process ownership, elevation, and GUI step tests."""

import tempfile
import unittest
from pathlib import Path
from unittest import mock

import quicken_automation
from settings import AppSettings


class QuickenAutomationTests(unittest.TestCase):
    def test_ensure_project_python_requires_the_repository_venv(self):
        missing = Path("absent-venv") / "Scripts" / "python.exe"
        with mock.patch.object(
            quicken_automation, "project_python", return_value=missing
        ):
            with self.assertRaisesRegex(
                FileNotFoundError, "Project virtual environment not found"
            ):
                quicken_automation.ensure_project_python()

        with tempfile.TemporaryDirectory() as temp_dir:
            interpreter = Path(temp_dir) / "python.exe"
            interpreter.write_text("", encoding="utf-8")
            with mock.patch.object(
                quicken_automation, "project_python", return_value=interpreter
            ):
                with self.assertRaisesRegex(
                    RuntimeError, "Run QuickenPrices with the project interpreter"
                ):
                    quicken_automation.ensure_project_python()

                with mock.patch.object(
                    quicken_automation.sys, "executable", str(interpreter)
                ):
                    self.assertIsNone(quicken_automation.ensure_project_python())

    def test_quicken_session_closes_only_its_owned_process(self):
        process = mock.Mock()
        process.poll.return_value = None
        session = quicken_automation.QuickenSession(process=process)

        self.assertTrue(quicken_automation.close_quicken(session))

        process.terminate.assert_called_once_with()
        process.wait.assert_called_once_with(timeout=10)
        process.kill.assert_not_called()
        self.assertIsNone(session.process)

    def test_quicken_session_leaves_existing_user_process_open(self):
        session = quicken_automation.QuickenSession()

        self.assertTrue(quicken_automation.close_quicken(session))

    def test_quicken_import_skips_gui_without_elevation(self):
        config = AppSettings(paths={"base": ".", "data_file": "data.csv"})
        with (
            mock.patch.object(quicken_automation, "is_elevated", return_value=False),
            mock.patch.object(quicken_automation, "setup_pyautogui") as setup_gui,
            self.assertLogs(level="WARNING") as captured,
        ):
            self.assertFalse(quicken_automation.quicken_import(config))

        setup_gui.assert_not_called()
        self.assertIn("CSV export is ready", captured.output[0])

    def test_execute_import_sequence_uses_uniform_steps_and_surfaces_failures(self):
        config = AppSettings(paths={"base": "."})
        session = quicken_automation.QuickenSession()
        with (
            mock.patch.object(quicken_automation, "open_quicken", return_value=True) as open_app,
            mock.patch.object(quicken_automation, "navigate_to_portfolio", return_value=True) as navigate,
            mock.patch.object(quicken_automation, "open_import_dialog", return_value=False) as open_dialog,
            mock.patch.object(quicken_automation, "import_data_file", return_value=True) as import_file,
        ):
            with self.assertRaisesRegex(RuntimeError, "Opening import dialog"):
                quicken_automation.execute_import_sequence(config, session)

        open_app.assert_called_once_with(config, session)
        navigate.assert_called_once_with()
        open_dialog.assert_called_once_with()
        import_file.assert_not_called()

    def test_open_quicken_reports_a_missing_executable(self):
        config = AppSettings(paths={"base": "."})
        session = quicken_automation.QuickenSession()

        with (
            mock.patch.object(quicken_automation.subprocess, "Popen") as popen,
            self.assertLogs(level="ERROR") as captured,
        ):
            self.assertFalse(quicken_automation.open_quicken(config, session))

        popen.assert_not_called()
        self.assertIn("Quicken executable not found", captured.output[0])

    def test_open_quicken_reuses_an_already_running_session(self):
        config = AppSettings(paths={"base": ".", "quicken": __file__})
        session = quicken_automation.QuickenSession()

        with (
            mock.patch.object(
                quicken_automation.gw, "getWindowsWithTitle", return_value=[mock.Mock()]
            ),
            mock.patch.object(
                quicken_automation, "activate_window_by_title", return_value=True
            ) as activate,
            mock.patch.object(quicken_automation.subprocess, "Popen") as popen,
            self.assertLogs(level="INFO") as captured,
        ):
            self.assertTrue(quicken_automation.open_quicken(config, session))

        activate.assert_called_once_with(
            quicken_automation.QUICKEN_MAIN_WINDOW_TITLE, timeout=1
        )
        popen.assert_not_called()
        # Nothing was launched here, so nothing here may close it.
        self.assertIsNone(session.process)
        self.assertIn("Quicken is already open", "\n".join(captured.output))

    def test_open_quicken_forgets_a_process_that_exits_immediately(self):
        config = AppSettings(paths={"base": ".", "quicken": __file__})
        session = quicken_automation.QuickenSession()
        process = mock.Mock()
        process.poll.return_value = 1

        with (
            mock.patch.object(quicken_automation.gw, "getWindowsWithTitle", return_value=[]),
            mock.patch.object(quicken_automation.subprocess, "Popen", return_value=process),
            self.assertLogs(level="ERROR") as captured,
        ):
            self.assertFalse(quicken_automation.open_quicken(config, session))

        self.assertIsNone(session.process)
        self.assertIn("Failed to start Quicken process", captured.output[0])

    def test_open_quicken_keeps_ownership_when_the_window_never_appears(self):
        config = AppSettings(paths={"base": ".", "quicken": __file__})
        session = quicken_automation.QuickenSession()
        process = mock.Mock()
        process.poll.return_value = None

        with (
            mock.patch.object(quicken_automation.gw, "getWindowsWithTitle", return_value=[]),
            mock.patch.object(quicken_automation.subprocess, "Popen", return_value=process),
            mock.patch.object(
                quicken_automation, "activate_window_by_title", return_value=False
            ),
            mock.patch.object(quicken_automation.time, "sleep"),
            mock.patch.object(quicken_automation.time, "time", side_effect=[0.0, 31.0]),
            self.assertLogs(level="ERROR") as captured,
        ):
            self.assertFalse(quicken_automation.open_quicken(config, session))

        # A half-started Quicken is still ours to clean up.
        self.assertIs(session.process, process)
        self.assertIn("did not appear within the expected time", captured.output[0])

    def test_quicken_import_closes_its_process_when_a_step_fails(self):
        config = AppSettings(paths={"base": "."})
        process = mock.Mock()
        process.poll.return_value = None

        def fail_after_launch(_config, session):
            session.process = process
            raise RuntimeError("Quicken import failed during: Importing data file.")

        with (
            mock.patch.object(quicken_automation, "is_elevated", return_value=True),
            mock.patch.object(quicken_automation, "setup_pyautogui"),
            mock.patch.object(
                quicken_automation,
                "execute_import_sequence",
                side_effect=fail_after_launch,
            ),
        ):
            with self.assertRaisesRegex(RuntimeError, "Importing data file"):
                quicken_automation.quicken_import(config)

        process.terminate.assert_called_once()

    def test_run_as_admin_is_a_noop_when_already_elevated(self):
        with (
            mock.patch.object(quicken_automation, "is_elevated", return_value=True),
            mock.patch.object(quicken_automation.ctypes, "windll") as windll,
            self.assertLogs(level="INFO") as captured,
        ):
            self.assertFalse(quicken_automation.run_as_admin())

        windll.shell32.ShellExecuteW.assert_not_called()
        self.assertIn("administrative privileges", "\n".join(captured.output))

    def test_run_as_admin_launches_the_project_interpreter(self):
        with (
            mock.patch.object(quicken_automation, "is_elevated", return_value=False),
            mock.patch.object(
                quicken_automation, "project_python", return_value=Path(__file__)
            ),
            mock.patch.object(quicken_automation.ctypes, "windll") as windll,
            self.assertLogs(level="INFO") as captured,
        ):
            windll.shell32.ShellExecuteW.return_value = 42
            self.assertTrue(quicken_automation.run_as_admin())

        launch = windll.shell32.ShellExecuteW.call_args.args
        self.assertEqual(launch[1], "runas")
        self.assertEqual(launch[2], str(Path(__file__)))
        self.assertIn("Successfully launched elevated process", "\n".join(captured.output))

    def test_run_as_admin_reports_a_refused_elevation_and_continues(self):
        with (
            mock.patch.object(quicken_automation, "is_elevated", return_value=False),
            mock.patch.object(
                quicken_automation, "project_python", return_value=Path(__file__)
            ),
            mock.patch.object(quicken_automation.ctypes, "windll") as windll,
            self.assertLogs(level="ERROR") as captured,
        ):
            windll.shell32.ShellExecuteW.return_value = 5
            self.assertFalse(quicken_automation.run_as_admin())

        self.assertIn("ShellExecuteW returned code: 5", captured.output[0])

    def test_import_data_file_reports_a_missing_import_dialog(self):
        config = AppSettings(paths={"base": "."})

        with (
            mock.patch.object(
                quicken_automation, "activate_window_by_title", return_value=False
            ),
            self.assertLogs(level="ERROR") as captured,
        ):
            self.assertFalse(quicken_automation.import_data_file(config))

        self.assertIn("Import Price Data dialog is not available", captured.output[0])

    def test_import_data_file_reports_a_missing_confirmation(self):
        config = AppSettings(paths={"base": "."})

        with (
            mock.patch.object(
                quicken_automation, "activate_window_by_title", return_value=True
            ),
            mock.patch.object(quicken_automation.pyautogui, "typewrite") as typewrite,
            mock.patch.object(quicken_automation.pyautogui, "press"),
            mock.patch.object(quicken_automation.time, "sleep"),
            mock.patch.object(quicken_automation.time, "time", side_effect=[0.0, 121.0]),
            self.assertLogs(level="ERROR") as captured,
        ):
            self.assertFalse(quicken_automation.import_data_file(config))

        typewrite.assert_called_once()
        self.assertIn(
            "Timed out waiting for the Quicken import confirmation", captured.output[0]
        )

    def test_activate_window_restores_a_minimized_window(self):
        window = mock.Mock()
        window.isMinimized = True

        with (
            mock.patch.object(
                quicken_automation.gw, "getWindowsWithTitle", return_value=[window]
            ),
            mock.patch.object(quicken_automation.time, "sleep"),
        ):
            self.assertTrue(
                quicken_automation.activate_window_by_title("Quicken", timeout=1)
            )

        window.restore.assert_called_once()
        window.activate.assert_called_once()

    def test_activate_window_reports_a_missing_window(self):
        with (
            mock.patch.object(quicken_automation.gw, "getWindowsWithTitle", return_value=[]),
            mock.patch.object(quicken_automation.time, "sleep"),
            mock.patch.object(quicken_automation.time, "time", side_effect=[0.0, 2.0]),
        ):
            self.assertFalse(
                quicken_automation.activate_window_by_title("Quicken", timeout=1)
            )


if __name__ == "__main__":
    unittest.main()
