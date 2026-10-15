"""Windows-only Quicken discovery and GUI automation."""

from __future__ import annotations

import ctypes
import logging
import os
import subprocess
import sys
import time
import traceback
import winreg
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional

import pandas as pd
import pyautogui
import pygetwindow as gw

from common import format_path
from settings import AppSettings

QUICKEN_SESSION_STARTED = False
QUICKEN_MAIN_WINDOW_TITLE = "Quicken XG 2004"
QUICKEN_PORTFOLIO_WINDOW_TITLE = "Quicken XG 2004 - Home - [Investing Centre]"
QUICKEN_IMPORT_DIALOG_TITLE = "Import Price Data"


@dataclass
class QuickenSession:
    """Own only a Quicken process started by this application."""

    process: Optional[subprocess.Popen] = None

    def close(self) -> bool:
        if self.process is None:
            logging.info("Leaving existing Quicken session open.")
            return True

        process = self.process
        self.process = None
        if process.poll() is not None:
            logging.info("Quicken process already exited.")
            return True

        try:
            logging.info("Closing Quicken process started by this application...")
            process.terminate()
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                logging.warning("Quicken did not close; terminating its process.")
                process.kill()
                process.wait()
            logging.info("Quicken closed successfully.")
            return True
        except OSError:
            logging.exception("Failed to close Quicken process.")
            raise


def validate_quicken_path() -> Optional[str]:
    """
    Validate and locate the Quicken executable.
    """

    return find_quicken_via_registry() or locate_quicken_in_standard_paths()

def find_quicken_via_registry() -> Optional[str]:
    """
    Locate Quicken executable path via Windows Registry.
    """
    registry_keys = [
        r"SOFTWARE\WOW6432Node\Microsoft\Windows\CurrentVersion\App Paths\qw.exe",
        r"SOFTWARE\WOW6432Node\Microsoft\Windows\CurrentVersion\Uninstall\{366CC735-543D-42CB-9C03-D7512314DE52}",
    ]
    for key_path in registry_keys:
        try:
            with winreg.OpenKey(
                winreg.HKEY_LOCAL_MACHINE, key_path, 0, winreg.KEY_READ
            ) as key:
                if "App Paths" in key_path:
                    # Explicitly handle `None` as a valid argument
                    quicken_path, _ = winreg.QueryValueEx(
                        key, ""
                    )  # Use empty string for the default value
                else:
                    install_location, _ = winreg.QueryValueEx(key, "InstallLocation")
                    quicken_path = os.path.join(install_location, "qw.exe")
                if os.path.isfile(quicken_path):
                    return quicken_path
        except (FileNotFoundError, OSError, winreg.error):
            continue
    return None

def locate_quicken_in_standard_paths() -> Optional[str]:
    """
    Locate Quicken executable in standard installation directories.
    """

    standard_paths = [
        r"C:\Program Files\Quicken\qw.exe",
        r"C:\Program Files (x86)\Quicken\qw.exe",
        r"C:\Quicken\qw.exe",
    ]
    for path in standard_paths:
        if os.path.isfile(path):
            logging.info(f"Quicken executable found at: {path}")
            return path
    return None

def project_python() -> Path:
    """Return the only supported application interpreter for this checkout."""
    return Path(__file__).resolve().parent / ".venv" / "Scripts" / "python.exe"

def ensure_project_python() -> None:
    interpreter = project_python()
    if not interpreter.is_file():
        raise FileNotFoundError(
            f"Project virtual environment not found: {interpreter}. "
            "Create it with 'py -3 -m venv .venv'."
        )
    if Path(sys.executable).resolve() != interpreter.resolve():
        raise RuntimeError(
            f"Run QuickenPrices with the project interpreter: {interpreter}"
        )

def run_as_admin() -> bool:
    """
    Re-runs the script with administrative privileges if not already elevated.

    Returns:
        bool: True when a new elevated process was launched and the current
        process should stop after notifying the user.
    """
    elevated = is_elevated()
    logging.info(
        f"Current elevation status: {'Elevated' if elevated else 'Not elevated'}"
    )

    if not elevated:
        # Re-run the script as admin
        try:
            script_path = str(Path(sys.argv[0]).resolve())
            interpreter = project_python()
            if not interpreter.is_file():
                raise FileNotFoundError(
                    f"Project virtual environment not found: {interpreter}"
                )
            logging.info("Attempting to re-run with administrative privileges...")
            return_code = ctypes.windll.shell32.ShellExecuteW(
                None,
                "runas",
                str(interpreter),
                f"\"{script_path}\"",
                str(Path(script_path).parent),
                1,
            )
            if return_code > 32:
                logging.info("Successfully launched elevated process in a new window.")
                return True
            else:
                # See ShellExecuteW documentation for error codes
                logging.error(
                    f"Failed to elevate privileges. ShellExecuteW returned code: {return_code}"
                )
                logging.error(
                    "Script will continue without elevation, but Quicken automation may fail."
                )
                # Do not exit here, allow the script to continue without elevation
        except Exception as e:
            logging.error(f"An error occurred while trying to elevate privileges: {e}")
    else:
        logging.info("Script is running with administrative privileges.")

    return False

def is_elevated():
    """
    Checks if the script is running with administrative privileges.
    """

    return bool(ctypes.windll.shell32.IsUserAnAdmin())

def import_data_file(config: AppSettings) -> bool:

    try:
        output_file_name = config.paths.data_file
        base_path = config.paths.base
        filename = str(Path(base_path) / output_file_name)

        if not activate_window_by_title(QUICKEN_IMPORT_DIALOG_TITLE, timeout=5):
            logging.error("Import Price Data dialog is not available for file entry.")
            return False

        # Type the file name into the dialog
        pyautogui.typewrite(filename, interval=0.01)
        time.sleep(0.5)  # Small delay before hitting enter for stability
        pyautogui.press("enter")

        # Wait to get to the price import success dialogue
        start_time = time.time()
        while time.time() - start_time < 120:
            if activate_window_by_title(QUICKEN_PORTFOLIO_WINDOW_TITLE, timeout=0.5):
                time.sleep(2)
                pyautogui.press("enter")
                logging.info(
                    f"Successfully imported {format_path(filename)} at {pd.Timestamp.now().strftime('%d/%m/%Y %H:%M')}"
                )
                time.sleep(0.5)
                return True
            time.sleep(0.5)

        logging.error("Timed out waiting for the Quicken import confirmation window.")
        return False

    except Exception as e:
        logging.error(f"Failed to import data file: {e} ({type(e).__name__})")
        raise

def open_import_dialog():

    try:
        if not activate_window_by_title(QUICKEN_PORTFOLIO_WINDOW_TITLE, timeout=5):
            logging.error("Quicken portfolio window is not active for opening import dialog.")
            return False

        with pyautogui.hold("alt"):
            pyautogui.press(["f", "i", "i"])

        # Wait to get to the price import dialogue box
        start_time = time.time()
        while time.time() - start_time < 10:
            if activate_window_by_title(QUICKEN_IMPORT_DIALOG_TITLE, timeout=0.5):
                time.sleep(0.5)
                return True
            time.sleep(0.5)

        logging.error("Timed out waiting for the Import Price Data dialog.")
        return False
    except Exception as e:
        logging.error(f"Failed to open price import dialogue box: {e}")
        logging.debug("Full traceback:\n%s", traceback.format_exc())
        return False

def navigate_to_portfolio():

    try:
        if not activate_window_by_title(QUICKEN_MAIN_WINDOW_TITLE, timeout=5):
            logging.error("Quicken window is not active for portfolio navigation.")
            return False

        pyautogui.hotkey("ctrl", "u")

        # Wait to get to the investing portfolio page
        start_time = time.time()
        while time.time() - start_time < 30:
            if activate_window_by_title(QUICKEN_PORTFOLIO_WINDOW_TITLE, timeout=0.5):
                time.sleep(0.5)
                return True
            time.sleep(0.5)

        logging.error("Could not get to portfolio page within the expected time.")
        return False

    except Exception as e:
        logging.error(f"Failed to navigate to portfolio: {e}")
        logging.debug("Full traceback:\n%s", traceback.format_exc())
        return False

def open_quicken(config: AppSettings, session: QuickenSession) -> bool:
    """Open Quicken and record ownership only for a process launched here."""
    quicken_path = config.paths.quicken
    try:
        if not quicken_path or not Path(quicken_path).is_file():
            logging.error(f"Quicken executable not found: {quicken_path}")
            return False

        windows = gw.getWindowsWithTitle(QUICKEN_MAIN_WINDOW_TITLE)
        if windows:
            activate_window_by_title(QUICKEN_MAIN_WINDOW_TITLE, timeout=1)
            logging.info("Quicken is already open.")
            return True

        logging.info("Launching Quicken...")
        session.process = subprocess.Popen([quicken_path], shell=False)
        if session.process.poll() is not None:
            logging.error("Failed to start Quicken process.")
            session.process = None
            return False

        start_time = time.time()
        while time.time() - start_time < 30:
            if activate_window_by_title(QUICKEN_MAIN_WINDOW_TITLE, timeout=0.5):
                logging.info("Quicken window found.")
                return True
            time.sleep(0.5)

        logging.error("Quicken window did not appear within the expected time.")
        return False
    except Exception as e:
        logging.error(f"An unexpected error occurred while opening Quicken: {e}")
        logging.debug("Full traceback:\n%s", traceback.format_exc())
        return False

def close_quicken(session: QuickenSession) -> bool:
    """Close only the Quicken process owned by the supplied session."""
    return session.close()

def activate_window_by_title(window_title: str, timeout: float = 5.0) -> bool:
    """
    Bring a window matching the supplied title to the foreground.
    """
    deadline = time.time() + timeout
    while time.time() < deadline:
        windows = gw.getWindowsWithTitle(window_title)
        if windows:
            window = windows[0]
            try:
                if getattr(window, "isMinimized", False):
                    window.restore()
                    time.sleep(0.2)
                window.activate()
                time.sleep(0.2)
                return True
            except Exception as e:
                logging.debug(f"Could not activate '{window_title}': {e}")
        time.sleep(0.2)
    return False

def execute_import_sequence(config: AppSettings, session: QuickenSession) -> bool:
    """
    Execute the sequence of steps for importing data.

    Returns:
        bool: True if all steps completed successfully
    """

    steps: list[tuple[str, Callable[[], bool]]] = [
        ("Opening Quicken", lambda: open_quicken(config, session)),
        ("Navigating to Portfolio view", navigate_to_portfolio),
        ("Opening import dialog", open_import_dialog),
        ("Importing data file", lambda: import_data_file(config)),
    ]

    for message, step in steps:
        logging.info(f"Starting step: {message}")
        try:
            result = step()
        except Exception as error:
            raise RuntimeError(f"Quicken import failed during: {message}.") from error
        if not result:
            raise RuntimeError(f"Quicken import failed during: {message}.")
        logging.info(f"Step '{message}' completed successfully.")

    logging.info("All steps completed successfully.")
    return True

def setup_pyautogui():
    """
    Configures pyautogui for automation.
    """

    pyautogui.FAILSAFE = True  # Abort ongoing automation by moving mouse to the top-left corner of screen.
    pyautogui.PAUSE = 0.3  # Delay between actions

def quicken_import(config: AppSettings) -> bool:
    """
    Automates the process of importing data into Quicken.

    Args:
        config: Configuration dictionary.

    Returns:
        bool: True if the import process completes successfully.
    """

    logging.info("\nImport to Quicken 2004")

    if not is_elevated():
        file_path = Path(config.paths.base) / config.paths.data_file
        short_csv_path = format_path(file_path)
        logging.warning("Quicken GUI automation requires elevation; CSV export is ready.")
        logging.info("To import manually:")
        logging.info("1. Close this window.")
        logging.info("2. Click the Quicken 'Update Prices' shortcut.")
        logging.info(f"3. If needed, import {short_csv_path} manually.")
        return False

    session = QuickenSession()
    try:
        setup_pyautogui()
        return execute_import_sequence(config, session)
    finally:
        close_quicken(session)
