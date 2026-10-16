"""The Start-menu launcher chain that starts this application on this machine.

These checks read a Start-menu shortcut and a wrapper that live outside the
checkout, so they are specific to this machine and skip anywhere else - including
a clean CI checkout, where those files cannot exist.
"""

import subprocess
import unittest
from pathlib import Path

CHECKOUT = Path(r"C:\Users\wayne\GitHub\Python\Projects\quicken_prices")
SHORTCUT = Path(r"C:\ProgramData\Microsoft\Windows\Start Menu\Programs\Update Prices.lnk")
WRAPPER = Path(r"C:\Programs\bat Files\quickenprice.bat")
LEGACY_LAUNCHER = CHECKOUT / "quickenprice.bat"
CURRENT_LAUNCHER = CHECKOUT / "quicken_prices.bat"

WRAPPER_CONTENT = """\
@echo off
call "C:\\Users\\wayne\\GitHub\\Python\\Projects\\quicken_prices\\quickenprice.bat"
exit /b %ERRORLEVEL%"""

LAUNCHER_CONTENT = """\
@echo off
setlocal

set "PROJECT_DIR=%~dp0"
set "VENV_PYTHON=%PROJECT_DIR%.venv\\Scripts\\python.exe"

if not exist "%PROJECT_DIR%quicken_prices.py" (
    echo [ERROR] Could not find quicken_prices.py at:
    echo         %PROJECT_DIR%
    pause
    exit /b 1
)

if not exist "%VENV_PYTHON%" (
    echo [ERROR] Project virtual environment is missing:
    echo         %VENV_PYTHON%
    echo Create it with: py -3.14 -m venv .venv
    pause
    exit /b 1
)

pushd "%PROJECT_DIR%" || (
    echo [ERROR] Failed to switch to project directory:
    echo         %PROJECT_DIR%
    pause
    exit /b 1
)

"%VENV_PYTHON%" "quicken_prices.py"
set "EXIT_CODE=%ERRORLEVEL%"

popd

if not "%EXIT_CODE%"=="0" (
    echo [ERROR] QuickenPrices exited with code %EXIT_CODE%.
    pause
)

exit /b %EXIT_CODE%"""


def normalise(text: str) -> str:
    """Compare content rather than line endings, since Windows writes CRLF."""
    return text.replace("\r\n", "\n").strip()


def read_script(path: Path) -> str:
    return normalise(path.read_text(encoding="utf-8"))


def shortcut_target(path: Path) -> str:
    """Ask the shell for the target Explorer would actually use."""
    completed = subprocess.run(
        [
            "powershell",
            "-NoProfile",
            "-NonInteractive",
            "-Command",
            (
                "$shell = New-Object -ComObject WScript.Shell; "
                f"$shell.CreateShortcut('{path}').TargetPath"
            ),
        ],
        capture_output=True,
        text=True,
        check=True,
    )
    return completed.stdout.strip()


@unittest.skipUnless(
    Path(__file__).resolve().parent == CHECKOUT,
    "the Start-menu launcher chain exists only on this machine's checkout",
)
class LauncherChainTests(unittest.TestCase):
    def test_the_start_menu_shortcut_points_at_the_wrapper(self):
        self.assertTrue(SHORTCUT.is_file(), f"shortcut missing: {SHORTCUT}")
        self.assertEqual(shortcut_target(SHORTCUT), str(WRAPPER))

    def test_the_wrapper_calls_the_repository_launcher(self):
        self.assertTrue(WRAPPER.is_file(), f"wrapper missing: {WRAPPER}")
        self.assertEqual(read_script(WRAPPER), WRAPPER_CONTENT)

    def test_the_legacy_launcher_runs_the_project_interpreter(self):
        # The wrapper outside the checkout still calls this name, so the file
        # the audit renamed has to keep existing under it.
        self.assertTrue(LEGACY_LAUNCHER.is_file(), f"launcher missing: {LEGACY_LAUNCHER}")
        self.assertEqual(read_script(LEGACY_LAUNCHER), LAUNCHER_CONTENT)

    def test_both_launcher_names_hold_the_same_script(self):
        self.assertEqual(read_script(LEGACY_LAUNCHER), read_script(CURRENT_LAUNCHER))
