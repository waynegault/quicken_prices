import ctypes
import subprocess
import logging
import time
import pyautogui
import sys

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)


def is_admin():
    """Checks if the script is run as admin"""
    try:
        return ctypes.windll.shell32.IsUserAnAdmin()
    except:
        return False


def run_as_admin():
    """Prompts the user for elevation"""
    if not is_admin():
        # Re-run the script as admin
        ctypes.windll.shell32.ShellExecuteW(
            None, "runas", sys.executable, __file__, None, 1
        )
        sys.exit(0)


def main():
    run_as_admin()

    logging.info("Opening Notepad with elevated privileges...")
    notepad_process = subprocess.Popen(["notepad.exe"], shell=True)

    # Give Notepad some time to open
    time.sleep(2)

    logging.info("Automating text input in Notepad...")
    # Temporarily disable Ctrl+Alt+Del to prevent accidental interruption
    pyautogui.hotkey("ctrl", "alt", "del")
    pyautogui.write("Hello")
    pyautogui.hotkey("ctrl", "alt", "del")

    logging.info("Waiting for user to close Notepad...")
    quicken_process.wait()

    logging.info("Notepad closed by user.")
    logging.info("Keeping terminal open for 10 seconds...")
    time.sleep(10)


if __name__ == "__main__":
    main()
