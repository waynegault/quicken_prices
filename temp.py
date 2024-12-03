import time
import win32gui


def get_active_window_title():
    """
    Get the title of the current active window.

    Returns:
        str: The title of the current active window.
    """
    window_handle = win32gui.GetForegroundWindow()
    window_title = win32gui.GetWindowText(window_handle)
    return window_title


def is_quicken_window(title):
    """
    Check if the window belongs to Quicken, including its dialogs.

    Args:
        title (str): The title of the window to check.

    Returns:
        bool: True if the window belongs to Quicken.
    """
    # Update this if you want to add more variations of possible dialog names
    quicken_main_title = "Quicken XG 2004"
    quicken_dialog_keywords = [
        "Import",
        "Export",
        "Warning",
        "Confirm",
        "Error",
        "Settings",
        "Preferences",
    ]

    # Check if the window title matches the main title
    if quicken_main_title in title:
        return True

    # Check if the title contains one of the typical dialog box keywords
    for keyword in quicken_dialog_keywords:
        if keyword in title:
            return True

    return False


def monitor_quicken_title():
    """
    Monitor active window titles and print them if they belong to Quicken or its dialogs.
    """
    previous_title = ""
    while True:
        current_title = get_active_window_title()

        # If the current title is different from the last title we saw
        if current_title != previous_title:
            # Check if the window title indicates it belongs to Quicken or its dialog
            if is_quicken_window(current_title):
                print(f"New Quicken window or dialog detected: {current_title}")

            # Update the previous title
            previous_title = current_title

        # Adjust the polling interval as needed
        time.sleep(1)


if __name__ == "__main__":
    monitor_quicken_title()
