"""Small shared helpers used across the application."""

from pathlib import Path

def pluralise(title: str, quantity: int) -> str:
    return title if quantity == 1 else title + "s"

def native_currencies(home_currency: str) -> set:
    """Currency codes already denominated in the home currency (no FX needed).

    Includes the pence variant yfinance reports for some GBP-listed securities.
    """
    native = {home_currency}
    if home_currency == "GBP":
        native.add("GBp")
    return native

def format_path(full_path):
    """Format a file path to show the last parent directory and filename."""
    path = Path(full_path)
    if path.parent == Path("."):
        return f"\\{path.name}"
    return f"\\{path.parent.name}\\{path.name}"
