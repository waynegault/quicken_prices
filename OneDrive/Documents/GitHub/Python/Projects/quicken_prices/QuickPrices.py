#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Shebang and coding declaration.

"""
QuickenPrices.py

This script fetches historical price data for a list of tickers specified in 'configuration.yaml' and saves the data to a CSV file.

It includes features such as configuration management, logging, error handling, data fetching, data processing, currency conversion, and more.

Wayne Gault
23/11/2024

MIT Licence
"""

import logging
import os
import sys
import winreg  # Added for registry access

# Set up a simple logging configuration

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    stream=sys.stdout,
)

import contextlib  # Used to redirect output streams temporarily
import ctypes  # Low-level OS interfaces
import functools  # Higher-order functions and decorators
from logging.handlers import (
    RotatingFileHandler,
)  # Additional logging handlers (e.g., RotatingFileHandler)
import pickle  # Object serialization/deserialization
import subprocess  # Process creation and management
import time  # Time-related functions
import traceback  # Exception handling and traceback printing
from datetime import date, datetime, timedelta  # Date and time handling
from pathlib import Path  # Object-oriented filesystem paths
from queue import Queue  # Thread-safe producer-consumer queues
from typing import Any, Dict, List, Optional, Tuple, Union  # Type annotations
from dataclasses import dataclass  # Easy class creation for storing data
from enum import Enum  # Define enumerations

import numpy as np
import yaml
import pandas as pd

# Removed Box import since it's no longer needed
import yfinance as yf
from colorama import init, Fore, Style
import pygetwindow as gw
import pyautogui
from tabulate import tabulate

init(autoreset=True)


# Utility to recursively apply default settings
def apply_defaults(
    defaults: Dict[str, Any], settings: Dict[str, Any]
) -> Dict[str, Any]:
    for key, value in defaults.items():
        if isinstance(value, dict):
            settings[key] = apply_defaults(value, settings.get(key, {}))
        else:
            settings.setdefault(key, value)
    return settings


def find_quicken_via_registry() -> Optional[str]:
    """
    Attempts to find Quicken executable path via Windows Registry.

    Returns:
        Path to Quicken executable if found, else None.
    """
    # 1. Attempt to read from App Paths
    try:
        registry_path = (
            r"SOFTWARE\WOW6432Node\Microsoft\Windows\CurrentVersion\App Paths\qw.exe"
        )
        with winreg.OpenKey(
            winreg.HKEY_LOCAL_MACHINE, registry_path, 0, winreg.KEY_READ
        ) as key:
            quicken_path, _ = winreg.QueryValueEx(key, None)  # Default value
            if os.path.isfile(quicken_path):
                logging.info(
                    f"Quicken executable found via App Paths registry: {quicken_path}"
                )
                return quicken_path
            else:
                logging.warning(
                    f"Quicken executable path from App Paths registry is invalid: {quicken_path}"
                )
    except FileNotFoundError:
        logging.warning("App Paths registry key for Quicken not found.")
    except Exception as e:
        logging.error(f"Error accessing App Paths registry key: {e}")

    # 2. Attempt to read from Uninstall key
    try:
        registry_path = r"SOFTWARE\WOW6432Node\Microsoft\Windows\CurrentVersion\Uninstall\{366CC735-543D-42CB-9C03-D7512314DE52}"
        with winreg.OpenKey(
            winreg.HKEY_LOCAL_MACHINE, registry_path, 0, winreg.KEY_READ
        ) as key:
            install_location, _ = winreg.QueryValueEx(key, "InstallLocation")
            quicken_path = os.path.join(install_location, "qw.exe")
            if os.path.isfile(quicken_path):
                logging.info(
                    f"Quicken executable found via Uninstall registry key: {quicken_path}"
                )
                return quicken_path
            else:
                logging.warning(
                    f"Quicken executable not found in InstallLocation registry path: {quicken_path}"
                )
    except FileNotFoundError:
        logging.warning("Uninstall registry key for Quicken not found.")
    except Exception as e:
        logging.error(f"Error accessing Uninstall registry key: {e}")

    # 3. Fallback to standard installation paths
    standard_paths = [
        r"C:\Program Files\Quicken\qw.exe",
        r"C:\Program Files (x86)\Quicken\qw.exe",
        r"C:\Quicken\qw.exe",
        # Add other standard paths if necessary
    ]
    for path in standard_paths:
        if os.path.isfile(path):
            logging.info(f"Quicken executable found at standard path: {path}")
            return path
    logging.error(
        "Quicken executable not found in registry or standard installation paths."
    )
    return None


def validate_quicken_path(provided_path: str) -> Optional[str]:
    """
    Validates the provided Quicken executable path.
    If invalid, searches registry and standard installation directories.

    Args:
        provided_path: Path to the Quicken executable from configuration.

    Returns:
        Validated path to the Quicken executable or None if not found.
    """
    if os.path.isfile(provided_path):
        logging.info(f"Quicken executable found at provided path: {provided_path}")
        return provided_path
    else:
        logging.warning(
            f"Quicken executable not found at provided path: {provided_path}"
        )
        # Attempt to find via registry
        registry_quicken_path = find_quicken_via_registry()
        if registry_quicken_path:
            return registry_quicken_path
        # If not found in registry, fallback to standard paths
        standard_paths = [
            r"C:\Program Files\Quicken\qw.exe",
            r"C:\Program Files (x86)\Quicken\qw.exe",
            r"C:\Quicken\qw.exe",
            # Add other standard paths if necessary
        ]
        for path in standard_paths:
            if os.path.isfile(path):
                logging.info(f"Quicken executable found at standard path: {path}")
                return path
        logging.error(
            "Quicken executable not found in provided, registry, or standard paths."
        )
        return None


# Load configuration with defaults
DEFAULT_CONFIG = {
    "tickers": {"^FTSE"},
    "paths": {
        "quicken": r"C:\Program Files (x86)\Quicken\qw.exe",
        "data_file": "data.csv",
        "log_file": "prices.log",
        "cache": "cache",
    },
    "collection": {"period_years": 0.08, "max_retries": 3, "retry_delay": 2},
    "cache": {"max_age_days": 30, "cleanup_threshold": 200},
    "logging": {
        "levels": {"file": "DEBUG", "terminal": "DEBUG"},
        "max_bytes": 5_242_880,
        "backup_count": 5,
    },
    "clear_startup": False,
}


def load_configuration(config_file: str = "configuration.yaml") -> Dict[str, Any]:
    # Determine the base path based on whether the script is packaged
    if getattr(sys, "frozen", False):
        # If the application is run as a bundle, the PyInstaller bootloader
        # sets the _MEIPASS attribute to the temp folder.
        base_path = sys._MEIPASS
    else:
        base_path = os.path.dirname(os.path.abspath(__file__))

    config_path = os.path.join(base_path, config_file)
    config = {}
    if os.path.exists(config_path):
        try:
            with open(config_path, "r") as f:
                config = yaml.safe_load(f) or {}

            if not isinstance(config, dict):
                logging.warning(f"Invalid format in {config_file}. Using defaults.")
                config = {}
        except Exception as e:
            logging.error(f"Error reading {config_file}: {e}. Using internal defaults.")

    # Merge defaults with user config
    final_config = apply_defaults(DEFAULT_CONFIG, config)

    # Dynamically add the base path to DEFAULT_CONFIG under "paths"
    DEFAULT_CONFIG.setdefault("paths", {})  # Ensure "paths" exists
    DEFAULT_CONFIG["paths"]["base"] = base_path  # Add "base" key
    final_config.setdefault("paths", {}).setdefault("base", base_path)

    # Validate Quicken path
    provided_quicken_path = final_config["paths"].get(
        "quicken", DEFAULT_CONFIG["paths"]["quicken"]
    )
    validated_quicken_path = validate_quicken_path(provided_quicken_path)

    if validated_quicken_path:
        final_config["paths"]["quicken"] = validated_quicken_path
    else:
        logging.error(
            "Quicken executable path is invalid. Please check your configuration."
        )
        # Exit the script if Quicken is essential
        sys.exit(1)

    return final_config  # Return as a standard dictionary


# Load the configuration
config = load_configuration()


def retry(exceptions, tries=config["collection"]["max_retries"], delay=1, backoff=2):

    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            mtries, mdelay = tries, delay
            for attempt in range(1, tries + 1):
                try:
                    return func(*args, **kwargs)
                except exceptions as e:
                    if attempt < tries:
                        logging.warning(
                            f"Attempt {attempt}/{tries}: {e}, Retrying in {mdelay} seconds..."
                        )
                    else:
                        logging.error(
                            f"Function '{func.__name__}' failed after {tries} attempts"
                        )
                        raise
                    time.sleep(mdelay)
                    mdelay *= backoff
            raise exceptions(
                f"Function '{func.__name__}' failed after {tries} attempts!"
            )

        return wrapper

    return decorator


def get_date_range() -> Tuple[datetime, datetime, int]:
    period_year = config["collection"]["period_years"]
    period_days = period_year * 365 + 1
    end_date = date.today()
    start_date = end_date - timedelta(days=int(period_days))
    # Convert dates to datetime objects
    end_date = pd.to_datetime(end_date)
    start_date = pd.to_datetime(start_date)
    # Convert to numpy datetime format for business day calculation
    start_date_np = np.datetime64(start_date, "D")
    end_date_np = np.datetime64(end_date, "D")
    # Calculate business days using np.busday_count
    business_days = np.busday_count(start_date_np, end_date_np)
    return start_date, end_date, business_days


def pluralise(title: str, quantity: int) -> str:
    if quantity == 1:
        return title
    else:
        return title + "s"


@retry(Exception, tries=3)
def validate_ticker(
    ticker_symbol: str,
) -> Optional[Dict[str, Union[str, pd.Timestamp]]]:

    if not isinstance(ticker_symbol, str):
        raise TypeError("ticker_symbol must be a string")

    try:
        ticker = yf.Ticker(ticker_symbol)

        # Suppress yfinance output
        with contextlib.redirect_stderr(open(os.devnull, "w")):
            # Attempt to fetch metadata
            info = ticker.info

        # If info is empty or lacks meaningful data, consider it invalid
        if not info or info.get("symbol") is None:
            logging.error(
                f"Ticker '{ticker_symbol}' is invalid. Ensure the symbol is correct."
            )
            return None

        # Safely handle date conversion
        first_trade_date = info.get(
            "firstTradeDateEpochUtc",
            info.get("firstTradeDateEpoch", None),
        )
        if first_trade_date is not None:
            try:
                earliest_date = pd.to_datetime(first_trade_date, unit="s")
            except (ValueError, TypeError):
                earliest_date = None
        else:
            earliest_date = None

        # Construct the data dictionary
        data = {
            "ticker": info.get("symbol", ticker_symbol).upper(),
            "earliest_date": earliest_date,
            "type": info.get("quoteType", "Unknown"),
            "currency": info.get("currency", "Unknown"),
        }

        # Check for missing data points
        missing_fields = []
        if data["earliest_date"] is None:
            missing_fields.append("earliest_date")
        if data["type"] == "Unknown":
            missing_fields.append("type")
        if data["currency"] == "Unknown":
            missing_fields.append("currency")

        if missing_fields and not info:
            logging.warning(
                f"{ticker_symbol} is valid but missing fields: {', '.join(missing_fields)}."
            )

        return data

    except Exception as e:
        logging.error(f"An unexpected error occurred fetching {ticker_symbol}: {e}")
        return None


def validate_tickers(
    tickers: List[str], max_tickers_in_logs: int = 5
) -> List[Tuple[str, Optional[datetime], str, str]]:
    """
    Validates tickers and returns a list of valid tickers with metadata.

    Args:
        tickers: List of ticker symbols.
        max_tickers_in_logs: Maximum number of tickers to display in logs.

    Returns:
        List of tuples containing ticker metadata.
    """
    valid_tickers = []

    for ticker_symbol in tickers:
        try:
            data = validate_ticker(ticker_symbol)
            if not data:
                continue

            ticker, earliest_date, type, currency = (
                data["ticker"],
                data["earliest_date"],
                data["type"],
                data["currency"],
            )

            missing_fields = [
                field
                for field, value in zip(
                    ["ticker", "earliest_date", "type", "currency"],
                    [ticker, earliest_date, type, currency],
                )
                if not value or value == "Unknown"
            ]

            if not missing_fields:
                valid_tickers.append((ticker, earliest_date, type, currency))

        except Exception as e:
            logging.error(
                f"An unexpected error occurred processing ticker '{ticker_symbol}': {e}"
            )

    def summarise_tickers(ticker_list: List[str], label: str, limit: int) -> str:
        num_tickers = len(ticker_list)
        displayed_tickers = ticker_list[:limit]
        continuation = "..." if num_tickers > limit else ""
        return f"{num_tickers} {label.capitalize()} {pluralise('ticker',num_tickers)} {displayed_tickers}{continuation}"

    logging.info(
        summarise_tickers([t[0] for t in valid_tickers], "valid", max_tickers_in_logs)
    )

    return valid_tickers


def get_tickers(
    tickers: List[str], set_maximum_tickers: int = 5
) -> List[Tuple[str, Optional[datetime], str, str]]:
    """
    Get and validate stock and FX tickers.

    Args:
        tickers: A list of ticker symbols to validate.
        set_maximum_tickers: Maximum number of tickers to display in logs.

    Returns:
        List of valid tickers with metadata.
    """
    if not tickers:
        logging.error("No tickers defined in YAML file. Exiting.")
        sys.exit(1)

    logging.info(f"Validating {len(tickers)} stock tickers...")
    valid_tickers = validate_tickers(tickers)

    if not valid_tickers:
        logging.error("No valid stock tickers found in YAML file. Exiting.")
        sys.exit(1)

    # Generate FX tickers based on the currencies in valid_tickers
    currencies = {t[3] for t in valid_tickers if t[3] not in ["GBP", "GBp"]}
    fx_tickers = {
        f"{currency}GBP=X" if currency != "USD" else "GBP=X" for currency in currencies
    }

    valid_fx_tickers = validate_tickers(list(fx_tickers))

    if not valid_fx_tickers:
        logging.warning("No valid FX tickers found. Processing GBP stocks only.")

    # Combine and deduplicate valid tickers (preserving full tuples)
    all_valid_tickers = list({t for t in valid_tickers + valid_fx_tickers})

    logging.info(
        f"Validated {len(all_valid_tickers)} {pluralise('ticker',len(all_valid_tickers))} in total."
    )

    return all_valid_tickers


@retry(Exception, tries=3)
def fetch_ticker_history(
    ticker_symbol: str, start_date: datetime, end_date: datetime
) -> pd.DataFrame:
    """
    Fetches price/rate data for a single ticker.
    First checks if the data is cached; if not,
    fetches the data using yfinance and caches it.
    """
    cache_dir = os.path.join(base_directory, config["paths"]["cache"])
    if not os.path.exists(cache_dir):
        os.makedirs(cache_dir)
    cache_file = os.path.join(
        cache_dir,
        f"{ticker_symbol}_{start_date.strftime('%Y%m%d')}_{end_date.strftime('%Y%m%d')}.pkl",
    )
    if os.path.exists(cache_file):
        with open(cache_file, "rb") as f:
            tidy_df = pickle.load(f)

    else:
        ticker = yf.Ticker(ticker_symbol)

        time.sleep(0.2)  # Rate limiting
        df = ticker.history(
            start=start_date,
            end=end_date,
            interval="1d",
        )
        df = df.rename(columns={"Close": "Old Price"})

        if df.empty:
            logging.warning(
                f"No data found online for {ticker_symbol}. Check if available during this time period."
            )
            return pd.DataFrame()

        # Drop superfluous columns
        df = df.reset_index()  # make the index a regular column named "Date"
        tidy_df = df[["Date", "Old Price"]].copy()
        tidy_df["Ticker"] = ticker_symbol

        with open(cache_file, "wb") as f:  # With automatically closes file
            pickle.dump(tidy_df, f)

    return tidy_df  # returns a pandas df


@retry(Exception, tries=3)
def fetch_historical_data(
    tickers: List[Tuple[str, Optional[datetime], str, str]],
) -> pd.DataFrame:
    """
    Fetch historical price data for all valid tickers.

    Args:
        tickers: List of valid tickers with metadata.

    Returns:
        Combined DataFrame containing historical data for all tickers.
    """
    records = []

    # Get date range
    start_date, end_date, business_days = get_date_range()

    for ticker, earliest_date, type, currency in tickers:
        # Adjusted start date is always the later of earliest_date and start_date unless one is None.
        adjusted_start_date = max(start_date, earliest_date or start_date)

        # Check if end_date is before earliest_date, if so skip the ticker
        if earliest_date and end_date < earliest_date:
            logging.warning(
                f"Skipping {ticker}: Period requested, ending {end_date.strftime('%d/%m/%Y')}, is before {ticker} existed (from {earliest_date.strftime('%d/%m/%Y')}).\n"
            )
            continue

        if adjusted_start_date > start_date:
            logging.warning(
                f"Earliest available data for {ticker} is from {adjusted_start_date.strftime('%d-%m-%Y')}.\n"
            )
        try:
            # Fetch historical data
            df = fetch_ticker_history(ticker, adjusted_start_date, end_date)
            if df.empty:
                logging.info(f"No data for {ticker}. Skipping.")
                continue

            # Add metadata columns to the DataFrame
            df["Ticker"] = ticker
            df["Type"] = type
            df["Original Currency"] = currency

            # Append to records
            records.append(df)
        except Exception as e:
            logging.warning(f"Failed to fetch data for {ticker}: {e}")

    if not records:
        logging.error(
            f"No data fetched for the {business_days} business days between "
            f"{start_date.strftime('%d/%m/%Y')} and {end_date.strftime('%d/%m/%Y')}!"
        )
        return pd.DataFrame()  # Return an empty DataFrame when no data is fetched.

    # Combine all individual DataFrames into a single DataFrame
    combined_df = pd.concat(records, ignore_index=True)

    # Reorder columns to ensure consistency
    combined_df = combined_df[
        [
            "Ticker",
            "Date",
            "Type",
            "Old Price",
            "Original Currency",
        ]
    ]

    return combined_df


def convert_prices(df: pd.DataFrame) -> pd.DataFrame:
    """
    Converts prices to GBP based on ticker type and currency.

    Args:
        df: DataFrame containing historical data.

    Returns:
        DataFrame with prices converted to GBP where applicable.
    """

    # Standardise the Date column to datetime.date
    def standardise_date(value):
        try:
            dt = pd.to_datetime(value, errors="coerce")
            if pd.isna(dt):
                return None  # Handle invalid dates
            return dt.date()  # Convert to date only
        except Exception as e:
            logging.warning(f"Error parsing date '{value}': {e}")
            return None

    # Apply the standardisation function to the Date column
    df["Date"] = df["Date"].apply(standardise_date)

    # Validate Date column
    if df["Date"].isna().any():
        logging.warning("Some rows contain invalid or missing dates.")
        # Remove these rows
        df = df.dropna(subset=["Date"])

    # Create exchange rate DataFrame with valid `Date` index
    exchange_rate_df = (
        df[df["Type"] == "CURRENCY"]
        .rename(columns={"Old Price": "FX Rate"})
        .set_index(["Date", "Ticker"])
    )

    # Prepare the output DataFrame
    final_df = df.copy()
    final_df["FX Code"] = ""
    final_df["FX Rate"] = np.nan
    final_df["Price"] = np.nan
    final_df["Currency"] = "GBP"

    # Process each row
    for index, row in final_df.iterrows():
        currency = row["Original Currency"]
        date = row["Date"]
        price = row["Old Price"]
        type = row["Type"]

        try:
            # Skip conversion for specific quote types
            if type in ["CURRENCY", "INDEX"]:
                final_df.at[index, "FX Code"] = "-"
                final_df.at[index, "FX Rate"] = 1
                final_df.at[index, "Price"] = price
                final_df.at[index, "Currency"] = currency
                continue

            # Handle GBP and GBp cases
            if currency == "GBP":
                fx_ticker = "-"
                exchange_rate = 1
            elif currency == "GBp":
                fx_ticker = "-"
                exchange_rate = 0.01
            else:
                # Create FX ticker for non-GBP currencies
                fx_ticker = f"{currency}GBP=X" if currency != "USD" else "GBP=X"
                # Retrieve exchange rate using date objects
                try:
                    exchange_rate = exchange_rate_df.loc[(date, fx_ticker), "FX Rate"]
                except KeyError:
                    logging.error(
                        f"FX rate not found for {currency} ({fx_ticker}) on date {date}."
                    )
                    exchange_rate = np.nan

            # Calculate the converted price
            if not np.isnan(exchange_rate):
                converted_price = price * exchange_rate
            else:
                converted_price = np.nan

            # Populate the row with updated data
            final_df.at[index, "FX Code"] = fx_ticker
            final_df.at[index, "FX Rate"] = exchange_rate
            final_df.at[index, "Price"] = converted_price
            final_df.at[index, "Currency"] = "GBP"

        except Exception as e:
            logging.error(f"Error processing row {index}: {e}")

    # Remove rows where conversion failed (Price is NaN)
    final_df = final_df.dropna(subset=["Price"])

    return final_df


def process_converted_prices(
    converted_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Prepares the final DataFrame for CSV export.

    Args:
        converted_df: DataFrame with converted prices.

    Returns:
        Processed DataFrame ready to be saved as CSV.
    """
    try:
        output_csv = converted_df.copy()

        # Select required columns
        output_csv = output_csv[["Ticker", "Price", "Date"]]

        # Order by descending date
        output_csv = output_csv.sort_values(by="Date", ascending=False)

        # Reformat date into dd/mm/yyyy
        output_csv["Date"] = pd.to_datetime(output_csv["Date"]).dt.strftime("%d/%m/%Y")

        return output_csv

    except Exception as e:
        logging.error(f"An error occurred during processing: {e}")
        raise


def format_path(full_path):
    """Format a file path to show the last parent directory and filename."""
    path_parts = full_path.split("\\")
    if len(path_parts) > 1:
        return "\\" + "\\".join(path_parts[-2:])  # Show parent dir and filename
    return "\\" + full_path  # Just in case there's no parent directory


@retry(Exception, tries=3)
def save_to_csv(df: pd.DataFrame) -> bool:
    """
    Saves the DataFrame to a CSV file without headers.

    Args:
        df: DataFrame to save.

    Returns:
        True if saved successfully, False otherwise.
    """
    try:
        # Build the output file path
        CSV_file_name = config["paths"]["data_file"]
        CSV_path = os.path.join(base_directory, CSV_file_name)
        short_CSV_path = format_path(CSV_path)

        required_columns = ["Ticker", "Price", "Date"]
        if not all(col in df.columns for col in required_columns):
            logging.error(
                f"Required columns are missing. Available columns are: {df.columns.tolist()}"
            )
            return False

        # Save the DataFrame to CSV without headers
        df.to_csv(CSV_path, index=False, header=False)

        # Log the save location
        logging.info(f"Data successfully saved to: {short_CSV_path}")

        return True

    except Exception as e:
        logging.exception(
            f"An error occurred while saving the dataframe to a file called {CSV_file_name}"
        )
        return False


def clean_cache():
    """
    Cleans up cached data files older than the specified maximum age.
    """
    cache_dir = config["paths"]["cache"]
    cache_dir_path = os.path.join(base_directory, cache_dir)
    max_age_days = config["cache"]["max_age_days"]
    now = time.time()
    cutoff = now - (max_age_days * 86400)  # in seconds
    num_cleaned_files = 0
    try:
        for filename in os.listdir(cache_dir_path):
            file_path = os.path.join(cache_dir_path, filename)
            file_stats = os.stat(file_path)
            # mod_time = the number of seconds since January 1, 1970, 00:00:00 UTC that the file was last modified.
            mod_time = file_stats.st_mtime
            file_age_in_seconds = now - mod_time
            if os.path.isfile(file_path) and file_age_in_seconds > cutoff:
                os.remove(file_path)
                logging.info(
                    f"File {cache_dir}/{filename} older than {max_age_days} days, so deleted!"
                )
                num_cleaned_files += 1
        logging.info(
            f"{num_cleaned_files} cache files were cleaned for being older than {max_age_days} days."
        )
    except Exception as e:
        logging.error(f"Error cleaning cache: {e}")
        logging.error(traceback.format_exc())
        return False
    return True


def is_elevated():
    """
    Checks if the script is running with administrative privileges.
    """
    try:
        return ctypes.windll.shell32.IsUserAnAdmin()
    except:
        return False


def is_admin():
    """
    Alias for is_elevated to maintain consistency.
    """
    return is_elevated()


def run_as_admin():
    """
    Re-runs the script with administrative privileges if not already elevated.
    """
    if not is_admin():
        # Re-run the script as admin

        return_code = ctypes.windll.shell32.ShellExecuteW(
            None, "runas", sys.executable, __file__, None, 1
        )
        if return_code == 5:
            # User rejected the elevation prompt
            logging.info("Elevation request cancelled.")

        elif return_code == 42:
            # User accepted elevation prompt
            exit(0)
        else:
            logging.warning(f"Unknown situation. Code: {return_code}")
            exit(0)


@retry(exceptions=(RuntimeError, IOError), tries=3, delay=2, backoff=2)
def import_data_file():

    try:
        output_file_name = config["paths"]["data_file"]
        filename = os.path.join(base_directory, output_file_name)

        # Type the file name into the dialog
        pyautogui.typewrite(filename, interval=0.01)
        time.sleep(0.3)  # Small delay before hitting enter for stability
        pyautogui.press("enter")

        # Wait to get to the price import success dialogue
        start_time = time.time()
        while time.time() - start_time < 10:
            windows = gw.getWindowsWithTitle(
                "Quicken XG 2004 - Home - [Investing Centre]"
            )
            if windows:

                pyautogui.press("enter")
                logging.info(
                    f"Successfully imported {format_path(filename)} to Quicken at {datetime.now().strftime('%d/%m/%Y %H:%M')}"
                )
                return True
            time.sleep(0.5)

    except Exception as e:
        logging.error(f"Failed to import data file: {e} ({type(e).__name__})")
        raise


@retry(exceptions=(Exception,), tries=3, delay=2, backoff=2)
def open_import_dialog():

    try:
        with pyautogui.hold("alt"):
            pyautogui.press(["f", "i", "i"])

        # Wait to get to the price import dialogue box
        start_time = time.time()
        while time.time() - start_time < 10:
            windows = gw.getWindowsWithTitle("Import Price Data")
            if windows:

                return True
            time.sleep(1)

        return True
    except Exception as e:
        logging.error(f"Failed to open price import dialogue box: {e}")
        return False


@retry(exceptions=(Exception,), tries=3, delay=2, backoff=2)
def navigate_to_portfolio():

    try:
        pyautogui.hotkey("ctrl", "u")

        # Wait to get to the investing portfolio page
        start_time = time.time()
        while time.time() - start_time < 30:
            windows = gw.getWindowsWithTitle(
                "Quicken XG 2004 - Home - [Investing Centre]"
            )
            if windows:

                return True
            time.sleep(1)

        logging.error("Could not get to portfolio page within the expected time.")
        return False

    except Exception as e:
        logging.error(f"Failed to navigate to portfolio: {e}")
        return False


@retry(exceptions=(Exception,), tries=3, delay=2, backoff=2)
def open_quicken():

    quicken_path = config["paths"]["quicken"]
    try:
        # Check if Quicken is already open
        windows = gw.getWindowsWithTitle("Quicken XG 2004")
        if len(windows) > 0:
            quicken_window = windows[0]
            quicken_window.activate()  # Bring the existing Quicken window to the foreground
            return True

        # If not open, launch Quicken

        subprocess.Popen([quicken_path], shell=True)

        # Wait for the Quicken window to appear
        start_time = time.time()
        while time.time() - start_time < 30:
            windows = gw.getWindowsWithTitle("Quicken XG 2004")
            if windows:

                # Temporarily disable Ctrl+Alt+Del to prevent accidental interruption
                pyautogui.hotkey("ctrl", "alt", "del")
                return True
            time.sleep(1)

        # If we cannot find the window
        logging.error("Quicken window did not appear within the expected time.")
        return False

    except Exception as e:
        logging.error(f"Failed to open Quicken: {e}")
        return False


def execute_import_sequence():
    """
    Execute the sequence of steps for importing data.

    Returns:
        bool: True if all steps completed successfully
    """
    steps = [
        (open_quicken, "Opening Quicken..."),
        (navigate_to_portfolio, "Navigating to Portfolio view..."),
        (open_import_dialog, "Opening import dialog..."),
        (import_data_file, "Importing data file..."),
    ]

    for step_function, message in steps:
        logging.info(message)
        if not step_function():
            logging.error(f"Step failed: {message}")
            return False

    # Reenabling Ctrl+Alt+Del previously disabled to prevent accidental interruption
    pyautogui.hotkey("ctrl", "alt", "del")
    return True


def setup_pyautogui():

    pyautogui.FAILSAFE = True  # Abort ongoing automation by moving mouse to the top-left corner of screen.
    pyautogui.PAUSE = 0.3  # Delay between actions


def quicken_import():

    try:

        if is_elevated():

            return execute_import_sequence()

        return

    except Exception as e:
        logging.error(f"Error during Quicken import: {e}")
        return


def main():

    try:

        # Conditional clear screen at beginning for clean output
        if config["clear_startup"] == True:
            os.system("cls" if os.name == "nt" else "clear")

        # Get elevation
        run_as_admin()

        # Get 'raw' tickers from YAML file and FX tickers if required, validate and acquire metadata.
        ticker_list = config["tickers"]
        valid_tickers = get_tickers(ticker_list, set_maximum_tickers=5)

        # Fetch historical price and FX data
        price_data = fetch_historical_data(valid_tickers)

        # Convert prices to GBP except INDEX and CURRENCY
        processed_data = convert_prices(price_data)

        # Process and create output DataFrame
        output_csv = process_converted_prices(processed_data)

        # Save data to CSV
        save_to_csv(output_csv)

        # Clean up cache
        clean_cache()

        #  Open Quicken
        setup_pyautogui()
        quicken_import()

        # Pause to allow reading of terminal
        input("\n\t Press Enter to exit...\n\t")

    except KeyboardInterrupt:
        logging.warning("Script interrupted by user.")
    except Exception as e:
        logging.error(f"An unexpected error occurred: {e}")
        logging.error(traceback.format_exc())


if __name__ == "__main__":

    # Ensure base directory exists
    base_directory = config["paths"]["base"]
    os.makedirs(base_directory, exist_ok=True)

    main()
