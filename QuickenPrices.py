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

# Imports #####################################

# Standard Library Imports
import calendar
import ctypes
import datetime
import inspect
import logging
import os
import pickle
import subprocess
import sys
import time
import traceback
import warnings
from functools import wraps
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Type, Union

# Windows-Specific Imports
import winreg  # Registry access

# Third-Party Data and Scientific Computing Imports
import numpy as np
from numpy import busday_count
import pandas as pd
from pandas import Timestamp, Timedelta
import yaml
import yfinance as yf

# GUI and Automation Imports
import pyautogui
import pygetwindow as gw


# Constants ###################################

SECONDS_IN_A_DAY = 86400
CACHE_COLUMNS = ("Ticker", "Old Price", "Date")
DEFAULT_CONFIG = {
    "home_currency": "GBP",
    "collection_period_years": 0.08,
    "retries": {"max_retries": 3, "retry_delay": 2, "backoff": 2},
    "logging_level": {"file": "DEBUG", "terminal": "DEBUG"},
    "log_file": {"max_bytes": 5242880, "backup_count": 5},
    "tickers": ["^FTSE"],
    "paths": {"data_file": "data.csv", "log_file": "prices.log", "cache": "cache"},
    "cache": {"max_age_days": 30},
}


# Configuration ###############################


def load_configuration(config_file: str = "configuration.yaml") -> Dict[str, Any]:
    """
    Load and validate YAML configuration with defaults.
    Exits if critical values are missing or invalid.
    """

    base_path = Path(__file__).resolve().parent
    config_path = base_path / config_file

    # Ensure configuration file exists
    if not config_path.exists():
        logging.warning(
            f"{config_file} not found. Creating default configuration. Please update to suit."
        )
        with config_path.open("w", encoding="utf-8") as f:
            yaml.dump(DEFAULT_CONFIG, f, default_flow_style=False)
        logging.info(f"Default {config_file} created at {config_path}.")
    else:
        # Log that the local YAML file is found and being used
        logging.info(f"Using {config_file} found at {base_path}.")

    # Load user configuration
    try:
        with config_path.open("r", encoding="utf-8") as f:
            user_config = yaml.safe_load(f) or {}

        if not isinstance(user_config, dict):
            logging.warning(f"Invalid format in {config_file}. Using defaults.")
            user_config = {}

    except Exception as e:
        logging.error(f"Error reading {config_file}: {e}. Using internal defaults.")
        user_config = {}

    # Merge defaults with user config
    final_config = apply_defaults(DEFAULT_CONFIG, user_config)

    # Add base path to paths in the config
    final_config.setdefault("paths", {}).setdefault("base", str(base_path))

    # Validate Quicken path
    quicken_path = validate_quicken_path()
    if quicken_path:
        final_config["paths"]["quicken"] = quicken_path
        logging.info(f"Using Quicken found at: {quicken_path}")
    else:
        quicken_not_found_error()

    # Validate tickers
    if not final_config["tickers"]:
        logging.error(
            "No tickers defined in configuration. Please add at least one ticker."
        )
        sys.exit(1)

    return final_config


def apply_defaults(
    defaults: Dict[str, Any], settings: Dict[str, Any]
) -> Dict[str, Any]:

    for key, value in defaults.items():
        if isinstance(value, dict):
            settings[key] = apply_defaults(value, settings.get(key, {}))
        else:
            settings.setdefault(key, value)
    return settings


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
                    quicken_path, _ = winreg.QueryValueEx(key, None)
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


def quicken_not_found_error():
    """
    Handle the case where the Quicken executable cannot be located.
    """
    print(
        "\n❌ Critical Error: Quicken executable could not be located.\n"
        "✅ Suggestions:\n"
        "  1. Ensure Quicken is installed on your system.\n"
        "  2. Verify installation paths:\n"
        "     - C:\\Program Files (x86)\\Quicken\\qw.exe\n"
        "     - C:\\Program Files\\Quicken\\qw.exe\n"
        "  3. If using an older version, confirm proper installation.\n"
        "\n⚠️ Manually specify the Quicken path in configuration.yaml if needed.\n"
    )
    sys.exit(1)


# Load the configuration after all config related functions loaded
config = load_configuration("configuration.yaml")

# Coding utilities ############################


def retry(
    exceptions,
    config=config,
    tries=config["retries"]["max_retries"],
    delay=config["retries"]["retry_delay"],
    backoff=config["retries"]["backoff"],
):

    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            attempt = 0
            while attempt < tries:
                try:
                    return func(*args, **kwargs)
                except exceptions as e:
                    attempt += 1
                    if attempt == tries:
                        logging.error(
                            f"Max retries reached for {func.__name__}. Error: {e}"
                        )
                        raise
                    logging.warning(
                        f"Retry {attempt}/{tries} for {func.__name__} in {delay} seconds after error: {e}"
                    )
                    time.sleep(delay)
                    delay *= backoff

        return wrapper

    return decorator


def setup_logging(config: Dict[str, Any]) -> None:
    """
    Sets up logging configuration.
    """

    base_path = config["paths"]["base"]
    file_log_level = config["logging_level"]["file"].upper()
    terminal_log_level = config["logging_level"]["terminal"].upper()

    # Determine the log directory based on the operating system
    if os.name == "nt":  # Windows
        log_dir = os.path.join(os.getenv("APPDATA", base_path), "YourApp", "logs")
    else:  # Linux/macOS
        log_dir = os.path.join(os.path.expanduser("~"), ".your_app", "logs")

    # Ensure the logs directory exists
    os.makedirs(log_dir, exist_ok=True)

    # Use the log file path from the configuration or default to the generated log directory
    log_file_path = os.path.join(base_path, config["paths"]["log_file"])

    # Create a rotating file handler with a simpler format for readability
    file_handler = RotatingFileHandler(
        log_file_path,
        maxBytes=config["log_file"]["max_bytes"],
        backupCount=config["log_file"]["backup_count"],
        encoding="utf-8",
    )
    file_handler.setLevel(getattr(logging, file_log_level, logging.DEBUG))
    file_formatter = logging.Formatter(
        "%(asctime)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s"
    )
    file_handler.setFormatter(file_formatter)

    # Create a console handler with the same format
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(getattr(logging, terminal_log_level, logging.DEBUG))
    console_formatter = logging.Formatter(
        "%(asctime)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s"
    )
    console_handler.setFormatter(console_formatter)

    logging.getLogger().handlers = []
    logging.getLogger().addHandler(file_handler)
    logging.getLogger().addHandler(console_handler)
    logging.getLogger().setLevel(logging.DEBUG)

    logging.info(f"Log at: {log_file_path}")


# Initialisation ##############################


def run_as_admin():
    """
    Re-runs the script with administrative privileges if not already elevated.
    """
    if not is_elevated():
        # Re-run the script as admin
        return_code = ctypes.windll.shell32.ShellExecuteW(
            None, "runas", sys.executable, __file__, None, 1
        )
        if return_code == 5:
            # User rejected the elevation prompt
            logging.info("Elevation request cancelled.")
        elif return_code == 42:
            # User accepted elevation prompt
            logging.info("Continuing elevated.")
            exit(0)
        else:
            logging.warning(f"Unknown elevation situation. Code: {return_code}")
            exit(0)


def is_elevated():
    """
    Checks if the script is running with administrative privileges.
    """

    try:
        return ctypes.windll.shell32.IsUserAnAdmin()
    except:
        return False


# Helper functions ############################


def pluralise(title: str, quantity: int) -> str:
    return title if quantity == 1 else title + "s"


def panda_date(
    date_obj: Union[
        str, datetime.date, datetime.datetime, pd.Timestamp, int, float, None
    ]
) -> pd.Timestamp:
    """Converts a date-like object to a UTC-normalized pandas Timestamp.

    Handles strings, datetime objects, pandas Timestamps, and Unix epoch
    timestamps (seconds or milliseconds). Returns pd.NaT for invalid input.

    Args:
        date_obj: The date object to convert.

    Returns:
        A UTC-normalized pandas Timestamp, or pd.NaT if the input is invalid.
    """
    if date_obj is None or pd.isna(date_obj):  # Explicitly handle None
        return pd.NaT

    try:
        if isinstance(date_obj, (int, float)):
            unit = "ms" if date_obj > 10**10 else "s"
            date_obj = pd.Timestamp(date_obj, unit=unit, tz="UTC")
        elif isinstance(date_obj, (datetime.datetime, datetime.date)):
            date_obj = pd.Timestamp(date_obj)
            date_obj = (
                date_obj.tz_localize("UTC")
                if date_obj.tzinfo is None
                else date_obj.tz_convert("UTC")
            )
        elif isinstance(date_obj, pd.Timestamp):
            date_obj = (
                date_obj.tz_localize("UTC")
                if date_obj.tzinfo is None
                else date_obj.tz_convert("UTC")
            )
        else:  # String or other
            date_obj = pd.to_datetime(date_obj, utc=True)
            date_obj = (
                date_obj.tz_localize("UTC")
                if date_obj.tzinfo is None
                else date_obj.tz_convert("UTC")
            )

        return date_obj

    except (ValueError, TypeError) as e:
        logging.warning(f"Invalid date format: {date_obj}. Error: {e}")
        return pd.NaT


def adjust_for_weekend(
    date: Union[pd.Timestamp, datetime.date, datetime.datetime, None],
    direction: str = "forward",
) -> Union[pd.Timestamp, datetime.date, datetime.datetime, None]:
    """Adjusts a date for weekends.

    If the date falls on a weekend (Saturday or Sunday), it's adjusted
    to the following Monday ("forward") or the preceding Friday ("back").
    Returns the original date if it's not a weekend.

    Args:
        date: The date to adjust. Can be a pandas Timestamp, datetime.date,
            datetime.datetime, or None.
        direction: The adjustment direction ("forward" or "back").

    Returns:
        The adjusted date (same type as input), or None if the input is None.

    Raises:
        ValueError: If the direction is invalid.
    """

    if date is None or pd.isna(date):
        return None

    if not isinstance(date, (pd.Timestamp, datetime.date, datetime.datetime)):
        raise TypeError(
            "Input date must be a pandas Timestamp or datetime.date/datetime object"
        )

    weekday = date.weekday()

    if weekday in (5, 6):  # Saturday or Sunday (More explicit)
        if direction == "forward":
            adjustment = calendar.MONDAY - weekday
        elif direction == "back":
            adjustment = calendar.FRIDAY - weekday
        else:
            raise ValueError("Invalid direction. Must be 'forward' or 'back'.")
        return date + pd.Timedelta(days=adjustment)
    else:
        return date


def get_date_range(config: dict) -> Tuple[pd.Timestamp, pd.Timestamp]:
    """Calculates the start and end dates (naive Timestamps).

    Considers weekends and the specified collection period (in years).
    Returns naive Timestamps to avoid timezone comparison issues later in the code.

    Args:
        config: Configuration with "collection_period_years".

    Returns:
        A tuple containing the start and end dates (naive Timestamps).
    """
    collection_period_years = config["collection_period_years"]
    collection_period_days = int(round(collection_period_years * 365.25))

    today = pd.Timestamp.now().floor("D")

    end_date = adjust_for_weekend(today, direction="back")
    start_date = end_date - pd.Timedelta(days=collection_period_days)
    start_date = adjust_for_weekend(start_date, direction="forward")

    return start_date, end_date

def get_business_days(
    start_date: pd.Timestamp, end_date: pd.Timestamp
) -> List[pd.Timestamp]:
    """Generates a list of business days between two dates (inclusive).

    Args:
        start_date: The start date.
        end_date: The end date.

    Returns:
        A list of pandas Timestamps representing business days. Returns an empty list if either input date is pd.NaT or if start date is after end date.
    """

    if pd.isna(start_date) or pd.isna(end_date) or start_date > end_date:
        return []

    date_range = pd.date_range(start=start_date, end=end_date)
    is_weekday = np.logical_not(date_range.weekday.isin([5, 6]))
    business_days = date_range[is_weekday]

    return business_days.to_list()


def find_missing_ranges(
    start_date: pd.Timestamp,
    end_date: pd.Timestamp,
    first_cached_date: pd.Timestamp,
    last_cached_date: pd.Timestamp,
) -> List[Tuple[pd.Timestamp, pd.Timestamp]]:
    """Finds missing date ranges between a download period and cached data.

    Args:
        start_date: The start date of the download period.
        end_date: The end date of the download period.
        first_cached_date: The first date in the cached data.
        last_cached_date: The last date in the cached data.

    Returns:
        A list of (start, end) tuples representing missing date ranges.
        Returns an empty list if no ranges are missing or if any input is pd.NaT.
    """

    if any(
        pd.isna(x) for x in [start_date, end_date, first_cached_date, last_cached_date]
    ):
        return []

    assert start_date <= end_date, "start_date must be less than or equal to end_date"
    assert (
        first_cached_date <= last_cached_date
    ), "first_cached_date must be less than or equal to last_cached_date"

    missing_ranges: List[Tuple[pd.Timestamp, pd.Timestamp]] = []

    if start_date < first_cached_date:
        missing_ranges.append(
            (start_date, min(end_date, first_cached_date - pd.Timedelta(days=1)))
        )

    if end_date > last_cached_date:
        missing_ranges.append(
            (max(start_date, last_cached_date + pd.Timedelta(days=1)), end_date)
        )

    return missing_ranges


def format_path(full_path):
    """Format a file path to show the last parent directory and filename."""
    path_parts = full_path.split("\\")
    if len(path_parts) > 1:
        return "\\" + "\\".join(path_parts[-2:])  # Show parent dir and filename
    return "\\" + full_path  # Just in case there's no parent directory


# Validate tickers ############################


def get_tickers(
    tickers: List[str], max_show: int = 5
) -> List[Tuple[str, Optional[pd.Timestamp], str, str]]:
    """
    Get and validate stock and FX tickers.

    Args:
        tickers: A list of ticker symbols to validate.
        max_show: Maximum number of tickers to display in logs.

    Returns:
        List of valid tickers with metadata.
    """

    if not tickers:
        logging.error("No tickers defined in YAML file. Exiting.")
        sys.exit(1)

    logging.info(
        f"Validating {len(tickers)} stock {pluralise('ticker', len(tickers))}..."
    )
    valid_tickers = validate_tickers(tickers, max_show)

    if not valid_tickers:
        logging.error(
            "No valid tickers found. Please update your configuration or check network access. Exiting."
        )
        sys.exit(1)

    # Generate FX tickers based on the currencies in valid_tickers
    currencies = {t[3] for t in valid_tickers if t[3] not in ["GBP", "GBp"]}
    fx_tickers = {
        f"{currency}GBP=X" if currency != "USD" else "GBP=X" for currency in currencies
    }

    # Exclude tickers from fx_tickers that might already be in valid_tickers
    valid_ticker_strings = {ticker[0] for ticker in valid_tickers}
    fx_tickers = {ticker for ticker in fx_tickers if ticker not in valid_ticker_strings}

    if fx_tickers:
        logging.info(
            f"Validating {len(fx_tickers)} FX {pluralise('ticker', len(fx_tickers))}..."
        )
        valid_fx_tickers = validate_tickers(list(fx_tickers), max_show)
        all_valid_tickers = list(set(valid_tickers + valid_fx_tickers))
    else:
        logging.info("No new FX tickers required.")
        all_valid_tickers = valid_tickers

    logging.info(
        f"Validated {len(all_valid_tickers)} {pluralise('ticker', len(all_valid_tickers))} in total.\n"
    )
    return all_valid_tickers


def validate_tickers(
    tickers: List[str], max_show: int = 5
) -> List[Tuple[str, Optional[pd.Timestamp], str, str]]:
    """
    Validates tickers and returns a list of valid tickers with metadata.

    Args:
        tickers: List of ticker symbols.
        max_show: Maximum number of tickers to display in logs.

    Returns:
        List of tuples containing ticker metadata.
    """
    valid_tickers = []

    for ticker_symbol in tickers:
        try:
            data = validate_ticker(ticker_symbol)
            if (
                data
                and data["ticker"]
                and data["type"] != "Unknown"
                and data["currency"] != "Unknown"
            ):
                valid_tickers.append(
                    (
                        data["ticker"],
                        data["earliest_date"],
                        data["type"],
                        data["currency"],
                        data["current_price"]
                    )
                )

        except Exception as e:
            logging.error(
                f"An unexpected error occurred processing ticker '{ticker_symbol}': {e}"
            )

    logging.info(
        f"{len(valid_tickers)} {pluralise('ticker', len(valid_tickers))} validated: "
        f"{[ticker[0] for ticker in valid_tickers[:max_show]]}"
        f"{'...' if len(valid_tickers) > max_show else ''}"
    )

    return valid_tickers


@retry(Exception, config=config)
def validate_ticker(
    ticker_symbol: str,
) -> Optional[Dict[str, Union[str, pd.Timestamp]]]:
    """
    Validate a ticker symbol and fetch metadata using yfinance.

    Args:
        ticker_symbol (str): The ticker symbol to validate.

    Returns:
        Optional[Dict[str, Union[str, pd.Timestamp]]]: Metadata about the ticker, or None if invalid.
    """
    if not isinstance(ticker_symbol, str):
        logging.error("Ticker must be a string.")
        return None

    try:
        ticker = yf.Ticker(ticker_symbol)
        info = ticker.info
        if not info or info.get("symbol") is None:
            logging.error(
                f"Ticker '{ticker_symbol}' is invalid. Check if the symbol is correct or available on yfinance."
            )
            return None

        # Safely handle date conversion
        first_trade_date = info.get("firstTradeDateEpochUtc", None)
        earliest_date = panda_date(first_trade_date) if first_trade_date else pd.nat

        # Construct the data dictionary
        data = {
            "ticker": info.get("symbol", ticker_symbol).upper(),
            "earliest_date": earliest_date,
            "type": info.get("quoteType", "Unknown"),
            "currency": info.get("currency", "Unknown"),
            "current_price": info.get("bid", "Unknown"),
        }

        return data

    except Exception as e:
        logging.error(f"An unexpected error occurred fetching {ticker_symbol}: {e}")
        return None

# Get data ####################################

def fetch_historical_data(
    tickers: List[Tuple[str, Optional[pd.Timestamp], str, str, float]],
    config: Dict[str, Any],
) -> pd.DataFrame:
    """Fetch historical data for a list of tickers."""

    if not tickers:
        logging.info("No tickers provided. Returning empty DataFrame.")
        return pd.DataFrame(columns=list(CACHE_COLUMNS) + ["Type", "Original Currency"])

    start_date, end_date = get_date_range(config)
    business_days = get_business_days(start_date, end_date)
    num_business_days = len(business_days)

    logging.info(
        f"Seeking data from {start_date.strftime('%d/%m/%Y')} to {end_date.strftime('%d/%m/%Y')} ({num_business_days} business days).\n"
    )

    records = []
    for ticker, earliest_date, ticker_type, currency, current_price in tickers:
        if earliest_date is not None:
            earliest_date = earliest_date.tz_localize(None)

        if earliest_date and end_date < earliest_date:
            logging.warning(
                f"Skipping {ticker}: Period requested ends before ticker existed."
            )
            continue

        data_start_date = max(start_date, earliest_date or start_date)

        try:
            df = fetch_ticker_data(
                ticker, data_start_date, end_date, current_price, config
            )

            if df is None or df.empty:
                logging.warning(f"No data found for ticker {ticker}. Skipping.")
                continue

            if "Type" not in df.columns:
                df["Type"] = ticker_type
            if "Original Currency" not in df.columns:
                df["Original Currency"] = currency

            df["Date"] = df["Date"].apply(panda_date)

            records.append(df)

        except requests.exceptions.RequestException as e:
            logging.error(f"Error fetching data for {ticker}: {e}")
        except Exception as e:
            logging.error(f"Unexpected error fetching data for {ticker}: {e}")

    valid_records = [df for df in records if not df.empty and df is not None]
    if valid_records:
        combined_df = pd.concat(valid_records, ignore_index=True)
        logging.info(
            f"Finished getting {len(combined_df)} {pluralise('record',len(combined_df))} for {len(valid_records)} {pluralise('ticker',len(valid_records))}."
        )
        return combined_df
    else:
        logging.warning("No valid data fetched for any tickers.")
        return pd.DataFrame(columns=list(CACHE_COLUMNS) + ["Type", "Original Currency"])


def fetch_ticker_data(ticker, start_date, end_date, current_price, config):
    """
    Fetch and manage historical data for a ticker either as cache, download or a combination of the two.
    """

    cache_dir = os.path.join(config["paths"]["base"], config["paths"]["cache"])
    os.makedirs(cache_dir, exist_ok=True)

    # Initialize empty DataFrame to store downloaded data
    df = pd.DataFrame(columns=CACHE_COLUMNS)

    # Ensure input dates are in the correct format
    start_date = panda_date(start_date).date()
    end_date = panda_date(end_date).date()

    # Load cache
    cache_data = load_cache(ticker, cache_dir)
    cache_data = cache_data.reset_index(drop=True)

    # If cache is empty, set first and last cache dates to pd.NaT
    if cache_data.empty or len(cache_data) == 0:
        logging.info(
            f"{ticker}: Trying to download {start_date.strftime('%d/%m/%Y')} to {end_date.strftime('%d/%m/%Y')}."
        )
        first_cache = pd.NaT
        last_cache = pd.NaT
    else:
        # Get first and last cache dates
        # Safely handle missing "Date" column rather than just using cache_data["Date"]
        raw_dates = set(cache_data.get("Date", []))

        # Ensure dates correct format and sorted
        cached_dates = sorted([panda_date(date) for date in raw_dates if date])
        # Get the first and last dates, or pd.NaT if the list is empty
        first_cache = cached_dates[0].date() if cached_dates else pd.NaT
        last_cache = cached_dates[-1].date() if cached_dates else pd.NaT

        # safety check if there is something wrong with first_cache or last_cache
        if pd.isna(first_cache) or pd.isna(last_cache):
            logging.info(
                f"{ticker}: Invalid cache dates. Trying to download from {start_date.strftime('%d/%m/%Y')} to {end_date.strftime('%d/%m/%Y')}."
            )
            first_cache = pd.NaT
            last_cache = pd.NaT
        else:
            # Cache is ok
            logging.info(
                f"{ticker}: Cache loaded {first_cache.strftime('%d/%m/%Y')} to {last_cache.strftime('%d/%m/%Y')}."
            )

    # Determine missing date ranges (if no cache data, will return download start to finish)
    missing_ranges = find_missing_ranges(start_date, end_date, first_cache, last_cache)

    # initiate downloaded_data
    downloaded_data = pd.DataFrame(columns=CACHE_COLUMNS)

    # Get missing data
    if missing_ranges:
        if isinstance(missing_ranges, (list, tuple)):
            # Handle single tuple or list of tuples case
            for start_date, end_date in missing_ranges:
                try:

                    downloaded_data = download_data(ticker, start_date, end_date)

                except Exception as e:
                    logging.error(f"Error downloading data from {start_date} to {end_date}: {e}")
                    continue  # Skip to next iteration on error

                # Concatenate downloaded data to the main DataFrame
                if df.empty:
                    df = downloaded_data
                elif downloaded_data.empty:
                    pass
                else:
                    df = pd.concat([df, downloaded_data], ignore_index=True)

    # Concatenate downloaded data with cache if both df's have data
    download_count = len(df)
    cache_count = len(cache_data)
    if cache_count == 0 and download_count != 0:
        combined_data = df.copy()
    elif cache_count != 0 and download_count == 0:
        combined_data = cache_data.copy()
    else:
        combined_data = pd.concat([df, cache_data], ignore_index=True)

    # Identify duplicates
    duplicates = combined_data.duplicated(keep="first")  # Keep the first occurrence

    # Check if duplicates exist and warn
    num_duplicates = 0
    if duplicates.any():
        num_duplicates = duplicates.sum()
        print(f"WARNING: {num_duplicates} duplicate rows found.")

        # Display the duplicate rows
        print("Duplicate rows:")
        print(df[duplicates])

        # Remove duplicates
        combined_data.drop_duplicates(
            keep="first", inplace=True
        )  # using inplace is more memory efficient

        print("Dataframe after removing duplicates:")
        print(df)

    combined_data_count = len(combined_data)

    if not combined_data.equals(cache_data):
        save_cache(ticker, combined_data, cache_dir)

    # get today's latest prices
    if isinstance(current_price, float):
        new_row = pd.Series({"Ticker": ticker, "Old Price": current_price, "Date": pd.Timestamp.now().date()})
        # takes the series and converts it into a DataFrame with one row and the correct column names
        combined_data = pd.concat([combined_data, new_row.to_frame().T], ignore_index=True)  # more efficient than append
        download_count += 1
        combined_data_count += 1

    logging.info(
        f"{ticker}: Retrieved {cache_count} {pluralise('day',cache_count)} cache, {download_count} {pluralise('day',download_count)} download, with {num_duplicates} {pluralise('day',num_duplicates)} duplicated. {combined_data_count} {pluralise('day',combined_data_count)} in total.\n"
    )
    return combined_data


@retry(Exception, config=config)
def load_cache(ticker: str, cache_dir: str) -> pd.DataFrame:
    """
    Loads cached data for a specific ticker using pickle.

    Args:
        ticker (str): The ticker symbol.
        cache_dir (str): Path to the cache directory.

    Returns:
        pd.DataFrame: DataFrame containing cached data, or an empty DataFrame if the cache file is not found or cannot be read.
    """
    cache_file = os.path.join(cache_dir, f"{ticker}.pkl")

    if os.path.exists(cache_file):
        try:
            # Use pickle.load to deserialize the data
            with open(cache_file, "rb") as f:
                data = pickle.load(f)
            return data
        except (FileNotFoundError, IOError) as e:
            logging.error(f"Failed to load cache for {ticker}: {e}")
            return pd.DataFrame(columns=CACHE_COLUMNS)
    else:
        logging.info(f"{ticker}: No cache found.")
        return pd.DataFrame(columns=CACHE_COLUMNS)


@retry(Exception, config=config)
def download_data(ticker_symbol, start, end):
    """
    Downloads historical data for a ticker from yfinance and ensures consistency.
    Args:
        ticker_symbol: The ticker symbol to fetch data for.
        start: Start date (tz-aware).
        end: End date (tz-aware).
    Returns:
        A cleaned DataFrame with historical data.
    """

    # Make sure dates in right format
    start = panda_date(start).date()
    end = panda_date(end).date()
    today = pd.Timestamp.now().date()

    # Dont look for 'close' today as it probably wont exist yet
    if start == today:
        return pd.DataFrame(columns=CACHE_COLUMNS)

    # Fetch data using yfinance
    try:
        ticker_obj = yf.Ticker(ticker_symbol)
        raw_data = ticker_obj.history(start=start, end=end, interval="1d")

        if raw_data.empty:
            logging.info(
                f"{ticker_symbol}: yfinance returned no data from {start.strftime('%d/%m/%Y')} to {end.strftime('%d/%m/%Y')}."
            )
            return pd.DataFrame(columns=CACHE_COLUMNS)

        # Reset index to ensure 'Date' is a column
        raw_data.reset_index(inplace=True)

        # Rename `Close` to `Old Price`
        raw_data = raw_data.rename(columns={"Close": "Old Price"})

        # Apply panda_date function to the entire 'Date' column
        raw_data["Date"] = raw_data["Date"].apply(panda_date)

        # Add ticker to df
        raw_data["Ticker"] = ticker_symbol

        # Subset df
        raw_data = raw_data[list(CACHE_COLUMNS)]

        # Get the earliest and latest dates
        earliest_date = raw_data["Date"].min()
        latest_date = raw_data["Date"].max()

        logging.info(
            f"{ticker_symbol}: Downloaded {earliest_date.strftime('%d/%m/%Y')} to {latest_date.strftime('%d/%m/%Y')}."
        )

        return raw_data

    except Exception as e:
        logging.error(f"Error fetching data for {ticker_symbol}: {e}")
        return pd.DataFrame(columns=CACHE_COLUMNS)


@retry(Exception, config=config)
def save_cache(ticker: str, data: pd.DataFrame, cache_dir: str) -> None:
    """
    Save cache for a specific ticker using pickle.

    Args:
        ticker (str): The ticker symbol.
        data (pd.DataFrame): DataFrame containing data to cache.
    """
    cache_file = os.path.join(cache_dir, f"{ticker}.pkl")  # Use .pkl extension
    try:
        os.makedirs(cache_dir, exist_ok=True)

        # Filter to ensure only valid columns are saved
        if "Ticker" not in data.columns:
            raise ValueError(
                f"Ticker column missing in data for {ticker}. Columns: {data.columns.tolist()}"
            )
        data = data[list(CACHE_COLUMNS)].dropna()

        # Use pickle.dump to serialize the DataFrame
        with open(cache_file, "wb") as f:
            pickle.dump(data, f)

        logging.info(f"{ticker}: Cache saved.")
    except Exception as e:
        logging.error(f"Failed to save cache for {ticker}: {e}")


@retry(Exception, config=config)
def clean_cache(config):
    """
    Cleans up cache files older than the configured maximum age.

    Args:
        config (dict): Configuration dictionary containing cache path and max age settings.

    Returns:
        bool: True if cleaning completes successfully, False otherwise.
    """

    cache_dir = os.path.join(config["paths"]["base"], config["paths"]["cache"])
    max_age_seconds = config["cache"]["max_age_days"] * SECONDS_IN_A_DAY
    now = time.time()

    if not os.path.exists(cache_dir) or not os.path.isdir(cache_dir):
        logging.warning(
            f"Cache cleaning: can't find {cache_dir}"
        )
        return False

    deleted_files = 0
    try:
        with os.scandir(cache_dir) as it:
            for entry in it:
                if entry.is_file() and (now - entry.stat().st_mtime) > max_age_seconds:
                    try:
                        os.remove(entry.path)
                        deleted_files += 1
                        logging.info(
                            f"Cache cleaning: {entry.name} deleted. Older than {config['cache']['max_age_days']} {pluralise('day', config['cache']['max_age_days'])}."
                        )
                    except Exception as remove_error:
                        logging.error(
                            f"Cache cleaning: Failed to delete file {entry.path}: {remove_error}"
                        )

        logging.info(
            f"Cache cleaning: {deleted_files} cache {pluralise('file', deleted_files)} deleted.\n"
        )
    except Exception as e:
        logging.error(
            f"Cache cleaning: Error with {cache_dir}: {e}. Check file permissions or existence of the directory."
        )
        return False

    return True


# Process Data ################################
def convert_prices(df: pd.DataFrame) -> pd.DataFrame:
    """
    Converts prices to GBP based on ticker type and currency.

    Args:
        df: DataFrame containing historical data.

    Returns:
        DataFrame with prices converted to GBP where applicable.
    """
    if df.empty:
        logging.error(
            "Empty DataFrame received in `convert_prices`. No data to process."
        )
        return pd.DataFrame()

    # Remove time component to enable better matching of FX with stock.
    df["Date"] = pd.to_datetime(df["Date"]).dt.date

    # Extract exchange rates for CURRENCY type
    exchange_rate_df = (
        df[df["Type"] == "CURRENCY"]
        .rename(columns={"Old Price": "FX Rate", "Ticker": "FX Ticker"})
        .loc[:, ["Date", "FX Ticker", "FX Rate"]]
        .sort_values(by=["Date", "FX Ticker"])
    ).copy()

    # Initialise FX-related columns
    df = df.copy()
    df["FX Ticker"] = "-"
    df = df.rename(columns={"Price": "Old Price"})

    # Non-GBP FX Ticker Assignment
    non_gbp_mask = ~df["Original Currency"].isin(["GBP", "GBp"])
    df.loc[non_gbp_mask, "FX Ticker"] = (
        df.loc[non_gbp_mask, "Original Currency"] + "GBP=X"
    )

    # Handle USD Special Case
    df.loc[df["Original Currency"] == "USD", "FX Ticker"] = "GBP=X"

    # Merge FX rates into the main DataFrame
    df = df.merge(
        exchange_rate_df,
        how="left",
        on=["Date", "FX Ticker"],
        suffixes=("", "_merge"),
    )

    # GBP and GBp Handling
    gbp_mask = df["Original Currency"] == "GBP"
    gbp_pence_mask = df["Original Currency"] == "GBp"
    df.loc[gbp_mask, "FX Rate"] = 1.0
    df.loc[gbp_pence_mask, "FX Rate"] = 0.01
    # df.loc[gbp_pence_mask, "Price"] = df["Old Price"] * 0.01

    # Identify rows with missing FX Rates
    missing_fx_mask = df["FX Rate"].isna() & non_gbp_mask
    if missing_fx_mask.any():
        missing_count = missing_fx_mask.sum()
        logging.error(
            f"{missing_count} rows have missing FX rates and will be removed."
        )
        print(f"\nRows with Missing FX Rates:\n{df.loc[missing_fx_mask]}\n")
        # Remove rows with missing FX rates for non-GBP currencies
        df = df[~missing_fx_mask].copy()

    # Calculate final prices for non-GBP currencies
    df["Price"] = df["Old Price"] * df["FX Rate"]

    # Log the number of successfully converted prices
    converted_count = df[~df["FX Ticker"].str.contains("-")].shape[0]
    logging.info(f"{converted_count} prices successfully converted to GBP.")

    # Return the required columns
    return df[["Ticker", "Price", "Date"]]


def process_converted_prices(converted_df: pd.DataFrame) -> pd.DataFrame:
    """Prepares the final DataFrame for CSV export.

    Args:
        converted_df: DataFrame with "Ticker", "Date", "Price"

    Returns:
        Processed DataFrame ready to be saved as CSV, or an empty DataFrame if input is empty.
        Exits if a critical error occurs.
    """

    if converted_df.empty:
        logging.warning(
            "Input DataFrame is empty. Returning empty DataFrame."
        )
        return pd.DataFrame(
            columns=["Ticker", "Price", "Date"]
        )  # return empty dataframe with correct columns

    try:
        output_csv = converted_df[["Ticker", "Price", "Date"]].copy()

        output_csv["Date"] = output_csv["Date"].apply(panda_date)

        # Sort by descending date IN PLACE
        output_csv.sort_values(by="Date", ascending=False, inplace=True)

        # Format date as string using .dt.strftime()
        output_csv["Date"] = output_csv["Date"].dt.strftime("%d/%m/%Y")

        return output_csv

    except Exception as e:
        logging.exception(
            f"A critical error occurred: {e}"
        )
        sys.exit(1)  # Exit program


@retry(Exception, config=config)
def save_to_csv(df: pd.DataFrame, config: Dict[str, Any]) -> bool:
    """
    Saves the DataFrame to a CSV file without headers.

    Args:
        df: DataFrame to save.
        config: Configuration dictionary.

    Returns:
        True if saved successfully, False otherwise.
    """

    try:
        if df.empty:
            logging.error("No data to save. Check earlier steps for errors.")
            return False
        required_columns = ["Ticker", "Price", "Date"]
        if not all(col in df.columns for col in required_columns):
            logging.error(
                f"Missing required columns {required_columns} in data. Cannot save CSV. Exiting"
            )
            sys.exit(1)

        # Build the output file path
        file_path = os.path.join(config["paths"]["base"], config["paths"]["data_file"])

        # Save the DataFrame to CSV without headers
        df.to_csv(file_path, index=False, header=False)

        # Log the save location
        short_CSV_path = format_path(file_path)
        logging.info(f"{short_CSV_path} saved successfully.\n")

        return True

    except Exception as e:
        logging.exception(f"An error occurred while saving to {short_CSV_path}")
        return False


# Quicken GUI #################################

@retry(exceptions=(RuntimeError, IOError), config=config)
def import_data_file(config):

    try:
        output_file_name = config["paths"]["data_file"]
        base_path = config["paths"]["base"]
        filename = os.path.join(base_path, output_file_name)

        # Type the file name into the dialog
        pyautogui.typewrite(filename, interval=0.01)
        time.sleep(0.5)  # Small delay before hitting enter for stability
        pyautogui.press("enter")

        # Wait to get to the price import success dialogue
        start_time = time.time()
        while time.time() - start_time < 10:
            windows = gw.getWindowsWithTitle(
                "Quicken XG 2004 - Home - [Investing Centre]"
            )
            if windows:
                time.sleep(2)
                pyautogui.press("enter")
                logging.info(
                    f"Successfully imported {format_path(filename)} at {pd.Timestamp.now().strftime('%d/%m/%Y %H:%M')}"
                )
                time.sleep(0.5)
                return True
            time.sleep(0.5)

    except Exception as e:
        logging.error(f"Failed to import data file: {e} ({type(e).__name__})")
        raise


@retry(exceptions=(Exception), config=config)
def open_import_dialog():

    try:
        with pyautogui.hold("alt"):
            pyautogui.press(["f", "i", "i"])

        # Wait to get to the price import dialogue box
        start_time = time.time()
        while time.time() - start_time < 10:
            windows = gw.getWindowsWithTitle("Import Price Data")
            if windows:
                time.sleep(0.5)
                return True
            time.sleep(0.5)

        return True
    except Exception as e:
        logging.error(f"Failed to open price import dialogue box: {e}")
        return False


@retry(exceptions=(Exception,), config=config)
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
                time.sleep(0.5)
                return True
            time.sleep(0.5)

        logging.error("Could not get to portfolio page within the expected time.")
        return False

    except Exception as e:
        logging.error(f"Failed to navigate to portfolio: {e}")
        return False


@retry(exceptions=(Exception,), config=config)
def open_quicken(config):

    quicken_path = config["paths"]["quicken"]
    try:
        # Check if Quicken is already open
        windows = gw.getWindowsWithTitle("Quicken XG 2004")
        if len(windows) > 0:
            quicken_window = windows[0]
            quicken_window.activate()  # Bring the existing Quicken window to the foreground
            logging.info("Quicken is already open.")
            return True

        # If not open, launch Quicken
        logging.info("Launching Quicken...")
        process = subprocess.Popen([quicken_path], shell=True)
        if process.poll() is not None:
            raise RuntimeError(
                f"Failed to launch Quicken. Process terminated immediately."
            )

        # Wait for the Quicken window to appear
        start_time = time.time()
        while time.time() - start_time < 30:
            windows = gw.getWindowsWithTitle("Quicken XG 2004")
            if windows:
                logging.info("Quicken window found.")
                return True
            time.sleep(1)

        logging.error("Quicken window did not appear within the expected time.")
        return False

    except FileNotFoundError:
        logging.error(
            f"Quicken executable not found at {quicken_path}. Please check the path."
        )
        return False
    except Exception as e:
        logging.error(f"Failed to open Quicken: {e}")
        return False


def execute_import_sequence(config):
    """
    Execute the sequence of steps for importing data.

    Returns:
        bool: True if all steps completed successfully
    """

    steps = [
        (open_quicken, "Opening Quicken"),
        (navigate_to_portfolio, "Navigating to Portfolio view"),
        (open_import_dialog, "Opening import dialog"),
        (import_data_file, "Importing data file"),
    ]

    for step_function, message in steps:
        logging.info(f"Starting step: {message}")
        try:
            # Call the step function with necessary arguments if applicable
            if step_function == import_data_file:
                result = step_function(config)
            elif step_function == open_quicken:
                result = step_function(config)
            else:
                result = step_function()

            if not result:
                logging.error(f"Step '{message}' failed.")
                return False

            logging.info(f"Step '{message}' completed successfully.")
        except Exception as e:
            logging.error(f"Error in step '{message}': {e}")
            return False

    logging.info("All steps completed successfully.")
    return True


def setup_pyautogui():
    """
    Configures pyautogui for automation.
    """

    pyautogui.FAILSAFE = True  # Abort ongoing automation by moving mouse to the top-left corner of screen.
    pyautogui.PAUSE = 0.3  # Delay between actions


def quicken_import(config):
    """
    Automates the process of importing data into Quicken.

    Args:
        config: Configuration dictionary.

    Returns:
        bool: True if the import process completes successfully.
    """

    try:
        if is_elevated():
            return execute_import_sequence(config)
        else:
            file_path = os.path.join(
                config["paths"]["base"], config["paths"]["data_file"]
            )
            short_CSV_path = format_path(file_path)
            logging.info(
                f"Can't automate file import. Open Quicken and upload {short_CSV_path} manually."
            )
            return False
    except Exception as e:
        logging.error(f"Error during Quicken import: {e}")
        return False


# Main ########################################

def main():
    """
    Main script execution.
    """
    try:
        # Initialise
        os.system("cls" if os.name == "nt" else "clear")
        print("Starting script.\n")

        # Call setup_logging early in the script
        setup_logging(config=config)

        # Get elevation
        run_as_admin()

        # Get 'raw' tickers from YAML file
        tickers = config.get("tickers", [])
        if not tickers:
            logging.error(
                "No tickers found. Please update your configuration. Exiting."
            )
            sys.exit(1)

        # Validate and acquire metadata.
        valid_tickers = get_tickers(tickers)
        if not valid_tickers:
            logging.error("No valid tickers to process. Exiting.")
            sys.exit(1)

        # Fetch historical data
        price_data = fetch_historical_data(valid_tickers, config=config)

        if price_data is None or price_data.empty: #check for None as well
            logging.error("No valid data fetched. Exiting.")
            sys.exit(1)

        # Convert prices to GBP
        processed_data = convert_prices(price_data)

        # Process and create output DataFrame
        output_csv = process_converted_prices(processed_data)

        # Save data to CSV
        save_to_csv(output_csv, config=config)

        # Clean up cache
        clean_cache(config=config)

        # Quicken import sequence
        setup_pyautogui()
        quicken_import(config=config)

        # Pause to allow reading of terminal
        logging.info("Script completed successfully.")
        if is_elevated():
            input("\n\t Press Enter to exit...\n\t")

    except KeyboardInterrupt:
        logging.warning("Script interrupted by user.")
    except Exception as e:
        logging.error(f"An unexpected error occurred: {e}")
        logging.error(traceback.format_exc())


if __name__ == "__main__":

    main()
