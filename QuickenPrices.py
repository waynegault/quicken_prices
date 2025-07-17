#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Shebang and coding declaration.

"""
QuickenPrices.py

This script fetches historical price data for a list of tickers specified in 'configuration.yaml' and saves the data to a CSV file.

It includes features such as configuration management, logging, error handling, data fetching, data processing, currency conversion, and more.

Wayne Gault
23/11/2024

Updated 17/7/25 with revised error handling and logging improvements and concurrent processing for faster ticker validation.

MIT Licence
"""

# Imports #####################################

# Standard Library Imports
import concurrent.futures
import time
import calendar
import os
import sys
import pickle
import subprocess
import ctypes
import winreg
import datetime
import inspect
import logging
import traceback
import warnings
from functools import wraps
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Type, Union
import requests

# Third-Party Data and Scientific Computing Imports
import numpy as np
from numpy import busday_count
import pandas as pd
from pandas import Timestamp, Timedelta, isna
from pandas._libs.tslibs.nattype import NaTType
import yaml
import yfinance as yf

# GUI and Automation Imports
import pyautogui
import pygetwindow as gw


# Constants ###################################

SECONDS_IN_A_DAY = 86400
CACHE_COLUMNS = ("Ticker", "Old Price", "Date")
PUBLIC_HOLIDAYS = {
    "01-01",  # New Year's Day
    "12-25",  # Christmas Day
    "12-26",  # Boxing Day
}
DEFAULT_TRANSIENT_EXCEPTIONS = (
    requests.exceptions.ConnectionError,
    requests.exceptions.Timeout,
    requests.exceptions.HTTPError,  # Sometimes a retry helps if it's 5xx
)
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


def quicken_not_found_error():
    """
    Handle the case where the Quicken executable cannot be located.
    """
    logging.error(
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


# Note: yfinance now handles its own session internally, no custom session needed

# Coding utilities ############################


def retry(
    exceptions: Union[Type[BaseException], Tuple[Type[BaseException], ...]] = Exception,
    retry_config: Optional[Dict[str, Dict[str, float]]] = None,
):
    """
    A decorator that retries a function call in case of specified exceptions.
    By default, uses global config to get max_retries, retry_delay, backoff.

    Args:
        exceptions:
            Exception class or tuple of exception classes to catch.
            Defaults to `Exception` if not specified.
        retry_config:
            A config dict with structure similar to config. If None,
            uses the globally defined config variable.

    Raises:
        The last exception if the maximum number of retries is reached.
    """

    if retry_config is None:
        retry_config = config  # Use the loaded config instead of DEFAULT_CONFIG

    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            # Explicitly cast `max_retries` to int
            tries = int(retry_config["retries"]["max_retries"])
            delay = retry_config["retries"]["retry_delay"]
            backoff = retry_config["retries"]["backoff"]

            for attempt in range(1, tries + 1):
                try:
                    return func(*args, **kwargs)
                except exceptions as e:
                    if attempt == tries:
                        logging.error(
                            f"[{func.__name__}] Max retries reached ({tries}). "
                            f"Last error: {e}"
                        )
                        raise
                    logging.warning(
                        f"[{func.__name__}] Attempt {attempt}/{tries} failed "
                        f"with error: {e}. Retrying in {delay:.1f} seconds..."
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
            logging.info("Elevation request cancelled")
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
        str, datetime.date, datetime.datetime, pd.Timestamp, int, float, NaTType, None
    ],
) -> Union[pd.Timestamp, NaTType]:
    """Converts a date-like object to a UTC midnight pandas Timestamp.

    Handles strings, datetime objects, pandas Timestamps, and Unix epoch
    timestamps (seconds or milliseconds). Assumes UTC for naive datetimes.
    Returns pd.NaT for invalid input.

    Args:
        date_obj: The date object to convert.

    Returns:
        A UTC midnight pandas Timestamp, or pd.NaT if the input is invalid.
    """
    if pd.isna(date_obj):  # Explicitly handle None or NaN
        return pd.NaT

    try:
        if isinstance(date_obj, (int, float)):
            # Handle epoch timestamps
            date_obj = abs(date_obj)  # Handle negative epoch timestamps
            unit = "s" if date_obj < 10**10 else "ms"  # Determine units
            date_obj = pd.Timestamp(date_obj, unit=unit, tz="UTC").floor("D")
            logging.debug(f"Converted epoch to UTC: {date_obj}")

        elif isinstance(date_obj, (datetime.datetime, datetime.date, pd.Timestamp)):
            # Handle datetime-like objects
            date_obj = pd.to_datetime(date_obj)  # Ensure it's a pandas Timestamp
            if date_obj.tz is None:  # Localise naive datetimes
                date_obj = date_obj.tz_localize("UTC")
            else:  # Convert timezone-aware to UTC
                date_obj = date_obj.tz_convert("UTC")
            date_obj = date_obj.floor("D")  # Floor to midnight
            logging.debug(f"Normalized datetime: {date_obj}")

        elif isinstance(date_obj, str):  # Handle strings
            date_obj = pd.to_datetime(date_obj, utc=True, dayfirst=True).floor("D")
            logging.debug(f"Parsed string date: {date_obj}")

        else:
            logging.warning(f"Unhandled date type: {type(date_obj)}")
            return pd.NaT

        return date_obj

    except (ValueError, TypeError) as e:
        logging.warning(f"Invalid date format: {date_obj}. Error: {e}")
        return pd.NaT


def adjust_date(date: pd.Timestamp, direction: str) -> pd.Timestamp:
    """
    Adjusts a date to the nearest valid business day (avoiding weekends and public holidays).

    Args:
        date (pd.Timestamp): The initial date to adjust.
        direction (str): "start" to move forwards, "end" to move backwards.

    Returns:
        pd.Timestamp: The adjusted date.
    """
    if direction not in {"start", "end"}:
        raise ValueError("Direction must be 'start' or 'end'.")

    while True:
        # Adjust for weekends
        if date.weekday() >= 5:  # Weekend check (5=Saturday, 6=Sunday)
            if direction == "start":
                days_to_add = (calendar.MONDAY - date.weekday()) % 7
                date += pd.Timedelta(days=days_to_add)
            elif direction == "end":
                date -= pd.Timedelta(days=date.weekday() - calendar.FRIDAY)
        else:
            # Check for public holidays
            if date.strftime("%m-%d") not in PUBLIC_HOLIDAYS:
                break  # Valid date found
            # Move the date if it's a holiday
            if direction == "start":
                date += pd.Timedelta(days=1)
            elif direction == "end":
                date -= pd.Timedelta(days=1)

    return date.floor("D")  # Normalize to remove time component


def get_date_range(config: dict) -> Tuple[pd.Timestamp, pd.Timestamp]:
    """
    Calculates start and end dates, avoiding weekends and public holidays.

    Args:
        config (dict): Configuration containing 'collection_period_years'.

    Returns:
        Tuple[pd.Timestamp, pd.Timestamp]: The adjusted start and end dates, localized to UTC.
    """
    if "collection_period_years" not in config:
        raise KeyError("Config is missing 'collection_period_years' key.")

    collection_period_years = config["collection_period_years"]
    if (
        not isinstance(collection_period_years, (int, float))
        or collection_period_years <= 0
    ):
        raise ValueError("'collection_period_years' must be a positive number.")

    # Calculate the number of days in the collection period
    collection_period_days = int(round(collection_period_years * 365.25))

    # Get today's date in UTC
    today_utc = pd.Timestamp.now(tz="UTC").normalize()

    # Calculate initial start and end dates
    end_date = today_utc
    start_date = end_date - pd.Timedelta(days=collection_period_days)

    # Adjust for weekends and public holidays
    end_date = adjust_date(end_date.tz_localize(None), direction="end").tz_localize(
        "UTC"
    )
    start_date = adjust_date(
        start_date.tz_localize(None), direction="start"
    ).tz_localize("UTC")

    # Log the collection period details
    business_days_range = pd.bdate_range(start_date, end_date, tz="UTC")
    non_weekend_days = len(business_days_range)

    logging.info(
        f"Collection period: {collection_period_years} years "
        f"({collection_period_days} days, {non_weekend_days} business days)"
    )

    return start_date, end_date


def find_missing_ranges(
    start_date: pd.Timestamp,
    end_date: pd.Timestamp,
    first_cache: Optional[pd.Timestamp],
    last_cache: Optional[pd.Timestamp],
) -> List[Tuple[pd.Timestamp, pd.Timestamp]]:
    """Finds missing date ranges, no longer adjusting for weekends."""

    if start_date > end_date:
        logging.warning("start_date is after end_date; no valid range.")
        return []

    # If no cache data, just return the entire [start_date, end_date].
    if pd.isna(first_cache) or pd.isna(last_cache):
        return [(start_date, end_date)]

    if first_cache > last_cache:
        first_cache, last_cache = last_cache, first_cache

    # If entire requested range is already in cache, no missing
    if start_date >= first_cache and end_date <= last_cache:
        return []

    missing_ranges = []

    # If there's a portion before the cache
    if start_date < first_cache:
        missing_end = min(end_date, first_cache - pd.Timedelta(days=1))
        if start_date <= missing_end:
            missing_ranges.append((start_date, missing_end))

    # If there's a portion after the cache
    if end_date > last_cache:
        missing_start = max(start_date, last_cache + pd.Timedelta(days=1))
        if missing_start <= end_date:
            missing_ranges.append((missing_start, end_date))

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
) -> List[Tuple[str, Optional[pd.Timestamp], str, str, float]]:
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
    valid_tickers: List[Tuple[str, Optional[pd.Timestamp], str, str, float]] = (
        validate_tickers(tickers, max_show)
    )

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
        f"Validated {len(all_valid_tickers)} {pluralise('ticker', len(all_valid_tickers))} in total\n"
    )
    return all_valid_tickers


def validate_tickers(
    tickers: List[str], max_show: int = 5
) -> List[Tuple[str, Optional[pd.Timestamp], str, str, float]]:
    valid_tickers = []
    # Use a ThreadPoolExecutor for concurrent validation
    with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
        # Map each ticker to the validate_ticker function
        future_to_ticker = {
            executor.submit(validate_ticker, ticker): ticker for ticker in tickers
        }
        for future in concurrent.futures.as_completed(future_to_ticker):
            ticker_symbol = future_to_ticker[future]
            try:
                data = future.result()
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
                            data["current_price"],
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


@retry(
    exceptions=(
        requests.exceptions.HTTPError,
        requests.exceptions.ConnectionError,
        requests.exceptions.Timeout,
    )
)
def validate_ticker(
    ticker_symbol: str,
) -> Optional[Dict[str, Union[str, pd.Timestamp, float]]]:
    """
    Validate a ticker symbol and fetch metadata using yfinance.

    Args:
        ticker_symbol (str): The ticker symbol to validate.

    Returns:
        Optional[Dict[str, Union[str, pd.Timestamp, float]]]: Metadata about the ticker, or None if invalid.
    """
    if not isinstance(ticker_symbol, str):
        logging.error("Ticker must be a string.")
        return None

    try:
        # Let yfinance handle its own session (no longer accepts custom sessions)
        ticker = yf.Ticker(ticker_symbol)

        # In yfinance 2.x, ticker.info is now a method, so we must call it:
        info = ticker.info

        if not info or info.get("symbol") is None:
            logging.error(
                f"Ticker '{ticker_symbol}' is invalid. Check if the symbol is correct or available on yfinance."
            )
            return None

        # Safely parse the first trade date (Unix epoch in seconds)
        first_trade_date = info.get("firstTradeDateEpochUtc", None)
        if first_trade_date is not None:
            try:
                earliest_date = pd.Timestamp(first_trade_date, unit="s").tz_localize(
                    "UTC"
                )
            except (ValueError, OverflowError) as e:
                logging.warning(
                    f"Failed to parse firstTradeDateEpochUtc={first_trade_date} for '{ticker_symbol}': {e}"
                )
                earliest_date = pd.NaT
        else:
            earliest_date = pd.NaT

        # Construct the data dictionary
        data = {
            "ticker": info.get("symbol", ticker_symbol).upper(),
            "earliest_date": earliest_date,
            "type": info.get("quoteType", "Unknown"),
            "currency": info.get("currency", "Unknown"),
            "current_price": info.get("bid", float("nan")),
        }

        return data

    except requests.exceptions.HTTPError as e:
        if "429" in str(e):
            logging.warning(
                f"Rate limited for {ticker_symbol}. This will be retried automatically..."
            )
            # Don't sleep here - let the retry decorator handle it
            raise  # Re-raise to trigger retry
        else:
            logging.error(f"HTTP error fetching {ticker_symbol}: {e}")
            return None
    except (ValueError, TypeError) as e:
        # Handle JSON parsing errors that can occur when rate limited
        if "Expecting value" in str(e):
            logging.warning(
                f"JSON parsing error for {ticker_symbol} (likely rate limited). Will retry..."
            )
            # Convert to HTTPError to trigger retry
            raise requests.exceptions.HTTPError(
                "429 Rate Limited - JSON parsing failed"
            )
        else:
            logging.error(f"Data parsing error for {ticker_symbol}: {e}")
            return None
    except Exception as e:
        logging.error(f"An unexpected error occurred fetching {ticker_symbol}: {e}")
        return None


# Get data ####################################


def fetch_historical_data(
    tickers: List[Tuple[str, Optional[pd.Timestamp], str, str, float]],
    start_date: pd.Timestamp,
    end_date: pd.Timestamp,
    cache_dir: str,
) -> pd.DataFrame:
    """
    Fetch historical data for a list of tickers using daily yfinance data
    and caching logic (handled in fetch_ticker_data).

    Args:
        tickers: A list of tuples containing:
            (ticker_symbol, earliest_date, ticker_type, currency, current_price).
        config: A dictionary with necessary configuration values
            (e.g., for cache paths, date ranges, etc.).

    Returns:
        A combined DataFrame with columns in CACHE_COLUMNS plus 'Type'
        and 'Original Currency'. Returns empty DataFrame if no data.
    """

    if not tickers:
        logging.info("No tickers provided. Returning empty DataFrame.")
        return pd.DataFrame(columns=list(CACHE_COLUMNS) + ["Type", "Original Currency"])

    logging.info(
        f"Seeking data from {start_date.strftime('%d/%m/%Y')} "
        f"to {end_date.strftime('%d/%m/%Y')}\n"
    )
    records = []

    # For each ticker, fetch data
    for ticker, earliest_date, ticker_type, currency, current_price in tickers:

        # Ensure earliest_date and end_date are directly comparable
        if earliest_date is not None and earliest_date.tzinfo:

            if earliest_date.tzinfo != end_date.tzinfo:
                earliest_date = earliest_date.tz_convert(
                    end_date.tzinfo
                )  # Convert to the same timezone as end_date

        # Skip tickers that began trading after our end_date
        if earliest_date and end_date < earliest_date:
            logging.warning(
                f"Skipping {ticker}: Requested period ends before ticker existed."
            )
            continue

        # Only fetch from max(start_date, earliest_date)
        data_start_date = max(start_date, earliest_date or start_date)

        try:
            df = fetch_ticker_data(
                ticker, data_start_date, end_date, current_price, cache_dir
            )
            if df is None or df.empty:
                logging.warning(f"No data found for ticker {ticker}. Skipping.")
                continue

            # Ensure metadata columns exist
            if "Type" not in df.columns:
                df["Type"] = ticker_type
            if "Original Currency" not in df.columns:
                df["Original Currency"] = currency

            records.append(df)

        except requests.exceptions.RequestException as e:
            logging.error(f"Error fetching data for {ticker}: {e}")
        except Exception as e:
            logging.error(f"Unexpected error fetching data for {ticker}: {e}")

    # Combine records
    valid_records = [df for df in records if not df.empty and df is not None]
    if valid_records:
        combined_df = pd.concat(valid_records, ignore_index=True)
        logging.info(
            f"Finished retrieving {len(combined_df)} {pluralise('record', len(combined_df))} "
            f"for {len(valid_records)} {pluralise('ticker', len(valid_records))}\n"
        )
        return combined_df

    logging.warning("No valid data fetched for any tickers.")
    return pd.DataFrame(columns=list(CACHE_COLUMNS) + ["Type", "Original Currency"])


def fetch_ticker_data(
    ticker: str,
    start_date: pd.Timestamp,
    end_date: pd.Timestamp,
    current_price: float,
    cache_dir: str,
) -> pd.DataFrame:
    """
    Fetch historical data (daily) for a ticker using local cache + yfinance,
    only calling yfinance for missing date ranges.
    Forward-fills missing dates if this is a CURRENCY ticker.
    """

    # 1) Load cache
    os.makedirs(cache_dir, exist_ok=True)
    cache_data, cache_status = load_cache(ticker, cache_dir)
    needs_save = False  # Track whether the cache needs saving

    # 2) Identify missing date ranges in cache
    if cache_status == "not_found":
        logging.info(f"{ticker}: No cache file found. Will fetch entire range.")
        first_cache, last_cache = None, None
        cache_data = pd.DataFrame(columns=CACHE_COLUMNS)
    elif cache_status == "empty":
        logging.info(f"{ticker}: Cache file is empty. Will fetch entire range.")
        first_cache, last_cache = None, None
        cache_data = pd.DataFrame(columns=CACHE_COLUMNS)
    elif cache_status == "loaded":
        # Ensure 'Date' column is datetime and validate cache data
        if "Date" in cache_data.columns:
            cache_data["Date"] = pd.to_datetime(
                cache_data["Date"], errors="coerce", utc=True
            )
            first_cache = cache_data["Date"].min()
            last_cache = cache_data["Date"].max()
            logging.info(
                f"{ticker}: Cache loaded from {first_cache.strftime('%d/%m/%Y')} to {last_cache.strftime('%d/%m/%Y')} "
                f"({len(cache_data)} {pluralise('day', len(cache_data))})"
            )
        else:
            logging.error(
                f"{ticker}: Cache is missing 'Date' column. Will fetch entire range."
            )
            first_cache, last_cache = None, None
            cache_data = pd.DataFrame(columns=CACHE_COLUMNS)

    # 3) Determine missing date ranges
    missing_ranges = find_missing_ranges(start_date, end_date, first_cache, last_cache)

    # 4) Download missing data via yfinance
    downloaded_data = pd.DataFrame()
    if missing_ranges:
        for rng_start, rng_end in missing_ranges:
            partial_df = download_data(ticker, rng_start, rng_end)
            if not (partial_df is None or partial_df.empty):
                downloaded_data = pd.concat(
                    [downloaded_data, partial_df], ignore_index=True
                )

    # 5) Merge newly downloaded data with cache

    if cache_data is None or cache_data.empty:
        combined_data = downloaded_data
        needs_save = True  # Flag that new data is to be saved
    elif downloaded_data is None or downloaded_data.empty:
        combined_data = cache_data
    else:
        combined_data = pd.concat([cache_data, downloaded_data], ignore_index=True)
        combined_data.drop_duplicates(subset=["Date"], keep="last", inplace=True)
        needs_save = True  # Flag that merged data is to be saved

    combined_data.sort_values("Date", inplace=True)

    # 6) Save cache only if changes occurred (and before forwards-fill)
    if needs_save:
        save_cache(ticker, combined_data, cache_dir)

    # 7) Forward-fill logic (for FX tickers)
    if "=X" in ticker.upper():
        min_date = combined_data["Date"].min().normalize()
        max_date = combined_data["Date"].max().normalize()
        # Create full range of dates for every calendar day even if there are gaps in the original data
        all_days = pd.date_range(min_date, max_date, freq="D")

        # Create a full date range for each ticker where all calendar days are present, and missing data is covered by forward-filling the last known value.
        # Get a list of all unique tickers in the dataset
        all_tickers = combined_data["Ticker"].unique()
        # create a MultiIndex that combines all dates and all tickers
        all_ticker_dates = pd.MultiIndex.from_product(
            [all_days, all_tickers], names=["Date", "Ticker"]
        )
        # Create a df where all calendar days are present for every ticker, and missing data is covered by forward-filling the last known value.
        forward_filled_data = (
            # Temporarily use Date and Ticker as a MultiIndex
            combined_data.set_index(["Date", "Ticker"])
            # Reindex to include every day in all_days, (adding rows missing in combined_data) filling any missing rows with the last available data (forward-filling).
            .reindex(all_ticker_dates, method="ffill")
            # Convert the MultiIndex back into regular columns.
            .reset_index()
        )

        if not combined_data.equals(forward_filled_data):
            logging.info(f"{ticker}: Forward-filled FX data ({len(all_days)} days)")
            combined_data = forward_filled_data

    # 8) Append latest price if today's closing not available
    max_date = combined_data["Date"].max()
    today = pd.Timestamp.now(tz="UTC").normalize()

    if (
        max_date.normalize() < today
        and isinstance(current_price, float)
        and current_price > 0
    ):
        #  Create new_row as a DataFrame with explicit dtypes
        new_row = pd.DataFrame(
            [
                {
                    "Ticker": ticker,
                    "Old Price": current_price,
                    "Date": pd.to_datetime(
                        today, utc=True
                    ),  # Ensure Date is datetime64[ns, UTC]
                }
            ]
        )
        # Ensure combined_data['Date'] remains datetime64[ns, UTC]
        combined_data["Date"] = pd.to_datetime(combined_data["Date"], utc=True)

        # Concatenate new_row into combined_data
        combined_data = pd.concat([combined_data, new_row], ignore_index=True)
        logging.info(
            f"{ticker}: Appended today's {today.strftime('%d/%m/%Y')} latest price = {current_price:.3f}"
        )

    # 9) Final log
    logging.info(
        f"{ticker}: Got {len(combined_data)} {pluralise('day', len(combined_data))} data "
        f"from {combined_data['Date'].min().strftime('%d/%m/%Y')} to {combined_data['Date'].max().strftime('%d/%m/%Y')}\n"
    )
    return combined_data


@retry()
def load_cache(ticker: str, cache_dir: str) -> Tuple[pd.DataFrame, str]:
    """
    Loads cached data for a specific ticker using pickle.

    Args:
        ticker (str): The ticker symbol.
        cache_dir (str): Path to the cache directory.

    Returns:
        Tuple[pd.DataFrame, str]: A tuple containing:
            - A DataFrame with cached data, or an empty DataFrame if the file is not found or empty.
            - A string indicating the cache status: "not_found", "empty", or "loaded".
    """
    cache_file = os.path.join(cache_dir, f"{ticker}.pkl")

    if os.path.exists(cache_file):
        try:
            # Check if the file is empty
            if os.path.getsize(cache_file) == 0:
                logging.warning(f"{ticker}: Cache file exists but is empty.")
                return pd.DataFrame(columns=CACHE_COLUMNS), "empty"

            # File exists and is not empty, attempt to load
            with open(cache_file, "rb") as f:
                data = pickle.load(f)

            # Ensure data is a DataFrame
            if isinstance(data, pd.DataFrame):
                return data, "loaded"
            else:
                logging.error(f"{ticker}: Cache file is invalid or corrupted.")
                return pd.DataFrame(columns=CACHE_COLUMNS), "empty"

        except (FileNotFoundError, IOError, pickle.UnpicklingError) as e:
            logging.error(f"Failed to load cache for {ticker}: {e}")
            return pd.DataFrame(columns=CACHE_COLUMNS), "empty"
    else:
        return pd.DataFrame(columns=CACHE_COLUMNS), "not_found"


@retry()
def download_data(
    ticker_symbol: str,
    rng_start: pd.Timestamp,
    rng_end: pd.Timestamp,
) -> Optional[pd.DataFrame]:
    """
    Downloads daily data for the specified date range [rng_start, rng_end] from yfinance.
    Returns a DataFrame with columns including "Date".
    """
    try:
        # dont collect data if start is today
        today = pd.Timestamp.now(tz="UTC").normalize()
        if today == rng_start:
            return pd.DataFrame()

        # Make range end inclusive
        rng_end = rng_end + pd.Timedelta(days=1)

        # Let yfinance handle its own session (no longer accepts custom sessions)
        ticker = yf.Ticker(ticker_symbol)
        raw_df = ticker.history(
            start=rng_start,
            end=rng_end,
            interval="1d",
        )

        if raw_df is None or raw_df.empty:
            logging.warning(
                f"{ticker_symbol}: No data from {rng_start.strftime('%d/%m/%Y')} to {rng_end.strftime('%d/%m/%Y')}."
            )
            return pd.DataFrame()

        raw_df.reset_index(inplace=True)
        raw_df.rename(columns={"Close": "Old Price"}, inplace=True)
        raw_df["Ticker"] = ticker_symbol

        # Make sure "Date" column is in correct format
        raw_df["Date"] = raw_df["Date"].dt.tz_convert("UTC")

        # Select columns
        raw_df = raw_df[list(CACHE_COLUMNS)]

        first_data = raw_df["Date"].min().strftime("%d/%m/%Y")
        last_data = raw_df["Date"].max().strftime("%d/%m/%Y")

        logging.info(
            f"{ticker_symbol}: Downloaded from {first_data} to {last_data} ({len(raw_df)} days.)"
        )

        return raw_df

    except Exception as e:
        logging.error(
            f"Failed to download data for {ticker_symbol} from {rng_start} to {rng_end}: {e}"
        )
        return pd.DataFrame()


@retry()
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

        first_data = data["Date"].min().strftime("%d/%m/%Y")
        last_data = data["Date"].max().strftime("%d/%m/%Y")

        logging.info(
            f"{ticker}: Cache saved from {first_data} to {last_data} "
            f"({len(data)} days)."
        )
    except Exception as e:
        logging.error(f"Failed to save cache for {ticker}: {e}")


@retry()
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
        logging.warning(f"Cache cleaning: can't find {cache_dir}")
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
                            f"Cache cleaning: {entry.name} deleted. Older than {config['cache']['max_age_days']} {pluralise('day', config['cache']['max_age_days'])}"
                        )
                    except Exception as remove_error:
                        logging.error(
                            f"Cache cleaning: Failed to delete file {entry.path}: {remove_error}"
                        )

        logging.info(
            f"Cache cleaning: {deleted_files} cache {pluralise('file', deleted_files)} deleted\n"
        )
    except Exception as e:
        logging.error(
            f"Cache cleaning: Error with {cache_dir}: {e}. Check file permissions or existence of the directory"
        )
        return False

    return True


# Process Data ################################


def convert_prices(df: pd.DataFrame) -> pd.DataFrame:
    """
    Converts prices to GBP based on ticker type and currency (daily data),
    while also logging mismatched days between equity and FX.

    Logic Overview:
      1) Separate equity rows (Type != "CURRENCY") from FX rows (Type == "CURRENCY").
      2) Log how many dates appear in equity vs. FX vs. both (mismatched days).
      3) For currency rows, group by (Date, Ticker) to build an 'FX Rate'.
      4) Merge 'FX Rate' into equity rows on (Date, FX Ticker).
      5) Set FX Rate = 1.0 or 0.01 for GBP or GBp rows respectively.
      6) Raise an error if non-GBP rows have missing FX rates.
      7) Compute final Price in GBP and return the result.

    Args:
        df: DataFrame containing historical data. Must include columns:
            'Date', 'Type', 'Ticker', 'Old Price', 'Original Currency', 'Price'.

    Returns:
        DataFrame with a new 'Price' column converted to GBP where applicable.
        Raises ValueError if missing FX rates are found for non-GBP instruments.
        Returns empty DataFrame if 'df' is empty or errors occur.
    """
    # Quick check
    if df.empty:
        logging.error("Empty DataFrame received.")
        return pd.DataFrame()

    # 1) Separate equity rows vs. FX rows
    equity_df = df[df["Type"] != "CURRENCY"].copy()
    fx_df = df[df["Type"] == "CURRENCY"].copy()
    equity_df["Date"] = equity_df["Date"].dt.normalize()
    fx_df["Date"] = fx_df["Date"].dt.normalize()

    # 2) Log day coverage mismatches
    equity_days = set(equity_df["Date"])
    fx_days = set(fx_df["Date"])
    common_days = equity_days.intersection(fx_days)
    unmatched_equity_days = equity_days - fx_days
    unmatched_fx_days = fx_days - equity_days
    logging.info(
        f"Total equity {pluralise('day',len(equity_days))}: {len(equity_days)}"
    )
    logging.info(f"Total FX {pluralise('day',len(fx_days))}: {len(fx_days)}")
    logging.info(f"Common {pluralise('day',len(common_days))}: {len(common_days)}")
    logging.info(
        f"Equity-only {pluralise('day',len(unmatched_equity_days))}: {len(unmatched_equity_days)}"
    )
    logging.info(
        f"FX-only {pluralise('day',len(unmatched_fx_days))}: {len(unmatched_fx_days)}\n"
    )

    # 3) Build a minimal FX DataFrame -> 'FX Rate'
    exchange_rate_df = (
        fx_df.groupby(["Date", "Ticker"])["Old Price"]
        .first()  # one price per date+ticker
        .rename("FX Rate")
        .reset_index()
        .rename(columns={"Ticker": "FX Ticker"})
    )

    # 4) Work on a copy of the equity data to avoid mutating the original
    equity_df = equity_df.copy()

    # We'll rename 'Price' -> 'Old Price' so we can store a new 'Price'
    equity_df = equity_df.rename(columns={"Price": "Old Price"})

    # Prepare an 'FX Ticker' column for non-GBP
    equity_df["FX Ticker"] = "-"
    non_gbp_mask = ~equity_df["Original Currency"].isin(["GBP", "GBp"])
    equity_df.loc[non_gbp_mask, "FX Ticker"] = (
        equity_df.loc[non_gbp_mask, "Original Currency"] + "GBP=X"
    )

    # Special case for USD => "GBP=X" (adjust if your logic differs)
    equity_df.loc[equity_df["Original Currency"] == "USD", "FX Ticker"] = "GBP=X"

    # Ensure data types match for merging
    equity_df["FX Ticker"] = equity_df["FX Ticker"].astype(str)
    exchange_rate_df["FX Ticker"] = exchange_rate_df["FX Ticker"].astype(str)

    # 5) Merge the 'FX Rate' onto the equity rows
    equity_df = pd.merge(
        equity_df, exchange_rate_df, on=["Date", "FX Ticker"], how="left"
    )

    # 6) Hard-code FX Rate for GBP & GBp
    gbp_mask = equity_df["Original Currency"] == "GBP"
    gbp_pence_mask = equity_df["Original Currency"] == "GBp"
    equity_df.loc[gbp_mask, "FX Rate"] = 1.0
    equity_df.loc[gbp_pence_mask, "FX Rate"] = 0.01

    # 7) Error if missing FX Rate for non-GBP
    missing_fx_mask = equity_df["FX Rate"].isna() & non_gbp_mask
    if missing_fx_mask.any():
        missing_count = missing_fx_mask.sum()
        raise ValueError(
            f"{missing_count} rows have missing FX rates. This is an error."
        )

    # 8) Calculate final Price in GBP
    equity_df["Price"] = equity_df["Old Price"] * equity_df["FX Rate"]

    # Log how many conversions happened (where 'FX Ticker' != '-')
    converted_count = equity_df[~equity_df["FX Ticker"].str.contains("-")].shape[0]
    logging.info(f"{converted_count} prices successfully converted to GBP\n")

    # 9) Validate floating-point multiplication
    equity_df["Conversion Check"] = equity_df.apply(
        lambda row: (
            "Correct"
            if abs(row["FX Rate"] * row["Old Price"] - row["Price"]) < 1e-6
            else "Error"
        ),
        axis=1,
    )
    incorrect_rows = equity_df[equity_df["Conversion Check"] == "Error"]
    if not incorrect_rows.empty:
        logging.warning(f"The following rows have conversion errors:\n{incorrect_rows}")

    # 10) merge equity and fx df's with "Ticker", "Price", "Date" cols
    equity_df = equity_df[["Ticker", "Price", "Date"]]
    fx_df = fx_df.rename(columns={"Old Price": "Price"})
    fx_df = fx_df[["Ticker", "Price", "Date"]]
    combined_df = pd.concat([equity_df, fx_df])
    # Sort by Date then Ticker and reset index
    combined_df = combined_df.sort_values(by=["Date", "Ticker"]).reset_index(drop=True)

    return combined_df


def process_converted_prices(
    converted_df: pd.DataFrame, start_date: pd.Timestamp
) -> pd.DataFrame:
    """
    Prepares the final DataFrame for CSV export.

    Args:
        converted_df: DataFrame with "Ticker", "Date", "Price"

    Returns:
        Processed DataFrame ready to be saved as CSV, or an empty DataFrame if input is empty.
        Exits if a critical error occurs.
    """

    if converted_df.empty:
        logging.warning("Input DataFrame is empty. Returning empty DataFrame.")
        return pd.DataFrame(columns=["Ticker", "Price", "Date"])

    try:
        output_csv = converted_df[["Ticker", "Price", "Date"]].copy()

        # Ensure 'Date' is a proper datetime type
        output_csv["Date"] = pd.to_datetime(output_csv["Date"], errors="coerce")

        # Sort by descending date (in place)
        output_csv.sort_values(by="Date", ascending=False, inplace=True)

        # Filter rows to include only those with 'Date' >= start_date
        output_csv = output_csv[output_csv["Date"] >= start_date]

        # Format date as string using .dt.strftime() -> DD/MM/YYYY
        output_csv["Date"] = output_csv["Date"].dt.strftime("%d/%m/%Y")

        return output_csv

    except Exception as e:
        logging.exception(f"A critical error occurred: {e}")
        sys.exit(1)  # Exit program


@retry()
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
        logging.info(f"Data saved successfully to {short_CSV_path}\n")

        return True

    except Exception as e:
        logging.exception(f"An error occurred while saving to {short_CSV_path}")
        return False


# Quicken GUI #################################


@retry()
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
        while time.time() - start_time < 120:
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


@retry()
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


@retry()
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


@retry()
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
                f"Can't automate file import. Open Quicken and upload {short_CSV_path} manually"
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

        # 0) Initialise
        os.system("cls" if os.name == "nt" else "clear")
        print("Starting script\n")
        print("sys.executable", sys.executable)
        print("yfinance version:", yf.__version__, "\n")

        # 0) Configure logging from YAML/ini settings
        setup_logging(config=config)

        # 2) Attempt to elevate privileges (if needed on Windows)
        run_as_admin()

        # 3) Read ticker config from your YAML
        tickers = config.get("tickers", [])
        if not tickers:
            logging.error(
                "No tickers found. Please update your configuration. Exiting."
            )
            sys.exit(1)

        # 4) Validate and acquire ticker metadata
        valid_tickers: List[Tuple[str, Optional[pd.Timestamp], str, str, float]] = []
        valid_tickers = get_tickers(tickers)
        if not valid_tickers:
            logging.error("No valid tickers to process. Exiting.")
            sys.exit(1)

        # 5) Fetch historical data

        # Retrieve date range and cache_dir from config
        start_date, end_date = get_date_range(config)
        cache_dir = os.path.join(config["paths"]["base"], config["paths"]["cache"])
        price_data = fetch_historical_data(
            valid_tickers, start_date, end_date, cache_dir
        )

        if price_data is None or price_data.empty:
            logging.error("No valid data fetched. Exiting.")
            sys.exit(1)

        # 6) Convert prices to GBP
        processed_data = convert_prices(price_data)

        # 7) Build the final output (e.g., pivoting, filtering, etc.)
        output_csv = process_converted_prices(processed_data, start_date)

        # 8) Save to CSV or your preferred data store
        save_to_csv(output_csv, config=config)

        # 9) Clean up the cache if desired
        clean_cache(config=config)

        # 10) Optionally automate an import sequence
        setup_pyautogui()
        quicken_import(config=config)

        # End
        logging.info("Script completed successfully")
        if is_elevated():
            input("\n\t Press Enter to exit...\n\t")

    except KeyboardInterrupt:
        logging.warning("Script interrupted by user.")
    except Exception as e:
        logging.error(f"An unexpected error occurred: {e}")
        logging.error(traceback.format_exc())


if __name__ == "__main__":

    main()
