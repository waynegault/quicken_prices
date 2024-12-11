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

# Imports
import logging
import os
import sys
import winreg  # Added for registry access

# Set up a simple logging configuration

logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s - %(levelname)s - %(message)s",
    stream=sys.stdout,
)
import traceback

# import contextlib  # Used to redirect output streams temporarily
import ctypes  # Low-level OS interfaces

# import functools  # Higher-order functions and decorators
from logging.handlers import (
    RotatingFileHandler,
)  # Additional logging handlers (e.g., RotatingFileHandler)
import pickle  # Object serialization/deserialization
import subprocess  # Process creation and management
import time  # Time-related functions
from functools import wraps
from datetime import date, datetime, timedelta  # Date and time handling
from pathlib import Path  # Object-oriented filesystem paths
from typing import Any, Callable, Dict, List, Optional, Tuple, Type, Union
import numpy as np
from numpy import busday_count
import yaml
import pandas as pd
import yfinance as yf
import pygetwindow as gw
import pyautogui
from pythonjsonlogger import jsonlogger


# Constants
SECONDS_IN_A_DAY = 86400


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
                return quicken_path
            else:
                logging.warning(
                    f"Quicken executable path from App Paths registry is invalid: {quicken_path}"
                )
    except FileNotFoundError:
        logging.warning("App Paths registry key for Quicken not found.")
    except OSError as e:
        logging.error(f"OS error accessing App Paths registry key: {e}")
    except winreg.error as e:
        logging.error(f"Registry access error: {e}")

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
    except OSError as e:
        logging.error(f"OS error accessing Uninstall registry key: {e}")
    except winreg.error as e:
        logging.error(f"Registry access error: {e}")

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
    "home_currency": "GBP",
    "collection_period_years": 0.08,
    "retries": {"max_retries": 3, "retry_delay": 2},
    "logging_level": {"file": "DEBUG", "terminal": "DEBUG"},
    "log_file": {"max_bytes": 5242880, "backup_count": 5},
    "tickers": ["^FTSE"],
    "paths": {"data_file": "data.csv", "log_file": "prices.log", "cache": "cache"},
    "cache": {"max_age_days": 30},
}


def load_configuration(config_file: str = "configuration.yaml") -> Dict[str, Any]:
    """
    Load and validate YAML configuration with defaults.
    Exits if critical values are missing or invalid.
    """
    base_path = Path(__file__).resolve().parent
    config_path = base_path / config_file

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

    # Attempt to load configuration file
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

    # Automatically find Quicken path
    validated_quicken_path = find_quicken_via_registry()

    if validated_quicken_path:
        final_config["paths"]["quicken"] = validated_quicken_path
        logging.info(f"Using Quicken found at: {validated_quicken_path}")
    else:
        logging.error("Quicken executable not found.")
        print(
            "\n❌ Critical Error: Quicken executable could not be located.\n"
            "✅ Suggestions to resolve this issue:\n"
            "  1. Ensure Quicken is installed on your system.\n"
            "  2. Verify that Quicken is installed in one of the standard paths, such as:\n"
            "     - C:\\Program Files (x86)\\Quicken\\qw.exe\n"
            "     - C:\\Program Files\\Quicken\\qw.exe\n"
            "  3. If using an older version of Quicken, confirm that it is properly installed.\n"
            "\n⚠️ If the problem persists, you may manually specify the Quicken path in configuration.yaml.\n"
        )
        sys.exit(1)

    # Basic validation
    if not final_config["tickers"]:
        logging.error(
            "No tickers defined in configuration. Please add at least one ticker."
        )
        sys.exit(1)

    return final_config


def setup_logging(config: Dict[str, any]) -> None:

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

    logging.info(f"Log file at: {log_file_path}")


def retry(
    exceptions: Tuple[Type[BaseException], ...],
    tries: int = 3,
    delay: float = 1.0,
    backoff: float = 2.0,
):
    """
    Retry decorator to automatically re-run a function if it raises specified exceptions.
    """

    def decorator(func: Callable):
        @wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            current_delay = delay
            for attempt in range(1, tries + 1):
                try:
                    return func(*args, **kwargs)
                except exceptions as e:
                    if attempt < tries:
                        logging.warning(
                            f"Attempt {attempt} failed with error '{e}'. Retrying in {current_delay} seconds..."
                        )
                        time.sleep(current_delay)
                        current_delay *= backoff
                    else:
                        logging.error(
                            f"Function '{func.__name__}' failed after {tries} attempts. Error: {e}. "
                            f"Consider checking network connectivity or ticker symbol correctness."
                        )
                        raise

        return wrapper

    return decorator


from datetime import date, timedelta
import pandas as pd


def get_date_range(config: dict) -> tuple[pd.Timestamp.date, pd.Timestamp.date, int]:
    """
    Calculates the start and end dates for data collection, and the number of business days between them,
    considering weekends and the specified period.

    Args:
        config (dict): Configuration dictionary containing "collection_period_years" key.

    Returns:
        tuple[pd.Timestamp.date, pd.Timestamp.date, int]: Tuple containing start date, end date, and number of business days.
    """

    period_year = config["collection_period_years"]
    period_days = period_year * 365

    # Determine the end date, adjusting for weekends
    today = date.today()
    if today.weekday() >= 5:  # Friday or Saturday
        end_date = today - timedelta(days=today.weekday() - 4)  # Previous Friday
    else:
        end_date = today

    # Determine the start date, adjusting for weekends
    start_date = end_date - timedelta(days=period_days)
    if start_date.weekday() >= 5:  # Friday or Saturday
        start_date = start_date + timedelta(
            days=7 - start_date.weekday()
        )  # Following Monday

    # Convert both start and end dates to pandas Timestamps and then extract their date components
    start_datetime = pd.Timestamp(start_date).date()
    end_datetime = pd.Timestamp(end_date).date()

    # Calculate business days
    business_days = busday_count(
        start_datetime.strftime("%Y-%m-%d"), end_datetime.strftime("%Y-%m-%d")
    )+1
    return start_datetime, end_datetime, business_days


def pluralise(title: str, quantity: int) -> str:
    return title if quantity == 1 else title + "s"


@retry((Exception,), tries=3)
def validate_ticker(
    ticker_symbol: str,
) -> Optional[Dict[str, Union[str, pd.Timestamp]]]:
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
        earliest_date = (
            pd.to_datetime(first_trade_date, unit="s", utc=True).date()
            if first_trade_date
            else None
        )

        # Construct the data dictionary
        data = {
            "ticker": info.get("symbol", ticker_symbol).upper(),
            "earliest_date": earliest_date,
            "type": info.get("quoteType", "Unknown"),
            "currency": info.get("currency", "Unknown"),
        }

        return data

    except Exception as e:
        logging.error(f"An unexpected error occurred fetching {ticker_symbol}: {e}")
        return None


def validate_tickers(
    tickers: List[str], max_show: int = 5
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
                    )
                )

        except Exception as e:
            logging.error(
                f"An unexpected error occurred processing ticker '{ticker_symbol}': {e}"
            )

    return valid_tickers


def get_tickers(
    tickers: List[str], max_show: int = 5
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

    logging.info(
        f"Validating {len(tickers)} user {pluralise("ticker",len(tickers))}..."
    )
    valid_tickers = validate_tickers(tickers)

    if not valid_tickers:
        logging.error(
            "No valid tickers found. Please update your configuration or check network access. Exiting."
        )
        sys.exit(1)
    else:
        logging.info(
            f"{len(valid_tickers)} valid {pluralise('ticker', len(valid_tickers))} found: {[ticker[0] for ticker in valid_tickers[:max_show]]}{'...' if len(valid_tickers)>max_show else ''}"
        )

    # Generate FX tickers based on the currencies in valid_tickers
    currencies = {t[3] for t in valid_tickers if t[3] not in ["GBP", "GBp"]}
    fx_tickers = {
        f"{currency}GBP=X" if currency != "USD" else "GBP=X" for currency in currencies
    }

    if fx_tickers:
        # Exclude tickers from fx_tickers that might already be in valid_tickers

        # Extract only the ticker strings from valid_tickers
        valid_ticker_strings = {ticker[0] for ticker in valid_tickers}
        # Exclude tickers from fx_tickers that are already in valid_ticker_strings
        fx_tickers = {
            ticker for ticker in fx_tickers if ticker not in valid_ticker_strings
        }

    if fx_tickers:
        # Validate FX tickers
        logging.info(
            f"Validating {len(fx_tickers)} FX {pluralise("ticker",len(fx_tickers))}..."
        )
        valid_fx_tickers = validate_tickers(list(fx_tickers))
        logging.info(
            f"{len(valid_fx_tickers)} valid FX {pluralise('ticker', len(valid_fx_tickers))} found: {[ticker[0] for ticker in valid_fx_tickers[:max_show]]}{'...' if len(valid_fx_tickers)>max_show else ''}"
        )
        all_valid_tickers = list(set(valid_tickers + valid_fx_tickers))
    else:
        logging.warning("No new FX tickers required.")
        all_valid_tickers = valid_tickers

    logging.info(
        f"Validated {len(all_valid_tickers)} {pluralise('ticker',len(all_valid_tickers))} in total."
    )

    return all_valid_tickers


# Start inclusive, end exclusive
def download_data(ticker_symbol, start, end):
    # assume start and end are date datetime objects
    ticker = yf.Ticker(ticker_symbol)
    time.sleep(0.1)  # Rate limiting

    df = ticker.history(
        start=start,
        end=end + timedelta(days=1),
        interval="1d",
    )
    if df.empty:
        # Create an empty DataFrame with the desired columns and data types
        empty_df = pd.DataFrame(columns=["Date", "Old Price"])
        empty_df["Date"] = pd.to_datetime(empty_df["Date"], utc=True)
        empty_df["Old Price"] = np.nan
        return empty_df
    df = df.rename(columns={"Close": "Old Price"})
    df.reset_index(inplace=True)  # Reset the index to make Date a column
    # Convert the 'Date' column to UTC timezone
    df["Date"] = pd.to_datetime(df["Date"], utc=True).dt.date
    df = df[["Date", "Old Price"]]  # Select only the "Old Price" column
    return df


def fetch_ticker_history(
    ticker: str, start_date: datetime, end_date: datetime, config: Dict
) -> pd.DataFrame:
    """
    Fetches historical data for a ticker, using cached data where available and downloading missing data for business days.
    """
    cache_dir = os.path.join(config["paths"]["base"], config["paths"]["cache"])
    cache_file = os.path.join(
        cache_dir,
        f"{ticker}.pkl",
    )

    # Create the date range with timezone
    required_business_days = sorted(
        set(
            date.date()
            for date in pd.date_range(
                start=start_date, end=end_date, freq="B", tz="UTC"
            )
        )
    )

    all_data = pd.DataFrame(
        columns=["Date", "Old Price"]
    )  # create empty df with correct column types
    all_data["Date"] = pd.to_datetime(all_data["Date"], utc=True).dt.date
    all_data["Old Price"] = np.nan

    count = 0
    dont_save_cache = 1  # Flag to prevent resaving cache

    logging.info(f"############# Seeking data for {ticker}")
    if os.path.exists(cache_file):
        logging.info(f"Loading {format_path(cache_file)}")
        cached_data = pd.read_pickle(cache_file)
        cached_dates = sorted(set(cached_data["Date"]))

        if not cached_data.empty:
            cached_min_date = cached_data["Date"].min()
            cached_max_date = cached_data["Date"].max()

            # Identify missing business days - returns the required_business_days that are not in cached_data["Date"]
            missing_days = sorted(
                set(day for day in required_business_days if day not in cached_dates)
            )
            if not missing_days:
                logging.info(
                    f"Using 100% cache {cached_min_date.strftime("%d/%m/%y")} - {cached_max_date.strftime("%d/%m/%y")}."
                )
                all_data = cached_data
            else:
                logging.info(
                    f"Got {len(cached_data)} {pluralise('day', len(cached_data))} cache between {cached_min_date.strftime("%d/%m/%y")} - {cached_max_date.strftime("%d/%m/%y")}. Try to download {len(missing_days)} {pluralise("day", len(missing_days))} more."
                )

                # Try to download missing dates in batches if contiguous or individually of not to minimise api calls
                data_frames = []
                s_date = None
                e_date = None

                for date in missing_days:
                    if s_date is None:
                        # Start a new range
                        s_date = date
                        e_date = date
                    else:
                        # Calculate the number of business days between e_date and the current date
                        gap = busday_count(
                            e_date.strftime("%Y-%m-%d"), date.strftime("%Y-%m-%d")
                        )

                        if gap == 1:
                            # Dates are consecutive (excluding weekends)
                            e_date = date
                        else:
                            # Fetch data for the current range
                            try:
                                fetched_data = download_data(ticker, s_date, e_date)
                                if not fetched_data.empty:
                                    count += 1
                                    data_frames.append(fetched_data)
                            except Exception as e:
                                print(
                                    f"Error fetching data for range  {s_date} to {e_date}: {e}"
                                )
                            # Start a new range
                            s_date = date
                            e_date = date

                # Handle the final range
                if s_date:
                    try:
                        fetched_data = download_data(ticker, s_date, e_date)
                        if not fetched_data.empty:
                            count += 1
                            data_frames.append(fetched_data)
                    except Exception as e:
                        print(
                            f"Error fetching data for range {s_date} to {e_date}: {e}"
                        )

                print(f"data_frames {data_frames}")

                # Concatenate all fetched data into a single DataFrame
                if data_frames:
                    try:
                        fetched_data = pd.concat(data_frames, ignore_index=True)
                    except Exception as e:
                        logging.error(f"Error concatenating data frames: {e}")
                        fetched_data = pd.DataFrame()  # Fallback to an empty DataFrame
                else:
                    logging.debug("No data frames to concatenate. Returning empty DataFrame.")
                    fetched_data = pd.DataFrame()

                if not fetched_data.empty:
                    all_data = pd.concat(
                        [cached_data, fetched_data], ignore_index=True
                    ).drop_duplicates()
                    all_data = all_data.sort_values(by=["Date"], ascending=True)
                    dont_save_cache = 0  # Flag to save cache
                else:
                    logging.info(f"No new data available to download.")
                    all_data = cached_data
                    dont_save_cache = 1  # Flag to prevent resaving cache
        else:
            logging.info(f"Cache found but empty. Downloading data.")
            all_data = download_data(ticker, start_date, end_date)
            count += 1
            dont_save_cache = 0
    else:
        logging.info(f"No cache found. Downloading data.")
        all_data = download_data(ticker, start_date, end_date)
        count += 1
        dont_save_cache = 0

    print(f"Count = {count}")
    print(f"fetched_data\n{all_data}")

    # Process the fetched data
    if not all_data.empty and dont_save_cache == 0:
        required_columns = {"Date", "Old Price"}
        missing_columns = required_columns - set(all_data.columns)
        if missing_columns:
            logging.warning(
                f"Missing required columns, {missing_columns}. Available columns: {list(all_data.columns)}. Skipping."
            )
            logging.info(
                "Missing columns in fetch_ticker_history #######################################\n"
            )
            return pd.DataFrame()

        # Save updated data to cache
        if not os.path.exists(cache_dir):
            os.makedirs(cache_dir)
        all_data.to_pickle(cache_file)
        logging.info(
            f"Saved {len(all_data)} business days {all_data["Date"].min().strftime("%d/%m/%y")} - {all_data["Date"].max().strftime("%d/%m/%y")} to {format_path(cache_file)}."
        )
    logging.info(
        f"# end fetch_ticker_history for {ticker} #####################################\n"
    )

    return all_data


def fetch_historical_data(
    tickers: List[Tuple[str, Optional[datetime], str, str]], config: Dict[str, Any]
) -> pd.DataFrame:
    """
    Fetch historical price data for all valid tickers.

    Args:
        tickers: List of valid tickers with metadata.

    Returns:
        Combined DataFrame containing historical data for all tickers.
    """
    # Get date range
    start_date, end_date, bdays = get_date_range(config)
    records = []
    logging.info(
        f"Seeking {bdays} business days of data, {start_date.strftime("%d/%m/%y")} - {end_date.strftime("%d/%m/%y")}.\n"
    )

    for ticker, earliest_date, type, currency in tickers:
        # Adjusted start date is the later of earliest_date or start_date
        adjusted_start_date = max(start_date, earliest_date or start_date)

        # Check if end_date is before earliest_date
        if earliest_date and end_date < earliest_date:
            logging.warning(
                f"Skipping {ticker}: Period requested, ending {end_date.strftime('%d/%m/%Y')}, "
                f"is before {ticker} existed (from {earliest_date.strftime('%d/%m/%Y')})."
            )
            continue

        try:
            # Fetch historical data (from cache or yfinance)
            df = fetch_ticker_history(ticker, adjusted_start_date, end_date, config)

            # Validate fetched data
            if df.empty:
                logging.warning(f"No data found for {ticker}. Skipping.")
                continue

            # Check for required columns
            required_columns = {"Date", "Old Price"}
            if not required_columns.issubset(df.columns):
                logging.warning(
                    f"Data for {ticker} is missing required columns {required_columns}. "
                    f"Available columns: {df.columns.tolist()}. Skipping."
                )
                continue

            # Add metadata columns
            df["Ticker"] = ticker
            df["Type"] = type
            df["Original Currency"] = currency

            # Append valid data to records
            records.append(df)

        except Exception as e:
            logging.error(f"Failed to fetch data: {e}")
            logging.info(
                "fetch_historical_data exception #############################################\n"
            )

    # If no valid data was fetched, log an error and return an empty DataFrame
    if not records:
        logging.error(
            f"No data fetched for {bdays} business days of data, {start_date.strftime("%d/%m/%y")} - {end_date.strftime("%d/%m/%y")}. Check your internet connection, ticker validity, or date ranges."
        )
        return pd.DataFrame()

    # Combine all DataFrames into one
    combined_df = pd.concat(records, ignore_index=True)

    return combined_df


def convert_prices(df: pd.DataFrame) -> pd.DataFrame:
    """
    Converts prices to GBP based on ticker type and currency.

    Args:
        df: DataFrame containing historical data.

    Returns:
        DataFrame with prices converted to GBP where applicable.
    """

    df = df.copy()

    # Ensure the Date column is properly formatted
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce").dt.date
    df = df.dropna(subset=["Date"])  # Drop rows with invalid dates

    # Extract exchange rates for CURRENCY type into a new DataFrame
    exchange_rate_df = (
        df[df["Type"] == "CURRENCY"]
        .rename(columns={"Old Price": "FX Rate", "Ticker": "FX Ticker"})
        .loc[:, ["Date", "FX Ticker", "FX Rate"]]
        .sort_values(by=["Date", "FX Ticker"])
    )

    # Standardise FX Code and FX Rate columns in the main DataFrame
    df["FX Ticker"] = "-"
    df["FX Rate"] = 1.0  # Default FX Rate (decimal ensured floating type)
    df["Currency"] = "GBP"  # Default Currency
    df["Price"] = np.nan  # Placeholder for converted prices

    # Handle rows with GBP and GBp directly
    df.loc[df["Original Currency"] == "GBP", "FX Rate"] = 1.0
    df.loc[df["Original Currency"] == "GBp", "FX Rate"] = 0.01

    # Identify non-GBP rows and assign FX Code
    non_gbp_mask = ~df["Original Currency"].isin(["GBP", "GBp"])
    fx_codes = df.loc[non_gbp_mask, "Original Currency"] + "GBP=X"
    fx_codes[df["Original Currency"] == "USD"] = "GBP=X"  # USD treated as special case
    df.loc[non_gbp_mask, "FX Ticker"] = fx_codes

    # Merge exchange rates back into the main DataFrame
    df = df.merge(
        exchange_rate_df,
        how="left",
        left_on=["Date", "FX Ticker"],
        right_on=["Date", "FX Ticker"],
        suffixes=("", "_merge"),
    )

    # Populate FX Rate from merged data where available
    df["FX Rate"] = df["FX Rate"].combine_first(df["FX Rate_merge"])

    # Calculate the converted prices
    df["Price"] = df["Old Price"] * df["FX Rate"].fillna(1)

    # Clean up unnecessary columns
    return df[["Ticker", "Price", "Date"]]


def process_converted_prices(
    converted_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Prepares the final DataFrame for CSV export.

    Args:
        converted_df: DataFrame with "Ticker", "Date", "Price"

    Returns:
        Processed DataFrame ready to be saved as CSV.
    """
    try:

        if converted_df.empty:
            logging.error(
                "No data received. Verify that tickers and date ranges are correct. Exiting."
            )
            sys.exit(1)

        output_csv = converted_df[["Ticker", "Price", "Date"]].copy()

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


def save_to_csv(df: pd.DataFrame, config: Dict[str, Any]) -> bool:
    """
    Saves the DataFrame to a CSV file without headers.

    Args:
        df: DataFrame to save.

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
                f"Missing required columns {required_cols} in data. Cannot save CSV. Exiting"
            )
            sys.exit(1)

        # Build the output file path
        file_path = f"{config['paths']['base']}/{config['paths']['data_file']}"
        # Save the DataFrame to CSV without headers
        df.to_csv(file_path, index=False, header=False)
        # Log the save location
        short_CSV_path = format_path(file_path)
        logging.info(f"Data successfully saved to: {short_CSV_path}")

        return True

    except Exception as e:
        logging.exception(f"An error occurred while saving to {short_CSV_path}")
        return False


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
            f"Cache directory does not exist or is not a directory: {cache_dir}"
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
                            f"Deleted cache file: {entry.name}, older than {config['cache']['max_age_days']} days."
                        )
                    except Exception as remove_error:
                        logging.error(
                            f"Failed to delete file {entry.path}: {remove_error}"
                        )

        logging.debug(f"Total cache files deleted: {deleted_files}")
    except Exception as e:
        logging.error(
            f"Error cleaning cache at {cache_dir}: {e}. Check file permissions or existence of the directory."
        )
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
            exit(0)
        else:
            logging.warning(f"Unknown situation. Code: {return_code}")
            exit(0)


@retry(exceptions=(RuntimeError, IOError), tries=3, delay=2, backoff=2)
def import_data_file():

    try:
        output_file_name = config["paths"]["data_file"]
        filename = os.path.join(base_path, output_file_name)

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
                time.sleep(1.5)
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
                time.sleep(0.5)
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
                time.sleep(0.5)
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
                time.sleep(0.5)
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
        # Initialise
        os.system("cls" if os.name == "nt" else "clear")
        logging.info("Starting script.")

        # Load the configuration
        config = load_configuration("configuration.yaml")

        # Call setup_logging early in the script
        setup_logging(config)

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
        price_data = fetch_historical_data(valid_tickers, config)
        if price_data.empty:
            logging.error("No valid data fetched. Exiting.")
            sys.exit(1)

        # Convert prices to GBP
        processed_data = convert_prices(price_data)

        # Process and create output DataFrame
        output_csv = process_converted_prices(processed_data)

        # Save data to CSV
        save_to_csv(output_csv, config)

        # Clean up cache
        clean_cache(config)

        # Quicken import sequence
        setup_pyautogui()
        quicken_import()

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
