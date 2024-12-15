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
import inspect
import sys
import winreg  # Added for registry access
import traceback
import ctypes  # Low-level OS interfaces
from logging.handlers import RotatingFileHandler
import pickle  # Object serialization/deserialization
import subprocess  # Process creation and management
import calendar
import time  # Time-related functions
from functools import wraps
from pandas import Timestamp, Timedelta
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Type, Union
import numpy as np
from numpy import busday_count
import yaml
import pandas as pd
import yfinance as yf
import pygetwindow as gw
import pyautogui

# Constants

SECONDS_IN_A_DAY = 86400
CACHE_COLUMNS = ["Date", "Old Price", "Ticker"]
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


config = load_configuration("configuration.yaml")


# Retry decorator for fetching
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
        f"Validating {len(tickers)} user {pluralise('ticker', len(tickers))}..."
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
        logging.info(
            f"{len(valid_fx_tickers)} valid FX {pluralise('ticker', len(valid_fx_tickers))} found: "
            f"{[ticker[0] for ticker in valid_fx_tickers[:max_show]]}"
            f"{'...' if len(valid_fx_tickers) > max_show else ''}"
        )
        all_valid_tickers = list(set(valid_tickers + valid_fx_tickers))
    else:
        logging.warning("No new FX tickers required.")
        all_valid_tickers = valid_tickers

    logging.info(
        f"Validated {len(all_valid_tickers)} {pluralise('ticker', len(all_valid_tickers))} in total.\n"
    )

    return all_valid_tickers


def pluralise(title: str, quantity: int) -> str:
    return title if quantity == 1 else title + "s"


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
                    )
                )

        except Exception as e:
            logging.error(
                f"An unexpected error occurred processing ticker '{ticker_symbol}': {e}"
            )

    logging.info(
        f"{len(valid_tickers)} valid {pluralise('ticker', len(valid_tickers))} found: "
        f"{[ticker[0] for ticker in valid_tickers[:max_show]]}"
        f"{'...' if len(valid_tickers) > max_show else ''}"
    )

    return valid_tickers


@retry((Exception,), tries=3)
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
        earliest_date = (
            pd.Timestamp(first_trade_date, unit="s", tz="UTC").date()
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


def fetch_historical_data(
    tickers: List[Tuple[str, Optional[pd.Timestamp], str, str]], config: Dict[str, Any]
) -> pd.DataFrame:
    """
    Fetch historical data for a list of tickers.

    Args:
        tickers: List of tickers with metadata.
        config: Configuration dictionary.

    Returns:
        pd.DataFrame: Combined historical data for all tickers.
    """

    start_date, end_date, bdays = get_date_range(config)
    required_business_days = pd.date_range(
        start=start_date, end=end_date, freq="B"
    ).normalize()

    records = []
    logging.info(
        f"Seeking {bdays} business days of data ({start_date.strftime('%d/%m/%Y')} to {end_date.strftime('%d/%m/%Y')})."
    )

    for ticker, earliest_date, ticker_type, currency in tickers:
        adjusted_start_date = max(start_date, earliest_date or start_date)

        if earliest_date and end_date < earliest_date:
            logging.warning(
                f"Skipping {ticker}: Period requested ends before ticker existed."
            )
            continue

        try:
            df = fetch_ticker_history(
                ticker, adjusted_start_date, end_date, required_business_days, config
            )

            if df.empty:
                logging.warning(f"No valid data found for ticker {ticker}. Skipping.")
                continue

            # Add metadata columns
            df["Ticker"] = ticker
            df["Type"] = ticker_type
            df["Original Currency"] = currency
            records.append(df)

        except Exception as e:
            logging.error(f"Failed to fetch data for ticker {ticker}: {e}")

    if not records:
        logging.error("No valid data fetched.")
        return pd.DataFrame(
            columns=["Ticker", "Original Currency", "Date", "FX Rate", "Old Price"]
        )

    combined_df = pd.concat([df for df in records if not df.empty], ignore_index=True)
    combined_df.drop_duplicates(subset=["Ticker", "Date"], keep="first", inplace=True)

    logging.info(
        f"Fetched {len(combined_df)} total records across {len(records)} tickers."
    )
    return combined_df


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

    today = pd.Timestamp.now().normalize()

    def adjust_for_weekend(date):
        if date.weekday() >= 5:  # Saturday or Sunday
            return date + pd.Timedelta(days=calendar.MONDAY - date.weekday())
        else:
            return date

    # Determine end date, adjusting for weekends
    end_date = adjust_for_weekend(today)

    # Determine start date, adjusting for weekends
    start_date = end_date - pd.Timedelta(days=period_days)
    start_date = adjust_for_weekend(start_date)

    # Convert to pandas Timestamp.date
    start_pd_date = start_date.date()
    end_pd_date = end_date.date()

    # Calculate business days (using pandas functionality)
    business_days = pd.bdate_range(start_pd_date, end_pd_date).size

    return start_pd_date, end_pd_date, business_days


def fetch_ticker_history(ticker, start_date, end_date, required_business_days, config):
    """
    Fetch and manage historical data for a ticker.
    """
    print(
        f"\n start of {inspect.getframeinfo(inspect.currentframe()).function} function\n"
    )
    cache_dir = os.path.join(config["paths"]["base"], config["paths"]["cache"])
    os.makedirs(cache_dir, exist_ok=True)

    # Load cache
    cache_data = load_cache(ticker, cache_dir)

    # Identify cached and permanently missing dates
    cached_dates = set(cache_data["Date"])
    permanently_missing_dates = set(
        cache_data.loc[cache_data["Old Price"].isna(), "Date"]
    )

    # Compute missing dates
    missing_dates = compute_missing_dates(
        required_business_days, cached_dates, permanently_missing_dates
    )

    # Fetch and update cache
    cache_data = fetch_and_update_data(ticker, missing_dates, cache_dir, download_data)

    # Update permanently missing dates
    if missing_dates:
        remaining_missing = compute_missing_dates(
            required_business_days, set(cache_data["Date"]), permanently_missing_dates
        )
        update_cache_with_missing_dates(
            ticker, cache_data, remaining_missing, cache_dir
        )

    return cache_data


def load_cache(ticker: str, cache_dir: str) -> pd.DataFrame:
    """
    Load cache for a specific ticker from a CSV file.

    Args:
        ticker (str): The ticker symbol.
        cache_dir (str): Path to the cache directory.

    Returns:
        pd.DataFrame: DataFrame containing cached data.
    """
    print(
        f"\n start of {inspect.getframeinfo(inspect.currentframe()).function} function\n"
    )
    cache_file = Path(cache_dir) / f"{ticker}.csv"
    if cache_file.exists():
        try:
            df = pd.read_csv(cache_file, parse_dates=["Date"])
            df["Date"] = pd.to_datetime(df["Date"], utc=True).dt.normalize()
            if "Ticker" not in df.columns:
                df["Ticker"] = ticker
            return df
        except Exception as e:
            logging.error(f"Failed to load cache for {ticker}: {e}")
            return pd.DataFrame(columns=CACHE_COLUMNS)
    else:
        return pd.DataFrame(columns=CACHE_COLUMNS)


def compute_missing_dates(
    required_days: set, cached_dates: set, permanently_missing: set
) -> set:
    """
    Compute the set of missing dates that need to be fetched.

    Args:
        required_days (set): Set of required business days.
        cached_dates (set): Set of dates already in the cache.
        permanently_missing (set): Set of dates known to be permanently missing.

    Returns:
        set: Set of missing dates.
    """
    required_days = {pd.Timestamp(day).normalize() for day in required_days}
    cached_dates = {pd.Timestamp(day).normalize() for day in cached_dates}
    permanently_missing = {pd.Timestamp(day).normalize() for day in permanently_missing}
    return required_days - cached_dates - permanently_missing


def fetch_and_update_data(
    ticker: str, missing_dates: set, cache_dir: str, download_data
) -> pd.DataFrame:
    """
    Fetch data for missing dates, update the cache, and return the combined data.

    Args:
        ticker (str): The ticker symbol.
        missing_dates (set): Set of missing dates to fetch.
        cache_dir (str): Path to the cache directory.
        download_data (callable): Function to download data for a date range.

    Returns:
        pd.DataFrame: Updated cache data.
    """
    print(f"\n start of fetch_and_update_data function\n")
    cache_data = load_cache(ticker, cache_dir)
    cached_count = len(cache_data)
    missing_count = len(missing_dates)

    if cached_count > 0 and missing_count > 0:
        logging.info(
            f"Ticker {ticker}: {cached_count} days from cache, {missing_count} days to fetch from API."
        )
    elif cached_count > 0:
        logging.info(f"Ticker {ticker}: 100% of data retrieved from cache.")
    elif missing_count > 0:
        logging.info(f"Ticker {ticker}: 100% of data fetched from API.")

    print(f"\nmissing_dates: {missing_dates}")

    if missing_dates:
        start_date = min(missing_dates)
        end_date = max(missing_dates) + pd.Timedelta(days=1)
        fetched_data = download_data(ticker, start_date, end_date)

        # Log fetched data for inspection
        logging.info(
            f"Fetched data columns for {ticker}: {fetched_data.columns.tolist()}"
        )

        if not fetched_data.empty:
            fetched_data = fetched_data[["Date", "Old Price", "Ticker"]]
            fetched_data["Date"] = pd.to_datetime(
                fetched_data["Date"], utc=True
            ).dt.normalize()

            logging.info(f"Fetched data for {ticker}:\n{fetched_data.head()}")

            valid_data_frames = [
                df
                for df in [cache_data, fetched_data]
                if not df.empty and not df.dropna(how="all").empty
            ]

            print(f"valid_data_frames: {valid_data_frames}")

            if valid_data_frames:
                cache_data = pd.concat(
                    valid_data_frames, ignore_index=True
                ).drop_duplicates(subset=["Date", "Ticker"])
            else:
                logging.warning(f"No valid data for ticker {ticker} after fetch.")

        print(f"ticker: {ticker}")
        print(f"cache_data: {cache_data}")
        print(f"cache_dir: {cache_dir}")

        save_cache(ticker, cache_data, cache_dir)

    print(f"cache_data: {cache_data}")
    return cache_data


@retry(Exception, tries=3, backoff=2)
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
    print(
        f"\n start of {inspect.getframeinfo(inspect.currentframe()).function} function\n"
    )
    # Ensure the start and end dates are UTC-aware
    start = start.tz_localize("UTC") if start.tz is None else start.tz_convert("UTC")
    end = end.tz_localize("UTC") if end.tz is None else end.tz_convert("UTC")

    logging.info(
        f"ccc Downloading data for {ticker_symbol} from {start.date()} to {end.date()}"
    )

    # Fetch data using yfinance

    try:
        ticker = yf.Ticker(ticker_symbol)
        print(f"ticker: {ticker}")
        raw_data = ticker.history(start=start.date(), end=end.date(), interval="1d")
        print(f"raw_data: {raw_data}")
        if raw_data.empty:
            logging.warning(f"No data returned for {ticker_symbol} from yfinance.")
            return pd.DataFrame(columns=["Date", "Old Price", "Ticker"])
        raw_data.reset_index(inplace=True)
        # Prepare the DataFrame to match the expected format
        processed_data = pd.DataFrame(
            {
                "Date": pd.to_datetime(raw_data["Date"], utc=True).dt.normalize(),
                "Old Price": raw_data["Close"],
                "Ticker": ticker,
            }
        )

        return processed_data

    except Exception as e:
        logging.error(f"Error fetching data for {ticker_symbol}: {e}")
        return pd.DataFrame(columns=["Date", "Old Price", "Ticker"])

    # Reset index to ensure 'Date' is a column
    raw_data.reset_index(inplace=True)

    # Handle missing `Close` column and rename to `Old Price`
    if "Close" in raw_data.columns:
        raw_data.rename(columns={"Close": "Old Price"}, inplace=True)
    else:
        logging.warning(f"{ticker_symbol} missing 'Close' column. Adding placeholder.")
        raw_data["Old Price"] = pd.NA

    # Normalize datetime column
    raw_data = ensure_tz_aware(raw_data)

    # Cleaned data logging
    cleaned_data = raw_data[["Date", "Old Price"]]

    return cleaned_data


def ensure_tz_aware(df: pd.DataFrame) -> pd.DataFrame:
    """
    Ensures all date-related columns in the DataFrame are timezone-aware and localized to UTC.

    Args:
        df: Input DataFrame.

    Returns:
        DataFrame with all date columns tz-aware and localized to UTC.
    """
    print(
        f"\n start of {inspect.getframeinfo(inspect.currentframe()).function} function\n"
    )
    if "Date" in df.columns:
        df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
        # Handle both tz-naive and tz-aware cases
        if df["Date"].dt.tz is None:
            df["Date"] = df["Date"].dt.tz_localize(
                "UTC", ambiguous="NaT", nonexistent="shift_forward"
            )
        else:
            df["Date"] = df["Date"].dt.tz_convert("UTC")
    return df


#
#######################
#
#


def save_cache(ticker: str, data: pd.DataFrame, cache_dir: str) -> None:
    """
    Save cache for a specific ticker to a CSV file.

    Args:
        ticker (str): The ticker symbol.
        data (pd.DataFrame): DataFrame containing data to cache.
        cache_dir (str): Path to the cache directory.
    """
    print(
        f"\n start of {inspect.getframeinfo(inspect.currentframe()).function} function\n"
    )
    cache_file = Path(cache_dir) / f"{ticker}.csv"
    try:
        os.makedirs(cache_dir, exist_ok=True)
        # Filter to ensure only valid columns are saved
        if "Ticker" not in data.columns:
            raise ValueError(
                f"Ticker column missing in data for {ticker}. Columns: {data.columns.tolist()}"
            )
        data = data[CACHE_COLUMNS].dropna(subset=["Date", "Ticker"])
        data.to_csv(cache_file, index=False)
    except Exception as e:
        logging.error(f"Failed to save cache for {ticker}: {e}")


def update_cache_with_missing_dates(
    ticker: str, data: pd.DataFrame, missing_dates: set, cache_dir: str
) -> None:
    """
    Update the cache with permanently missing dates.

    Args:
        ticker (str): The ticker symbol.
        data (pd.DataFrame): Current cache data.
        missing_dates (set): Set of permanently missing dates.
        cache_dir (str): Path to the cache directory.
    """
    print(
        f"\n start of {inspect.getframeinfo(inspect.currentframe()).function} function\n"
    )
    if missing_dates:
        missing_df = pd.DataFrame(
            {"Date": list(missing_dates), "Ticker": ticker, "Old Price": None}
        )
        data = pd.concat([data, missing_df], ignore_index=True).drop_duplicates(
            subset="Date"
        )
        save_cache(ticker, data, cache_dir)
        logging.info(
            f"Updated cache for {ticker} with {len(missing_dates)} permanently missing dates."
        )


def convert_prices(df: pd.DataFrame) -> pd.DataFrame:
    """
    Converts prices to GBP based on ticker type and currency.

    Args:
        df: DataFrame containing historical data.

    Returns:
        DataFrame with prices converted to GBP where applicable.
    """
    print(
        f"\n start of {inspect.getframeinfo(inspect.currentframe()).function} function\n"
    )
    if df.empty:
        logging.error(
            "Empty DataFrame received in `convert_prices`. No data to process."
        )
        return pd.DataFrame()

    required_columns = {"Original Currency", "Old Price", "Date", "Ticker"}
    missing_columns = required_columns - set(df.columns)
    if missing_columns:
        logging.error(f"Missing required columns in data: {missing_columns}")
        return pd.DataFrame()

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
    c = df[~df["FX Ticker"].str.contains("-")].shape[0]
    logging.info(f"{c} prices converted.\n")
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
    print(
        f"\n start of {inspect.getframeinfo(inspect.currentframe()).function} function\n"
    )
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
        output_csv["Date"] = pd.to_datetime(
            output_csv["Date"], errors="coerce"
        ).dt.strftime("%d/%m/%Y")

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
        config: Configuration dictionary.

    Returns:
        True if saved successfully, False otherwise.
    """
    print(
        f"\n start of {inspect.getframeinfo(inspect.currentframe()).function} function\n"
    )
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
        logging.info(f"Data successfully saved to: {short_CSV_path}\n")

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
    print(
        f"\n start of {inspect.getframeinfo(inspect.currentframe()).function} function\n"
    )
    cache_dir = os.path.join(config["paths"]["base"], config["paths"]["cache"])
    max_age_seconds = config["cache"]["max_age_days"] * SECONDS_IN_A_DAY
    now = time.time()
    logging.info("Cache cleaning.")

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
                            f"{entry.name} deleted. Older than {config['cache']['max_age_days']} {pluralise('day', config['cache']['max_age_days'])}."
                        )
                    except Exception as remove_error:
                        logging.error(
                            f"Failed to delete file {entry.path}: {remove_error}"
                        )

        logging.info(
            f"{deleted_files} cache {pluralise('file', deleted_files)} deleted.\n"
        )
    except Exception as e:
        logging.error(
            f"Error cleaning cache at {cache_dir}: {e}. Check file permissions or existence of the directory."
        )
        return False

    return True


@retry(exceptions=(RuntimeError, IOError), tries=3, backoff=2)
def import_data_file(config):
    print(
        f"\n start of {inspect.getframeinfo(inspect.currentframe()).function} function\n"
    )
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


@retry(exceptions=(Exception,), tries=3, backoff=2)
def open_import_dialog():
    print(
        f"\n start of {inspect.getframeinfo(inspect.currentframe()).function} function\n"
    )
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


@retry(exceptions=(Exception,), tries=3, backoff=2)
def navigate_to_portfolio():
    print(
        f"\n start of {inspect.getframeinfo(inspect.currentframe()).function} function\n"
    )
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


@retry(exceptions=(Exception,), tries=3, backoff=2)
def open_quicken(config):
    print(
        f"\n start of {inspect.getframeinfo(inspect.currentframe()).function} function\n"
    )
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
    print(
        f"\n start of {inspect.getframeinfo(inspect.currentframe()).function} function\n"
    )
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
    print(
        f"\n start of {inspect.getframeinfo(inspect.currentframe()).function} function\n"
    )
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
    print(
        f"\n start of {inspect.getframeinfo(inspect.currentframe()).function} function\n"
    )
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


def main():
    """
    Main script execution.
    """
    try:
        # Initialise
        os.system("cls" if os.name == "nt" else "clear")
        print("Starting script.")

        # Load the configuration

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
        quicken_import(config)

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
