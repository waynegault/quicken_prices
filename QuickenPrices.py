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
    level=logging.DEBUG,
    format="%(asctime)s - %(levelname)s - %(message)s",
    stream=sys.stdout,
)
import traceback
import contextlib  # Used to redirect output streams temporarily
import ctypes  # Low-level OS interfaces
import functools  # Higher-order functions and decorators
from logging.handlers import (
    RotatingFileHandler,
)  # Additional logging handlers (e.g., RotatingFileHandler)
import pickle  # Object serialization/deserialization
import subprocess  # Process creation and management
import time  # Time-related functions
from datetime import date, datetime, timedelta  # Date and time handling
from pathlib import Path  # Object-oriented filesystem paths
from typing import Any, Dict, List, Optional, Tuple, Union  # Type annotations
import numpy as np
import yaml
import pandas as pd
import yfinance as yf
from colorama import init, Fore, Style
import pygetwindow as gw
import pyautogui
from pythonjsonlogger import jsonlogger

init(autoreset=True)

os.system("cls" if os.name == "nt" else "clear")

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
    "tickers": ["^FTSE"],  # Changed from set to list
    "paths": {
        "data_file": "data.csv",
        "log_file": "prices.log",
        "cache": "cache",
    },
    "home_currency": "GBP",  # Added explicitly
    "collection": {"period_years": 0.08, "max_retries": 3, "retry_delay": 2},
    "cache": {"max_age_days": 30, "cleanup_threshold": 200},
    "logging": {
        "levels": {"file": "DEBUG", "terminal": "DEBUG"},
        "max_bytes": 5_242_880,
        "backup_count": 5,
    },
}


def load_configuration(config_file: str = "configuration.yaml") -> Dict[str, Any]:

    base_path = os.path.dirname(os.path.abspath(__file__))

    config_path = os.path.join(base_path, config_file)
    config = {}

    if not os.path.exists(config_path):
        # If the configuration file doesn't exist, create it with defaults
        logging.warning(f"{config_file} not found. Creating with default settings.")
        with open(config_path, "w") as f:
            yaml.dump(
                DEFAULT_CONFIG, f, default_flow_style=False
            )  # Ensures readable format
        logging.info(f"Default {config_file} created at {config_path}.")
    else:
        # Log that the local YAML file is found and being used
        logging.info(f"Local {config_file} found at {config_path}. Using it.")

    # Attempt to load configuration file
    try:
        with open(config_path, "r") as f:
            config = yaml.safe_load(f) or {}

        if not isinstance(config, dict):
            logging.warning(f"Invalid format in {config_file}. Using defaults.")
            config = {}

    except Exception as e:
        logging.error(f"Error reading {config_file}: {e}. Using internal defaults.")
        config = {}

    # Merge defaults with user config
    final_config = apply_defaults(DEFAULT_CONFIG, config)

    # Add base path to paths in the config
    final_config.setdefault("paths", {}).setdefault("base", base_path)

    # Automatically find Quicken path
    validated_quicken_path = find_quicken_via_registry()

    if validated_quicken_path:
        final_config["paths"]["quicken"] = validated_quicken_path
        logging.info(f"Quicken found at: {validated_quicken_path}")
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

    return final_config


# Load the configuration
config = load_configuration()
# Ensure base directory exists
base_path = config["paths"]["base"]
os.makedirs(base_path, exist_ok=True)


def setup_logging():
    # Determine the log directory based on the operating system
    if os.name == "nt":  # Windows
        log_dir = os.path.join(os.getenv("APPDATA", base_path), "YourApp", "logs")
    else:  # Linux/macOS
        log_dir = os.path.join(os.path.expanduser("~"), ".your_app", "logs")

    # Ensure the logs directory exists
    os.makedirs(log_dir, exist_ok=True)

    # Use the log file path from the configuration or default to the generated log directory
    log_file = os.path.join(log_dir, config["paths"].get("log_file", "prices.log"))

    # Create a rotating file handler with a simpler format for readability
    file_handler = RotatingFileHandler(
        log_file,
        maxBytes=config["logging"]["max_bytes"],
        backupCount=config["logging"]["backup_count"],
        encoding="utf-8",
    )
    file_handler.setLevel(
        getattr(logging, config["logging"]["levels"]["file"].upper(), logging.DEBUG)
    )
    # Simple, human-readable format for the file logs
    file_formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    file_handler.setFormatter(file_formatter)

    # Create a console handler with the same format
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(
        getattr(logging, config["logging"]["levels"]["terminal"].upper(), logging.DEBUG)
    )
    console_formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    console_handler.setFormatter(console_formatter)

    # Add handlers to the root logger
    logging.getLogger().handlers = []  # Clear existing handlers
    logging.getLogger().addHandler(file_handler)
    logging.getLogger().addHandler(console_handler)

    # Log the location of the log file
    logging.info(f"Logs are being written to: {log_file}")


# Call setup_logging early in the script
setup_logging()


def retry(exceptions, tries=3, delay=1, backoff=2):
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            for attempt in range(tries):
                try:
                    return func(*args, **kwargs)
                except exceptions as e:
                    if attempt < tries - 1:
                        logging.warning(f"Retry {attempt + 1}/{tries} failed: {e}")
                        time.sleep(delay)
                        delay *= backoff
                    else:
                        logging.error(
                            f"Function '{func.__name__}' failed after {tries} attempts."
                        )
                        raise

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

    logging.info(f"Validating {len(tickers)} user {pluralise("ticker",len(tickers))}...")
    valid_tickers = validate_tickers(tickers)

    if not valid_tickers:
        logging.error("No valid stock tickers found in YAML file. Exiting.")
        sys.exit(1)

    # Generate FX tickers based on the currencies in valid_tickers
    currencies = {t[3] for t in valid_tickers if t[3] not in ["GBP", "GBp"]}
    fx_tickers = {
        f"{currency}GBP=X" if currency != "USD" else "GBP=X" for currency in currencies
    }
    if fx_tickers:
        logging.info(f"Validating {len(fx_tickers)} imported FX {pluralise("ticker",len(fx_tickers))}...")
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
    First checks if the data is cached; if not, fetches the data using yfinance and caches it.
    If cache is corrupted or missing, reload the data. The cache can be supplemented with missing data.
    Takes weekends and holidays into account when checking for missing data.
    Handles delisted tickers and avoids warnings during concatenation.
    """
    cache_dir = os.path.join(base_path, config["paths"]["cache"])
    if not os.path.exists(cache_dir):
        os.makedirs(cache_dir)

    cache_file = os.path.join(
        cache_dir,
        f"{ticker_symbol}_{start_date.strftime('%Y%m%d')}_{end_date.strftime('%Y%m%d')}.pkl",
    )

    # Generate list of business days between start_date and end_date
    business_days = pd.bdate_range(start=start_date, end=end_date).date

    try:
        # Attempt to load cached data
        if os.path.exists(cache_file):
            try:
                with open(cache_file, "rb") as f:
                    cached_data = pickle.load(f)

                # Ensure cached data has the required structure
                required_columns = {"Date", "Old Price"}
                if required_columns.issubset(cached_data.columns):
                    logging.info(f"Loaded cached data for {ticker_symbol}.")
                    logging.debug(f"Cached data for {ticker_symbol}:\n{cached_data.head()}")
                    
                    # Check which dates are actually present in the cached data
                    cached_dates = pd.to_datetime(cached_data["Date"]).dt.date.tolist()

                    # Compare with business days to detect missing dates
                    missing_dates = [d for d in business_days if d not in cached_dates]
                    if missing_dates:
                        logging.info(f"Found missing data for {ticker_symbol} on {missing_dates}. Fetching missing dates.")
                        # Fetch missing data only for the missing dates
                        try:
                            missing_data = yf.Ticker(ticker_symbol).history(
                                start=min(missing_dates),
                                end=max(missing_dates),
                                interval="1d",
                            ).reset_index()

                            if missing_data.empty:
                                logging.warning(f"No price data found for missing dates of {ticker_symbol}.")
                                return cached_data  # Return cached data if no data found

                            missing_data = missing_data.rename(columns={"Close": "Old Price"})
                            missing_data = missing_data[["Date", "Old Price"]].copy()
                            missing_data["Ticker"] = ticker_symbol

                            # Combine cached data with new data
                            combined_data = pd.concat([cached_data, missing_data], ignore_index=True)

                            # Handle potential future pandas warning regarding empty columns
                            combined_data = combined_data.dropna(how='all', axis=1)  # Drop all-NA columns

                            # Save the combined data back to the cache
                            with open(cache_file, "wb") as f:
                                pickle.dump(combined_data, f)

                            return combined_data

                        except Exception as e:
                            logging.error(f"Error fetching missing data for {ticker_symbol}: {e}")
                            return cached_data  # Return cached data if fetching fails

                    # No missing data, return cached data
                    return cached_data

            except Exception as e:
                logging.warning(f"Error reading cache for {ticker_symbol}: {e}. Reloading data.")
                # Delete the corrupted cache
                os.remove(cache_file)

        # If no valid cache, fetch from yfinance
        ticker = yf.Ticker(ticker_symbol)
        time.sleep(0.2)  # Rate limiting
        df = ticker.history(
            start=start_date,
            end=end_date,
            interval="1d",
        ).reset_index()

        if df.empty:
            logging.warning(
                f"No data retrieved for {ticker_symbol} between {start_date} and {end_date}."
            )
            return pd.DataFrame()

        logging.info(f"Data for {ticker_symbol} retrieved from yfinance.")
        logging.debug(f"Fetched data for {ticker_symbol}:\n{df.head()}")

        # Rename columns and clean up DataFrame
        df = df.rename(columns={"Close": "Old Price"})
        df = df[["Date", "Old Price"]].copy()
        df["Ticker"] = ticker_symbol

        # Save to cache
        with open(cache_file, "wb") as f:
            pickle.dump(df, f)

        return df

    except Exception as e:
        logging.error(f"Failed to fetch or process data for {ticker_symbol}: {e}")
        return pd.DataFrame()

def fetch_historical_data(
    tickers: List[Tuple[str, Optional[datetime], str, str]]
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
        # Adjusted start date is the later of earliest_date and start_date
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
            df = fetch_ticker_history(ticker, adjusted_start_date, end_date)

            # Log the data from cache or download for debugging
            logging.debug(
                f"Fetched data for {ticker} from cache/download: \n{df.head()}"
            )

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
            logging.error(f"Failed to fetch data for {ticker}: {e}")

    # If no valid data was fetched, log an error and return an empty DataFrame
    if not records:
        logging.error(
            f"No data fetched for the {business_days} business days between "
            f"{start_date.strftime('%d/%m/%Y')} and {end_date.strftime('%d/%m/%Y')}!"
        )
        return pd.DataFrame()

    # Combine all DataFrames into one
    combined_df = pd.concat(records, ignore_index=True)

    # Reorder columns for consistency
    combined_df = combined_df[
        ["Ticker", "Date", "Type", "Old Price", "Original Currency"]
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

    # Create exchange rate DataFrame with valid Date index
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

            # Log the conversion process
            logging.debug(
                f"Converted {row['Ticker']}: {price} {currency} -> {converted_price} GBP"
            )

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
        CSV_path = os.path.join(base_path, CSV_file_name)
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
    cache_dir = os.path.join(base_path, config["paths"]["cache"])
    max_age_seconds = config["cache"]["max_age_days"] * 86400
    now = time.time()

    try:
        with os.scandir(cache_dir) as it:
            for entry in it:
                if entry.is_file() and (now - entry.stat().st_mtime) > max_age_seconds:
                    os.remove(entry.path)
                    logging.info(
                        f"Deleting cache older than {config["cache"]["max_age_days"]} days: {entry.name}"
                    )
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

        # Get elevation
        run_as_admin()

        # Get 'raw' tickers from YAML file and FX tickers if required, validate and acquire metadata.
        ticker_list = config["tickers"]
        valid_tickers = get_tickers(ticker_list, set_maximum_tickers=5)

        # Fetch historical data
        price_data = fetch_historical_data(valid_tickers)

        # Validate if price_data is empty
        if price_data.empty:
            logging.error("No valid data fetched. Exiting script.")
            return

        # Convert prices to GBP
        processed_data = convert_prices(price_data)

        # Process and create output DataFrame
        output_csv = process_converted_prices(processed_data)

        # Save data to CSV
        save_to_csv(output_csv)

        # Clean up cache
        clean_cache()

        # Quicken import sequence
        setup_pyautogui()
        quicken_import()

        # Pause to allow reading of terminal
        if is_elevated():
            input("\n\t Press Enter to exit...\n\t")

    except KeyboardInterrupt:
        logging.warning("Script interrupted by user.")
    except Exception as e:
        logging.error(f"An unexpected error occurred: {e}")
        logging.error(traceback.format_exc())


if __name__ == "__main__":

   

    main()
