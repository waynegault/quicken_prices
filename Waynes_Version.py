#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Shebang and coding declaration.

"""
QuickenPrices.py

This script fetches historical price data for a list of tickers specified in 'config.yaml' and saves the data to a CSV file.
It supports using a simulator for API calls or fetching real data using yfinance.

It includes features such as configuration management, logging, error handling, data fetching, data processing,currency conversion, and more.

Wayne Gault
23/11/2024

MIT Licence
"""
#
# -----------------------------------------------------------------------------------
# Imports                                                                           |
# -----------------------------------------------------------------------------------
#
try:
    # Standard Python Library Imports
    import ctypes  # Provides low-level operating system interfaces
    import sys  # Provides access to system-specific parameters and functions
    import subprocess  # Enables spawning new processes
    import time  # Offers time-related functions
    import calendar  # Provides calendar-related functions
    import os  # Provides a way of using operating system dependent functionality
    import uuid  # Generates universally unique identifiers (UUIDs)
    import gc  # Garbage Collector interface
    import threading  # Provides high-level threading interface
    from queue import Queue  # Implements multi-producer, multi-consumer queues
    from datetime import datetime, timedelta, date  # Classes for working with dates and times
    from pathlib import Path  # Object-oriented interface to filesystem paths
    from collections import namedtuple   # Create custom data structures with named fields.
    from dateutil import parser

    # Third-Party Library Imports
    import numpy as np
    import yaml  # YAML parser and emitter
    import json  # JSON encoder and decoder
    import functools  # Higher-order functions and operations
    import traceback  # Exception handling and traceback information
    from io import StringIO  # In-memory file-like object
    import pytz  # Time zone handling
    import pyperclip  # Clipboard operations
    from colorama import init, Fore, Style  # Colored terminal text
    import logging  # Logging framework
    from logging.handlers import RotatingFileHandler  # Rotating log file handler
    import hashlib  # Cryptographic hash functions
    import pickle  # Object serialization and deserialization
    from concurrent.futures import (
        ThreadPoolExecutor,
        as_completed,
    )  # Asynchronous task execution
    from typing import List, Iterator, Optional, Dict, Any, Set # Type hints
    from dataclasses import dataclass  # Data class decorator
    from enum import Enum  # Enum class definition
    from contextlib import contextmanager  # Context manager utilities
    import pyautogui  # GUI automation
    import yfinance as yf  # Yahoo Finance API
    import pandas as pd  # Data analysis and manipulation
    import psutil  # System and process utilities
    import weakref  # Weak reference support

except ImportError as e:
    print(
        f"The required library '{e.name}' is not installed. Please install it using 'pip install {e.name}'"
    )
    sys.exit(1)
#
# -----------------------------------------------------------------------------------
# Configuration Management                                                          |
# -----------------------------------------------------------------------------------
# Purpose: Load and manage configuration settings from config.yaml.


def load_configuration(config_file: str = "config.yaml") -> Dict[str, Any]:
    # Load configuration from the specified YAML file and applies defaults
    with open(config_file, "r") as f:
        # ensure config is always a dictionary, even if the yaml is empty
        config = yaml.safe_load(f) or {}

    # Default values dictionary incase of  missing settings in yaml
    default_settings = {
        "use_simulator": True,
        "tickers": {
            # Ticker is given by config["tickers"].keys() description = config["tickers"].get("AMGN", None)
            "0P00013P6I.L": "HSBC FTSE All-World Index C Acc (GBP)",
            "0P00018XAP.L": "Vanguard FTSE 100 Idx Unit Tr £ Acc (GBP)",
            "AMGN": "Amgen Inc. (USD)",
            "BZ=F": "Brent Crude Oil Last Day Finance (USD)",
            "CUKX.L": "iShares VII PLC - iShares Core FTSE 100 ETF GBP Acc (GBP)",
            "CYSE.L": "WisdomTree Cybersecurity UCITS ETF USD Acc (GBP)",
            "ELIX.L": "Elixirr International plc (GBP)",
            "EURGBP=X": "EUR/GBP Fx",
            "fake": "fake",
            "^FTAS": "FTSE All-share (GBP)",
            "^FTSE": "FTSE 100 (GBP)",
            "GBP=X": "USD/GBP Fx",
            "GL1S.MU": "Invesco Markets II Plc - Invesco UK Gilt UCITS ETF (EUR)",
            "GLTA.L": "Invesco UK Gilts UCITS ETF (GBP)",
            "GC=F": "Gold (USD)",
            "HMWO.L": "HSBC MSCI World UCITS ETF (GBP)",
            "IHCU.L": "iShares S&P 500 Health Care Sector UCITS ETF USD (Acc) (GBP)",
            "IIND.L": "iShares MSCI India UCITS ETF USD Acc (GBP)",
            "IUIT.L": "iShares S&P 500 Information Technology Sector UCITS ETF USD (Acc) (USD)",
            "LAU.AX": "(AUD)",
            "SEE.L": "Seeing Machines Limited (GBP)",
            "^OEX": "S&P 100 INDEX (USD)",
            "^GSPC": "S&P 500 INDEX (USD)",
            "SQX.CN": "(CAD)",
            "VHVG.L": "Vanguard FTSE Developed World UCITS ETF USD Accumulation (GBP)",
            "VNRX": "VolitionRX Limited (USD)",
            "VUKG.L": "Vanguard FTSE 100 UCITS ETF GBP Accumulation (GBP)",
        },
        # Debug Configuration -  populated for testing specific tickers
        "debug": {
            "VHVG.L",
            "ELIX.L",
            "VNRX",
            "EURGBP=X",
            "GC=F",
        },
        "paths": {
            "base": "C:\\Users\\wayne\\OneDrive\\Documents\\GitHub\\Python\\Projects\\quicken_prices\\",
            "quicken": "C:\\Program Files (x86)\\Quicken\\qw.exe",
            "data_file": "data.csv",
            "log_file": "prices.log",
            "cache": "cache",
        },
        # Default data collection periods for each known quoteType:
        "collection": {"period_years": 0.08, "max_retries": 3, "retry_delay": 2},
        "default_periods": {
            "EQUITY": "5d",
            "ETF": "5d",
            "MUTUALFUND": "1mo",
            "FUTURE": "5d",
            "CURRENCY": "5d",
            "INDEX": "5d",
        },
        # # ERROR - Indicates a serious problem that prevents the program from executing correctly
        # WARNING - Indicates a potential issue that may lead to problems in the future
        # INFO - Useful for understanding the program's behavior and for auditing purposes
        # DEBUG - For detailed information to trace the execution flow of a program
        # These are hierarchial levels. Python scores them as ERROR=40, WARNING=30, INFO=20, DEBUG=10. So selecting a lower level includes all higher levels too.
        "cache": {"max_age_hours": 168, "cleanup_threshold": 200},
        "api": {"rate_limit": {"max_requests": 30, "time_window": 60}},
        "memory": {"max_memory_percent": 75, "chunk_size": 1000},
        "validation": {"required_columns": ["Ticker", "Close", "Date"]},
        "logging": {
            "levels": {"file": "DEBUG", "terminal": "DEBUG"},
            "message_formats": {
                "file": {
                    "error": "%(asctime)s %(levelname)s:- %(filename)s:%(lineno)d - %(message)s - %(exc_info)s",
                    "warning": "%(asctime)s %(levelname)s:- %(filename)s:%(lineno)d - %(message)s",
                    "info": "%(asctime)s %(levelname)s:- %(message)s",
                    "debug": "%(asctime)s %(levelname)s:- %(filename)s:%(lineno)d - %(message)s - %(funcName)s - %(module)s - %(process)d - %(thread)d - %(relativeCreated)d - %(threadName)s",
                },
                "terminal": {
                    "error": "%(levelname)s:- %(filename)s:%(lineno)d - %(message)s - %(exc_info)s",
                    "warning": "%(levelname)s:- %(filename)s:%(lineno)d - %(message)s",
                    "info": "%(levelname)s:- %(message)s",
                    "debug": "%(levelname)s:- %(filename)s:%(lineno)d - %(message)s",
                },
            },
            "colors": {
                "error": "red",
                "warning": "yellow",
                "info": "green",
                "debug": "blue",
            },
            "max_bytes": 5242880,
            "backup_count": 5,
        },
    }

    for key, value in default_settings.items():
        # check if key exists in the loaded config dictionary, otherwise default is applied to config dictionary
        config[key] = config.get(key, value)

    return config

# Invoke the load_configuration function and assign settings to config
config = load_configuration()

# Initialize colorama to enable cross-platform colored terminal output
init(autoreset=True)

# Create namedtuple classes
valid_ticker = namedtuple("valid_ticker", ["ticker", "quote_type", "currency"])
ticker_data = namedtuple("ticker_data", ["ticker", "quote_type", "currency", "price", "date"])

# turn on yFinance debug mode
# yf.enable_debug_mode()

# Global Variables
base_directory = config.get("paths", {}).get("base", ".")
if not os.path.exists(base_directory):
    os.makedirs(base_directory)


#
# -----------------------------------------------------------------------------------
# Centralised Logger                                                                |
# -----------------------------------------------------------------------------------
# Purpose: Set up a centralized logging system to capture and record events, errors, and informational messages.
#
def setup_logging():
    """
    Set up logging for the script with log and terminal handlers
    """
    # Dict key:value pairs
    log_levels = {
        "DEBUG": logging.DEBUG,
        "INFO": logging.INFO,
        "WARNING": logging.WARNING,
        "ERROR": logging.ERROR,
    }

    # create a logger instance named "QuickenPrices"
    logger = logging.getLogger("QuickenPrices")
    logger.setLevel(logging.DEBUG)

    # Create terminal handler
    # create a handler that logs messages to the terminal
    terminal_handler = logging.StreamHandler()
    # retrieve the desired log level from config dictionary
    terminal_level = config["logging"]["levels"]["terminal"].upper()
    # set the log level of the handler
    terminal_handler.setLevel(log_levels.get(terminal_level))
    # Create a formatter based on the terminal level
    if terminal_level == "DEBUG":
        terminal_formatter = logging.Formatter(
            config["logging"]["message_formats"]["terminal"]["debug"]
        )
    elif terminal_level == "INFO":
        terminal_formatter = logging.Formatter(
            config["logging"]["message_formats"]["terminal"]["info"]
        )
    elif terminal_level == "WARNING":
        terminal_formatter = logging.Formatter(
            config["logging"]["message_formats"]["terminal"]["warning"]
        )
    else:
        # Default terminal formatter is error
        terminal_formatter = logging.Formatter(
            config["logging"]["message_formats"]["terminal"]["error"]
        )
    terminal_handler.setFormatter(terminal_formatter)
    # assign the formatter to the handler
    terminal_handler.setFormatter(terminal_formatter)
    logger.addHandler(
        terminal_handler
    )  # Adds terminal handler to the QuickenPrices logger.

    # Create file handler with rotating logs
    log_file_path = os.path.join(base_directory, config["paths"]["log_file"])
    file_handler = logging.handlers.RotatingFileHandler(
        log_file_path,
        maxBytes=config["logging"]["max_bytes"],
        backupCount=config["logging"]["backup_count"],
    )
    file_level = config["logging"]["levels"]["file"].upper()
    file_handler.setLevel(log_levels.get(file_level))
    # Create a formatter based on the terminal level
    if file_level == "ERROR":
        file_formatter = logging.Formatter(
            config["logging"]["message_formats"]["file"]["error"]
        )
    elif file_level == "INFO":
        file_formatter = logging.Formatter(
            config["logging"]["message_formats"]["file"]["info"]
        )
    elif file_level == "WARNING":
        file_formatter = logging.Formatter(
            config["logging"]["message_formats"]["file"]["warning"]
        )
    else:
        # Default formatter is debug
        file_formatter = logging.Formatter(
            config["logging"]["message_formats"]["file"]["debug"]
        )
    file_handler.setFormatter(file_formatter)
    logger.addHandler(file_handler)

    return logger


logger = setup_logging()


#
# -----------------------------------------------------------------------------------
# Decorators                                                                        |
# -----------------------------------------------------------------------------------
# Purpose: Provide reusable functionalities such as logging function entry and exit, retry mechanisms for transient errors, and clipboard operations.
#
def log_function(func):
    """
    A decorator that logs when a function is entered and exited. It also logs any errors that occur within the function.
    """

    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        logger.debug(f"Start {func.__name__}")
        try:
            result = func(*args, **kwargs)
            logger.debug(f"End {func.__name__}")
            return result
        except Exception as e:
            logger.error(f"{func.__name__} failed! Error: {e}")
            logger.error(traceback.format_exc())
            raise

    return wrapper


def retry(exceptions, tries=3, delay=1, backoff=2):
    """
    A decorator that retries a function upon encountering specified exceptions. It implements exponential backoff between retries.
    """

    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            mtries, mdelay = tries, delay
            while mtries > 0:
                try:
                    return func(*args, **kwargs)
                except exceptions as e:
                    mtries -= 1
                    if mtries == 0:
                        logger.error(f"{func.__name__} failed after {tries} attempts")
                        raise
                    logger.warning(f"{e}, Retrying in {mdelay} seconds...")
                    time.sleep(mdelay)
                    mdelay *= backoff
            raise exceptions(f"{func.__name__} failed after {tries} attempts")

        return wrapper

    return decorator


#
# -----------------------------------------------------------------------------------
# Utilities                                                                         |
# -----------------------------------------------------------------------------------
#
def copy_to_clipboard(text: str):
    """
    Copy the given text to the clipboard.
    """
    pyperclip.copy(text)

def get_date_range():
    """ Get period from config dictionary and calculate date range.
    """
    start_date = None
    end_date = None
    period_year= config["collection"]["period_years"]
    period_days = period_year *365
    end_date = date.today()
    start_date = end_date - timedelta(days=int(period_days))
    # Convert dates to numpy datetime64[D] format
    start_date_np = np.datetime64(start_date)
    end_date_np = np.datetime64(end_date)
    # Calculate business days using np.busday_count
    business_days = np.busday_count(start_date_np, end_date_np)

    logger.info(
        f"Obtained date range {start_date.strftime("%d/%m/%Y")} to {end_date.strftime("%d/%m/%Y")} ({business_days} business days)"
    )
    return start_date, end_date, business_days

def format_path(full_path):
    """Format a file path to show the last parent directory and filename."""
    path_parts = full_path.split("\\")
    if len(path_parts) > 1:
        return "\\" + "\\".join(path_parts[-2:])  # Show parent dir and filename
    return "\\" + full_path  # Just in case there's no parent directory


def normalise_dates(df, date_column="Date"):
    """
    Normalises the specified date column in the DataFrame by performing the following steps:

    1. **Check Column Existence**: Verifies that the specified date column exists in the DataFrame.
    2. **Trim Spaces**: Removes any leading or trailing spaces from the date strings.
    3. **Parse and Convert to UTC**: Converts all date strings to UTC, handling both timezone-aware and timezone-naive dates.
    4. **Extract Date Component**: Removes time and timezone information, retaining only the date in 'YYYY-MM-DD' format.
    5. **Remove Invalid Dates**: Drops rows where the date parsing failed.
    6. **Sort DataFrame**: Sorts the DataFrame by the normalized date in descending order.
    7. **Clean Up**: Replaces the original date column with the normalized dates and removes any auxiliary columns.

    Parameters:
    ----------
    df : pd.DataFrame
        The input DataFrame containing the date column to normalise.

    date_column : str, default 'Date'
        The name of the column containing date strings to be normalised.

    Returns:
    -------
    pd.DataFrame
        The DataFrame with the normalised date column and invalid date rows removed.
    """

    # Step 1: Check that the specified date column exists
    if date_column not in df.columns:
        raise ValueError(f"The DataFrame does not contain a '{date_column}' column.")

    # Step 2: Remove any leading or trailing spaces
    df[date_column] = df[date_column].astype(str).str.strip()

    # Step 3: Parse dates using dateutil.parser.parse and convert to UTC
    parsed_dates = []
    for idx, date_str in df[date_column].items():
        if pd.isna(date_str) or date_str == "":
            # Handle NaN and empty strings
            parsed_dates.append(pd.NaT)
            continue
        try:
            # Parse the date string with fuzzy parsing to handle various formats
            dt = parser.parse(date_str, fuzzy=True)
            if dt.tzinfo is None:
                # If no timezone info, assume UTC
                dt = dt.replace(tzinfo=pytz.UTC)
            else:
                # Convert to UTC
                dt = dt.astimezone(pytz.UTC)
            parsed_dates.append(dt)
        except (ValueError, TypeError):
            # If parsing fails, append NaT
            parsed_dates.append(pd.NaT)

    df["Date_Parsed"] = parsed_dates

    # Step 4: Extract the date component and remove timezone and time information
    df["Date_Normalised"] = df["Date_Parsed"].dt.date.astype(str)

    # Step 5: Remove rows with invalid dates (NaT)
    df_cleaned = df.dropna(subset=["Date_Parsed"]).copy()

    # Step 6: Sort the DataFrame by the normalized date in descending order
    df_cleaned = df_cleaned.sort_values(
        by="Date_Normalised", ascending=False
    ).reset_index(drop=True)

    # Step 7: Replace the original date column with the normalized dates
    df_cleaned[date_column] = df_cleaned["Date_Normalised"]
    # Drop auxiliary columns
    df_final = df_cleaned.drop(columns=["Date_Parsed", "Date_Normalised"])

    return df_final

#
# -----------------------------------------------------------------------------------
# Ticker Handling                                                                   |
# -----------------------------------------------------------------------------------
# Manage and validate tickers, ensuring that only valid and correctly formatted tickers are processed as well as any necessary FX rate tickers.
#
@log_function  # Decorator to log entry and exit of functions.
def get_tickers(config: Dict[str, Any], set_maximum_tickers=3) -> Set[valid_ticker]:
    """
    Get the list of tickers from the configuration, including currency pairs.
    """
    # Get tickers from config
    stock_tickers = list(config["tickers"].keys())

    if not stock_tickers:
        logger.error("No tickers defined in YAML file. Exiting.")
        sys.exit(1)
    else:
        logger.info(
            f"Received {len(stock_tickers)} unvalidated stock tickers:\n{f'First {set_maximum_tickers}' if set_maximum_tickers<len(stock_tickers) else ''}: {list(stock_tickers)[:set_maximum_tickers]}{f'...' if len(stock_tickers)>set_maximum_tickers else ''}"
        )

    # Validate stock tickers
    logger.info("Validating stock tickers...")
    valid_tickers, invalid_tickers = validate_tickers(stock_tickers)

    if not valid_tickers:
        logger.error("No valid tickers in YAML file. Exiting.")
        sys.exit(1)
    else:
        logger.info(
            f"Received {len(valid_tickers)} validated stock tickers:\n{f'First {set_maximum_tickers}' if set_maximum_tickers<len(valid_tickers) else ''}: {list(valid_tickers)[:set_maximum_tickers]}{f'...' if len(valid_tickers)>set_maximum_tickers else ''}"
        )

    # iterate through the currencies to extract unique currencies, excluding GBP and GBp
    currencies = { currency for _, _, currency in valid_tickers if currency not in ["GBP", "GBp"]}

    logger.info(
        f"Obtaining {len(currencies)} FX tickers for currencies: {currencies}"
    )

    # Get list of FX tickers for those currencies we have
    fx_tickers = []  # create empty list
    for currency in currencies:
        if currency == "USD":
            fx_tickers.append("GBP=X")
        else:
            fx_tickers.append(f"{currency}GBP=X")

    # join the elements of the list using a list comprehension:
    fx_tickers_list = ", ".join(str(ticker) for ticker in fx_tickers)
    logger.info(f"Got {len(fx_tickers)} FX tickers: {fx_tickers_list}")

    # Validate FX tickers
    logger.info("Validating FX tickers...")
    valid_FX_tickers, invalid_FX_tickers = validate_tickers(fx_tickers)

    if not valid_FX_tickers:
        logger.error("No valid FX tickers. Can only process GBP stock")
        return

    # Combine stock and FX ticker lists
    all_tickers = valid_tickers + valid_FX_tickers
    # Remove duplicates
    all_tickers = set(ticker for ticker in all_tickers)

    logger.debug(
        f"Received {len(all_tickers)} stock and FX tickers{f' - First {set_maximum_tickers}' if len(all_tickers)>set_maximum_tickers else ''}: {list(all_tickers)[:set_maximum_tickers]}{f'...' if len(all_tickers)>set_maximum_tickers else ''}"
    )

    return all_tickers


def validate_tickers(
    tickers: list[str], set_maximum_tickers=5
) -> tuple[list[tuple[str, str, str]], list[str]]:
    """
    Validates a list of ticker symbols and returns list of valid ticker tuples and list of invalid tickers strings.

    Args:
        tickers: A list of ticker symbols to validate.

    Returns:
        A tuple containing two lists:
            - The first list contains tuples of valid tickers (symbol, quoteType, currency).
            - The second list contains invalid ticker symbols.
    """
    valid_tickers = []
    invalid_tickers = []

    for ticker_symbol in tickers:
        try:
            dat = get_ticker(ticker_symbol)  # Get yFinance object
            info = dat.info  # Get info from yf object

            # Retrieve data with default values for missing keys
            quote_type = info.get("quoteType", None)
            currency = info.get("currency", None)
            # Make all tickers uppercase whist we have it as a string
            ticker = info.get("symbol", ticker_symbol)
            if ticker:
                ticker = ticker.upper()

            # Only append to valid_tickers list if all values are present
            if ticker and quote_type and currency:
                new_valid_ticker = valid_ticker(ticker, quote_type, currency)
                valid_tickers.append(new_valid_ticker)
            else:
                # Only append to invalid_tickers list if there is a ticker
                if ticker:
                    invalid_tickers.append(ticker)
                    warn = "missing quote type" if not quote_type else ""
                    warn2 = "missing currency" if not currency else ""
                    logger.warning(
                        f"Ticker {ticker} is invalid due to {warn if warn else ''}{' and ' if warn and warn2 else ''}{warn2 if warn2 else ''}."
                    )

        except Exception as e:
            if ticker:
                logger.warning(f"Ticker {ticker} is invalid: {e}")
                invalid_tickers.append(ticker)

    # Get a string of all valid ticker names for reporting
    if not valid_tickers:
        logger.error("No valid tickers found.")
        short_valid_ticker_list = ""
    else:
        # Extract names of valid tickers
        valid_ticker_names = [
            vt.ticker for vt in valid_tickers
        ]  # List comprehension for names
        num_valid_tickers = len(valid_ticker_names)
        # Restrict length of ticker output
        short_valid_ticker_list = f"{f'First {set_maximum_tickers} valid' if set_maximum_tickers<num_valid_tickers else 'Valid'}: {list(valid_ticker_names)[:set_maximum_tickers]}{f'...' if num_valid_tickers>set_maximum_tickers else ''}"

    # Get a string of all invalid ticker names
    num_invalid_tickers = len(invalid_tickers)
    if num_invalid_tickers > 0:
        # Restrict length of ticker output
        short_invalid_ticker_list = f"{f'\nFirst {set_maximum_tickers} invalid' if set_maximum_tickers<num_invalid_tickers else '\nInvalid'}: {list(invalid_tickers)[:set_maximum_tickers]}{f'...' if num_invalid_tickers>set_maximum_tickers else ''}"
    else:
        short_invalid_ticker_list =""

    # Report outcome
    logger.info(
        f"Validation summary - Valid: {num_valid_tickers}, Invalid: {num_invalid_tickers}"
    )
    logger.debug(f"{short_valid_ticker_list}{short_invalid_ticker_list}")

    return valid_tickers, invalid_tickers


#
# -----------------------------------------------------------------------------------
# Data Fetching and Caching                                                         |
# -----------------------------------------------------------------------------------
# Fetch historical price data for each ticker, utilizing caching to minimize redundant
# API calls and implement rate limiting to adhere to API usage policies.
#
class YFinanceSimulator:
    """A simulator class that mimics the behavior of the yfinance library by loading data from a JSON file (simulator_data.json).
    It's used when use_simulator is set to true in the configuration."""

    def __init__(self, ticker: str):
        # Initialise the simulator with data for a specific ticker
        self.ticker = ticker.upper()
        self.data = self.load_data()
        logger.debug(f"Getting simulated data for {self.ticker}")
        if self.ticker not in self.data.get("tickers", {}):
            raise ValueError(f"No data found in simulator for {self.ticker}")
        ticker_data = self.data["tickers"][self.ticker]
        self.info = ticker_data.get("info", {})
        history_records = ticker_data.get("history", [])
        if history_records:
            df = pd.DataFrame.from_records(history_records)
            df["Date"] = pd.to_datetime(df["Date"])
            df.set_index("Date", inplace=True)
            self._history = df
        else:
            self._history = pd.DataFrame()

    def load_data(self) -> Dict[str, Any]:
        # Load simulated data from the JSON file.
        with open("simulator_data.json", "r") as f:
            return json.load(f)

    def history(self, period: str = "1mo", start=None, end=None) -> pd.DataFrame:
        # Retrieves data within a specified date range.
        if start or end:
            # Filter the history based on start and end dates
            df = self._history
            if start:
                df = df[df.index >= pd.to_datetime(start)]
            if end:
                df = df[df.index <= pd.to_datetime(end)]
            return df
        else:
            return self._history


@retry(Exception, tries=3)
def get_ticker(ticker_symbol: str) -> Optional[yf.Ticker]:
    """
    Fetches ticker data from Yahoo Finance or a simulator based on configuration.

    Args:
        ticker_symbol: The ticker symbol to fetch data for.

    Returns:
        A yfinance Ticker object if successful, None otherwise.
    """
    try:
        if config.get("use_simulator", False):
            ticker = YFinanceSimulator(ticker_symbol)
        else:
            ticker = yf.Ticker(ticker_symbol)

        return ticker
    except Exception as e:
        if "404" in str(e) or "Client" in str(e):
            logger.error(f"Invalid ticker symbol: {ticker_symbol}")
        else:
            logger.error(f"Error fetching ticker data for {ticker_symbol}: {e}")

        return None

@retry(Exception, tries=3)
def fetch_ticker_history(
    ticker_symbol: str, start_date: datetime, end_date: datetime
) -> pd.DataFrame:
    """
    Fetches price/ rate data for a single ticker.
    First checks if the data is cached; if not,
    fetches the data using the appropriate ticker object and caches it.
    """
    cache_dir = os.path.join(base_directory, "cache")
    if not os.path.exists(cache_dir):
        os.makedirs(cache_dir)
    cache_file = os.path.join(
        cache_dir,
        f"{ticker_symbol}_{start_date.strftime('%Y%m%d')}_{end_date.strftime('%Y%m%d')}.pkl",
    )
    if os.path.exists(cache_file):
        with open(cache_file, "rb") as f:
            df = pickle.load(f)
        logger.debug(f"Using cache file {format_path(cache_file)}")
    else:
        ticker = get_ticker(ticker_symbol)
        time.sleep(0.2)  # Rate limiting
        df = ticker.history(
            start=start_date.strftime("%Y-%m-%d"),
            end=end_date.strftime("%Y-%m-%d"),
            interval="1d",
        ).rename(columns={"Close": "Price"})
        if df.empty:
            logger.warning(f"No data found online for {ticker_symbol}")
            return pd.DataFrame()
        else:
            logger.debug(f"Downloaded historical data for {ticker_symbol}")

        with open(cache_file, "wb") as f:# With automatically closes file
            pickle.dump(df, f)
            logger.debug(f"Save cache to {format_path(cache_file)}")

    # Drop superfluous columns
    df = df.reset_index()  # make the index a regular column named "Date"
    tidy_df = df[['Date','Price']]

    logger.debug(
        f" Got dataframe for {ticker_symbol} of {tidy_df.shape[0]} rows and {tidy_df.shape[1]} columns ({list(tidy_df.columns)})"
    )
    return tidy_df  # returns a pandas df


@log_function
def fetch_historical_data(
    tickers: Set[valid_ticker],
    start_date: datetime,
    end_date: datetime,
    business_days: int,
) -> pd.DataFrame:
    """
    Iterates over the list of valid tickers, fetching their data using fetch_ticker_history,
    and compiles the results into a single DataFrame with columns: 'Ticker', 'Price', 'Date', 'QuoteType', 'Currency'.
    """
    records = []
    for ticker_symbol, quote_type, currency in tickers:
        try:
            # Get pandas df of historical data
            df = fetch_ticker_history(ticker_symbol, start_date, end_date)
            if df.empty:
                continue
            df["Ticker"] = ticker_symbol
            df["QuoteType"] = quote_type
            df["Currency"] = currency
            records.append(df)
        except Exception as e:
            logger.warning(f"Failed to fetch data for ticker {ticker_symbol}: {e}")

    if records:
        # combine multiple DataFrames into a single DataFrame.
        combined_df = pd.concat(
            records, ignore_index=True
        )  
        # Normalise dates
        combined_df = normalise_dates(combined_df, date_column="Date")

        # Reorder columns
        combined_df = combined_df[["Ticker", "Price", "Date", "QuoteType", "Currency"]]

        logger.info(
                f"Got dataframe for {business_days} business days between {start_date.strftime("%d/%m/%Y")} and {end_date.strftime("%d/%m/%Y")}\n      {combined_df.shape[0]} rows and {combined_df.shape[1]} columns ({list(combined_df.columns)})"
            )
        return combined_df
    else:
        logger.error(
            f"Did not obtain data for the {business_days} business days between {start_date.strftime("%d/%m/%Y")} and {end_date.strftime("%d/%m/%Y")}!"
        )
    return pd.DataFrame()  # return an empty DataFrame


#
# -----------------------------------------------------------------------------------
# Data Processing                                                                    |
# -----------------------------------------------------------------------------------
# Purpose: Process the raw fetched data by performing currency conversions and organizing the data for output.
#
@log_function
def convert_prices(ticker_dataframe: pd.DataFrame) -> pd.DataFrame:
    """
    Converts the Price column to GBP based on the Currency and QuoteType columns.
    Rows with QuoteType 'CURRENCY' or 'INDEX' are included in the final dataframe unchanged.

    Parameters:
    - ticker_dataframe (pd.DataFrame): The input dataframe containing ticker information.

    Returns:
    - pd.DataFrame: A new dataframe with prices converted to GBP and including all original rows.
    """
    # Step 1: Separate exchange rates from the main dataframe
    exchange_rates_df = ticker_dataframe[ticker_dataframe["QuoteType"] == "CURRENCY"]

    # Step 2: Create a mapping for exchange rates
    # Key: (Date, Ticker), Value: Price (exchange rate)
    exchange_rate_map = exchange_rates_df.set_index(["Date", "Ticker"])[
        "Price"
    ].to_dict()

    # Step 3: Identify rows that require conversion and those that don't
    # Rows to convert: QuoteType not in ['CURRENCY', 'INDEX']
    rows_to_convert = ticker_dataframe[
        # note: The ~ operator negates the boolean Series!
        ~ticker_dataframe["QuoteType"].isin(["CURRENCY", "INDEX"])
    ].copy()

    # Rows to keep as-is: QuoteType in ['CURRENCY', 'INDEX']
    rows_to_keep = ticker_dataframe[
        ticker_dataframe["QuoteType"].isin(["CURRENCY", "INDEX"])
    ].copy()

    # Initialize a list to store converted rows
    converted_rows = []

    # Step 4: Iterate over each row to perform conversion
    for idx, row in rows_to_convert.iterrows():
        currency = row["Currency"]
        date = row["Date"]
        price = row["Price"]
        ticker = row["Ticker"]

        try:
            if pd.isna(price):
                logger.error(
                    f"Missing price for Ticker '{ticker}' on {date.date()}. Skipping this row."
                )
                continue  # Skip this row and continue processing

            if currency == "GBP":
                # No conversion needed
                price_gbp = price
            elif currency == "GBp":
                # Convert GBp to GBP by dividing by 100
                price_gbp = price / 100
            else:
                # Determine the exchange rate ticker based on Currency
                if currency == "USD":
                    exchange_ticker = "GBP=X"
                else:
                    exchange_ticker = f"{currency}GBP=X"

                # Retrieve the exchange rate using (Date, exchange_ticker)
                exchange_rate_key = (date, exchange_ticker)
                exchange_rate = exchange_rate_map.get(exchange_rate_key)

                if exchange_rate is not None:
                    price_gbp = price * exchange_rate
                else:
                    # Exchange rate not found for the given date and ticker
                    logger.warning(
                        f"Exchange rate for ticker '{exchange_ticker}' on {date.date()} not found. "
                        f"Skipping row with Ticker '{ticker}'."
                    )
                    continue  # Skip this row

            # Create a new row with the converted price
            new_row = row.copy()
            new_row["Price"] = price_gbp
            new_row["Currency"] = "GBP"

            # Append the converted row to the list
            converted_rows.append(new_row)

        except Exception as e:
            # Log the error and skip the row
            logger.error(
                f"Error converting price for Ticker '{ticker}' on {date}: {e}"
            )
            continue  # Skip this row and continue processing

    # Step 5: Create a dataframe from converted rows
    if converted_rows:
        converted_df = pd.DataFrame(converted_rows)
    else:
        converted_df = pd.DataFrame(columns=rows_to_convert.columns)
        logger.warning("No rows were converted.")

    # Step 6: Combine converted rows with rows to keep as-is
    final_dataframe = pd.concat([converted_df, rows_to_keep], ignore_index=True)

    # Optional: Reset index if necessary (already handled by ignore_index=True)

    return final_dataframe


#
# -----------------------------------------------------------------------------------
# Output Generation                                                                 |
# -----------------------------------------------------------------------------------
# Purpose: Save the processed data to a CSV file without headers and log the file path.
#


@log_function
def save_to_csv(df: pd.DataFrame, output_file: str):
    """
    Save the DataFrame to a CSV file without headers.
    """
    df2 = df[["Ticker", "Price", "Date"]]
    # Format date as 'dd/mm/yyyy'
    df2["Date"] = df2["Date"].dt.strftime("%d/%m/%Y")
    df2.to_csv(output_file, index=False, header=False)
    parent_dir = os.path.basename(os.path.dirname(output_file))
    filename = os.path.basename(output_file)
    logger.info(f"Data saved to: {parent_dir}/{filename}")
    return True


#
#
# -----------------------------------------------------------------------------------
# Cache Cleanup                                                                     |
# -----------------------------------------------------------------------------------
# Purpose: Manage the cache by deleting outdated cached files to conserve storage and maintain performance.
#
@log_function
def clean_cache(cache_dir: str, max_age_days: int):
    """
    Clean up cache files older than max_age_days.
    """
    now = time.time()
    cutoff = now - (max_age_days * 86400)  # 86400 seconds in a day
    try:
        for filename in os.listdir(cache_dir):
            file_path = os.path.join(cache_dir, filename)
            if os.path.isfile(file_path) and os.stat(file_path).st_mtime < cutoff:
                os.remove(file_path)
                parent_dir = os.path.basename(os.path.dirname(file_path))
                file_name = os.path.basename(file_path)
                logger.info(
                    f"{parent_dir}/{file_name} older than {max_age_days} days, so deleted!"
                )
    except Exception as e:
        logger.error(f"Error cleaning cache: {e}")
        logger.error(traceback.format_exc())
        return False
    return True


#
# -----------------------------------------------------------------------------------
# Quicken UI                                                                        |
# -----------------------------------------------------------------------------------
# Seeks elevation of Quicken XG 2004 to have it import the csv file.


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


def _setup_pyautogui(self):
    """
    Configure PyAutoGUI safety settings for automated GUI interaction.
    Sets fail-safe and timing parameters to ensure reliable automation.
    """
    pyautogui.FAILSAFE = True  # Move mouse to corner to abort
    pyautogui.PAUSE = 0.5  # Delay between actions


def _is_elevated(self):
    """Check if script is running with admin privileges."""
    try:
        return ctypes.windll.shell32.IsUserAnAdmin()
    except:
        return False


def handle_import(self):
    """
    Main entry point for handling Quicken import.
    """
    try:
        self.formatter.print_section("\nImport to Quicken 2004")
        if self._is_elevated():
            return self._execute_import_sequence()
        else:
            return self._show_elevation_message()

    except Exception as e:
        self.logger.error(f"Error during Quicken import: {e}")
        return False

    finally:
        pyautogui.FAILSAFE = True


def _show_elevation_message(self):
    """Display message when running without elevation."""
    self.formatter.capture_print("Quicken cannot be opened from here.\n\nInstead:\n")
    self.formatter.capture_print("1. Close this window.")
    self.formatter.capture_print("2. Click the Quicken 'Update Prices' shortcut.\n")
    self.formatter.print_section("END", major=True)
    return True


def _execute_import_sequence(self):
    """
    Execute the sequence of steps for importing data.

    Returns:
        bool: True if all steps completed successfully
    """
    steps = [
        (self._open_quicken, "Opening Quicken..."),
        (self._navigate_to_portfolio, "Navigating to Portfolio view..."),
        (self._open_import_dialog, "Opening import dialog..."),
        (self._import_data_file, "Importing data file..."),
    ]

    for step_function, message in steps:
        self.logger.info(message)
        if not step_function():
            return False

    # Log successful completion of entire sequence
    self.logger.info(
        f"Successfully imported {self.config.data_file_name} to Quicken at {datetime.now().strftime('%d-%m-%Y %H:%M')}"
    )
    self.formatter.capture_print("\nImport complete!")
    return True


def _open_quicken(self):
    """
    Launch Quicken application.

    Returns:
        bool: True if Quicken started successfully
    """
    try:
        subprocess.Popen([self.config.QUICKEN_PATH])
        time.sleep(8)  # Allow time for Quicken to start
        return True
    except Exception as e:
        self.logger.error(f"Failed to open Quicken: {e}")
        return False


def _navigate_to_portfolio(self):
    """
    Navigate to the portfolio view in Quicken.

    Returns:
        bool: True if navigation successful
    """
    try:
        pyautogui.hotkey("ctrl", "u")
        time.sleep(1)
        return True
    except Exception as e:
        logging.error(f"Failed to navigate to portfolio: {e}")
        return False


def _open_import_dialog(self):
    """
    Open the import dialog window using keyboard shortcuts.

    Returns:
        bool: True if dialog opened successfully
    """
    try:
        with pyautogui.hold("alt"):
            pyautogui.press(["f", "i", "i"])
        time.sleep(1)
        return True
    except Exception as e:
        logging.error(f"Failed to open import dialog: {e}")
        return False


def _import_data_file(self):
    """
    Import the data file through the import dialog.

    Returns:
        bool: True if import successful
    """
    try:
        filename = f"{self.config.PATH}data.csv"
        pyautogui.typewrite(filename, interval=0.03)
        time.sleep(1)
        pyautogui.press("enter")
        time.sleep(5)
        return True
    except Exception as e:
        logging.error(f"Failed to import data file: {e}")
        return False


#
# -----------------------------------------------------------------------------------
# Main Execution Flow                                                               |
# -----------------------------------------------------------------------------------
# Purpose: Coordinate the overall workflow of the script, orchestrating the fetching, processing, and saving of data.
#
@log_function
def main():
    """
    Main function to orchestrate data fetching and processing.
    """
    try:
        # Get elevation
        # run_as_admin()

        # Clear screen for clean output
        os.system("cls" if os.name == "nt" else "clear")
        logger.debug("Starting program...")

        # Get date range
        start_date, end_date, business_days = get_date_range()

        # Get tickers - returns a set of validated ticker namedtuples for defined stocks and currency FX rates required
        tickers = get_tickers(config) 
        if not tickers:  # checks for Falsy values eg empty set.
            logger.error("A serious error has occurred with no data being returned!")
            exit(1)

        # Fetch Ticker Dataframe
        ticker_dataframe = fetch_historical_data(
            tickers, start_date, end_date, business_days
        )
        if ticker_dataframe.empty: # checks for empty Pandas DataFrame
            logger.error(
                "No data fetched from fetch_historical_data for any tickers. Exiting."
            )
            exit(1)
            return

        print(f"ticker_dataframe: {ticker_dataframe}")
        
        # Check data types of all columns
        print(ticker_dataframe.dtypes)

        # Specifically check the 'Date' column
        print(ticker_dataframe['Date'].dtype)
        
        # Identify rows where 'Date' is not a datetime object
    problematic_rows = ticker_dataframe[
        ticker_dataframe['QuoteType'].isin(['ETF', 'EQUITY']) &  # Assuming these require conversion
        ticker_dataframe['Ticker'].isin(['GL1S.MU', 'LAU.AX']) &  # Specific tickers causing issues
        ticker_dataframe['Date'].apply(lambda x: isinstance(x, str))  # 'Date' is string
    ]

    print(problematic_rows)

        # Convert prices to GBP except INDEX and CURRENCY
        ticker_dataframe = convert_prices(ticker_dataframe)

        # processed_data = convert_prices(ticker_dataframe, exchange_rates)
        # if processed_data.empty:
        #     logger.error(
        #         "Something went wrong when converting prices.No data returned! Exiting."
        #     )
        #     exit(1)
        #     return
        # else:
        #     logger.info("Prices converted")

        # # Save data to CSV
        # output_file = os.path.join(
        #     base_directory, config.get("paths", {}).get("output_file", "data.csv")
        # )
        # logger.info(f"Saving to {output_file}")
        # save = save_to_csv(processed_data, output_file)
        # if save == True:
        #     logger.info("File saved")
        # else:
        #     logger.error(f"Something went wrong. {output_file} not saved!")

        # # Clean up cache
        # logger.info(f"Cleaning out cache")
        # cache_dir = os.path.join(base_directory, "cache")
        # clean = clean_cache(
        #     cache_dir, max_age_days=fetch_config.get("cache_max_age_days", 7)
        # )
        # if clean == True:
        #     logger.info("Cache cleaned")
        # else:
        #     logger.error(f"Something went wrong cleaning cache")

        # # Open Quicken
        # logger.info(f"Opening Quicken")
        # quicken_process = subprocess.Popen(["notepad.exe"], shell=True)

        # # Temporarily disable Ctrl+Alt+Del to prevent accidental interruption
        # pyautogui.hotkey("ctrl", "alt", "del")
        # # quicken does stuff

        # # Give Quicken some time to open
        # time.sleep(100)

        # navigate in quicken
        #
        #

        # logging.info("Waiting for user to close Quicken...")
        # # Renable Ctrl+Alt+Del
        # pyautogui.hotkey("ctrl", "alt", "del")
        # pyautogui.write("Hello")
        # pyautogui.hotkey("ctrl", "alt", "del")
        # # Wait for quicken to close
        # quicken_process.wait()

        # logging.info("Quicken closed by user.")
        # logging.info("Keeping terminal open for 10 seconds...")
        # time.sleep(10)

    except KeyboardInterrupt:
        logger.warning("Script interrupted by user.")
    except Exception as e:
        logger.error(f"An unexpected error occurred: {e}")
        logger.error(traceback.format_exc())


if __name__ == "__main__":
    # Starting script
    logger.info("Starting program")
    main()

    # Retrieve log messages from StringIO
    # terminal_output = log_capture_string.getvalue()
    # copy_to_clipboard(terminal_output)
    # logger.info("Terminal output copied to clipboard.")

    logger.info("Program finished")
#
# -----------------------------------------------------------------------------------
# END                                                                               |
# -----------------------------------------------------------------------------------
