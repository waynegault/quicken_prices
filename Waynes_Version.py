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
# -----------------------------------------------------------------------------------
# Imports                                                                           |
# -----------------------------------------------------------------------------------
#
try:
    # Standard Library Imports
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
    from datetime import datetime, timedelta  # Classes for working with dates and times
    from pathlib import Path  # Object-oriented interface to filesystem paths

    # Third-Party Library Imports
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
    from typing import List, Iterator, Optional, Dict, Any  # Type hints
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
        # Currency pairs available from Yahoo Finance
        "currency_pairs": {
            "EURGBP=X": "Euro to British Pound",
            "GBP=X": "US Dollar to British Pound",
            "JPYGBP=X": "Japanese Yen to British Pound",
            "AUDGBP=X": "Australian Dollar to British Pound",
            "CADGBP=X": "Canadian Dollar to British Pound",
            "NZDGBP=X": "New Zealand Dollar to British Pound",
            "CHFGBP=X": "Swiss Franc to British Pound",
            "CNYGBP=X": "Chinese Yuan to British Pound",
            "HKDGBP=X": "Hong Kong Dollar to British Pound",
            "SGDGBP=X": "Singapore Dollar to British Pound",
            "INRGBP=X": "Indian Rupee to British Pound",
            "MXNGBP=X": "Mexican Peso to British Pound",
            "PHPGBP=X": "Philippine Peso to British Pound",
            "MYRGBP=X": "Malaysian Ringgit to British Pound",
            "ZARGBP=X": "South African Rand to British Pound",
            "RUBGBP=X": "Russian Ruble to British Pound",
            "TRYGBP=X": "Turkish Lira to British Pound",
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

# Ensure the base directory exists
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
        logger.debug(f"Entering function: {func.__name__}")
        try:
            result = func(*args, **kwargs)
            logger.debug(f"Succeeded, exiting function: {func.__name__}")
            return result
        except Exception as e:
            logger.error(f"Failed, error: {e}\nexiting function {func.__name__}")
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


#
# -----------------------------------------------------------------------------------
# Ticker Handling                                                                   |
# -----------------------------------------------------------------------------------
# Manage and validate tickers, ensuring that only valid and correctly formatted tickers are processed as well as any necessary FX rate tickers.
#
@log_function  # Decorator to log entry and exit of functions.
def get_tickers(config: Dict[str, Any]) -> List[str]:
    """
    Get the list of tickers from the configuration, including currency pairs.
    """
    # Get tickers from config
    logger.info("Getting tickers...")
    stock_tickers = config["tickers"].keys()
    if not stock_tickers:
        logger.error("No tickers defined in YAML file. Exiting.")
        sys.exit(1)
    else:
        logger.info("Got tickers")

    # Validate stock tickers
    logger.info("Validating stock tickers...")
    valid_tickers, invalid_tickers = validate_tickers(stock_tickers)

    if not valid_tickers:
        logger.error("No valid tickers in YAML file. Exiting.")
        sys.exit(1)
    else:
        num_valid_tickers = len(valid_tickers)
        num_invalid_tickers = len(invalid_tickers)
        valid_ticker_list = ", ".join(str(ticker) for ticker in valid_tickers) if num_valid_tickers > 0 else "none"
        invalid_ticker_list = ", ".join(str(ticker) for ticker in invalid_tickers) if num_invalid_tickers > 0 else "none"    

        valid_plural = "" if num_valid_tickers == 1 else "s"
        invalid_plural = "" if num_invalid_tickers == 1 else "s"

        logger.info(f"{num_valid_tickers} valid stock ticker{valid_plural} {valid_ticker_list}, and {num_invalid_tickers} invalid stock ticker{invalid_plural} {invalid_ticker_list}.")

    # Extract unique currencies, excluding GBP and GBp
    currencies = {
        currency for _, _, currency in valid_tickers if currency not in ["GBP", "GBp"]
    }
    logger.info(
        f"Obtaining {len(currencies)} FX tickers for these currencies, {currencies}"
    )
    # Map currencies to their corresponding FX tickers
    fx_tickers = {}
    for currency in currencies:
        if currency == "USD":
            fx_tickers[currency] = "GBP=X"
        else:
            fx_tickers[currency] = f"{currency}GBP=X"

    # Validate FX tickers
    logger.info("Validating FX tickers...")
    valid_FX_tickers, invalid_FX_tickers = validate_tickers(fx_tickers[currency])
    # Extract ticker only
    valid_FX_tickers = [ticker for ticker, _, _ in valid_FX_tickers]

    if not valid_FX_tickers:
        logger.error("No valid FX tickers. Can only process GBP stock")
        return
    else:
        num_valid_FX_tickers = len(valid_FX_tickers)
        num_invalid_FX_tickers = len(invalid_FX_tickers)
        valid_FX_ticker_list = ", ".join(str(ticker) for ticker in valid_FX_tickers) if num_valid_FX_tickers > 0 else "none"
        invalid_FX_ticker_list = (
            ", ".join(str(ticker) for ticker in invalid_FX_tickers)
            if num_invalid_FX_tickers > 0
            else "none"
        )

        valid_FX_plural = "" if num_valid_FX_tickers == 1 else "s"
        invalid_FX_plural = "" if num_invalid_FX_tickers == 1 else "s"

        logger.info(
            f"{num_valid_FX_tickers} valid stock ticker{valid_FX_plural} {valid_FX_ticker_list}, and {num_invalid_FX_tickers} invalid stock ticker{invalid_FX_plural} {invalid_FX_ticker_list}."
        )

    # Combine stock and FX ticker lists
    all_tickers = valid_tickers + valid_FX_tickers
    # Remove duplicates and ensure tickers are uppercase
    all_tickers = set(ticker for ticker in all_tickers)
    return all_tickers

def validate_tickers(
    tickers: list[str],
) -> tuple[list[tuple[str, str, str]], list[str]]:
    """
    Validates a list of ticker symbols and returns valid and invalid tickers.

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
            ticker = get_ticker(ticker_symbol)  # see Data Fetching and Caching
            info = ticker.info

            quote_type = info.get("quoteType", None)
            currency = info.get("currency", None)
            ticker = info.get("symbol", None)
            # Make all tickers uppercase whist we have it as a string
            ticker = ticker.upper()

            if quote_type and currency:
                valid_tickers.append((ticker, quote_type, currency))
            else:
                invalid_tickers.append(ticker)
                if not quote_type:
                    logger.warning(f"Ticker {ticker} is missing quote type.")
                if not currency:
                    logger.warning(f"Ticker {ticker} is missing currency.")
                logger.warning(f"Ticker {ticker} is invalid.")

        except Exception as e:
            logger.warning(f"Ticker {ticker} is invalid: {e}")
            invalid_tickers.append(ticker)

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
@log_function
def get_ticker(ticker_symbol: str):
    """
    Get ticker object from yfinance or simulator based on configuration.
    """
    if config.get("use_simulator", False):
        return YFinanceSimulator(ticker_symbol)
    else:
        # Get ticker info dictionary from Yahoo
        dat = yf.Ticker(ticker_symbol)
        return dat


@retry(Exception, tries=3)
@log_function
def fetch_ticker_history(
    ticker_symbol: str, start_date: datetime, end_date: datetime
) -> pd.DataFrame:
    """
    Fetches metadata for a single ticker.
    It first checks if the data is cached; if not,
    it fetches the data using the appropriate ticker object and caches it.
    """
    cache_dir = os.path.join(base_directory, "cache")
    if not os.path.exists(cache_dir):
        os.makedirs(cache_dir)
    cache_file = os.path.join(
        cache_dir,
        f"{ticker_symbol}_{start_date.strftime('%Y%m%d')}_{end_date.strftime('%Y%m%d')}.pkl",
    )
    if os.path.exists(cache_file):
        logger.debug(f"Loading cached data for {ticker_symbol}")
        with open(cache_file, "rb") as f:
            df = pickle.load(f)
    else:
        ticker = get_ticker(ticker_symbol)
        time.sleep(0.2)  # Rate limiting
        df = ticker.history(
            start=start_date.strftime("%Y-%m-%d"), end=end_date.strftime("%Y-%m-%d")
        )
        if df.empty:
            logger.warning(f"No data found online for {ticker_symbol}")
            return pd.DataFrame()
        with open(cache_file, "wb") as f:
            pickle.dump(df, f)
    return df


@log_function
def fetch_data(
    tickers: List[tuple], start_date: datetime, end_date: datetime
) -> pd.DataFrame:
    """
    Iterates over the list of valid tickers, fetching their data using fetch_ticker_history,
    and compiles the results into a single DataFrame with columns: 'Ticker', 'Price', 'Date', 'QuoteType', 'Currency'.
    """
    records = []
    for ticker_symbol, quote_type, currency in tickers:
        try:
            df = fetch_ticker_history(ticker_symbol, start_date, end_date)
            if df.empty:
                continue
            df = df[["Close"]].copy()
            df.reset_index(inplace=True)
            df["Ticker"] = ticker_symbol
            df["QuoteType"] = quote_type
            df["Currency"] = currency
            records.append(df)
            logger.info(f"Fetched data for ticker {ticker_symbol}")
        except Exception as e:
            logger.warning(f"Failed to fetch data for ticker {ticker_symbol}: {e}")
    if records:
        combined_df = pd.concat(records, ignore_index=True)
        return combined_df
    else:
        return pd.DataFrame()


#
# -----------------------------------------------------------------------------------
# Data Processing                                                                    |
# -----------------------------------------------------------------------------------
# Purpose: Process the raw fetched data by performing currency conversions,
# formatting dates, and organizing the data for output.
#
@log_function
def process_data(df: pd.DataFrame, currency_rates: Dict[str, float]) -> pd.DataFrame:
    """
    Process the data to have columns: 'Ticker', 'Price', 'Date', "QuoteType", "Currency" ordered by descending date.
    """
    if df.empty:
        return df
    # Filter and reorder columns
    df = df[["Ticker", "Close", "Date", "QuoteType", "Currency"]]
    df.rename(columns={"Close": "Price"}, inplace=True)
    # Ensure 'Date' column is datetime without timezone
    df["Date"] = pd.to_datetime(df["Date"], utc=True).dt.tz_convert(None)
    # Reorder by iso date (dont order by non-iso date!)
    df.sort_values(by="Date", ascending=False, inplace=True)

    # Convert prices to GBP where applicable
    for index, row in df.iterrows():
        ticker = row["Ticker"]
        price = row["Price"]
        quote_type = row["QuoteType"]
        currency = row["Currency"]

        # For quote_type 'CURRENCY', keep in original denomination
        if quote_type == "CURRENCY":
            pass

        if currency == "GBp":
            # Convert GBp to GBP by dividing by 100
            df.at[index, "Price"] = price / 100
        elif currency != "GBP":
            # Convert price to GBP using exchange rates
            if currency == "USD":
                rate_ticker = "GBP=X"  # Ticker for USD to GBP
            else:
                rate_ticker = currency + "GBP=X"  # Correct format for other currencies
            rate = currency_rates.get(rate_ticker, None)
            if rate:
                df.at[index, "Price"] = price * rate
            else:
                logger.warning(f"Missing exchange rate for {currency} to GBP")


@log_function
def get_exchange_rates(tickers: List[str]) -> Dict[str, float]:
    """
    Get exchange rates for the required currencies.
    """
    currencies = set()
    for ticker_symbol in tickers:
        currency = get_currency_for_ticker(ticker_symbol)
        if currency not in ["GBP", "GBp"]:
            if currency == "USD":
                currencies.add("GBP=X")  # Ticker for USD to GBP
            else:
                currencies.add(
                    currency + "GBP=X"
                )  # Correct format for other currencies
    exchange_rates = {}
    for currency_pair in currencies:
        try:
            ticker = get_ticker(currency_pair)
            time.sleep(0.3)  # Rate limiting
            df = ticker.history(period="1d")
            if not df.empty:
                rate = df["Close"].iloc[-1]
                exchange_rates[currency_pair] = rate
            else:
                logger.warning(f"No data available for {currency_pair}")
        except Exception as e:
            logger.warning(f"Failed to fetch data for {currency_pair}: {e}")
    return exchange_rates


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

        # Get tickers
        tickers = get_tickers(config)

        # Calculate date range
        logger.info("Getting date range...")
        start_date = None
        end_date = None

        fetch_config = config.get("fetch", {})
        if "start_date" in fetch_config and "end_date" in fetch_config:
            try:
                start_date = datetime.datetime.strptime(
                    fetch_config["start_date"], "%Y-%m-%d"
                )
                end_date = datetime.datetime.strptime(
                    fetch_config["end_date"], "%Y-%m-%d"
                )
            except ValueError as ve:
                logger.error(
                    f"Check the YAML file. Invalid date format in configuration: {ve}"
                )
                return
        else:
            end_date = datetime.datetime.now()
            start_date = end_date - datetime.timedelta(
                days=fetch_config.get("days", 30)
            )
        logger.info("Got date range")

        # Fetch exchange rates
        logger.info(
            f"Fetching FX data from {start_date.strftime('%d/%m/%Y')} to {end_date.strftime('%d/%m/%Y')}..."
        )
        exchange_rates = get_exchange_rates([t[0] for t in valid_tickers])
        logger.info("Got FX data")

        # Fetch Ticker Data
        logger.info(
            f"Fetching price data from {start_date.strftime('%d/%m/%Y')} to {end_date.strftime('%d/%m/%Y')}"
        )
        raw_data = fetch_data(valid_tickers, start_date, end_date)
        if raw_data.empty:
            logger.error("No data fetched for any tickers. Exiting.")
            return
        else:
            logger.info("Got price data")

        # Process data
        logger.info("Converting prices...")
        processed_data = process_data(raw_data, exchange_rates)
        if processed_data.empty:
            logger.error(
                "Something went wrong when converting prices.No data returned!"
            )
            return
        else:
            logger.info("Prices converted")

        # Save data to CSV
        output_file = os.path.join(
            base_directory, config.get("paths", {}).get("output_file", "data.csv")
        )
        logger.info(f"Saving to {output_file}")
        save = save_to_csv(processed_data, output_file)
        if save == True:
            logger.info("File saved")
        else:
            logger.error(f"Something went wrong. {output_file} not saved!")

        # Clean up cache
        logger.info(f"Cleaning out cache")
        cache_dir = os.path.join(base_directory, "cache")
        clean = clean_cache(
            cache_dir, max_age_days=fetch_config.get("cache_max_age_days", 7)
        )
        if clean == True:
            logger.info("Cache cleaned")
        else:
            logger.error(f"Something went wrong cleaning cache")

        # Open Quicken
        logger.info(f"Opening Quicken")
        quicken_process = subprocess.Popen(["notepad.exe"], shell=True)

        # Temporarily disable Ctrl+Alt+Del to prevent accidental interruption
        pyautogui.hotkey("ctrl", "alt", "del")
        # quicken does stuff

        # Give Quicken some time to open
        time.sleep(100)

        # navigate in quicken
        #
        #

        logging.info("Waiting for user to close Quicken...")
        # Renable Ctrl+Alt+Del
        pyautogui.hotkey("ctrl", "alt", "del")
        pyautogui.write("Hello")
        pyautogui.hotkey("ctrl", "alt", "del")
        # Wait for quicken to close
        quicken_process.wait()

        logging.info("Quicken closed by user.")
        logging.info("Keeping terminal open for 10 seconds...")
        time.sleep(10)

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
