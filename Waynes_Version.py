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
print("Debugging: Starting program...")
# -----------------------------------------------------------------------------------
# Imports                                                                           |
# -----------------------------------------------------------------------------------
#
try:

    import yaml
    import json
    import functools
    import traceback
    from io import StringIO
    import pytz
    import pyperclip
    from colorama import init, Fore, Style  # Colored terminal text library
    import ctypes  # Provides low-level operating system interfaces, used for system-level operations
    import subprocess  # Enables spawning of new processes and interaction with their I/O
    import sys  # Provides access to Python interpreter variables and functions
    import time  # Offers time-related functions, used for delays and timing operations
    import calendar  # Provides calendar-related functions, used for date calculations
    import logging  # Implements a flexible event logging system for tracking program execution
    import logging.handlers
    from datetime import datetime, timedelta  # Classes for working with dates and times
    import hashlib  # Implements various secure hash and message digest algorithms
    import pickle  # Implements binary protocols for serializing and de-serializing Python objects
    from pathlib import Path  # Object-oriented interface to filesystem paths
    import os  # Provides a way of using operating system dependent functionality
    import uuid  # Generates universally unique identifiers (UUIDs)
    import gc  # Garbage Collector interface, used for memory management
    import threading  # Provides high-level threading interface
    from queue import Queue  # Implements multi-producer, multi-consumer queues
    from concurrent.futures import (
        ThreadPoolExecutor,
        as_completed,
    )  # Tools for asynchronous execution
    from typing import (
        List,
        Iterator,
        Optional,
        Dict,
        Any,
    )  # Type hints for better code documentation
    from dataclasses import (
        dataclass,
    )  # Decorator for automatically adding generated special methods
    from enum import Enum  # Base class for creating enumerated constants
    from contextlib import contextmanager  # Utilities for working with context managers
    from logging.handlers import RotatingFileHandler
    import pyautogui  # Provides cross-platform GUI automation tools
    import yfinance as yf  # Yahoo Finance API interface for financial data
    import pandas as pd  # Data manipulation and analysis library
    import psutil  # Cross-platform utilities for retrieving information on running processes
    import weakref  # Support for weak references and finalization

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
        
        "prices": 
            [
                {"ticker": "0P00013P6I.L", "description": "HSBC FTSE All-World Index C Acc (GBP)"},
                {"ticker": "0P00018XAP.L", "description": "Vanguard FTSE 100 Idx Unit Tr £ Acc (GBP)"},
                {"ticker": "AMGN", "description": "Amgen Inc. (USD)"},
                {"ticker": "BZ=F", "description": "Brent Crude Oil Last Day Finance (USD)"},
                {"ticker": "CUKX.L", "description": "iShares VII PLC - iShares Core FTSE 100 ETF GBP Acc (GBP)"},
                {"ticker": "CYSE.L", "description": "WisdomTree Cybersecurity UCITS ETF USD Acc (GBP)"},
                {"ticker": "ELIX.L", "description": "Elixirr International plc (GBP)"},
                {"ticker": "EURGBP=X", "description": "EUR/GBP Fx"},
                {"ticker": "fake", "description": "fake"},
                {"ticker": "^FTAS", "description": "FTSE All-share (GBP)"},    
                {"ticker": "^FTSE", "description": "FTSE 100 (GBP)"},
                {"ticker": "GBP=X", "description": "USD/GBP Fx"},
                {"ticker": "GL1S.MU", "description": "Invesco Markets II Plc - Invesco UK Gilt UCITS ETF (EUR)"},
                {"ticker": "GLTA.L", "description": "Invesco UK Gilts UCITS ETF (GBP)"},
                {"ticker": "GC=F", "description": "Gold (USD)"},
                {"ticker": "HMWO.L", "description": "HSBC MSCI World UCITS ETF (GBP)"},
                {"ticker": "IHCU.L", "description": "iShares S&P 500 Health Care Sector UCITS ETF USD (Acc) (GBP)"},
                {"ticker": "IIND.L", "description": "iShares MSCI India UCITS ETF USD Acc (GBP)"},
                {"ticker": "IUIT.L", "description": "iShares S&P 500 Information Technology Sector UCITS ETF USD (Acc) (USD)"},
                {"ticker": "LAU.AX", "description": "(AUD)"},
                {"ticker": "SEE.L", "description": "Seeing Machines Limited (GBP)"},
                {"ticker": "^OEX", "description": "S&P 100 INDEX (USD)"},
                {"ticker": "^GSPC", "description": "S&P 500 INDEX (USD)"},
                {"ticker": "SQX.CN", "description": "(CAD)"},
                {"ticker": "VHVG.L", "description": "Vanguard FTSE Developed World UCITS ETF USD Accumulation (GBP)"},
                {"ticker": "VNRX", "description": "VolitionRX Limited (USD)"},
                {"ticker": "VUKG.L", "description": "Vanguard FTSE 100 UCITS ETF GBP Accumulation (GBP)"},
            ],

        # Debug Configuration -  populated for testing specific tickers
        "debug":
            [
                {"ticker": "VHVG.L", "description": "Vanguard FTSE Developed World UCITS ETF USD Accumulation (GBP)"},
                {"ticker": "ELIX.L", "description": "Elixirr International plc (GBP)"},
                {"ticker": "VNRX", "description": "VolitionRX Limited (USD)"},
                {"ticker": "EURGBP=X", "description": "EUR/GBP Fx"},
                {"ticker": "GC=F", "description": "Gold (USD)"}
            ],
            
        "paths": 
            {
                "base": "C:\\Users\\wayne\\OneDrive\\Documents\\GitHub\\Python\\Projects\\quicken_prices\\",
                "quicken": "C:\\Program Files (x86)\\Quicken\\qw.exe",
                "data_file": "data.csv",
                "log_file": "prices.log",
                "cache": "cache",
            },
            
        # Default data collection periods for each known quoteType:
        "collection": {"period_years": 0.08, "max_retries": 3, "retry_delay": 2},
        "default_periods": 
            {
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
                    "error": "%(asctime)s %(levelname)s: %(status)s - %(filename)s:%(lineno)d - %(message)s - %(exc_info)s",
                    "warning": "%(asctime)s %(levelname)s: %(status)s - %(filename)s:%(lineno)d - %(message)s",
                    "info": "%(asctime)s %(levelname)s: %(status)s - %(message)s",
                    "debug": "%(asctime)s %(levelname)s: %(status)s - %(filename)s:%(lineno)d - %(message)s - %(funcName)s - %(module)s - %(process)d - %(thread)d - %(relativeCreated)d - %(threadName)s"
                    },
                "terminal": {
                    "error": "%(levelname)s: %(status)s - %(filename)s:%(lineno)d - %(message)s - %(exc_info)s",
                    "warning": "%(levelname)s: %(status)s - %(filename)s:%(lineno)d - %(message)s",
                    "info": "%(levelname)s: %(status)s - %(message)s",
                    "debug": "%(levelname)s: %(status)s - %(filename)s:%(lineno)d - %(message)s"
                    }
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
        "currency_pairs": 
            [
                {"ticker": "EURGBP=X", "description": "Euro to British Pound"},
                {"ticker": "GBP=X", "description": "US Dollar to British Pound"},
                {"ticker": "JPYGBP=X", "description": "Japanese Yen to British Pound"},
                {"ticker": "AUDGBP=X", "description": "Australian Dollar to British Pound"},
                {"ticker": "CADGBP=X", "description": "Canadian Dollar to British Pound"},
                {"ticker": "NZDGBP=X", "description": "New Zealand Dollar to British Pound"},
                {"ticker": "CHFGBP=X", "description": "Swiss Franc to British Pound"},
                {"ticker": "CNYGBP=X", "description": "Chinese Yuan to British Pound"},
                {"ticker": "HKDGBP=X", "description": "Hong Kong Dollar to British Pound"},
                {"ticker": "SGDGBP=X", "description": "Singapore Dollar to British Pound"},
                {"ticker": "INRGBP=X", "description": "Indian Rupee to British Pound"},
                {"ticker": "MXNGBP=X", "description": "Mexican Peso to British Pound"},
                {"ticker": "PHPGBP=X", "description": "Philippine Peso to British Pound"},
                {"ticker": "MYRGBP=X", "description": "Malaysian Ringgit to British Pound"},
                {"ticker": "ZARGBP=X", "description": "South African Rand to British Pound"},
                {"ticker": "RUBGBP=X", "description": "Russian Ruble to British Pound"},
                {"ticker": "TRYGBP=X", "description": "Turkish Lira to British Pound"},
            ]
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
    terminal_handler = (logging.StreamHandler())  
    # retrieve the desired log level from config dictionary
    terminal_level = config["logging"]["levels"]["terminal"].upper()
    # set the log level of the handler
    terminal_handler.setLevel(log_levels.get(terminal_level))
    # Create a formatter based on the terminal level
    if terminal_level == "DEBUG":
        terminal_formatter = logging.Formatter(config["logging"]["message_formats"]["terminal"]["debug"])
    elif terminal_level == "INFO":
        terminal_formatter = logging.Formatter(config["logging"]["message_formats"]["terminal"]["info"])
    elif terminal_level == "WARNING":
        terminal_formatter = logging.Formatter(config["logging"]["message_formats"]["terminal"]["warning"])
    else:
        # Default terminal formatter is error
        terminal_formatter = logging.Formatter(config["logging"]["message_formats"]["terminal"]["error"])  
    terminal_handler.setFormatter(terminal_formatter)
    # assign the formatter to the handler
    terminal_handler.setFormatter(terminal_formatter)  
    logger.addHandler(terminal_handler) #Adds terminal handler to the QuickenPrices logger.

    # Create file handler with rotating logs
    log_file_path = os.path.join(base_directory, config["paths"]["log_file"]) 
    file_handler = logging.handlers.RotatingFileHandler(
        log_file_path, 
        maxBytes=config["logging"]["max_bytes"],
        backupCount=config["logging"]["backup_count"])
    file_level = config["logging"]["levels"]["file"].upper()
    file_handler.setLevel(log_levels.get(file_level))   
    # Create a formatter based on the terminal level
    if file_level == "ERROR":
        file_formatter = logging.Formatter(config["logging"]["message_formats"]["file"]["error"])
    elif file_level == "INFO":
        file_formatter = logging.Formatter(config["logging"]["message_formats"]["file"]["info"])
    elif file_level == "WARNING":
        file_formatter = logging.Formatter(config["logging"]["message_formats"]["file"]["warning"])
    else:
        # Default formatter is debug
        file_formatter = logging.Formatter(config["logging"]["message_formats"]["file"]["debug"])  
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
# Manage and validate tickers, ensuring that only valid and correctly formatted tickers are processed.
#
@log_function  # Decorator to log entry and exit of functions.
def get_tickers(config: Dict[str, Any]) -> List[str]:
    """
    Get the list of tickers from the configuration, including currency pairs.
    """
    # Get tickers from config.
    stock_tickers = config["prices"]["tickers"]
    # Get tickers from 'currency_pairs'
    currency_tickers = config["currency_pairs"]["tickers"]
    # Combine both lists
    all_tickers = stock_tickers + currency_tickers
    # Remove duplicates and ensure tickers are uppercase
    all_tickers = list(set([ticker.upper() for ticker in all_tickers]))
    return all_tickers


@log_function  # Decorator to log entry and exit of functions.
def validate_tickers(tickers: List[str]) -> List[tuple]:
    """
    Validate tickers and categorize by quoteType.
    """
    valid_tickers = []
    invalid_tickers = []
    for ticker_symbol in tickers:
        try:
            ticker = get_ticker(ticker_symbol)  # see Data Fetching and Caching
            info = ticker.info
            quote_type = info.get("quoteType", None)
            if not quote_type:
                raise ValueError(f"Could not find a type category for {ticker_symbol}")
            valid_tickers.append((ticker_symbol, quote_type))
        except Exception as e:
            logger.warning(f"Ticker {ticker_symbol} is invalid: {e}")
            invalid_tickers.append(ticker_symbol)
    if invalid_tickers:
        logger.warning(f"Invalid tickers: {', '.join(invalid_tickers)}")
    return valid_tickers
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


def get_ticker(ticker_symbol: str):
    """
    Get ticker object from yfinance or simulator based on configuration.
    """
    if config.get("use_simulator", False):
        return YFinanceSimulator(ticker_symbol)
    else:
        return yf.Ticker(ticker_symbol)


@retry(Exception, tries=3)
@log_function
def fetch_ticker_data(
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
    Iterates over the list of valid tickers, fetching their data using fetch_ticker_data,
    and compiles the results into a single DataFrame with columns: 'Ticker', 'Price', 'Date'.
    """
    records = []
    for ticker_symbol, quote_type in tickers:
        try:
            df = fetch_ticker_data(ticker_symbol, start_date, end_date)
            if df.empty:
                continue
            df = df[["Close"]].copy()
            df.reset_index(inplace=True)
            df["Ticker"] = ticker_symbol
            df["QuoteType"] = quote_type
            records.append(df)
            
            # Reorder columns
            reordered_columns = ["Ticker", "Date", "Close", "QuoteType"]
            df = df[reordered_columns]
            
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
# Purpose: Process the raw fetched data by performing currency conversions, formatting dates, and organizing the data for output.
#
#
#
# -----------------------------------------------------------------------------------
# Output Generation                                                                 |
# -----------------------------------------------------------------------------------
# Purpose: Save the processed data to a CSV file without headers and log the file path.
#
#
#
#
# -----------------------------------------------------------------------------------
# Cache Cleanup                                                                     |
# -----------------------------------------------------------------------------------
# Purpose: Manage the cache by deleting outdated cached files to conserve storage and maintain performance.
#
#
#
#
# -----------------------------------------------------------------------------------
# Quicken UI                                                                        |
# -----------------------------------------------------------------------------------
#
#
#
#
#
# -----------------------------------------------------------------------------------
# Main Execution Flow                                                               |
# -----------------------------------------------------------------------------------
# Purpose: Coordinate the overall workflow of the script, orchestrating the fetching, processing, and saving of data.
#
#
#
#
# -----------------------------------------------------------------------------------
# END                                                                               |
# -----------------------------------------------------------------------------------
