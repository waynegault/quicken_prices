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
# Import core functionality to support wider import process

import logging  # Additional logging handlers (e.g., RotatingFileHandler)
import os  # Operating system interfaces
import sys  # System-specific parameters and functions

# Set up interim python root logger, 'logging' only for import errors prior to setting up our custom 'logger'
logging.basicConfig(
    level=logging.DEBUG,
    format="Startup: %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logging.info("Beginning script\n=========================")

# Standard Library Imports
try:
    import calendar  # Provides calendar-related functions
    import ctypes  # Low-level OS interfaces
    import functools  # Higher-order functions and decorators
    import gc  # Garbage collection interface
    import hashlib  # Cryptographic hash functions
    import importlib  # Provides utilities for importing modules dynamically
    import json  # JSON file encoding and decoding
    from logging.handlers import (
        RotatingFileHandler,
    )  # Additional logging handlers (e.g., RotatingFileHandler)
    import pickle  # Object serialization/deserialization
    import subprocess  # Process creation and management
    import threading  # High-level threading interface
    import time  # Time-related functions
    import traceback  # Exception handling and traceback printing
    import uuid  # Generate unique identifiers
    import weakref  # Support for weak references
    from collections import namedtuple  # Lightweight data structures
    from contextlib import contextmanager  # Context manager utilities
    from datetime import date, datetime, timedelta  # Date and time handling
    from pathlib import Path  # Object-oriented filesystem paths
    from queue import Queue  # Thread-safe producer-consumer queues
    from typing import (
        Any,
        Dict,
        Iterator,
        List,
        Optional,
        Set,
        Tuple,
    )  # Type annotations
    from dataclasses import dataclass  # Easy class creation for storing data
    from enum import Enum  # Define enumerations
except ImportError as e:
    logging.error(
        f"Error importing a standard library module: {e.name}. This should not happen in a standard Python installation."
    )
    sys.exit(1)

# Third-Party Library Imports with Mapping
# Format: module_name: {"install_name": pip_install_name, "import_as": alias, "from": specific_import}
third_party_libs = {
    "numpy": {"install_name": "numpy", "import_as": "np"},  # Numerical computing
    "yaml": {"install_name": "pyyaml"},  # YAML file parsing and emitting
    "pandas": {"install_name": "pandas", "import_as": "pd"},  # Data manipulation
    "box": {
        "install_name": "python-box",
        "from": ["Box"],
    },  # Dot notation for dictionaries
    "yfinance": {"install_name": "yfinance", "import_as": "yf"},  # Yahoo Finance API
    "pytz": {"install_name": "pytz"},  # Time zone handling
    "pyperclip": {"install_name": "pyperclip"},  # Clipboard operations
    "colorama": {
        "install_name": "colorama",
        "import_as": "colorama",  # Import full module for initialization purposes
        "from": ["init", "Fore", "Style"],
    },  # Terminal text styling
    "pyautogui": {"install_name": "pyautogui"},  # GUI automation
    "psutil": {"install_name": "psutil"},  # Process and system monitoring
    "tabulate": {"install_name": "tabulate",
                 "from": ["tabulate"]}
}

# Track Missing Packages
missing_packages = []


# Function to Check if pip Is Available
def is_pip_installed() -> bool:
    try:
        subprocess.run(
            [sys.executable, "-m", "pip", "--version"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=True,
        )
        return True
    except (FileNotFoundError, subprocess.CalledProcessError):
        return False


# Function to Automatically Run pip install
def install_missing_packages(packages: list):
    """
    Automatically run pip install for a list of missing packages.
    """
    if not packages:
        return  # Nothing to install

    try:
        command = [sys.executable, "-m", "pip", "install"] + packages
        print(f"Running: {' '.join(command)}")
        subprocess.run(command, check=True)
        logging.info("Successfully installed missing packages.")
    except subprocess.CalledProcessError as e:
        missing_packages_str = " ".join(packages)
        logging.error(f"Failed to install packages: {e}")
        logging.error(
            f"To manually install missing libraries, run: 'pip install {missing_packages_str}'"
        )
        sys.exit(1)


# Step 1: Detect Missing Packages Without Accessing Specific Attributes
for module_name, details in third_party_libs.items():
    try:
        if "import_as" in details and "from" in details:
            # Import the module to use by its alias
            globals()[details["import_as"]] = importlib.import_module(module_name)
            module = globals()[details["import_as"]]
            # Import specific items from the module
            for obj in details["from"]:
                globals()[obj] = getattr(module, obj)
            logging.info(
                f"Imported '{module_name}' as '{details['import_as']}' and from: {details['from']}"
            )
        elif "import_as" in details:
            # Import module and assign alias
            globals()[details["import_as"]] = importlib.import_module(module_name)
            logging.info(f"Imported '{module_name}' as '{details['import_as']}'")
        elif "from" in details:
            # Only import specific attributes
            module = importlib.import_module(module_name)
            for obj in details["from"]:
                globals()[obj] = getattr(module, obj)
            logging.info(f"Imported '{module_name}' with specific imports")
        else:
            globals()[module_name] = importlib.import_module(module_name)
            logging.info(f"Imported '{module_name}'")
    except ImportError as e:
        logging.warning(f"The required library '{module_name}' is missing.")
        missing_packages.append(details["install_name"])
    except AttributeError as e:
        logging.warning(f"Failed to access attributes from '{module_name}': {e}")
        missing_packages.append(details["install_name"])

# Step 2: Install Missing Packages
if missing_packages:
    if is_pip_installed():
        logging.info("Attempting to install missing packages...")
        install_missing_packages(missing_packages)
    else:
        logging.error("`pip` is not installed on this system.")
        logging.error(
            "To install `pip`, download the installer script from https://bootstrap.pypa.io/get-pip.py "
            "and run: 'python get-pip.py'."
        )
        sys.exit(1)

    # Step 3: Retry Importing Missing Packages, Including Specific Attributes
    for module_name, details in third_party_libs.items():
        if details["install_name"] in missing_packages:
            try:
                if "import_as" in details:
                    globals()[details["import_as"]] = importlib.import_module(
                        module_name
                    )
                    logging.info(
                        f"Successfully imported '{module_name}' as '{details['import_as']}' after installation."
                    )
                elif "from" in details:
                    module = importlib.import_module(module_name)
                    for obj in details["from"]:
                        globals()[obj] = getattr(module, obj)
                        logging.info(
                            f"Successfully imported '{obj}' from '{module_name}' after installation."
                        )
                else:
                    globals()[module_name] = importlib.import_module(module_name)
                    logging.info(
                        f"Successfully imported '{module_name}' after installation."
                    )
            except ImportError:
                logging.error(
                    f"Failed to import '{module_name}' even after installation."
                )
                sys.exit(1)
            except AttributeError as e:
                logging.error(
                    f"Failed to access attributes from '{module_name}' after installation: {e}.\n"
                )
                sys.exit(1)

# Step 4 report success
logging.info("All required libraries imported successfully.\n")

#
# -----------------------------------------------------------------------------------
# Configuration                                                                     |
# -----------------------------------------------------------------------------------
# Purpose: Load and manage configuration settings from config.yaml.
#
# Definitions required before functions defined
# Initialise Colorama for cross-platform terminal colours (defined in Global Scope)
init(autoreset=True)
# Define named tuples for application (defined in Global Scope)
valid_ticker = namedtuple("valid_ticker", ["ticker", "quote_type", "currency"])
ticker_data = namedtuple("ticker_data", ["ticker", "quote_type", "currency", "price", "date"])

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


# Load configuration with defaults
def load_configuration(config_file: str = "config.yaml") -> Box:
    """
    Load configuration from a YAML file and apply default values.
    """
    default_settings = {
        "tickers": {
            "^FTAS": "FTSE All-share (GBP)",
            "^FTSE": "FTSE 100 (GBP)",
            "^OEX": "S&P 100 INDEX (USD)",
            "^GSPC": "S&P 500 INDEX (USD)",
        },
        "paths": {
            "base": "C:\\Users\\wayne\\OneDrive\\Documents\\GitHub\\Python\\Projects\\quicken_prices\\",
            "quicken": "C:\\Program Files (x86)\\Quicken\\qw.exe",
            "data_file": "data.csv",
            "log_file": "prices.log",
            "cache": "cache",
        },
        "collection": {"period_years": 0.08, "max_retries": 3, "retry_delay": 2},
        "default_periods": {
            "EQUITY": "5d",
            "ETF": "5d",
            "MUTUALFUND": "1mo",
            "FUTURE": "5d",
            "CURRENCY": "5d",
            "INDEX": "5d",
        },
        "cache": {"max_age_days": 30, "cleanup_threshold": 200},
        "api": {"rate_limit": {"max_requests": 30, "time_window": 60}},
        "memory": {"max_memory_percent": 75, "chunk_size": 1000},
        "logging": {
            "levels": {"file": "DEBUG", "terminal": "DEBUG"},
            "max_bytes": 5_242_880,
            "backup_count": 5,
        },
        "clear_startup": False,
    }

    # Load user configuration
    config = {}
    if os.path.exists(config_file):
        try:
            with open(config_file, "r") as f:
                config = yaml.safe_load(f) or {}
            if not isinstance(config, dict):
                logging.warning(f"Invalid format in {config_file}. Using defaults.")
                config = {}
        except Exception as e:
            logging.error(f"Error reading {config_file}: {e}. Using defaults.")

    # Merge defaults with user config
    final_config = apply_defaults(default_settings, config)
    return Box(final_config)


#
# -----------------------------------------------------------------------------------
# Logging                                                                           |
# -----------------------------------------------------------------------------------
#
class UnifiedFormatter(logging.Formatter):
    """A single formatter for terminal and file outputs with optional colour."""

    LEVEL_COLOURS = {
        "DEBUG": Fore.BLUE,
        "INFO": Fore.GREEN,
        "WARNING": Fore.YELLOW,
        "ERROR": Fore.RED,
        "TEXT": Fore.CYAN,
    }

    def __init__(
        self, level_formats, max_level_length, use_colour=False, *args, **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.level_formats = level_formats
        self.max_level_length = max_level_length
        self.use_colour = use_colour

    def format(self, record):
        # Preserve the original levelname
        original_levelname = record.levelname

        # Pad the log level for alignment
        padded_levelname = f"{original_levelname}:".ljust(self.max_level_length + 1)

        # Add colour to the log level name if enabled
        if self.use_colour and original_levelname in self.LEVEL_COLOURS:
            colour = self.LEVEL_COLOURS[original_levelname]
            record.levelname = f"{colour}{padded_levelname}{Style.RESET_ALL}"
        else:
            record.levelname = padded_levelname

        # Select the format for the current log level
        log_fmt = self.level_formats.get(original_levelname, self._fmt)
        self._style._fmt = log_fmt

        # Format the message
        result = super().format(record)

        # Restore the original levelname
        record.levelname = original_levelname

        return result


def setup_logging(config: Box) -> logging.Logger:
    """
    Configure logging with separate level-specific formats for terminal and logfile outputs.
    """
    # Clear existing handlers
    logging.getLogger().handlers.clear()

    logger = logging.getLogger("QuickenPrices")
    logger.setLevel(logging.DEBUG)

    # Define the custom TEXT level
    TEXT_LEVEL = 45
    logging.addLevelName(TEXT_LEVEL, "TEXT")

    # Add a method for plain text logging
    def text(self, message, *args, **kwargs):
        """Custom method for plain text logging."""
        self.log(TEXT_LEVEL, message, *args, **kwargs)

    setattr(logging.Logger, "text", text)

    # Calculate maximum level length using raw log level names
    raw_levels = ["DEBUG", "INFO", "WARNING", "ERROR", "TEXT"]
    max_level_length = max(len(level) for level in raw_levels)

    # Define level-specific formats for terminal
    level_formats_terminal = {
        "DEBUG": f"%(levelname)s %(message)s",
        "INFO": f"%(levelname)s %(message)s",
        "WARNING": f"%(levelname)s %(message)s",
        "ERROR": f"%(levelname)s %(message)s",
        "TEXT": f"         %(message)s",  # Aligned plain text for TEXT level
    }

    # Define level-specific formats for logfile
    level_formats_file = {
        "DEBUG": f"%(asctime)s [DEBUG]: %(message)s",
        "INFO": f"%(asctime)s [INFO]: %(message)s",
        "WARNING": f"%(asctime)s [WARNING]: %(message)s",
        "ERROR": f"%(asctime)s [ERROR]: %(message)s",
        "TEXT": f"%(message)s",  # Plain text for TEXT level
    }

    # Terminal Handler with UnifiedFormatter
    terminal_handler = logging.StreamHandler(sys.stdout)
    terminal_handler.setLevel(
        getattr(logging, config.logging.levels.terminal.upper(), "DEBUG")
    )
    terminal_handler.setFormatter(
        UnifiedFormatter(level_formats_terminal, max_level_length, use_colour=True)
    )
    logger.addHandler(terminal_handler)

    # File Handler with UnifiedFormatter (no colour)
    log_file_path = os.path.join(config.paths.base, config.paths.log_file)
    file_handler = logging.handlers.RotatingFileHandler(
        log_file_path,
        maxBytes=config.logging.max_bytes,
        backupCount=config.logging.backup_count,
    )
    file_handler.setLevel(getattr(logging, config.logging.levels.file.upper(), "DEBUG"))
    file_handler.setFormatter(
        UnifiedFormatter(level_formats_file, max_level_length, use_colour=False)
    )
    logger.addHandler(file_handler)

    return logger


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
        logger.debug((f"Start of function '{func.__name__}'..."))
        try:
            result = func(*args, **kwargs)
            logger.debug(f"End of function '{func.__name__}'.\n")
            return result
        except Exception as e:
            logger.error(f"Function '{func.__name__}' failed! Error: {e}\n")
            raise

    return wrapper


def retry(exceptions, tries=3, delay=1, backoff=2):
    """
    A decorator that retries a function upon encountering specified exceptions.
    It implements exponential backoff between retries and prints retry attempts.
    """

    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            mtries, mdelay = tries, delay
            for attempt in range(1, tries + 1):
                try:
                    return func(*args, **kwargs)
                except exceptions as e:
                    if attempt < tries:
                        logger.warning(
                            f"Attempt {attempt}/{tries}: {e}, Retrying in {mdelay} seconds..."
                        )
                    else:
                        logger.error(
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
    """Get period from config dictionary and calculate date range."""
    start_date = None
    end_date = None
    period_year = config.collection.period_years
    period_days = period_year * 365
    end_date = date.today()
    start_date = end_date - timedelta(days=int(period_days))
    # Convert dates to numpy datetime64[D] format
    start_date_np = np.datetime64(start_date)
    end_date_np = np.datetime64(end_date)
    # Calculate business days using np.busday_count
    business_days = np.busday_count(start_date_np, end_date_np)
    return start_date, end_date, business_days


def normalise_dates(df: pd.DataFrame, date_column: str = "Date") -> pd.DataFrame:
    """
    Normalizes all date values in the specified column of the DataFrame to UTC and returns
    them in 'yyyy-mm-dd' format. Invalid dates are handled gracefully and returned as a separate DataFrame.

    Parameters:
    - df: pandas DataFrame containing the date column.
    - date_column: The name of the date column. Defaults to 'Date'.

    Returns:
    - normalized_df: DataFrame with normalized dates in 'yyyy-mm-dd' format.
    """

    normalized_dates = []
    invalid_rows = []

    for index, row in df.iterrows():
        date_value = row[date_column]

        try:
            # Parse date, coercing invalid formats to NaT
            parsed_date = pd.to_datetime(date_value, errors="coerce")

            if pd.isna(parsed_date):
                # If parsing fails, log the invalid date
                invalid_rows.append(row)
                normalized_dates.append(None)
            else:
                # If timezone-naive, assume it's in UTC and localize it
                if parsed_date.tzinfo is None or parsed_date.tz is None:
                    localized_date = parsed_date.tz_localize(
                        "UTC", ambiguous="NaT", nonexistent="NaT"
                    )
                else:
                    # If timezone-aware, convert it to UTC
                    localized_date = parsed_date.tz_convert("UTC")

                # Append the date in 'yyyy-mm-dd' format
                normalized_dates.append(localized_date.strftime("%Y-%m-%d"))

        except Exception as e:
            # If any unexpected error occurs, log the invalid date
            invalid_rows.append(row)
            normalized_dates.append(None)

    # Add the normalized dates to the DataFrame
    df[date_column] = normalized_dates

    # Convert the 'Date' column explicitly to datetime (in case some values are still objects)
    df[date_column] = pd.to_datetime(
        df[date_column], errors="coerce", format="%Y-%m-%d"
    )

    # Create a DataFrame of invalid rows
    invalid_dates_df = pd.DataFrame(invalid_rows, columns=df.columns)

    # Print rows that were not successfully transformed
    if not invalid_dates_df.empty:
        logger.debug(f"Invalid dates found at: {invalid_dates_df}")
    return df


def pluralise(title: str, quantity: int) -> str:
    if quantity == 1:
        return title
    else:
        return title + "s"

#
# -----------------------------------------------------------------------------------
# Ticker Handling                                                                   |
# -----------------------------------------------------------------------------------
# Manage and validate tickers, ensuring that only valid and correctly formatted tickers are processed as well as any necessary FX rate tickers.
#
@log_function  # Decorator to log entry and exit of functions.
def get_tickers(config: Dict[str, Any], set_maximum_tickers=5) -> Set[valid_ticker]:
    """
    Get the list of tickers from the configuration, including currency pairs.
    """
    # Get tickers from config
    stock_tickers = list(config.tickers.keys())

    if not stock_tickers:
        logger.error("No tickers defined in YAML file. Exiting.")
        sys.exit(1)
    else:
        logger.info(
            f"Received {len(stock_tickers)} unvalidated stock {pluralise('ticker', len(stock_tickers))},: {f'First {set_maximum_tickers}' if set_maximum_tickers<len(stock_tickers) else ''}: {list(stock_tickers)[:set_maximum_tickers]}{f'...\n' if len(stock_tickers)>set_maximum_tickers else '\n'}"
        )

    # Validate stock tickers
    logger.info(f"Validating stock {pluralise('ticker', len(stock_tickers))}...\n")
    valid_tickers, invalid_tickers = validate_tickers(stock_tickers)

    if not valid_tickers:
        logger.error("No valid tickers in YAML file. Exiting.")
        sys.exit(1)

    # iterate through the currencies to extract unique currencies, excluding GBP and GBp
    currencies = {
        currency for _, _, currency in valid_tickers if currency not in ["GBP", "GBp"]
    }

    # Get list of FX tickers for those currencies we have
    fx_tickers = []  # create empty list
    for currency in currencies:
        if currency == "USD":
            fx_tickers.append("GBP=X")
        else:
            fx_tickers.append(f"{currency}GBP=X")

    # join the elements of the list using a list comprehension:
    fx_tickers_list = ", ".join(str(ticker) for ticker in fx_tickers)
    logger.info(
        f"Retrieved {len(fx_tickers)} unvalidated FX {pluralise('ticker',len(fx_tickers))}: {fx_tickers_list}\n"
    )

    # Validate FX tickers
    logger.info(f"Validating FX {pluralise('ticker', len(fx_tickers))}...\n")
    valid_FX_tickers, invalid_FX_tickers = validate_tickers(fx_tickers)

    if not valid_FX_tickers:
        logger.error("No valid FX tickers. Can only process GBP stock")
        return

    # Combine stock and FX ticker lists
    all_tickers = valid_tickers + valid_FX_tickers
    # Remove duplicates
    all_tickers = set(ticker for ticker in all_tickers)

    if not all_tickers:  # checks for Falsy values eg empty set.
        logger.error("A serious error has occurred with no data being returned!")
        exit(1)
    else:
        # Extracting ticker values from valid_tickers
        all_ticker_list = [vt.ticker for vt in all_tickers]
        logger.info(
            f"In total, received {len(all_ticker_list)} stock and FX {pluralise('ticker', len(all_ticker_list))}"
            f"{f' - First {set_maximum_tickers}' if len(all_ticker_list)>set_maximum_tickers else ''}:"
            f"{list(all_ticker_list)[:set_maximum_tickers]}{f'...' if len(all_ticker_list)>set_maximum_tickers else ''}"
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
        short_invalid_ticker_list = ""

    # Report outcome
    logger.info(
        f"Summary: {num_valid_tickers} validated {pluralise('ticker',num_valid_tickers)} & {num_invalid_tickers} non-validated {pluralise('ticker',num_invalid_tickers)}:"
    )
    logger.info(f"{short_valid_ticker_list}{short_invalid_ticker_list}\n")

    return valid_tickers, invalid_tickers

#
# -----------------------------------------------------------------------------------
# Data Fetching and Caching                                                         |
# -----------------------------------------------------------------------------------
# Fetch historical price data for each ticker, utilizing caching to minimize redundant
# API calls and implement rate limiting to adhere to API usage policies.
#
@retry(Exception, tries=3)
@log_function
def get_ticker(ticker_symbol: str) -> Optional[yf.Ticker]:
    """
    Fetches ticker data from Yahoo Finance

    Args:
        ticker_symbol: The ticker symbol to fetch data for.

    Returns:
        A yfinance Ticker object if successful, None otherwise.
    """
    try:
        ticker = yf.Ticker(ticker_symbol)
        logger.debug(f"Validated {ticker_symbol}.")
        return ticker
    except Exception as e:
        if "404" in str(e) or "Client" in str(e):
            logger.error(f"Invalid ticker symbol: {ticker_symbol}")
        else:
            logger.error(f"Error fetching ticker data for {ticker_symbol}: {e}")
        return None

@retry(Exception, tries=3)
@log_function
def fetch_ticker_history(
    ticker_symbol: str, start_date: datetime, end_date: datetime
) -> pd.DataFrame:
    """
    Fetches price/ rate data for a single ticker.
    First checks if the data is cached; if not,
    fetches the data using the appropriate ticker object and caches it.
    """
    cache_dir = os.path.join(base_directory, config.paths.cache)
    if not os.path.exists(cache_dir):
        os.makedirs(cache_dir)
    cache_file = os.path.join(
        cache_dir,
        f"{ticker_symbol}_{start_date.strftime('%Y%m%d')}_{end_date.strftime('%Y%m%d')}.pkl",
    )
    if os.path.exists(cache_file):
        with open(cache_file, "rb") as f:
            df = pickle.load(f)
        logger.debug(f"Fetching {format_path(cache_file)}.")
    else:
        ticker = get_ticker(ticker_symbol)
        time.sleep(0.2)  # Rate limiting
        df = ticker.history(
            start=start_date.strftime("%Y-%m-%d"),
            end=end_date.strftime("%Y-%m-%d"),
            interval="1d",
        ).rename(columns={"Close": "Price"})
        if df.empty:
            logger.warning(f"No data found online for {ticker_symbol}.")
            return pd.DataFrame()
        else:
            logger.debug(f"Downloading data for {ticker_symbol}.")

        with open(cache_file, "wb") as f:  # With automatically closes file
            pickle.dump(df, f)
            logger.debug(f"Save cache to {format_path(cache_file)}.")

    # Drop superfluous columns
    df = df.reset_index()  # make the index a regular column named "Date"
    tidy_df = df[["Date", "Price"]]

    logger.debug(
        f"Obtained {tidy_df.shape[0]} {pluralise("day", tidy_df.shape[0])} of data for {ticker_symbol}."
    )
    return tidy_df  # returns a pandas df


@retry(Exception, tries=3)
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
        combined_df = pd.concat(records, ignore_index=True)
        # Normalise dates
        combined_df = normalise_dates(combined_df, date_column="Date")

        # Reorder columns
        combined_df = combined_df[["Ticker", "Price", "Date", "QuoteType", "Currency"]]

        logger.info(
            f"The {business_days} business days from {start_date.strftime("%d/%m/%Y")} - {end_date.strftime("%d/%m/%Y")} returned {combined_df.shape[0]} records."
        )
        return combined_df
    else:
        logger.error(
            f"Did not obtain data for the {business_days} business days between {start_date.strftime("%d/%m/%Y")} and {end_date.strftime("%d/%m/%Y")}!"
        )
    if combined_df.empty:  # checks for empty Pandas DataFrame
        logger.error(
            "No data fetched from fetch_historical_data for any tickers. Exiting."
        )
        exit(1)
    else:
        return pd.DataFrame()  # return an empty DataFrame

#
# -----------------------------------------------------------------------------------
# Data Processing                                                                   |
# -----------------------------------------------------------------------------------
# Purpose: Process the raw fetched data by performing currency conversions and organizing the data for output.
#
@log_function
def convert_prices(df):

    # Create exchange rate map from rows where QuoteType is 'CURRENCY'
    exchange_rate_df = df[df['QuoteType'] == 'CURRENCY'][['Ticker', 'Date', 'Price']]
    exchange_rate_df = exchange_rate_df.rename(columns={'Price': 'Exchange_rate'})
    exchange_rate_df.set_index(['Date', 'Ticker'], inplace=True)

    # Log exchange tickers in the map
    unique_FX_names = exchange_rate_df.index.get_level_values('Ticker').unique()
    unique_FX_names_str = ", ".join(unique_FX_names)
    logger.info(
        f"A {len(exchange_rate_df)} item FX rate map created for {unique_FX_names_str}."
    )

    # Initialize new DF
    final_df = pd.DataFrame(
        {
            "Ticker": df["Ticker"],
            "Date": df["Date"],
            "Type": df["QuoteType"],
            "Original Price": df["Price"].astype(float),
            "Original Currency": df["Currency"],
            "FX Ticker": "",
            "Exchange Rate": np.nan,
            "Price": np.nan,
            "Currency": "",
            "Conversion": "",
            "Outcome": "",
        }
    )

    logger.debug(
        f"A {len(final_df)} item 'final_df' dataframe created."
    )

    # Populate final_df 'FX Ticker' and 'Exchange Rate' columns
    error_occurred = False # a counter to see if this look runs completely ok
    for index, row in final_df.iterrows():
        currency = row['Original Currency']
        date = row['Date']
        old_price =row['Original Price']
        outcome=""

        if row['Type'] == 'CURRENCY' or row['Type'] == 'INDEX':
            final_df.loc[index, 'FX Ticker'] = "-"
            final_df.loc[index, 'Exchange Rate'] = 1
            final_df.loc[index, "Price"] = old_price
            final_df.loc[index, "Currency"] = currency
            final_df.loc[index, "Conversion"] = "No change"
            final_df.loc[index, "Outcome"] = "No change"
            continue  # Skip to the next iteration

        if currency == 'GBP':
            exchange_rate = 1.0
            FX_ticker = '-'
            conversion = "No change"
        elif currency == 'GBp':
            exchange_rate = 0.01
            FX_ticker = '-'
            conversion = "GBp to GBP"
        else:
            # Determine FX_ticker
            if currency == 'USD':
                FX_ticker = 'GBP=X'
                conversion = "USD to GBP"
            else:
                FX_ticker = f"{currency}GBP=X"
                conversion = f"{currency} to GBP"

            # Get exchange rates from exchange_rate_df
            try:
                exchange_rate = exchange_rate_df.loc[(date, FX_ticker), 'Exchange_rate']
                logger.debug(
                    f"Numeric: {isinstance(exchange_rate, (int, float))} FX rate is {exchange_rate} for {currency} ({FX_ticker}) on date {date.strftime('%d/%m/%Y')} and applied to {row['Ticker']}."
                )
            except Exception as e:
                exchange_rate = "-"
                outcome = "No FX rate"
                error_occurred = True
                logger.error(
                    f"FX rate not found for {currency} ({FX_ticker}) on date {date.strftime('%d/%m/%Y')}: {e}."
                )
        try:
            if currency =="GBP":
                new_price = old_price
                new_currency = "GBP"
                outcome = "No change"
            elif outcome != "No FX rate":
                new_price = old_price * exchange_rate
                new_currency = "GBP"
                outcome = "Success"
        except Exception as e:
            new_price, new_currency = "?"
            outcome = outcome + "No price"
            error_occurred = True
            logger.error(
                    f"Price conversion not possible using {currency} ({FX_ticker}) for {row['Ticker']} on date {date.strftime('%d/%m/%Y')}: {e}."
                )

        # Assign new values to the 'final_df'
        final_df.loc[index, 'FX Ticker'] = FX_ticker
        final_df.loc[index, 'Exchange Rate'] = exchange_rate
        final_df.loc[index, 'Price'] = new_price
        final_df.loc[index, 'Currency'] = new_currency
        final_df.loc[index, "Conversion"] = conversion
        final_df.loc[index, "Outcome"] = outcome

    # report on whether anything problematic happened during the for loop or not
    if not error_occurred:
        logger.info("All currency conversions processed successfully.")

    # Final check that the length of the output df is the same as the input df
    if len(final_df) != len(df):
        logger.error(
            "The length of the output DataFrame does not match the input DataFrame."
        )
    else:
        logger.info(f"{len(final_df)} records successfully processed from {len(df)} inputs.")

    return final_df


@log_function
def process_converted_prices(
    converted_df: pd.DataFrame,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    try:
        # Make Output CSV DataFrame ######################################################
        output_csv = converted_df.copy()

        # Select required columns
        output_csv = output_csv[["Ticker", "Price", "Date"]]

        # Order by descending date before changing format
        output_csv = output_csv.sort_values(by="Date", ascending=False)

        # Reformat date into dd/mm/yyyy
        output_csv["Date"] = pd.to_datetime(output_csv["Date"]).dt.strftime("%d/%m/%Y")

        # Ensure the DataFrame has the same number of rows as the input DataFrame
        if len(output_csv) != len(converted_df):
            logger.error(
                "Output CSV DataFrame row count does not match input DataFrame row count."
            )
        else:
            logger.info(f"{len(output_csv)} item 'output_csv' dataframe created successfully.")

        # Round all prices for the remaining df's
        converted_df[["Original Price", "Exchange Rate", "Price"]] = converted_df[
            ["Original Price", "Exchange Rate", "Price"]
            ].round(3)

        # Combine the price and currency columns
        converted_df["Original Price"] = (
            converted_df["Original Price"].astype(str)
            + " "
            + converted_df["Original Currency"]
        )
        converted_df["Price"] = (
            converted_df["Price"].astype(str) + " " + converted_df["Currency"]
        )

        # Calculate how many days of data were collected per ticker (not the diffrence between last date and first date)
        # Calculate the number of distinct days for each ticker
        grouped_data = (
            converted_df.groupby("Ticker")["Date"]
            .nunique()
            .reset_index(name="Days of Data")
        )
        converted_df = converted_df.merge(grouped_data, on="Ticker", how="left")

        # Make Success DataFrame ##########################################################
        success = converted_df[
            converted_df["Outcome"] == "Success"
            ]
        logger.debug(f"{len(success)} item 'success' dataframe created successfully.")

        # Make Failure DataFrame ##########################################################
        failure = converted_df[
            (converted_df["Outcome"] != "Success")
            & (converted_df["Outcome"] != "No change")
        ]
        logger.debug(f"{len(failure)} item 'failure' dataframe created successfully.")

        # Make No change DataFrame ########################################################
        no_change = converted_df[
            (converted_df["Outcome"] == "No change")
            & (~converted_df["Type"].isin(["INDEX", "CURRENCY"]))
        ]
        logger.debug(f"{len(no_change)} item 'no change' dataframe created successfully.")

        # Make Index and Currency DataFrame ###############################################
        index_currency = converted_df[
            converted_df["Type"].isin(["INDEX", "CURRENCY"])
        ]
        logger.debug(
            f"{len(index_currency)} item 'index_currency' dataframe created successfully."
        )

        # Calculate totals before filtering
        total_rows_input = len(converted_df)
        total_rows_output_csv = len(output_csv)
        total_rows_success = len(success)
        total_rows_failure = len(failure)
        total_rows_no_change = len(no_change)
        total_rows_index_currency = len(index_currency)

        total_rows_calculated = (
            total_rows_success
            + total_rows_failure
            + total_rows_no_change
            + total_rows_index_currency
        )

        if total_rows_calculated != total_rows_input:
            logger.error(
                "Total rows after processing do not match the original total rows."
            )
        else:
            logger.info(
                f"Inputs {total_rows_input} = outputs {total_rows_output_csv} = successes {total_rows_success} + failures {total_rows_failure} + not changed {total_rows_no_change} + Index & FX {total_rows_index_currency}."
            )

        # Return the DataFrames directly
        return output_csv, success, failure, no_change, index_currency

    except Exception as e:
        logger.error(f"An error occurred during processing: {e}")
        raise

#
# -----------------------------------------------------------------------------------
# Formatting                                                                        |
# -----------------------------------------------------------------------------------
#
def format_path(full_path):
    """Format a file path to show the last parent directory and filename."""
    path_parts = full_path.split("\\")
    if len(path_parts) > 1:
        return "\\" + "\\".join(path_parts[-2:])  # Show parent dir and filename
    return "\\" + full_path  # Just in case there's no parent directory


def format_status(status):
    """Format status indicators with colors."""
    status_map = {
        "new data": f"{Fore.GREEN}✓{Style.RESET_ALL}",
        "using cache": f"{Fore.BLUE}#{Style.RESET_ALL}",
        "failed": f"{Fore.RED}✗{Style.RESET_ALL}",
    }
    return status_map.get(status, f"{Fore.RED}?{Style.RESET_ALL}")


def format_ticker_status(ticker, status, currency=None, width=30):
    """Format ticker status with currency information."""
    status_symbol = format_status(status)
    ticker_with_currency = f"{ticker} ({currency})" if currency else ticker
    return f"{status_symbol} {ticker_with_currency:<{width}}"


def header(title, logger=None, major=False):
    """
    Create a section header, either printed or logged.

    Args:
        title (str): Header text
        logger (logging.Logger, optional): Logger to use instead of print
        major (bool, optional): Use '=' for major headers, '-' for minor
    """
    header_line = "=" * len(title) if major else "-" * len(title)

    if logger:
        logger.text(title)
        logger.text(header_line)
    else:
        print(f"\t{title}")
        print(f"\t{header_line}")


def _strip_ansi(text):
    """Remove ANSI escape sequences from text."""
    import re

    ansi_escape = re.compile(r"\x1B(?:[@-Z\\-_]|\[[0-?]*[ -/]*[@-~])")
    return ansi_escape.sub("", text)


def capture_print(*args, **kwargs):
    """Print and store output in buffer."""
    output = " ".join(str(arg) for arg in args)
    # Store the raw output with ANSI codes for terminal display
    print(output, **kwargs)
    # Store the clean output without ANSI codes for clipboard
    self.output_buffer.append(self._strip_ansi(output))


#
# -----------------------------------------------------------------------------------
# Output Generation                                                                 |
# -----------------------------------------------------------------------------------
# Purpose: Save the processed data to a CSV file without headers and log the file path.
#
@log_function
def prepare_tables(
    success: pd.DataFrame= pd.DataFrame(),
    failure: pd.DataFrame =pd.DataFrame(),
    no_change: pd.DataFrame =pd.DataFrame(),
    index_currency: pd.DataFrame= pd.DataFrame(),
):
    """
    Prepares and logs summary and 4 tables:
    success, failure, no_change, index_currency
    Summary of currency conversions and success rate.
    Table 1: Latest Prices.
    Table 2. No changes
    Table 3: Index and Currency Prices.
    Table 4: Failed Conversions.
    """
    # Conversion success rate
    success_rate = len(success) / (len(success) + len(failure)) * 100

    # Create latest_prices
    dfs_to_concat = []
    if not success.empty:
        # Get the latest date for each ticker in success
        success['latest_date'] = success.groupby('Ticker')['Date'].transform('max')
        # Filter for rows with the latest date for each ticker
        success = success[success['Date'] == success['latest_date']]
        dfs_to_concat.append(success)
    if not no_change.empty:
        no_change['latest_date'] = no_change.groupby('Ticker')['Date'].transform('max')
        no_change = no_change[no_change['Date'] == no_change['latest_date']]
        dfs_to_concat.append(no_change)
    if dfs_to_concat:
        latest_prices = pd.concat(dfs_to_concat, ignore_index=True)
        latest_prices.loc[:, "Date"] = latest_prices["Date"].dt.strftime("%d/%m/%Y")
        latest_prices.sort_values(by=["Type", "Ticker"], inplace=True)
        latest_prices_columns = [
            "Ticker",
            "Date",
            "Type",
            "Original Price",
            "FX Ticker",
            "Exchange Rate",
            "Days of Data",
            "Price",
        ]
    else:
        latest_prices = pd.DataFrame()

    # Prepare index_currency_prices
    if not index_currency.empty:
        index_currency['latest_date'] = index_currency.groupby('Ticker')['Date'].transform('max')
        index_currency = index_currency[index_currency['Date'] == index_currency['latest_date']]
        index_currency.loc[:, "Date"] = index_currency["Date"].dt.strftime("%d/%m/%Y")
        index_currency.sort_values(by=["Type", "Ticker"])
        index_currency_columns = [
            "Ticker",
            "Date",
            "Type",
            "Days of Data",
            "Price",
        ]

    # Prepare failure
    if not failure.empty:
        failure.loc[:, "Date"] = failure["Date"].dt.strftime("%d/%m/%Y")
        failure.rename(columns={"Outcome": "Failure Reason" }, inplace=True )
        failure_columns = [
            "Ticker",
            "Date",
            "Type",
            "Original Price",
            "FX Ticker",
            "Exchange Rate",
            "Failure Reason",
        ]

    def log_table(table_title, table_data, columns_to_show):
        if table_data.empty:
            logger.text(f"\n\t No {table_title} to display.\n")
        else:
            table_data.reset_index(drop=True, inplace=True)
            table_data.index += 1
            table_data.reset_index(inplace=True)
            table_data=table_data.rename(columns={"index": ""})

            table_data_str = (
                "\n\t "
                + table_title
                + ":\n\t "
                + "=" * len(table_title)
                + "\n\t "
                + tabulate(
                    table_data[columns_to_show],  # Specify columns to show
                    headers="keys",
                    tablefmt="grid",
                    showindex=False,
                ).replace("\n", "\n\t ")
            )
            logger.text(table_data_str)

    # Prep summary

    # Count tickers for specific conditions
    unchanged_gbp_tickers = no_change[
        (no_change["Original Currency"] == "GBP")][
        "Ticker" ].nunique()
    unchanged_index_tickers = index_currency[
        index_currency["Type"] == "INDEX"][
        "Ticker" ].nunique()
    unchanged_currency_tickers = index_currency[
        index_currency["Type"] == "CURRENCY"][
        "Ticker"].nunique()

    # Calculate conversion counts (key-value pair eg USD to GBP: 2)
    conversion_counts = success['Conversion'].value_counts()

    summary = []
    start_date, end_date, business_days = get_date_range()
    summary.append(
        f"{start_date.strftime("%d/%m/%Y")} - {end_date.strftime("%d/%m/%Y")}\n\t {business_days} business days"
    )
    for conversion, count in conversion_counts.items():
        summary.append(
            f"{pluralise('Ticker', count)} converted from {conversion}: {count}"
        )
    summary.append(f"Unchanged GBP tickers: {unchanged_gbp_tickers}")
    summary.append(f"Index tickers: {unchanged_index_tickers}")
    summary.append(f"Currency tickers obtained: {unchanged_currency_tickers}")
    summary.append(f"Total processed: {len(config.tickers)}")
    summary.append(f"Conversion success rate: {success_rate:.1f}%")
    summary = "\n\t ".join(summary)

    # --------------------
    # Generate Outputs
    # --------------------
    print("")
    header("Stock Price Update", logger=logger,major=True)
    start_date, end_date, business_days = get_date_range()
    logger.text(summary)

    if not latest_prices.empty:
        log_table("Latest Prices", latest_prices, latest_prices_columns)
    else:
        logger.text("\nThere were no converted prices.")
    if not index_currency.empty:
        log_table("Index and Currency Prices", index_currency, index_currency_columns)
    else:
        logger.text("\nThere were no index or currency prices.")
    if not failure.empty:
        log_table("Failed Conversions", failure, failure_columns)
    else:
        logger.text("\nThere were no failed conversions.")
    print("")

@retry(Exception, tries=3)
@log_function
def save_to_csv(df: pd.DataFrame) -> bool:
    """
    Save the DataFrame to a CSV file without headers.
    3 Columns: Ticker, Price in GBP, Date (dd/mm/yyyy).
    """
    try:
        # Build the output file path
        CSV_file_name = config.paths.data_file
        CSV_path = os.path.join(base_directory, CSV_file_name)
        short_CSV_path = format_path(CSV_path)

        required_columns = ["Ticker", "Price", "Date"]
        if not all(col in df.columns for col in required_columns):
            logger.error(
                f"Required columns are missing. Available columns are: {df.columns.tolist()}"
            )
            return False

        # This should have been done already.
        # # Select and sort the DataFrame
        # df2 = df[required_columns].sort_values(by="Date", ascending=False)

        # # Format 'Date' column as 'dd/mm/yyyy'
        # df2["Date"] = df2["Date"].dt.strftime("%d/%m/%Y")

        # Save the DataFrame to CSV without headers
        df.to_csv(CSV_path, index=False, header=False)

        # Log the save location
        logger.info(f"Data successfully saved to: {short_CSV_path}")

        return True

    except Exception as e:
        logger.exception(
            f"An error occurred while saving the dataframe to a file called {CSV_file_name}"
        )
        return False

#
# -----------------------------------------------------------------------------------
# Cache Cleanup                                                                     |
# -----------------------------------------------------------------------------------
# Purpose: Manage the cache by deleting outdated cached files to conserve storage and maintain performance.
#
@log_function
def clean_cache():
    """
    Clean up cache files older than max_age_days.
    """
    cache_dir = config.paths.cache
    cache_dir_path = os.path.join(base_directory, cache_dir)
    max_age_days = config.cache.max_age_days
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
                logger.info(
                    f"File {cache_dir}/{filename} older than {max_age_days} days, so deleted!"
                )
                num_cleaned_files += 1
        logger.info(
            f"{num_cleaned_files} cache files were cleaned for being older than {max_age_days} days."
        )
    except Exception as e:
        logger.error(f"Error cleaning cache: {e}")
        logger.error(traceback.format_exc())
        return False
    return True
#
# ------------------------------------------------------------------------------------
# Seek elevation                                                                    |
# ------------------------------------------------------------------------------------
#
def is_elevated():
    """Check if script is running with admin privileges."""
    try:
        return ctypes.windll.shell32.IsUserAnAdmin()
    except:
        return False


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
        print("")
        logger.debug("Attempting to open new elevated script...")
        ctypes.windll.shell32.ShellExecuteW(
            None, "runas", sys.executable, __file__, None, 1
        )
        logger.warning("Elevation request cancelled, continuing unelevated...\n")
        # exit(0)
        return

#
# -----------------------------------------------------------------------------------
# Quicken UI                                                                        |
# -----------------------------------------------------------------------------------
# Opens and navigates Quicken XG 2004 to import the csv file.
#
def import_data_file():
    """
    Import the data file through the import dialog.

    Returns:
        bool: True if import successful
    """
    try:
        output_file_name = config.paths.data_file
        filename = os.path.join(base_directory, output_file_name)
        pyautogui.typewrite(filename, interval=0.03)
        time.sleep(1)
        pyautogui.press("enter")
        time.sleep(5)
        logger.info(
            f"Successfully imported {format_path(filename)} to Quicken at {datetime.now().strftime('%d/%m/%Y %H:%M')}"
        )
        return True
    except Exception as e:
        logger.error(f"Failed to import data file: {e}")
        return False


def open_import_dialog():
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
        logger.error(f"Failed to open import dialog: {e}")
        return False


def navigate_to_portfolio():
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
        logger.error(f"Failed to navigate to portfolio: {e}")
        return False


def open_quicken():
    """
    Launch Quicken application.

    Returns:
        bool: True if Quicken started successfully
    """
    quicken_path = config.paths.quicken
    try:
        subprocess.Popen([quicken_path], shell=True)
        time.sleep(6)  # Allow time for Quicken to start
        # Temporarily disable Ctrl+Alt+Del to prevent accidental interruption
        pyautogui.hotkey("ctrl", "alt", "del")
        logger.info("Quicken opening...")
        return True
    except Exception as e:
        logger.error(f"Failed to open Quicken: {e}")
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
        logger.info(message)
        if not step_function():
            return False

    # Log successful completion of entire sequence
    logger.info(f"Import sequence complete!")
    # Reenabling Ctrl+Alt+Del previously disabled to prevent accidental interruption
    pyautogui.hotkey("ctrl", "alt", "del")
    return True


def setup_pyautogui():
    """
    Configure PyAutoGUI safety settings for automated GUI interaction.
    Sets fail-safe and timing parameters to ensure reliable automation.
    """
    pyautogui.FAILSAFE = True  # Move mouse to corner to abort
    pyautogui.PAUSE = 0.5  # Delay between actions


def quicken_import():
    """
    Main entry point for handling Quicken import.
    """
    try:
        header("Import to Quicken 2004", logger=logger, major=True)
        if is_elevated():
            logger.info("Confirmed 'is elevated'. Starting Quicken...")
            return execute_import_sequence()
        else:
            logger.warning("Quicken cannot be opened when unelevated.")
            logger.text("\nInstead:\n")
            logger.text("1. Close this window.")
            logger.text("2. Click the Quicken 'Update Prices' shortcut.\n")
            header("END", logger=logger,major=True)
        return

    except Exception as e:
        logger.error(f"Error during Quicken import: {e}")
        return

    finally:
        pyautogui.FAILSAFE = True


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
        
        # COnditional clear screen at beginning for clean output
        if config.clear_startup == True:
            os.system("cls" if os.name == "nt" else "clear")
        
        # Get elevation
        run_as_admin()

        # Get date range
        start_date, end_date, business_days = get_date_range()

        # Get tickers
        tickers = get_tickers(config)

        # Fetch Ticker Dataframe
        ticker_dataframe = fetch_historical_data(
            tickers, start_date, end_date, business_days
        )

        # Convert prices to GBP except INDEX and CURRENCY
        processed_data = convert_prices(ticker_dataframe)

        # Process and create output dataframes
        output_csv, success, failure, no_change, index_currency = (
            process_converted_prices(processed_data)
        )

        # Produce tables using processed_data instead of output_csv
        prepare_tables(success, failure, no_change, index_currency)

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
        logger.warning("Script interrupted by user.")
    except Exception as e:
        logger.error(f"An unexpected error occurred: {e}")
        logger.error(traceback.format_exc())


if __name__ == "__main__":
    
    # Items requiring defined in global scope (ie out with main or other function)
    config = load_configuration()
    
    # Ensure base directory exists
    base_directory = config.paths.base
    os.makedirs(base_directory, exist_ok=True)
    
    # Setup logging
    logger = setup_logging(config)
    
    main()

#
# -----------------------------------------------------------------------------------
# END                                                                               |
# -----------------------------------------------------------------------------------
