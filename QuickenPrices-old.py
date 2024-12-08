#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Shebang and coding declaration.

"""
QuickenPrices.py

This script fetches historical price data for a list of tickers specified in 'config.yaml' and saves the data to a CSV file.

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
    import contextlib  # Used to redirect output streams temporarily
    import ctypes  # Low-level OS interfaces
    import functools  # Higher-order functions and decorators
    import importlib  # Provides utilities for importing modules dynamically
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
    from typing import (
        Any,
        Dict,
        Iterator,
        List,
        Optional,
        Set,
        Tuple,
        Union,
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
    "tabulate": {"install_name": "tabulate", "from": ["tabulate"]},
    "cv2": {"install_name": "opencv-python"},  # For real-time computer vision task
    "pygetwindow": {
        "install_name": "pygetwindow", "import_as": "gw"
    },  # Interact with open windows in Windows
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
            "^FTAS",
            "^FTSE",
            "^OEX",
            "^GSPC",
            },
        "paths": {
            "base": "C:\\Users\\wayne\\OneDrive\\Documents\\GitHub\\Python\\Projects\\quicken_prices\\",
            "quicken": "C:\\Program Files (x86)\\Quicken\\qw.exe",
            "data_file": "data.csv",
            "log_file": "prices.log",
            "image": "images",
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

# Items requiring defined in global scope (ie out with main or other function)
config = load_configuration()

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
        "TEXT": f"\t %(message)s", 
    }

    # Define level-specific formats for logfile
    level_formats_file = {
        "DEBUG": f"%(asctime)s [DEBUG]: %(message)s",
        "INFO": f"%(asctime)s [INFO]: %(message)s",
        "WARNING": f"%(asctime)s [WARNING]: %(message)s",
        "ERROR": f"%(asctime)s [ERROR]: %(message)s",
        "TEXT": f"\t %(message)s",  # Plain text for TEXT level
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


def retry(exceptions, tries=config.collection.max_retries, delay=1, backoff=2):
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


def get_date_range() -> Tuple[datetime, datetime, int]:
    """
    Calculate the date range and business days for data collection.

    Returns:
        Tuple containing start_date, end_date, and business_days count.
    """
    period_year = config.collection.period_years
    period_days = period_year * 365 + 1
    end_date = date.today()
    start_date = end_date - timedelta(days=int(period_days))
    # Convert dates to datetime objects
    end_date = end_dt = pd.to_datetime(end_date)
    start_date = pd.to_datetime(start_date)
    # Convert to numpy datetime format for business day calculation
    start_date_np = np.datetime64(start_date, "D")
    end_date_np = np.datetime64(end_date, "D")
    # Calculate business days using np.busday_count
    business_days = np.busday_count(start_date_np, end_date_np)
    return start_date, end_date, business_days

import pandas as pd

def pluralise(title: str, quantity: int) -> str:
    if quantity == 1:
        return title
    else:
        return title + "s"

#
# -----------------------------------------------------------------------------------
# Ticker Validating, Data Fetching and Caching                                      |
# -----------------------------------------------------------------------------------
# Validate tickers, ensuring that only valid and correctly formatted tickers are processed
# as well as required FX rate tickers. Fetch historical price data for each ticker,
# utilizing caching to minimize redundant API calls and implement rate limiting to
# adhere to API usage policies.

# Get and Validate Tickers
#
@retry(Exception, tries=3)
@log_function
def validate_ticker(
    ticker_symbol: str,
) -> Optional[Dict[str, Union[str, pd.Timestamp]]]:
    """
    Validates a single ticker and returns a dictionary with metadata or None if validation fails.

    Args:
        ticker_symbol: The ticker symbol to validate.

    Returns:
        A dictionary with ticker metadata or None.
    """
    if not isinstance(ticker_symbol, str):
        raise TypeError("ticker_symbol must be a string")

    try:
        ticker = yf.Ticker(ticker_symbol)

        # Suppress yfinance output
        with contextlib.redirect_stderr(open(os.devnull, 'w')):
            # Attempt to fetch metadata
            info = ticker.info

        # If info is empty or lacks meaningful data, consider it invalid
        if not info or info.get("symbol") is None:
            logger.error(
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
            "name": info.get("longName", info.get("shortName", "Unknown")),
            "earliest_date": earliest_date,
            "time_zone": info.get("timeZoneShortName", "Unknown"),
            "type": info.get("quoteType", "Unknown"),
            "currency": info.get("currency", "Unknown"),
        }

        # Check for missing data points
        missing_fields = []
        if data["name"] == "Unknown":
            missing_fields.append("name")
        if data["earliest_date"] is None:
            missing_fields.append("earliest_date")
        if data["time_zone"] == "Unknown":
            missing_fields.append("time_zone")
        if data["type"] == "Unknown":
            missing_fields.append("type")
        if data["currency"] == "Unknown":
            missing_fields.append("currency")

        if missing_fields and not info:
            logger.warning(
                f"{ticker_symbol} is valid but missing fields: {', '.join(missing_fields)}."
            )
        else:
            logger.info(f"{ticker_symbol} is valid and metadata complete.")
        return data

    except Exception as e:
        logger.error(f"An unexpected error occurred fetching {ticker_symbol}: {e}")
        return None


@log_function
def validate_tickers(
    tickers: List[str], max_tickers_in_logs: int = 5
) -> Tuple[List[Tuple[str, str, Optional[datetime], str, str, str]], List[str]]:
    """
    Validates a list of tickers and separates them into valid and invalid tickers.

    Args:
        tickers: A list of ticker symbols to validate.
        max_tickers_in_logs: Maximum number of tickers to show in logs.

    Returns:
        A tuple containing:
            - List of valid tickers as tuples (ticker, name, earliest_date, time_zone, type, currency).
            - List of invalid ticker symbols as strings.
    """
    valid_tickers = []
    invalid_tickers = []

    for ticker_symbol in tickers:
        try:
            data = validate_ticker(ticker_symbol)
            if not data:
                invalid_tickers.append(ticker_symbol)
                continue

            ticker, name, earliest_date, time_zone, type, currency = (
                data["ticker"],
                data["name"],
                data["earliest_date"],
                data["time_zone"],
                data["type"],
                data["currency"],
            )

            missing_fields = [
                field for field, value in zip(
                    ["ticker", "earliest_date", "time_zone", "type", "currency"],
                    [ticker, earliest_date, time_zone, type, currency],
                ) if not value or value == "Unknown"
            ]

            if not missing_fields:
                valid_tickers.append((ticker, name, earliest_date, time_zone, type, currency))
            else:
                invalid_tickers.append(ticker_symbol)
                logger.warning(f"Ticker {ticker_symbol} is invalid due to missing fields: {', '.join(missing_fields)}.")
        except Exception as e:
            invalid_tickers.append(ticker_symbol)
            logger.error(f"An unexpected error occurred processing ticker '{ticker_symbol}': {e}")

    def summarise_tickers(ticker_list: List[str], label: str, limit: int) -> str:
        num_tickers = len(ticker_list)
        displayed_tickers = ticker_list[:limit]
        continuation = "..." if num_tickers > limit else ""
        return f"{num_tickers} {label.capitalize()} {pluralise("ticker",num_tickers)} {displayed_tickers}{continuation}"

    logger.info(summarise_tickers([t[0] for t in valid_tickers], "valid", max_tickers_in_logs))
    logger.info(summarise_tickers(invalid_tickers, "invalid", max_tickers_in_logs))

    return valid_tickers, invalid_tickers


@log_function
def get_tickers(
    tickers: List[str], set_maximum_tickers: int = 5
) -> Tuple[List[Tuple[str, str, Optional[datetime], str, str, str]], List[str]]:
    """
    Get and validate stock and FX tickers.

    Args:
        tickers: A list of ticker symbols to validate.
        set_maximum_tickers: Maximum number of tickers to display in logs.

    Returns:
        A tuple of:
            - List of valid tickers with full metadata (tuples).
            - List of invalid ticker symbols.
    """
    if not tickers:
        logger.error("No tickers defined in YAML file. Exiting.")
        sys.exit(1)

    logger.info(f"Validating {len(tickers)} stock tickers...")
    valid_tickers, invalid_tickers = validate_tickers(tickers)

    if not valid_tickers:
        logger.error("No valid stock tickers found in YAML file. Exiting.")
        sys.exit(1)

    # Generate FX tickers based on the currencies in valid_tickers
    currencies = {t[5] for t in valid_tickers if t[5] not in ["GBP", "GBp"]}
    fx_tickers = {
        f"{currency}GBP=X" if currency != "USD" else "GBP=X" for currency in currencies
    }

    logger.info(f"Validating {len(fx_tickers)} FX tickers...")
    valid_fx_tickers, invalid_fx_tickers = validate_tickers(list(fx_tickers))

    if not valid_fx_tickers:
        logger.warning("No valid FX tickers found. Processing GBP stocks only.")

    # Combine and deduplicate valid tickers (preserving full tuples)
    all_valid_tickers = list({t for t in valid_tickers + valid_fx_tickers})
    all_invalid_tickers = list(set(invalid_tickers + invalid_fx_tickers))

    logger.info(
        f"Validated {len(all_valid_tickers)} {pluralise("ticker",len(all_valid_tickers))} in total. Found {len(all_invalid_tickers)} invalid {pluralise("ticker",len(all_invalid_tickers))}."
    )

    return all_valid_tickers, all_invalid_tickers

#
#
# Get data
#
@retry(Exception, tries=3)
@log_function
def fetch_ticker_history(
    ticker_symbol: str, start_date: datetime, end_date: datetime
) -> pd.DataFrame:
    """
    Fetches price/ rate data for a single ticker.
    First checks if the data is cached; if not,
    fetches the data using the appropriate ticker object and caches it.
    Handles missing or delisted tickers and resolves warnings.
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
            tidy_df = pickle.load(f)
        logger.debug(f"Fetching {format_path(cache_file)}.")

    else:
        ticker = yf.Ticker(
            ticker_symbol
        )  # Already validated so can go straight to yf function rather than our validate_tickers function.
        logger.debug(f"Fetching data for {ticker_symbol} from Yahoo Finance.")

        time.sleep(0.2)  # Rate limiting
        df = ticker.history(
            start=start_date,
            end=end_date,
            interval="1d",
        )
        df=df.rename(columns={"Close": "Old Price"})

        if df.empty:
            logger.warning(
                f"No data found online for {ticker_symbol}. Check if available during this time period."
            )
            return pd.DataFrame()

        # Drop superfluous columns
        df = df.reset_index()  # make the index a regular column named "Date"
        tidy_df = df[["Date", "Old Price"]].copy()
        tidy_df["Ticker"] = ticker_symbol     

        with open(cache_file, "wb") as f:  # With automatically closes file
            pickle.dump(tidy_df, f)
            logger.debug(f"Save cache to {format_path(cache_file)}.")

    logger.debug(
        f"Obtained {tidy_df.shape[0]} {pluralise("day", tidy_df.shape[0])} of data for {ticker_symbol}."
    )
    return tidy_df  # returns a pandas df


@retry(Exception, tries=3)
@log_function
def fetch_historical_data(
    tickers: List[Tuple[str, str, Optional[datetime], str, str, str]],
) -> Tuple[pd.DataFrame, List[str]]:
    """
    Fetches historical data for a list of validated tickers and compiles it into a DataFrame.

    Args:
        tickers: A list of tuples containing valid ticker details:
            (ticker, name, earliest_date, time_zone, type, currency).
        start_date: The start date for the historical data range.
        end_date: The end date for the historical data range.
        business_days: Number of business days in the date range.

    Returns:
        A pandas DataFrame containing the fetched historical data.
    """
    records = []
    out_of_range =[]
    
    # Get date range
    start_date, end_date, business_days = get_date_range()

    for ticker, name, earliest_date, time_zone, type, currency in tickers:
        # Adjusted start date is always the later of earliest_date and start_date unless one is None.
        adjusted_start_date = max(start_date, earliest_date or start_date)

        # Check if end_date is before earliest_date, if so skip the ticker
        if earliest_date and end_date < earliest_date:
            logger.warning(
                f"Skipping {ticker}: Period requested, ending {end_date.strftime('%d/%m/%Y')}, is before {ticker} existed (from {earliest_date.strftime('%d/%m/%Y')}).\n"
            )
            out_of_range.append(ticker)
            continue

        if adjusted_start_date > start_date:
            logger.warning(
                f"Earliest available data for {ticker} is from {adjusted_start_date.strftime('%d-%m-%Y')}.\n"
            )
        try:
            # Fetch historical data
            df = fetch_ticker_history(ticker, adjusted_start_date, end_date)
            if df.empty:
                logger.info(f"No data for {ticker}. Skipping.")
                continue

            # Add metadata columns to the DataFrame
            df["Ticker"] = ticker
            df["Name"] = name
            df["TimeZone"] = time_zone
            df["Type"] = type
            df["Original Currency"] = currency

            # Append to records
            records.append(df)
        except Exception as e:
            logger.warning(f"Failed to fetch data for {ticker}: {e}")

    if not records:
        logger.error(
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
            "Name",
            "Date",
            "TimeZone",
            "Type",
            "Old Price",
            "Original Currency",
        ]
    ]

    logger.info(
        f"The {business_days} business days from {start_date.strftime('%d/%m/%Y')} "
        f"to {end_date.strftime('%d/%m/%Y')} returned {combined_df.shape[0]} records."
    )
    return combined_df, out_of_range


#
# -----------------------------------------------------------------------------------
# Data Processing                                                                   |
# -----------------------------------------------------------------------------------
# Purpose: Process the raw fetched data by performing currency conversions and organizing the data for output.
#
@log_function
def convert_prices(df: pd.DataFrame) -> pd.DataFrame:
    """
    Convert non-GBP currency to GBP except for Type CURRENCY and INDEX.

    Args:
        df: A pandas DataFrame with the following columns:
            - Ticker (str): The stock ticker symbol.
            - Name (str): The name of the stock or company.
            - Date (mixed): The date of the price data (varied formats).
            - TimeZone (str): The timezone of the data source.
            - Type (str): The type of quote (e.g., "Equity").
            - Old Price, (float): The stock price.          
            - Original Currency (str): The currency of the stock price.

    Returns:
        A pandas DataFrame with columns Ticker, Name, Date, TimeZone, Type, Old Price,  Original Currency, FX Code, FX Rate, Conversion, Price, Currency,  Outcome
    """
    # Standardise the Date column to datetime.date
    def standardise_date(value):
        try:
            dt = pd.to_datetime(value, errors="coerce")
            if pd.isna(dt):
                return None  # Handle invalid dates
            return dt.date()  # Convert to date only
        except Exception as e:
            logger.warning(f"Error parsing date '{value}': {e}")
            return None

    # Apply the standardisation function to the Date column
    df["Date"] = df["Date"].apply(standardise_date)

    # Validate Date column
    if df["Date"].isna().any():
        logger.warning("Some rows contain invalid or missing dates.")
        # Optionally, raise an exception or remove these rows
        df = df.dropna(subset=["Date"])

    # Create exchange rate DataFrame with valid `Date` index
    exchange_rate_df = (
        df[df["Type"] == "CURRENCY"]
        .rename(columns={"Old Price": "FX Rate"})
        .set_index(["Date", "Ticker"])
    )

    logger.info(f"Created exchange rate map with {len(exchange_rate_df)} items.")

    # Prepare the output DataFrame
    final_df = df.copy()
    final_df["FX Code"] = ""
    final_df["FX Rate"] = np.nan
    final_df["Price"] = np.nan
    final_df["Currency"] = "GBP"
    final_df["Conversion"] = ""
    final_df["Outcome"] = ""

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
                final_df.at[index, "Conversion"] = "no change"
                final_df.at[index, "Outcome"] = "no change"
                continue

            # Handle GBP and GBp cases
            if currency == "GBP":
                fx_ticker = "-"
                exchange_rate = 1
                conversion = "no change"
            elif currency == "GBp":
                fx_ticker = "-"
                exchange_rate = 0.01
                conversion = "GBp to GBP"
            else:
                # Create FX ticker for non-GBP currencies
                fx_ticker = f"{currency}GBP=X" if currency != "USD" else "GBP=X"
                conversion = f"{currency} to GBP"
                # Retrieve exchange rate using date objects
                exchange_rate = exchange_rate_df.loc[(date, fx_ticker), "FX Rate"]

            # Calculate the converted price
            converted_price = price * exchange_rate

            # Populate the row with updated data
            final_df.at[index, "FX Code"] = fx_ticker
            final_df.at[index, "FX Rate"] = exchange_rate
            final_df.at[index, "Price"] = converted_price
            final_df.at[index, "Currency"] = "GBP"
            final_df.at[index, "Conversion"] = conversion
            final_df.at[index, "Outcome"] = "Success"

        except KeyError:
            logger.error(
                f"FX rate not found for {currency} ({fx_ticker}) on date {date}."
            )
            final_df.at[index, "Outcome"] = "No FX rate"
        except Exception as e:
            logger.error(f"Error processing row {index}: {e}")
            final_df.at[index, "Outcome"] = "Error"

    # Log overall success or failure
    if final_df["Outcome"].eq("Error").any():
        logger.warning("Some tickers encountered errors during processing.")
    else:
        logger.info("All tickers processed successfully.")

    return final_df


@log_function
def process_converted_prices(
    converted_df: pd.DataFrame,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    try:
        # converted_df contains Ticker, Name, Date, TimeZone, Type, Old Price,  Original Currency, FX Code, FX Rate, Conversion, Price, Currency,  Outcome

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
        converted_df[["Old Price", "FX Rate", "Price"]] = converted_df[
            ["Old Price", "FX Rate", "Price"]
            ].round(3)

        # Combine the price and currency columns
        converted_df["Old Price"] = (
            converted_df["Old Price"].astype(str)
            + " " + converted_df["Original Currency"]
        )
        converted_df["Price"] = (
            converted_df["Price"].astype(str) 
            + " " + converted_df["Currency"]
        )

        # Calculate how many days of data were collected per ticker (not the diffrence between last date and first date)
        # Calculate the number of distinct days for each ticker
        grouped_data = (
            converted_df.groupby("Ticker")["Date"]
            .nunique()
            .reset_index(name="Days")
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
            & (converted_df["Outcome"] != "no change")
        ]
        logger.debug(f"{len(failure)} item 'failure' dataframe created successfully.")

        # Make No change DataFrame ########################################################
        no_change = converted_df[
            (converted_df["Outcome"] == "no change")
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
        print(f"\t {title}")
        print(f"\t {header_line}")


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
    success: pd.DataFrame = pd.DataFrame(),
    failure: pd.DataFrame = pd.DataFrame(),
    no_change: pd.DataFrame = pd.DataFrame(),
    index_currency: pd.DataFrame = pd.DataFrame(),
    invalid_tickers: List[str] = [],
    out_of_range: List[str] = []
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
    total = len(success) + len(failure)
    success_rate = 0 if total == 0 else (len(success) / total) * 100

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
        latest_prices.sort_values(by=["Type", "Ticker"], inplace=True)
        latest_prices_columns = [
            "Ticker",
            "Date",
            "Type",
            "Old Price",
            "FX Code",
            "FX Rate",
            "Days",
            "Price",
        ]
    else:
        latest_prices = pd.DataFrame()

    latest_prices = latest_prices.copy()
    latest_prices["Date"] = (
        pd.to_datetime(latest_prices["Date"]).dt.strftime("%d/%m/%Y")
    )

    # Prepare index_currency_prices
    if not index_currency.empty:
        index_currency['latest_date'] = index_currency.groupby('Ticker')['Date'].transform('max')
        index_currency = index_currency[index_currency['Date'] == index_currency['latest_date']]
        index_currency.sort_values(by=["Type", "Ticker"])
        index_currency_columns = [
            "Ticker",
            "Date",
            "Type",
            "Days",
            "Price",
        ]

    index_currency = index_currency.copy()
    index_currency["Date"] = pd.to_datetime(index_currency["Date"]).dt.strftime(
        "%d/%m/%Y"
    )

    # Prepare failure
    if not failure.empty:
        failure.rename(columns={"Outcome": "Failure Reason" }, inplace=True )
        failure_columns = [
            "Ticker",
            "Date",
            "Type",
            "Old Price",
            "FX Code",
            "FX Rate",
            "Failure Reason",
        ]
    failure = failure.copy()
    failure["Date"] = pd.to_datetime(failure["Date"]).dt.strftime("%d/%m/%Y")

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
    unchanged_index_tickers = index_currency[
        index_currency["Type"] == "INDEX"][
        "Ticker" ].nunique()
    unchanged_currency_tickers = index_currency[
        index_currency["Type"] == "CURRENCY"][
        "Ticker"].nunique()

    # Calculate conversion counts (key-value pair eg USD to GBP: 2)
    conversion_counts = success['Conversion'].value_counts()

    # Prepare invalid ticker and out_of_range strings
    invalid_tickers_str = ", ".join([f"'{item}'" for item in invalid_tickers])
    out_of_range_str = ", ".join([f"'{item}'" for item in out_of_range])
    invalid_tickers_count = len(invalid_tickers)
    out_of_range_count = len(out_of_range)

    # Get date range
    start_date, end_date, business_days = get_date_range()

    # Construct the summary message
    summary = f"""{start_date.strftime("%d/%m/%Y")} - {end_date.strftime("%d/%m/%Y")}
         {business_days} business days      
         {'\n\t '.join([f"{pluralise('Ticker', count)} {conversion}: {count}" for conversion, count in conversion_counts.items()])}
         Conversion success rate: {success_rate:.1f}%
         Index tickers: {unchanged_index_tickers}
         Currency tickers obtained: {unchanged_currency_tickers}
         Unrecognised {pluralise('ticker', invalid_tickers_count)}: {" and ".join(invalid_tickers) if invalid_tickers_count == 1 else f"{invalid_tickers_count} {invalid_tickers_str}"}
         {pluralise('Ticker', out_of_range_count)} with no data: {" and ".join(out_of_range) if out_of_range_count == 1 else f"{out_of_range_count} {out_of_range_str}"}
         Total processed: {len(config.tickers)}"""

    # --------------------
    # Generate Outputs
    # --------------------
    print("")
    header("Stock Price Update", logger=logger,major=True)

    logger.info(summary)

    if not latest_prices.empty:
        log_table("Latest Prices", latest_prices, latest_prices_columns)
    else:
        logger.text("\n\t There were no converted prices.")
    if not index_currency.empty:
        log_table("Index and Currency Prices", index_currency, index_currency_columns)
    else:
        logger.text("\n\t There were no index or currency prices.")
    if not failure.empty:
        log_table("Failed Conversions", failure, failure_columns)
    else:
        logger.text("\n\t There were no failed conversions.")
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
# Seek Elevation                                                                    |
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
        logger.debug("Attempting to open new elevated script...")
        return_code= ctypes.windll.shell32.ShellExecuteW(None, "runas", sys.executable, __file__, None, 1)
        if return_code == 5:
            # User rejected the elevation prompt
            logger.info("Elevation request cancelled.")
            logger.info("Quicken will not upload data automatically. Continuing unelevated...\n")
        elif return_code == 42:
            # User accepted elevation prompt
            logger.warning("Elevation request accepted, script restarting in terminal.")
            exit(0)
        else:
            logger.warning(f"Unknown situation. Code: {return_code}")
            exit(0)

#
# -----------------------------------------------------------------------------------
# Quicken UI                                                                        |
# -----------------------------------------------------------------------------------
# Opens and navigates Quicken XG 2004 to import the csv file.
#
import pyautogui
import time


@retry(exceptions=(RuntimeError, IOError), tries=3, delay=2, backoff=2)
def import_data_file():
    """
    Import the data file through the import dialog.

    Returns:
        bool: True if import successful
    """
    try:
        logger.debug("Entering filename...")
        output_file_name = config.paths.data_file
        filename = os.path.join(base_directory, output_file_name)

        # Type the file name into the dialog
        pyautogui.typewrite(filename, interval=0.01)
        time.sleep(0.3)  # Small delay before hitting enter for stability
        pyautogui.press("enter")
        logger.debug("Data upload in progress...")

        # Wait to get to the price import success dialogue (timeout = 5 seconds)
        start_time = time.time()
        while time.time() - start_time < 5:
            windows = gw.getWindowsWithTitle(
                "Quicken XG 2004 - Home - [Investing Centre]"
            )
            if windows:
                logger.debug("Price import success box opened.")
                pyautogui.press("enter")
                logger.text(
                    f"Successfully imported {format_path(filename)} to Quicken at {datetime.now().strftime('%d/%m/%Y %H:%M')}"
                )
                return True
            time.sleep(0.5)

    except Exception as e:
        logger.error(f"Failed to import data file: {e} ({type(e).__name__})")
        raise


@retry(exceptions=(Exception,), tries=3, delay=2, backoff=2)
def open_import_dialog():
    """
    Open the import dialog window using keyboard shortcuts.

    Returns:
        bool: True if dialog opened successfully
    """
    try:
        with pyautogui.hold("alt"):
            pyautogui.press(["f", "i", "i"])  # Open the import dialog in Quicken

        # Wait to get to the price import dialogue box (timeout = 5 seconds)
        start_time = time.time()
        while time.time() - start_time < 5:
            windows = gw.getWindowsWithTitle(
                "Import Price Data"
            )
            if windows:
                logger.debug("Price import dialogue box opened.")
                return True
            time.sleep(1)

        return True
    except Exception as e:
        logger.error(f"Failed to open price import dialogue box: {e}")
        return False


@retry(exceptions=(Exception,), tries=3, delay=2, backoff=2)
def navigate_to_portfolio():
    """
    Navigate to the portfolio view in Quicken.

    Returns:
        bool: True if navigation successful
    """
    try:
        pyautogui.hotkey("ctrl", "u")

        # Wait to get to the investing portfolio page (timeout = 5 seconds)
        start_time = time.time()
        while time.time() - start_time < 10:
            windows = gw.getWindowsWithTitle(
                "Quicken XG 2004 - Home - [Investing Centre]"
            )
            if windows:
                logger.debug("At portfolio page.")
                return True
            time.sleep(1)

        # If we cannot find the window after 10 seconds
        logger.error("Could not get to portfolio page within the expected time.")
        return False

    except Exception as e:
        logger.error(f"Failed to navigate to portfolio: {e}")
        return False


@retry(exceptions=(Exception,), tries=3, delay=2, backoff=2)
def open_quicken():
    """
    Launch Quicken application and wait until the window is open.

    Returns:
        bool: True if Quicken started successfully
    """
    quicken_path = config.paths.quicken
    try:
        # Check if Quicken is already open
        windows = gw.getWindowsWithTitle("Quicken XG 2004")
        if len(windows) > 0:
            logger.info("Quicken is already open.")
            quicken_window = windows[0]
            quicken_window.activate()  # Bring the existing Quicken window to the foreground
            return True

        # If not open, launch Quicken
        logger.debug("Quicken launching...")
        subprocess.Popen([quicken_path], shell=True)

        # Wait for the Quicken window to appear (timeout = 20 seconds)
        start_time = time.time()
        while time.time() - start_time < 20:
            windows = gw.getWindowsWithTitle("Quicken XG 2004")
            if windows:
                logger.debug("Quicken is ready.")
                # Temporarily disable Ctrl+Alt+Del to prevent accidental interruption
                pyautogui.hotkey("ctrl", "alt", "del")
                return True
            time.sleep(1)

        # If we cannot find the window after 20 seconds
        logger.error("Quicken window did not appear within the expected time.")
        return False

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
            logger.error(f"Step failed: {message}")
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
    pyautogui.FAILSAFE = True  # Abort ongoing automation by moving mouse to the top-left corner of screen.
    pyautogui.PAUSE = 0.3  # Delay between actions


def quicken_import():
    """
    Main entry point for handling Quicken import.
    """
    try:
        logger.text("")
        header("Import to Quicken 2004", logger=logger, major=True)
        if is_elevated():
            logger.info("Confirmed 'is elevated'.")
            return execute_import_sequence()
        else:
            header("END", logger=logger,major=True)
        return

    except Exception as e:
        logger.error(f"Error during Quicken import: {e}")
        return

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

        # Conditional clear screen at beginning for clean output
        if config.clear_startup == True:
            os.system("cls" if os.name == "nt" else "clear")

        # Get elevation
        run_as_admin()

        # Get date range
        start_date, end_date, business_days = get_date_range()

        # Get 'raw' tickers from YAML file and FX tickers if required, validate and acquire metadata.
        ticker_list= config.tickers    
        valid_tickers, invalid_tickers = get_tickers(ticker_list, set_maximum_tickers=5 )
        # valid_tickers includes: ticker, name, earliest_date, time_zone, type, currency

        # Fetch historical price and FX data
        price_data, out_of_range = fetch_historical_data(valid_tickers)
        # 'price_data' includes ticker, name, price, date, time_zone, type, currency.
        # 'out_of_range' contains tickers with insufficient historical data for the specified period.

        # Convert prices to GBP except INDEX and CURRENCY
        processed_data = convert_prices(price_data)
        # processed_data contains Ticker, Name, Date, TimeZone, Type,Old Price,  Original Currency, FX Code, FX Rate, Conversion, Price, Currency,  Outcome

        # Process and create output dataframes
        output_csv, success, failure, no_change, index_currency = (
            process_converted_prices(processed_data)
        )

        # Produce tables using processed_data instead of output_csv
        prepare_tables(success, failure, no_change, index_currency, invalid_tickers, out_of_range)

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
    
    
    
    
    # Ensure base directory exists
    base_directory = config.paths.base
    os.makedirs(base_directory, exist_ok=True)
    
    # Setup logging
    logger = setup_logging(config)
    
    # Run main sequence
    main()

#
# -----------------------------------------------------------------------------------
# END                                                                               |
# -----------------------------------------------------------------------------------
