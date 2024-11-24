#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
QuickenPrices.py

This script fetches historical price data for a list of tickers specified in 'config.yaml' and saves the data to a CSV file.
It supports using a simulator for API calls or fetching real data using yfinance.

It includes features such as configuration management, logging, error handling, data fetching, data processing,
currency conversion, and more, as specified in the master list of functionalities.
"""

import os
import sys
import logging
import logging.handlers
import yaml
import json
import datetime
import time
import pandas as pd
import numpy as np
import functools
import traceback
import shutil
import pickle
from typing import List, Dict, Any

# Check if required libraries are installed, if not, prompt the user
try:
    import yfinance as yf
    import pytz
    import colorama
    from colorama import Fore, Style
    import pyperclip
    import pyautogui
    import ctypes
except ImportError as e:
    print(
        f"The required library '{e.name}' is not installed. Please install it using 'pip install {e.name}'"
    )
    sys.exit(1)

# Initialize colorama
colorama.init(autoreset=True)

# -----------------------------
# Configuration Management
# -----------------------------


def load_configuration(config_file: str = "config.yaml") -> Dict[str, Any]:
    """
    Load configuration from the specified YAML file.
    """
    with open(config_file, "r") as f:
        config = yaml.safe_load(f)
    return config


config = load_configuration()

# Ensure the base directory exists
base_directory = config.get("paths", {}).get("base", ".")
if not os.path.exists(base_directory):
    os.makedirs(base_directory)

# -----------------------------
# Centralized Logger
# -----------------------------


def setup_logging():
    """
    Set up logging for the script.
    """
    log_levels = {
        "DEBUG": logging.DEBUG,
        "INFO": logging.INFO,
        "WARNING": logging.WARNING,
        "ERROR": logging.ERROR,
    }

    logger = logging.getLogger("QuickenPrices")
    logger.setLevel(logging.DEBUG)

    # Create console handler with color support
    console_handler = logging.StreamHandler()
    console_level = config.get("console_loglevel", "INFO").upper()
    console_handler.setLevel(log_levels.get(console_level, logging.INFO))
    console_formatter = CustomFormatter()
    console_handler.setFormatter(console_formatter)
    logger.addHandler(console_handler)

    # Create file handler with rotating logs
    log_file_path = os.path.join(
        base_directory, config.get("paths", {}).get("log_file", "prices.log")
    )
    file_handler = logging.handlers.RotatingFileHandler(
        log_file_path,
        maxBytes=config.get("logging", {}).get("max_bytes", 10485760),  # Default 10 MB
        backupCount=config.get("logging", {}).get("backup_count", 5),
    )
    file_level = config.get("file_loglevel", "DEBUG").upper()
    file_handler.setLevel(log_levels.get(file_level, logging.DEBUG))
    file_formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    file_handler.setFormatter(file_formatter)
    logger.addHandler(file_handler)

    return logger


class CustomFormatter(logging.Formatter):
    """
    Custom logging formatter with color support.
    """

    def format(self, record):
        log_fmt = "%(asctime)s - %(levelname)s - %(message)s"
        formatter = logging.Formatter(log_fmt)
        if record.levelno == logging.DEBUG:
            record.msg = f"{Fore.BLUE}{record.msg}{Style.RESET_ALL}"
        elif record.levelno == logging.INFO:
            record.msg = f"{Fore.GREEN}{record.msg}{Style.RESET_ALL}"
        elif record.levelno == logging.WARNING:
            record.msg = f"{Fore.YELLOW}{record.msg}{Style.RESET_ALL}"
        elif record.levelno == logging.ERROR:
            record.msg = f"{Fore.RED}{record.msg}{Style.RESET_ALL}"
        return formatter.format(record)


logger = setup_logging()

# -----------------------------
# Decorators and Utilities
# -----------------------------


def log_function(func):
    """
    Decorator to log entry and exit of functions.
    """

    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        logger.debug(f"Entering function: {func.__name__}")
        try:
            result = func(*args, **kwargs)
            logger.debug(f"Exiting function: {func.__name__}")
            return result
        except Exception as e:
            logger.error(f"Error in function {func.__name__}: {e}")
            logger.error(traceback.format_exc())
            raise

    return wrapper


def retry(exceptions, tries=3, delay=1, backoff=2):
    """
    Retry decorator with exponential backoff.
    """

    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            mtries, mdelay = tries, delay
            while mtries > 1:
                try:
                    return func(*args, **kwargs)
                except exceptions as e:
                    logger.warning(f"{e}, Retrying in {mdelay} seconds...")
                    time.sleep(mdelay)
                    mtries -= 1
                    mdelay *= backoff
            return func(*args, **kwargs)

        return wrapper

    return decorator


def is_admin():
    """
    Check if the script is running with administrative privileges.
    """
    try:
        return ctypes.windll.shell32.IsUserAnAdmin()
    except:
        return False


def run_as_admin():
    """
    Restart the script with administrative privileges.
    """
    ctypes.windll.shell32.ShellExecuteW(
        None, "runas", sys.executable, " ".join(sys.argv), None, 1
    )


def copy_to_clipboard(text: str):
    """
    Copy the given text to the clipboard.
    """
    pyperclip.copy(text)


# -----------------------------
# Ticker Handling
# -----------------------------


@log_function
def get_tickers(config: Dict[str, Any]) -> List[str]:
    """
    Get the list of tickers from the configuration, including currency pairs.
    """
    tickers = config.get("tickers", [])
    # Get tickers from 'currency_pairs'
    currency_pairs = config.get("currency_pairs", [])
    currency_tickers = [pair["symbol"] for pair in currency_pairs]
    # Combine both lists
    all_tickers = tickers + currency_tickers
    # Remove duplicates and ensure tickers are uppercase
    all_tickers = list(set([ticker.upper() for ticker in all_tickers]))
    return all_tickers


@log_function
def validate_tickers(tickers: List[str]) -> List[tuple]:
    """
    Validate tickers and categorize by quoteType.
    """
    valid_tickers = []
    invalid_tickers = []
    for ticker_symbol in tickers:
        try:
            ticker = get_ticker(ticker_symbol)
            info = ticker.info
            quote_type = info.get("quoteType", None)
            if not quote_type:
                raise ValueError(f"No quoteType found for ticker {ticker_symbol}")
            valid_tickers.append((ticker_symbol, quote_type))
        except Exception as e:
            logger.warning(f"Ticker {ticker_symbol} is invalid: {e}")
            invalid_tickers.append(ticker_symbol)
    if invalid_tickers:
        logger.warning(f"Invalid tickers: {', '.join(invalid_tickers)}")
    return valid_tickers


# -----------------------------
# Data Fetching and Caching
# -----------------------------


class YFinanceSimulator:
    def __init__(self, ticker: str):
        self.ticker = ticker.upper()
        self.data = self.load_data()
        if self.ticker not in self.data.get("tickers", {}):
            raise ValueError(f"No data available for ticker: {self.ticker}")
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
        with open("simulator_data.json", "r") as f:
            return json.load(f)

    def history(self, period: str = "1mo", start=None, end=None) -> pd.DataFrame:
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
    ticker_symbol: str, start_date: datetime.datetime, end_date: datetime.datetime
) -> pd.DataFrame:
    """
    Fetch historical data for a single ticker.
    """
    cache_dir = os.path.join(base_directory, "cache")
    if not os.path.exists(cache_dir):
        os.makedirs(cache_dir)
    cache_file = os.path.join(
        cache_dir,
        f"{ticker_symbol}_{start_date.strftime('%Y%m%d')}_{end_date.strftime('%Y%m%d')}.pkl",
    )
    if os.path.exists(cache_file):
        logger.info(f"Loading cached data for ticker {ticker_symbol}")
        with open(cache_file, "rb") as f:
            df = pickle.load(f)
    else:
        ticker = get_ticker(ticker_symbol)
        df = ticker.history(
            start=start_date.strftime("%Y-%m-%d"), end=end_date.strftime("%Y-%m-%d")
        )
        if df.empty:
            logger.warning(f"No historical data for ticker {ticker_symbol}")
            return pd.DataFrame()
        with open(cache_file, "wb") as f:
            pickle.dump(df, f)
    return df


@log_function
def fetch_data(
    tickers: List[tuple], start_date: datetime.datetime, end_date: datetime.datetime
) -> pd.DataFrame:
    """
    Fetch historical data for each ticker and return a DataFrame with columns: 'Ticker', 'Date', 'Price'.
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
            logger.info(
                f"{Fore.GREEN}✓ Fetched data for ticker {ticker_symbol}{Style.RESET_ALL}"
            )
        except Exception as e:
            logger.warning(f"Failed to fetch data for ticker {ticker_symbol}: {e}")
    if records:
        combined_df = pd.concat(records, ignore_index=True)
        return combined_df
    else:
        return pd.DataFrame()


# -----------------------------
# Data Processing
# -----------------------------


@log_function
def process_data(df: pd.DataFrame, currency_rates: Dict[str, float]) -> pd.DataFrame:
    """
    Process the data to have columns: 'Ticker', 'Date', 'Price'.
    """
    if df.empty:
        return df
    df = df[["Ticker", "Date", "Close", "QuoteType"]]
    df.rename(columns={"Close": "Price"}, inplace=True)
    # Convert prices to GBP where applicable
    for index, row in df.iterrows():
        ticker = row["Ticker"]
        price = row["Price"]
        quote_type = row["QuoteType"]
        if quote_type == "CURRENCY":
            # Invert the exchange rate to get the rate to GBP
            exchange_rate = price
            if exchange_rate != 0:
                df.at[index, "Price"] = 1 / exchange_rate
        elif quote_type == "FUND":
            # Handle funds differently if needed
            pass
        else:
            # Convert price to GBP if not already in GBP
            currency = get_currency_for_ticker(ticker)
            if currency != "GBP":
                rate = currency_rates.get(currency + "GBP=X", None)
                if rate:
                    df.at[index, "Price"] = price * rate
                else:
                    logger.warning(f"Missing exchange rate for {currency} to GBP")
    df["Date"] = df["Date"].dt.strftime("%d/%m/%Y")  # Format date as 'DD/MM/YYYY'
    return df[["Ticker", "Date", "Price"]]


def get_currency_for_ticker(ticker_symbol: str) -> str:
    """
    Get the currency for a given ticker symbol.
    """
    ticker = get_ticker(ticker_symbol)
    currency = ticker.info.get("currency", "GBP")
    return currency


@log_function
def get_exchange_rates(tickers: List[str]) -> Dict[str, float]:
    """
    Get exchange rates for the required currencies.
    """
    currencies = set()
    for ticker_symbol in tickers:
        currency = get_currency_for_ticker(ticker_symbol)
        if currency != "GBP":
            currencies.add(currency + "GBP=X")
    exchange_rates = {}
    for currency_pair in currencies:
        try:
            ticker = get_ticker(currency_pair)
            df = ticker.history(period="1d")
            if not df.empty:
                rate = df["Close"].iloc[-1]
                exchange_rates[currency_pair] = rate
            else:
                logger.warning(f"No data for exchange rate {currency_pair}")
        except Exception as e:
            logger.warning(f"Failed to fetch exchange rate {currency_pair}: {e}")
    return exchange_rates


# -----------------------------
# Output Generation
# -----------------------------


@log_function
def save_to_csv(df: pd.DataFrame, output_file: str):
    """
    Save the DataFrame to a CSV file without headers.
    """
    df.to_csv(output_file, index=False, header=False)
    logger.info(f"Data saved to: {output_file}")


@log_function
def copy_output_to_clipboard(df: pd.DataFrame):
    """
    Copy the CSV output to the clipboard.
    """
    csv_output = df.to_csv(index=False, header=False)
    copy_to_clipboard(csv_output)
    logger.info("CSV output copied to clipboard.")


# -----------------------------
# Cache Cleanup
# -----------------------------


@log_function
def clean_cache(cache_dir: str, max_age_days: int):
    """
    Clean up cache files older than max_age_days.
    """
    now = time.time()
    cutoff = now - (max_age_days * 86400)  # 86400 seconds in a day
    for filename in os.listdir(cache_dir):
        file_path = os.path.join(cache_dir, filename)
        if os.stat(file_path).st_mtime < cutoff:
            os.remove(file_path)
            logger.info(f"Deleted cache file: {file_path}")


# -----------------------------
# Main Function
# -----------------------------


@log_function
def main():
    """
    Main function to orchestrate data fetching and processing.
    """
    try:
        # Check for administrative privileges
        if not is_admin():
            logger.info(
                "Script is not running with administrative privileges. Restarting as admin..."
            )
            run_as_admin()
            sys.exit(0)

        tickers = get_tickers(config)
        valid_tickers = validate_tickers(tickers)
        if not valid_tickers:
            logger.error("No valid tickers to process. Exiting.")
            return

        # Calculate date range
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
                logger.error(f"Invalid date format in configuration: {ve}")
                return
        else:
            end_date = datetime.datetime.now()
            start_date = end_date - datetime.timedelta(
                days=fetch_config.get("days", 30)
            )

        logger.info(
            f"Fetching data from {start_date.strftime('%d/%m/%Y')} to {end_date.strftime('%d/%m/%Y')}"
        )

        # Fetch exchange rates
        exchange_rates = get_exchange_rates([t[0] for t in valid_tickers])

        # Fetch data
        raw_data = fetch_data(valid_tickers, start_date, end_date)

        if raw_data.empty:
            logger.error("No data fetched for any tickers. Exiting.")
            return

        # Process data
        processed_data = process_data(raw_data, exchange_rates)

        if processed_data.empty:
            logger.error("Processed DataFrame is empty. No data to save.")
            return

        # Save data to CSV
        output_file = os.path.join(
            base_directory, config.get("paths", {}).get("output_file", "data.csv")
        )
        save_to_csv(processed_data, output_file)

        # Copy output to clipboard
        copy_output_to_clipboard(processed_data)

        # Clean up cache
        cache_dir = os.path.join(base_directory, "cache")
        clean_cache(cache_dir, max_age_days=fetch_config.get("cache_max_age_days", 7))

        # GUI Automation for Quicken (Placeholder)
        # Automate Quicken GUI interactions if needed

    except KeyboardInterrupt:
        logger.warning("Script interrupted by user.")
    except Exception as e:
        logger.error(f"An unexpected error occurred: {e}")
        logger.error(traceback.format_exc())


if __name__ == "__main__":
    main()
