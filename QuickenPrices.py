#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
QuickenPrices.py

This script fetches stock and exchange rate data for specified tickers,
processes the data according to the configuration, and generates a CSV file
compatible with Quicken 2004 UK edition. It supports switching between
using the real yfinance API and a simulator for testing purposes.

Author: [Your Name]
Date: [Current Date]
"""

import os
import sys
import yaml
import json
import time
import logging
import logging.handlers
import datetime
import pytz
import traceback

# Import optional dependencies
# import pyperclip  # For clipboard functionality (to be implemented)
# import pyautogui  # For GUI automation (to be implemented)
# import ctypes     # For checking admin privileges (to be implemented)
from functools import wraps
from typing import List, Dict, Any, Optional
import pandas as pd

# Load configuration from config.yaml
with open("config.yaml", "r") as f:
    config = yaml.safe_load(f)

# Determine whether to use the simulator or the real yfinance API
use_simulator = config.get("use_simulator", True)

# Clear the terminal screen at the start of the script
os.system("cls" if os.name == "nt" else "clear")

# Configure logging based on settings from config.yaml
log_levels = {
    "DEBUG": logging.DEBUG,
    "INFO": logging.INFO,
    "WARNING": logging.WARNING,
    "ERROR": logging.ERROR,
}

# Create logger
logger = logging.getLogger("QuickenPrices")
logger.setLevel(
    log_levels.get(config["logging"]["levels"]["file"].upper(), logging.DEBUG)
)

# Create file handler with rotating logs
log_file_path = os.path.join(config["paths"]["base"], config["paths"]["log_file"])
file_handler = logging.handlers.RotatingFileHandler(
    log_file_path,
    maxBytes=config["logging"]["max_bytes"],
    backupCount=config["logging"]["backup_count"],
)
file_handler.setLevel(
    log_levels.get(config["logging"]["levels"]["file"].upper(), logging.DEBUG)
)

# Create console handler for terminal output
console_handler = logging.StreamHandler()
console_handler.setLevel(
    log_levels.get(config["logging"]["levels"]["terminal"].upper(), logging.INFO)
)

# Create formatter and add it to the handlers
formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
file_handler.setFormatter(formatter)
console_handler.setFormatter(formatter)

# Add the handlers to the logger
logger.addHandler(file_handler)
logger.addHandler(console_handler)

# Define utility functions and classes


def log_function(func):
    """
    Decorator to log the entry and exit of functions.
    """

    @wraps(func)
    def wrapper(*args, **kwargs):
        logger.debug(f"Entering function: {func.__name__}")
        result = func(*args, **kwargs)
        logger.debug(f"Exiting function: {func.__name__}")
        return result

    return wrapper


class DateFormatter:
    """
    Utility class for date formatting and conversion.
    """

    @staticmethod
    def to_utc_iso(date: datetime.datetime) -> datetime.datetime:
        """
        Convert a date to UTC and normalize to ISO format.
        """
        if date.tzinfo is None:
            date = date.replace(tzinfo=pytz.UTC)
        else:
            date = date.astimezone(pytz.UTC)
        return date

    @staticmethod
    def to_ddmmyyyy(date: datetime.datetime) -> str:
        """
        Format a date as dd/mm/yyyy for display.
        """
        return date.strftime("%d/%m/%Y")

    @staticmethod
    def calculate_date_range(
        period_years: float,
    ) -> (datetime.datetime, datetime.datetime):
        """
        Calculate the start and end dates based on the period in years.
        """
        end_date = datetime.datetime.now(pytz.UTC)
        start_date = end_date - datetime.timedelta(days=period_years * 365)
        return start_date, end_date


# Define the YFinanceSimulator class directly in this script
class YFinanceSimulator:
    def __init__(self, ticker: str):
        self.ticker = ticker.upper()
        self.data = self.load_data()
        if self.ticker not in self.data["tickers"]:
            raise ValueError(f"No data available for ticker: {self.ticker}")
        self.info = self.data["tickers"][self.ticker]["info"]

    @staticmethod
    def load_data() -> Dict[str, Any]:
        with open("simulator_data.json", "r") as f:
            return json.load(f)

    def history(self, period: str = "1mo") -> pd.DataFrame:
        records = self.data["tickers"][self.ticker]["history"]
        if not records:
            return pd.DataFrame()
        df = pd.DataFrame.from_records(records)
        df["Date"] = pd.to_datetime(df["Date"])
        df.set_index("Date", inplace=True)
        return df

    @property
    def actions(self) -> pd.DataFrame:
        return pd.DataFrame()

    @property
    def dividends(self) -> pd.DataFrame:
        return pd.DataFrame()

    @property
    def splits(self) -> pd.DataFrame:
        return pd.DataFrame()


# Determine whether to use the simulator or the real yfinance API
if use_simulator:
    # Use the YFinanceSimulator class defined above
    yf = YFinanceSimulator
    logger.info("Using simulator for API calls.")
else:
    # Import the real yfinance API
    import yfinance as yf

    logger.info("Using yfinance for API calls.")


def get_ticker(ticker_symbol: str):
    """
    Get the ticker object using the appropriate API.
    """
    if use_simulator:
        return yf(ticker_symbol)
    else:
        return yf.Ticker(ticker_symbol)


@log_function
def load_configuration() -> Dict[str, Any]:
    """
    Load and validate the configuration from config.yaml.
    """
    # Configuration has already been loaded at the top
    # Additional validation can be added here
    return config


@log_function
def validate_tickers(tickers: List[str]) -> List[str]:
    """
    Validate tickers by checking for duplicates and removing invalid ones.
    """
    unique_tickers = list(set(tickers))
    valid_tickers = []
    invalid_tickers = []

    for ticker_symbol in unique_tickers:
        try:
            ticker = get_ticker(ticker_symbol)
            info = ticker.info
            if not info:
                raise ValueError(f"No data found for ticker {ticker_symbol}")
            valid_tickers.append(ticker_symbol)
        except Exception as e:
            logger.warning(f"Ticker {ticker_symbol} is invalid: {e}")
            invalid_tickers.append(ticker_symbol)

    if invalid_tickers:
        logger.warning(f"Invalid tickers: {', '.join(invalid_tickers)}")

    return valid_tickers


@log_function
def fetch_data(tickers: List[str], period: str) -> Dict[str, Any]:
    """
    Fetch data for the given tickers and period.
    """
    data = {}
    for ticker_symbol in tickers:
        try:
            ticker = get_ticker(ticker_symbol)
            history = ticker.history(period=period)
            data[ticker_symbol] = {"info": ticker.info, "history": history}
            logger.info(f"Fetched data for ticker {ticker_symbol}")
        except Exception as e:
            logger.error(f"Failed to fetch data for ticker {ticker_symbol}: {e}")
    return data


@log_function
def process_data(data: Dict[str, Any]) -> pd.DataFrame:
    """
    Process the fetched data and prepare it for CSV output.
    """
    rows = []
    for ticker_symbol, ticker_data in data.items():
        history = ticker_data["history"]
        if history.empty:
            logger.warning(f"No historical data for ticker {ticker_symbol}")
            continue

        currency = ticker_data["info"].get("currency", "USD")
        for date, row in history.iterrows():
            date_utc = DateFormatter.to_utc_iso(date)
            close_price = row["Close"]
            # Apply currency conversion if necessary (to be implemented)
            rows.append(
                {
                    "Ticker": ticker_symbol,
                    "Date": DateFormatter.to_ddmmyyyy(date_utc),
                    "Price": close_price,
                    "Currency": currency,
                }
            )

    df = pd.DataFrame(rows)
    return df


@log_function
def save_to_csv(df: pd.DataFrame, file_path: str):
    """
    Save the processed DataFrame to a CSV file.
    """
    df.to_csv(file_path, index=False)
    logger.info(f"Data saved to: {file_path}")


@log_function
def main():
    """
    Main function to execute the script logic.
    """
    # Load configuration
    config = load_configuration()

    # Calculate date range
    period_years = config["collection"]["period_years"]
    start_date, end_date = DateFormatter.calculate_date_range(period_years)
    logger.info(
        f"Range {DateFormatter.to_ddmmyyyy(start_date)} - {DateFormatter.to_ddmmyyyy(end_date)}:"
    )
    # Calculate business days (to be implemented)

    # Validate tickers
    tickers = config["tickers"]
    valid_tickers = validate_tickers(tickers)

    if not valid_tickers:
        logger.error("No valid tickers to process. Exiting.")
        return

    # Fetch data
    default_period = "1mo"  # Default period; can be adjusted based on config
    data = fetch_data(valid_tickers, default_period)

    if not data:
        logger.error("No data fetched for any tickers. Exiting.")
        return

    # Process data
    df = process_data(data)

    if df.empty:
        logger.error("No data to save after processing. Exiting.")
        return

    # Save data to CSV
    data_file_path = os.path.join(config["paths"]["base"], config["paths"]["data_file"])
    save_to_csv(df, data_file_path)

    # Copy terminal output to clipboard (to be implemented)
    # Automate Quicken import (to be implemented)


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        logger.error(f"An unexpected error occurred: {e}")
        traceback.print_exc()
