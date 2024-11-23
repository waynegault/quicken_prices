#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
QuickenPrices.py

This script fetches historical price data for a list of tickers specified in 'config.yaml' and saves the data to a CSV file.
It supports using a simulator for API calls or fetching real data using yfinance.
"""

import os
import sys
import logging
import logging.handlers
import yaml
import json
import datetime
import pandas as pd
from typing import List, Dict, Any, Optional

# Check if yfinance is installed, if not, prompt the user
try:
    import yfinance as yf
except ImportError:
    print(
        "The 'yfinance' library is not installed. Please install it using 'pip install yfinance'"
    )
    sys.exit(1)

# -----------------------------
# Load Configuration
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
base_directory = config["paths"]["base"]
if not os.path.exists(base_directory):
    os.makedirs(base_directory)

# -----------------------------
# Configure Logging
# -----------------------------

log_levels = {
    "DEBUG": logging.DEBUG,
    "INFO": logging.INFO,
    "WARNING": logging.WARNING,
    "ERROR": logging.ERROR,
}

logger = logging.getLogger("QuickenPrices")
logger.setLevel(logging.DEBUG)

# Create console handler
console_handler = logging.StreamHandler()
console_handler.setLevel(
    log_levels.get(config["logging"]["levels"]["console"].upper(), logging.INFO)
)
console_formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
console_handler.setFormatter(console_formatter)
logger.addHandler(console_handler)

# Create file handler with rotating logs
log_file_path = os.path.join(base_directory, config["paths"]["log_file"])
file_handler = logging.handlers.RotatingFileHandler(
    log_file_path,
    maxBytes=config["logging"]["max_bytes"],
    backupCount=config["logging"]["backup_count"],
)
file_handler.setLevel(
    log_levels.get(config["logging"]["levels"]["file"].upper(), logging.DEBUG)
)
file_formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
file_handler.setFormatter(file_formatter)
logger.addHandler(file_handler)

# -----------------------------
# Decorator for Function Logging
# -----------------------------


def log_function(func):
    """
    Decorator to log entry and exit of functions.
    """

    def wrapper(*args, **kwargs):
        logger.debug(f"Entering function: {func.__name__}")
        result = func(*args, **kwargs)
        logger.debug(f"Exiting function: {func.__name__}")
        return result

    return wrapper


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
    return all_tickers


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
            history = ticker.history(period="1d")
            if not info or history.empty:
                raise ValueError(f"No data found for ticker {ticker_symbol}")
            valid_tickers.append(ticker_symbol)
        except Exception as e:
            logger.warning(f"Ticker {ticker_symbol} is invalid: {e}")
            invalid_tickers.append(ticker_symbol)

    if invalid_tickers:
        logger.warning(f"Invalid tickers: {', '.join(invalid_tickers)}")

    return valid_tickers


# -----------------------------
# Ticker Data Fetching
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
    if config["use_simulator"]:
        return YFinanceSimulator(ticker_symbol)
    else:
        return yf.Ticker(ticker_symbol)


@log_function
def fetch_data(
    tickers: List[str], start_date: datetime.datetime, end_date: datetime.datetime
) -> Dict[str, pd.DataFrame]:
    """
    Fetch historical data for each ticker.
    """
    data = {}
    for ticker_symbol in tickers:
        try:
            ticker = get_ticker(ticker_symbol)
            df = ticker.history(
                start=start_date.strftime("%Y-%m-%d"), end=end_date.strftime("%Y-%m-%d")
            )
            if df.empty:
                logger.warning(f"No historical data for ticker {ticker_symbol}")
                continue
            data[ticker_symbol] = df[["Close"]].copy()
            logger.info(f"Fetched data for ticker {ticker_symbol}")
        except Exception as e:
            logger.warning(f"Failed to fetch data for ticker {ticker_symbol}: {e}")
    return data


# -----------------------------
# Data Processing and Saving
# -----------------------------


@log_function
def process_data(data: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    """
    Process the data into a single DataFrame.
    """
    combined_df = pd.DataFrame()
    for ticker_symbol, df in data.items():
        df = df.rename(columns={"Close": ticker_symbol})
        if combined_df.empty:
            combined_df = df
        else:
            combined_df = combined_df.join(df, how="outer")
    combined_df.sort_index(inplace=True)
    return combined_df


@log_function
def save_to_csv(df: pd.DataFrame, output_file: str):
    """
    Save the DataFrame to a CSV file.
    """
    df.to_csv(output_file)
    logger.info(f"Data saved to: {output_file}")


# -----------------------------
# Main Function
# -----------------------------


@log_function
def main():
    """
    Main function to orchestrate data fetching and processing.
    """
    tickers = get_tickers(config)
    valid_tickers = validate_tickers(tickers)
    if not valid_tickers:
        logger.error("No valid tickers to process. Exiting.")
        return

    # Calculate date range
    end_date = datetime.datetime.now()
    start_date = end_date - datetime.timedelta(days=config["fetch"]["days"])
    logger.info(
        f"Range {start_date.strftime('%d/%m/%Y')} - {end_date.strftime('%d/%m/%Y')}:"
    )

    # Fetch data
    data = fetch_data(valid_tickers, start_date, end_date)

    if not data:
        logger.error("No data fetched for any tickers. Exiting.")
        return

    # Process data
    combined_df = process_data(data)

    # Save data to CSV
    output_file = os.path.join(base_directory, config["paths"]["output_file"])
    save_to_csv(combined_df, output_file)


if __name__ == "__main__":
    main()
