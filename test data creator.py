#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
generate_simulator_data.py

This script generates a simulator_data.json file by downloading two months
of historical data and metadata for all tickers specified in config.yaml,
including currency tickers listed under 'currency_pairs'.
"""

import os
import sys
import yaml
import json
import datetime
import logging
from typing import List, Dict, Any
import yfinance as yf
import pandas as pd

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("SimulatorDataGenerator")


def load_configuration(config_file: str = "config.yaml") -> Dict[str, Any]:
    """
    Load configuration from the specified YAML file.
    """
    with open(config_file, "r") as f:
        config = yaml.safe_load(f)
    return config


def get_tickers(config: Dict[str, Any]) -> List[str]:
    """
    Extract the list of tickers from the configuration, including currency pairs.
    """
    tickers = config.get("tickers", [])
    # Get tickers from 'currency_pairs'
    currency_pairs = config.get("currency_pairs", [])
    currency_tickers = [pair["symbol"] for pair in currency_pairs]
    # Combine both lists
    all_tickers = tickers + currency_tickers
    return all_tickers


def is_currency_ticker(ticker_symbol: str) -> bool:
    """
    Determine if the ticker symbol represents a currency pair.
    Currency tickers often end with '=X'.
    """
    return ticker_symbol.endswith("=X")


def fetch_ticker_data(
    ticker_symbol: str, start_date: str, end_date: str
) -> Dict[str, Any]:
    """
    Fetch historical data and metadata for a single ticker.
    """
    try:
        ticker = yf.Ticker(ticker_symbol)
        # Fetch metadata
        info = ticker.info

        # Identify if the ticker is a currency
        if is_currency_ticker(ticker_symbol):
            # For currency tickers, set the interval to daily
            interval = "1d"
            period = None  # Period parameter is None when using start and end dates
        else:
            interval = "1d"
            period = None

        # Fetch historical data
        history = ticker.history(start=start_date, end=end_date, interval=interval)
        if history.empty:
            logger.warning(f"No historical data for ticker {ticker_symbol}")
            return None
        # Reset index to get 'Date' as a column
        history.reset_index(inplace=True)
        # Convert datetime to string for JSON serialization
        history["Date"] = history["Date"].dt.strftime("%Y-%m-%d")
        # Keep only relevant columns
        history_records = history[["Date", "Close"]].to_dict(orient="records")
        # Structure the data
        data = {"info": info, "history": history_records}
        logger.info(f"Fetched data for ticker {ticker_symbol}")
        return data
    except Exception as e:
        logger.warning(f"Failed to fetch data for ticker {ticker_symbol}: {e}")
        return None


def generate_simulator_data(tickers: List[str]) -> Dict[str, Any]:
    """
    Generate simulator data for a list of tickers.
    """
    simulator_data = {"tickers": {}}
    # Calculate the date range for two months
    end_date = datetime.datetime.now()
    start_date = end_date - datetime.timedelta(days=60)  # Approximate two months
    end_date_str = end_date.strftime("%Y-%m-%d")
    start_date_str = start_date.strftime("%Y-%m-%d")
    logger.info(f"Fetching data from {start_date_str} to {end_date_str}")
    for ticker_symbol in tickers:
        logger.info(f"Processing ticker: {ticker_symbol}")
        data = fetch_ticker_data(
            ticker_symbol, start_date=start_date_str, end_date=end_date_str
        )
        if data:
            simulator_data["tickers"][ticker_symbol] = data
    return simulator_data


def save_simulator_data(
    simulator_data: Dict[str, Any], output_file: str = "simulator_data.json"
):
    """
    Save the simulator data to a JSON file.
    """
    with open(output_file, "w") as f:
        json.dump(simulator_data, f, indent=2)
    logger.info(f"Simulator data saved to {output_file}")


def main():
    """
    Main function to generate the simulator_data.json file.
    """
    # Load configuration
    config = load_configuration()
    # Get tickers from configuration
    tickers = get_tickers(config)
    if not tickers:
        logger.error("No tickers found in configuration. Exiting.")
        sys.exit(1)
    # Generate simulator data
    simulator_data = generate_simulator_data(tickers)
    if not simulator_data["tickers"]:
        logger.error("No data fetched for any tickers. Exiting.")
        sys.exit(1)
    # Save simulator data to JSON file
    save_simulator_data(simulator_data)


if __name__ == "__main__":
    main()
