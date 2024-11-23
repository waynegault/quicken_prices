import logging
import os
import time
from typing import List, Dict, Any


# YFinance Simulator for testing
class YFinanceSimulator:
    def fetch_data(self, ticker: str, period: str) -> Dict[str, Any]:
        simulated_data = {
            "AAPL": {"quoteType": "EQUITY", "prices": [150, 152, 151, 149, 150]},
            "EURUSD=X": {
                "quoteType": "CURRENCY",
                "prices": [1.1, 1.12, 1.11, 1.13, 1.1],
            },
            "GC=F": {"quoteType": "FUTURE", "prices": [1800, 1810, 1795, 1805, 1815]},
            "FAKE": {"quoteType": None, "prices": None},
            "NONE": {"quoteType": None, "prices": None},
        }
        if ticker in simulated_data:
            return simulated_data[ticker]
        raise ValueError(f"No data found for ticker: {ticker}")


# Logger Configuration
class CentralizedLogger:
    logger = None

    @staticmethod
    def setup(config: Dict[str, Any]):
        log_file = os.path.join(config["paths"]["base"], config["paths"]["log_file"])
        log_level_file = config["logging"]["levels"]["file"]
        log_level_terminal = config["logging"]["levels"]["terminal"]

        # Create logger
        CentralizedLogger.logger = logging.getLogger("QuickenPricesLogger")
        CentralizedLogger.logger.setLevel(logging.DEBUG)

        # File handler
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(getattr(logging, log_level_file))
        file_formatter = logging.Formatter(
            config["logging"]["message_formats"]["file"]["basic"]
        )
        file_handler.setFormatter(file_formatter)

        # Console handler
        console_handler = logging.StreamHandler()
        console_handler.setLevel(getattr(logging, log_level_terminal))
        console_formatter = logging.Formatter(
            config["logging"]["message_formats"]["terminal"]["basic"]
        )
        console_handler.setFormatter(console_formatter)

        # Add handlers
        CentralizedLogger.logger.addHandler(file_handler)
        CentralizedLogger.logger.addHandler(console_handler)


# Retry Decorator
def retry(max_retries: int, delay: int):
    def decorator(func):
        def wrapper(*args, **kwargs):
            for attempt in range(max_retries):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    if attempt < max_retries - 1:
                        CentralizedLogger.logger.warning(
                            f"Retry {attempt + 1}/{max_retries} failed. Reason: {str(e)}"
                        )
                        time.sleep(delay)
                    else:
                        raise e

        return wrapper

    return decorator


# Error Handler
def handle_error(message: str, error: Exception):
    CentralizedLogger.logger.error(f"{message}: {str(error)}")


# Data Fetcher
class DataFetcher:
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.simulator = YFinanceSimulator()

    @retry(max_retries=3, delay=2)
    def fetch_ticker_data(self, ticker: str):
        if not hasattr(self, "_fetch_attempts"):
            self._fetch_attempts = {}
        self._fetch_attempts[ticker] = self._fetch_attempts.get(ticker, 0) + 1

        if self._fetch_attempts[ticker] == 1:
            CentralizedLogger.logger.info(f"Fetching data for ticker: {ticker}")

        data = self.simulator.fetch_data(ticker, period="5d")
        if not data or "prices" not in data or not data["prices"]:
            raise ValueError("No data found or invalid ticker type")
        return data

    def process_tickers(self, tickers: List[str]):
        results = {"success": [], "failed": []}
        for ticker in tickers:
            try:
                data = self.fetch_ticker_data(ticker)
                results["success"].append({"ticker": ticker, "type": data["quoteType"]})
                CentralizedLogger.logger.info(
                    f"Ticker: {ticker} ({data['quoteType']}) processed successfully."
                )
            except Exception as e:
                results["failed"].append({"ticker": ticker, "reason": str(e)})
                if self._fetch_attempts.get(ticker, 1) == 3:
                    CentralizedLogger.logger.error(
                        f"Ticker: {ticker} failed after 3 retries. Reason: {str(e)}"
                    )
                else:
                    CentralizedLogger.logger.warning(
                        f"Ticker: {ticker} - Retry {self._fetch_attempts.get(ticker)} failed. Reason: {str(e)}"
                    )
        return results


# Main Script
if __name__ == "__main__":
    # Config placeholder
    config = {
        "paths": {
            "base": "./",
            "log_file": "test.log",
        },
        "logging": {
            "levels": {
                "file": "DEBUG",
                "terminal": "INFO",
            },
            "message_formats": {
                "file": {"basic": "%(asctime)s - %(levelname)s - %(message)s"},
                "terminal": {"basic": "%(asctime)s - %(levelname)s - %(message)s"},
            },
        },
    }

    # Setup Logger
    CentralizedLogger.setup(config)

    # Simulated tickers
    tickers = ["AAPL", "EURUSD=X", "GC=F", "FAKE", "NONE"]

    # Fetch and process data
    fetcher = DataFetcher(config)
    results = fetcher.process_tickers(tickers)

    # Final summary
    CentralizedLogger.logger.info("Processing complete.")
    CentralizedLogger.logger.info(f"Successful Tickers: {len(results['success'])}")
    CentralizedLogger.logger.info(f"Failed Tickers: {len(results['failed'])}")
