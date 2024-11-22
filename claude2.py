"""
QuickenPrices - Stock Price Fetcher and Quicken Data Import Tool
Fetches stock prices and exchange rates, converts to GBP, and prepares for Quicken import.
"""

#################################################################
# Imports and Initialization                                    #
#################################################################

# Standard library imports
from datetime import datetime, timedelta
from dataclasses import dataclass, asdict
from functools import wraps
from pathlib import Path
from typing import Dict, Optional, Any, List, Tuple
import ctypes
import hashlib
import json
import logging
import logging.handlers
import os
import psutil
import subprocess
import sys
import threading
import time
import traceback

# Third-party imports
import colorama
import pytz
from colorama import Fore, Style, init
import pandas as pd
import pyautogui
import pyperclip
import requests
import yaml
import yfinance as yf

# Initialize colorama for cross-platform terminal colour support
init(autoreset=True)

# clear terminal
os.system("cls" if os.name == "nt" else "clear")

# Terminal output formatting constants
SUCCESS = f"{Fore.GREEN}✓{Style.RESET_ALL}"
CACHED = f"{Fore.BLUE}#{Style.RESET_ALL}"
FAILED = f"{Fore.RED}✗{Style.RESET_ALL}"
HEADING = f"{Fore.CYAN}"

# Debugging colors
DEBUG_INFO = f"{Fore.BLUE}"
DEBUG_FUNCTION = f"{Fore.CYAN}"
WARNING = f"{Fore.YELLOW}"
ERROR = f"{Fore.RED}"

#################################################################
# Utilities: Metrics, Decorators, and Helper Functions          #
#################################################################


@dataclass
class SystemMetrics:
    """
    Tracks system performance metrics such as memory and CPU usage.
    """

    cpu_percent: float
    memory_used_mb: float
    memory_percent: float
    timestamp: float

    @classmethod
    def collect(cls) -> "SystemMetrics":
        process = psutil.Process()
        return cls(
            cpu_percent=psutil.cpu_percent(interval=0.1),
            memory_used_mb=process.memory_info().rss / (1024 * 1024),
            memory_percent=process.memory_percent(),
            timestamp=time.time(),
        )


def retry(max_attempts=3, base_delay=2):
    """Retry decorator with exponential backoff."""

    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            logger = logging.getLogger("quickenprices")
            last_exception = None
            for attempt in range(max_attempts):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    last_exception = e
                    if attempt < max_attempts - 1:
                        delay = base_delay * (2**attempt)
                        logger.warning(
                            f"Network error in {func.__name__}",
                            extra={
                                "error": {"type": type(e).__name__, "message": str(e)},
                                "retry": {
                                    "attempt": attempt + 1,
                                    "max_attempts": max_attempts,
                                    "next_delay": delay,
                                },
                            },
                        )
                        time.sleep(delay)
            raise last_exception

        return wrapper

    return decorator


def log_operation(func):
    """Log execution of a function with metrics."""

    @wraps(func)
    def wrapper(*args, **kwargs):
        logger = logging.getLogger("quickenprices")
        start_time = time.time()
        metrics = SystemMetrics.collect()

        logger.debug(
            f"Starting {func.__name__}",
            extra={
                "timestamp": datetime.now(),
                "user_args": kwargs,
                "system_metrics": asdict(metrics),
            },
        )

        try:
            result = func(*args, **kwargs)
            duration = time.time() - start_time
            end_metrics = SystemMetrics.collect()

            logger.debug(
                f"Completed {func.__name__}",
                extra={"duration_sec": duration, "system_metrics": asdict(end_metrics)},
            )
            return result

        except Exception as e:
            logger.error(
                f"Error in {func.__name__}",
                extra={
                    "error": {
                        "type": type(e).__name__,
                        "message": str(e),
                        "traceback": traceback.format_exc(),
                    },
                    "duration_sec": time.time() - start_time,
                },
            )
            raise

    return wrapper


def detect_currencies(tickers: List[str]) -> set:
    """Detect unique currencies for all tickers by checking their actual data."""
    currencies = set()
    for ticker in tickers:
        try:
            stock = yf.Ticker(ticker)
            currency = stock.info.get("currency")
            if currency and currency != "GBP":  # Skip GBP as it's our base
                currencies.add(currency)
        except Exception as e:
            logging.warning(f"Error getting currency for {ticker}: {e}")
    return currencies


def fetch_exchange_rate(symbol: str) -> Optional[float]:
    """Fetch the latest exchange rate for a given symbol."""
    try:
        ticker = yf.Ticker(symbol)
        hist = ticker.history(period="1d")
        if not hist.empty:
            return float(hist["Close"].iloc[-1])
        else:
            logging.warning(f"No data for {symbol}")
    except Exception as e:
        logging.warning(f"Error fetching exchange rate for {symbol}: {e}")
    return None


def triangulate_currency_to_gbp(currency: str, gbpusd_rate: float) -> Optional[float]:
    """Triangulate exchange rate to GBP via USD."""
    try:
        usd_symbol = f"{currency}USD=X"
        rate = fetch_exchange_rate(usd_symbol)
        if rate is not None:
            gbp_rate = float(rate) / float(gbpusd_rate)
            return gbp_rate
    except Exception as e:
        print(f"Error triangulating {currency} to GBP: {e}")
    return None


#################################################################
# Configuration, Cache and Logging Management                   #
#################################################################
class Config:
    """Configuration management using YAML file."""

    def __init__(self, config_path: str = "config.yaml"):
        self.config_path = Path(config_path)
        try:
            if not self.config_path.exists():
                raise FileNotFoundError(f"Config file not found: {config_path}")
            with open(self.config_path, "r") as f:
                self.config = yaml.safe_load(f)
        except Exception as e:
            print(f"Configuration error: {e}")
            sys.exit(1)

        # Set up attributes for easy access
        self.tickers = self.config.get("tickers", [])  # Now a simple list
        self.debug_tickers = self.config.get("debug", {}).get("tickers", [])
        self.paths = self.config.get("paths", {})  # Added paths initialization

        # Configure logging level for yfinance
        logging.getLogger("yfinance").setLevel(
            getattr(logging, self.config["logging"]["levels"]["terminal"])
        )

    def get(self, key: str, default=None):
        """Get a configuration value by key."""
        return self.config.get(key, default)


class AppLogger:

    def __init__(self, config: Config):
        self.config = config
        self.logger = self._setup_logging()

    def _setup_logging(self) -> logging.Logger:
        """Configure logging using YAML-defined formats and levels."""
        logger = logging.getLogger("quickenprices")
        logger.setLevel(logging.DEBUG)
        logger.handlers = []

        log_config = self.config.config["logging"]
        terminal_level = getattr(logging, log_config["levels"]["terminal"])

        # File handler - keeps everything
        base_path = Path(self.config.paths["base"])
        log_file = base_path / self.config.paths["log_file"]
        file_handler = logging.handlers.RotatingFileHandler(
            log_file,
            maxBytes=log_config["max_bytes"],
            backupCount=log_config["backup_count"],
        )
        file_handler.setLevel(getattr(logging, log_config["levels"]["file"]))
        file_handler.setFormatter(
            logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
        )
        logger.addHandler(file_handler)

        # Console handler - respects terminal level
        console_handler = logging.StreamHandler()
        console_handler.setLevel(terminal_level)

        # Custom formatter that omits INFO messages about tickers and currencies
        class CustomFormatter(logging.Formatter):
            def format(self, record):
                if record.levelno == logging.INFO and (
                    "Valid tickers" in record.msg or "Detected currencies" in record.msg
                ):
                    return ""
                return record.getMessage()

        console_handler.setFormatter(CustomFormatter())
        logger.addHandler(console_handler)

        # Configure yfinance logging
        yf_logger = logging.getLogger("yfinance")
        yf_logger.setLevel(terminal_level)
        yf_logger.handlers = []
        yf_handler = logging.StreamHandler()
        yf_handler.setLevel(terminal_level)
        yf_handler.setFormatter(logging.Formatter("%(message)s"))
        yf_logger.addHandler(yf_handler)

        return logger

    def debug(self, msg: str, *args, **kwargs):
        self.logger.debug(msg, *args, **kwargs)

    def info(self, msg: str, *args, **kwargs):
        self.logger.info(msg, *args, **kwargs)

    def warning(self, msg: str, *args, **kwargs):
        self.logger.warning(msg, *args, **kwargs)

    def error(self, msg: str, *args, **kwargs):
        self.logger.error(msg, *args, **kwargs)


class Cache:
    """File-based cache system."""

    def __init__(self, config: Config):
        self.config = config
        base_path = Path(config.paths["base"])
        self.cache_dir = base_path / "cache"
        self.cache_dir.mkdir(exist_ok=True)

        self.max_age = timedelta(hours=config.config["cache"]["max_age_hours"])
        self.cleanup_threshold = config.config["cache"]["cleanup_threshold"]

        # Create subdirectories
        for cache_type in ["price", "rate"]:
            (self.cache_dir / cache_type).mkdir(exist_ok=True)

    def get(self, key: str) -> Optional[Any]:
        """Retrieve a cache entry."""
        cache_file = self._get_cache_path(key)
        if not cache_file.exists():
            return None

        try:
            with cache_file.open("r") as f:
                data = json.load(f)

            # Check expiration
            if time.time() - data["timestamp"] > self.max_age.total_seconds():
                cache_file.unlink()
                return None

            return data["value"]
        except Exception as e:
            logging.debug(f"Cache read error for {key}: {e}")
            return None

    def set(self, key: str, value: Any) -> None:
        """Store a cache entry."""
        try:
            cache_file = self._get_cache_path(key)
            data = {"timestamp": time.time(), "value": value}

            with cache_file.open("w") as f:
                json.dump(data, f)

            self._cleanup_if_needed()
        except Exception as e:
            logging.debug(f"Cache write error for {key}: {e}")

    def _get_cache_path(self, key: str) -> Path:
        """Generate cache file path."""
        key_parts = key.split("_", 1)
        if len(key_parts) < 2:
            return (
                self.cache_dir
                / f"misc_{hashlib.sha256(key.encode()).hexdigest()[:12]}.json"
            )

        data_type = key_parts[0]
        cache_type_dir = self.cache_dir / data_type

        if data_type == "price":
            ticker = key_parts[1].split("_")[0]
            date_hash = hashlib.sha256(key.encode()).hexdigest()[:8]
            filename = f"{ticker}_{date_hash}.json"
        elif data_type == "rate":
            currency = key_parts[1].split("_")[0]
            date_hash = hashlib.sha256(key.encode()).hexdigest()[:8]
            filename = f"{currency}_{date_hash}.json"
        else:
            filename = f"misc_{hashlib.sha256(key.encode()).hexdigest()[:12]}.json"

        return cache_type_dir / filename

    def _cleanup_if_needed(self) -> None:
        """Clean up expired cache files."""
        cache_files = []
        for cache_type in ["price", "rate"]:
            cache_files.extend(list((self.cache_dir / cache_type).glob("*.json")))

        if len(cache_files) < self.cleanup_threshold:
            return

        current_time = time.time()
        max_age_seconds = self.max_age.total_seconds()

        for file in cache_files:
            try:
                if current_time - file.stat().st_mtime > max_age_seconds:
                    file.unlink()
            except Exception:
                continue

    def get_price_cache_key(self, ticker: str, start_date: str, end_date: str) -> str:
        """Generate cache key for price data."""
        return f"price_{ticker}_{start_date}_{end_date}"

    def get_rate_cache_key(self, currency: str, date: str) -> str:
        """Generate cache key for exchange rate data."""
        return f"rate_{currency}_{date}"


#################################################################
# Exchange Rate and Stock Price Management                      #
#################################################################
class ExchangeRateManager:
    """Handles fetching and processing exchange rates."""

    def __init__(self, config: Config, cache: Cache, logger: logging.Logger):
        self.config = config
        self.cache = cache
        self.logger = logger
        self.rates = {}
        self.known_pairs = [
            pair["symbol"] for pair in config.config.get("currency_pairs", [])
        ]

    def fetch_rates(self, currencies: set, start_date: str, end_date: str) -> bool:
        """Fetch exchange rates, using known pairs or constructing ticker if needed."""
        Application.header(self, title="Exchange Rates", major=False)
        print(f"Status: {SUCCESS} = New Data  # = Cached  {FAILED} = Failed\n")

        success = True
        for currency in currencies:
            if currency in ("GBP", "GBp"):
                continue

            # Construct the expected ticker
            ticker = f"{currency}GBP=X" if currency != "USD" else "GBP=X"

            rates = self._fetch_single_rate(ticker, start_date, end_date)
            if rates:
                self.rates[currency] = rates
                self._print_rate_info(currency, ticker, rates)
            else:
                self.logger.error(
                    f"Failed to fetch rates for {currency} using {ticker}"
                )
                success = False

        return success

    def _fetch_single_rate(
        self, ticker: str, start_date: str, end_date: str
    ) -> Optional[Dict[str, float]]:
        """Fetch exchange rates with latest market data."""
        try:
            # Get real-time rate for today first
            ticker_obj = yf.Ticker(ticker)
            today_data = ticker_obj.history(period="1d", interval="1m")

            if not today_data.empty:
                today_rate = float(today_data["Close"].iloc[-1])
                today = datetime.now().strftime("%Y-%m-%d")

            # Get historical data for previous days
            hist_data = ticker_obj.history(start=start_date, end=end_date)
            if hist_data.empty:
                return None

            # Build rates dictionary with historical data
            rates = {
                date.strftime("%Y-%m-%d"): float(row["Close"])
                for date, row in hist_data.iterrows()
            }

            # Add/update today's rate
            if not today_data.empty:
                rates[today] = today_rate

            return rates

        except Exception as e:
            self.logger.error(f"Error fetching rates for {ticker}: {e}")
            return None

    def _print_rate_info(
        self, currency: str, ticker: str, rates: Dict[str, float]
    ) -> None:
        if not rates:
            return

        print(f"\n{currency} exchange rates (using {ticker}):")

        # Get today and the previous two business days
        market_days = sorted(rates.keys(), reverse=True)[:3]

        for i, date_str in enumerate(market_days):
            rate = rates[date_str]
            formatted_date = datetime.strptime(date_str, "%Y-%m-%d").strftime(
                "%d/%m/%Y"
            )
            status_symbol = SUCCESS if i == 0 else CACHED
            print(f"  {status_symbol} {formatted_date} rate: {rate:.4f}")

        print(f"Number of days fetched: {len(rates)}")

    def get_rate(self, currency: str, date_str: str) -> Optional[float]:
        """Get exchange rate for a specific currency and date."""
        if currency not in self.rates:
            return None

        rates = self.rates[currency]
        if date_str in rates:
            return rates[date_str]

        # Find closest available date
        try:
            target_date = datetime.strptime(date_str, "%Y-%m-%d")
            available_dates = [datetime.strptime(d, "%Y-%m-%d") for d in rates.keys()]
            closest_date = min(available_dates, key=lambda x: abs(x - target_date))
            return rates[closest_date.strftime("%Y-%m-%d")]
        except Exception as e:
            self.logger.error(f"Error finding rate for {currency} on {date_str}: {e}")
            return None


class StockPriceManager:

    def __init__(
        self,
        config: Config,
        cache: Cache,
        logger: logging.Logger,
        exchange_rates: ExchangeRateManager,
    ):
        self.config = config
        self.cache = cache
        self.logger = logger
        self.exchange_rates = exchange_rates
        self.debug_info = []
        self.currency_conversions = {}
        self.tickers = self.config.tickers
        self.successful_conversions = 0
        self.total_processed = 0

    @log_operation
    def fetch_prices(self, start_date: str, end_date: str) -> Dict[str, pd.DataFrame]:
        """Fetch stock prices for all configured tickers."""
        print(f"\n{HEADING}Downloading Prices{Style.RESET_ALL}")
        print(f"{HEADING}------------------{Style.RESET_ALL}")
        print(f"Status: {SUCCESS} = New Data  {CACHED} = Cached  {FAILED} = Failed\n")

        ticker_data = {}
        current_line = []
        failed_tickers = []
        self.total_processed = 0
        self.successful_conversions = 0

        for ticker in self.tickers:
            self.total_processed += 1
            try:
                stock = yf.Ticker(ticker)
                hist_data = stock.history(start=start_date, end=end_date)

                if hist_data.empty or hist_data["Close"].isnull().all():
                    failed_tickers.append((ticker, "No data available"))
                    continue

                currency = self._get_currency(stock, ticker)
                processed_data = self._process_price_data(hist_data, ticker, currency)

                if processed_data is not None:
                    ticker_data[ticker] = processed_data
                    current_line.append(
                        self._format_ticker_status(ticker, "new data", currency)
                    )
                    self.successful_conversions += 1
                else:
                    failed_tickers.append((ticker, "Processing failed"))

                if len(current_line) >= 3:
                    print(" ".join(current_line))
                    current_line = []

            except Exception as e:
                if "possibly delisted" in str(e):
                    error_msg = (
                        f"${ticker.upper()}: possibly delisted; no timezone found"
                    )
                else:
                    error_msg = str(e)
                failed_tickers.append((ticker, error_msg))

        # Print any remaining successful tickers
        if current_line:
            print(" ".join(current_line))
            print()

        # Print failed tickers with warnings
        if failed_tickers:
            print()  # Add blank line before failed tickers
            for ticker, msg in failed_tickers:
                self.logger.warning(f"✗ {ticker} (Unknown)  WARNING: {msg}")

        print(
            f"\nNumber of days fetched: {len(next(iter(ticker_data.values()))) if ticker_data else 0}"
        )
        return ticker_data

    def _get_currency(self, stock: yf.Ticker, ticker: str) -> str:
        """Get currency for a ticker with proper checks."""
        try:
            if ticker.endswith("=X"):
                return "GBP"
            if ticker.endswith(".L"):
                if not stock.info:  # If no info available, assume GBp for .L
                    return "GBp"
                # Check actual stock info currency if available
                stock_currency = stock.info.get("currency", "GBp")
                return stock_currency
            return stock.info.get("currency", "unknown")
        except Exception as e:
            self.logger.warning(f"Could not determine currency for {ticker}: {e}")
            # Default to GBp for .L tickers when there's an error
            return "GBp" if ticker.endswith(".L") else "unknown"

    def _process_price_data(
        self, hist: pd.DataFrame, ticker: str, currency: str
    ) -> Optional[pd.DataFrame]:
        """Process historical price data."""
        try:
            result = hist.copy()
            latest_date = result.index[-1]
            latest_price = float(result["Close"].iloc[-1])

            if currency not in self.currency_conversions:
                self.currency_conversions[currency] = 0

            # Prepare debug info if this is a tracked ticker
            if ticker in self.config.debug_tickers:
                debug_lines = [
                    f"Debug - {ticker}:",
                    f"Date: {latest_date.strftime('%d/%m/%Y')}",
                    f"Currency: {currency}",
                    f"Original price: {latest_price:.4f} {currency}",
                ]

            # Process GBp conversions
            if currency == "GBp":
                result["Close"] = result["Close"] / 100
                revised_price = latest_price / 100
                self.currency_conversions[currency] += 1
                if ticker in self.config.debug_tickers:
                    debug_lines.extend(
                        [
                            "Converting GBp to GBP (divide by 100)",
                            f"Revised price = {latest_price:.4f} / 100 = {revised_price:.4f} GBP",
                        ]
                    )

            # Process other currency conversions
            elif currency not in ("GBP", "GBp"):
                rate = self.exchange_rates.get_rate(
                    currency, latest_date.strftime("%Y-%m-%d")
                )
                if rate:
                    result["Close"] = result["Close"] * rate
                    revised_price = latest_price * rate
                    self.currency_conversions[currency] += 1
                    if ticker in self.config.debug_tickers:
                        debug_lines.extend(
                            [
                                f"{latest_date.strftime('%d/%m/%Y')} Exchange rate ({currency}/GBP): {rate:.4f}",
                                f"Revised price = {latest_price:.4f} * {rate:.4f} = {revised_price:.4f} GBP",
                            ]
                        )
                elif ticker in self.config.debug_tickers:
                    debug_lines.append("Warning: No exchange rate available")

            # GBP needs no conversion
            elif ticker in self.config.debug_tickers:
                debug_lines.append("No need for conversion")

            # Store debug info if this is a tracked ticker
            if ticker in self.config.debug_tickers:
                self.debug_info.extend(debug_lines)

            return pd.DataFrame(
                {
                    "ticker": ticker,
                    "price": result["Close"],
                    "date": result.index.strftime("%d/%m/%Y"),
                }
            )

        except Exception as e:
            self.logger.error(f"Error processing price data for {ticker}: {e}")
            return None

    def _format_ticker_status(
        self, ticker: str, status: str, currency: str, width: int = 30
    ) -> str:
        """Format ticker status for display."""
        status_symbol = {
            "new data": SUCCESS,
            "using cache": CACHED,
            "failed": FAILED,
        }.get(status, FAILED)

        ticker_with_currency = f"{ticker} ({currency})"
        return f"{status_symbol} {ticker_with_currency:<{width}}"

    def _print_debug_info(self) -> None:
        """Print debugging information for tracked tickers."""
        if not self.debug_info:
            return

        print(f"\n{HEADING}Debugging{Style.RESET_ALL}")
        print(f"{HEADING}---------{Style.RESET_ALL}")

        # Group debug info by ticker
        ticker_info = []
        current_info = []

        for line in self.debug_info:
            if line.startswith("Debug - "):
                if current_info:
                    ticker_info.append(current_info)
                current_info = [line]
            else:
                current_info.append(line)
        if current_info:
            ticker_info.append(current_info)

        # Sort debug info by date (newest first) and print
        for info in sorted(ticker_info, key=self._get_debug_date, reverse=True):
            print("\n".join(info))
            print()  # Add blank line between tickers

    def _get_debug_date(self, info_lines: List[str]) -> datetime:
        """Extract date from debug info for sorting."""
        for line in info_lines:
            if line.startswith("Date: "):
                try:
                    return datetime.strptime(line[6:], "%d/%m/%Y")
                except:
                    pass
        return datetime.min


#################################################################
# QuickenInterface                                              #
#################################################################
class QuickenInterface:
    """Handles integration with Quicken."""

    def __init__(self, config: Config, logger: logging.Logger):
        self.config = config
        self.logger = logger
        pyautogui.FAILSAFE = True
        pyautogui.PAUSE = 0.5

    @log_operation
    def handle_import(self) -> bool:
        """Handle data import to Quicken."""
        try:
            Application.header(self, "Import to Quicken 2004")

            if self._is_elevated():
                return self._execute_import_sequence()
            else:
                return self._show_elevation_message()

        except Exception as e:
            self.logger.error(f"Error during Quicken import: {e}")
            return False

    def _is_elevated(self) -> bool:
        """Check for admin privileges."""
        try:
            return ctypes.windll.shell32.IsUserAnAdmin()
        except:
            return False

    def _show_elevation_message(self) -> bool:
        """Show message for manual Quicken update."""
        print("Quicken cannot be opened from here.\n")
        print("Instead:")
        print("1. Close this window.")
        print("2. Click the Quicken 'Update Prices' shortcut.\n")
        Application.header(self, "END", True)
        return True

    def _execute_import_sequence(self) -> bool:
        """Execute Quicken import sequence."""
        steps = [
            (self._open_quicken, "Opening Quicken..."),
            (self._navigate_to_portfolio, "Navigating to Portfolio view..."),
            (self._open_import_dialog, "Opening import dialog..."),
            (self._import_data_file, "Importing data file..."),
        ]

        for step_func, message in steps:
            print(message)
            if not step_func():
                return False

        print("\nImport complete!")
        return True

    def _open_quicken(self) -> bool:
        """Launch Quicken."""
        try:
            subprocess.Popen([str(self.config.paths["quicken"])])
            time.sleep(8)
            return True
        except Exception as e:
            self.logger.error(f"Failed to open Quicken: {e}")
            return False

    def _navigate_to_portfolio(self) -> bool:
        """Navigate to portfolio view."""
        try:
            pyautogui.hotkey("ctrl", "u")
            time.sleep(1)
            return True
        except Exception as e:
            self.logger.error(f"Failed to navigate to portfolio: {e}")
            return False

    def _open_import_dialog(self) -> bool:
        """Open import dialog."""
        try:
            with pyautogui.hold("alt"):
                pyautogui.press(["f", "i", "i"])
            time.sleep(1)
            return True
        except Exception as e:
            self.logger.error(f"Failed to open import dialog: {e}")
            return False

    def _import_data_file(self) -> bool:
        """Import data file."""
        try:
            filename = str(self.config.paths["data_file"])
            pyautogui.typewrite(filename, interval=0.03)
            time.sleep(1)
            pyautogui.press("enter")
            time.sleep(5)
            return True
        except Exception as e:
            self.logger.error(f"Failed to import data file: {e}")
            return False


#################################################################
# DataManager and Application Classes                           #
#################################################################
class DataManager:
    """Handles data processing and transformation."""

    def __init__(self, config: Config):
        self.config = config

    def process_large_dataset(
        self, data: pd.DataFrame, chunk_size: int = 1000
    ) -> pd.DataFrame:
        """Process large datasets in chunks."""
        return pd.concat(
            [
                self._process_chunk(data[i : i + chunk_size].copy())
                for i in range(0, len(data), chunk_size)
            ]
        )

    def _process_chunk(self, chunk: pd.DataFrame) -> pd.DataFrame:
        """Process a single chunk of data."""
        return chunk

    def sort_data_for_export(self, df: pd.DataFrame) -> pd.DataFrame:
        """Sort dataframe for export."""
        try:
            df["sort_date"] = pd.to_datetime(df["date"], format="%d/%m/%Y")
            df = df.sort_values(["sort_date", "ticker"], ascending=[False, True])
            return df.drop("sort_date", axis=1)
        except Exception as e:
            raise ValueError(f"Error sorting data: {e}")


class Application:
    """Main application class."""

    def __init__(self):
        try:
            self.config = Config()
            self.logger = AppLogger(self.config)
            self.cache = Cache(self.config)
            self._setup_tickers()  # Move this before creating managers
            self.exchange_rates = ExchangeRateManager(
                self.config, self.cache, self.logger.logger
            )
            self.stock_prices = StockPriceManager(
                self.config, self.cache, self.logger.logger, self.exchange_rates
            )
            self.quicken = QuickenInterface(self.config, self.logger.logger)
            self.data_manager = DataManager(self.config)

            self.start_time = time.time()

        except Exception as e:
            error_msg = f"Failed to initialize application: {str(e)}\n"
            error_msg += f"Traceback:\n{traceback.format_exc()}"
            sys.stderr.write(error_msg)
            sys.exit(1)

    def _setup_tickers(self):
        """Set up tickers and detect currencies."""
        try:
            valid_tickers = []
            valid_currencies = set()

            for ticker in self.config.tickers:
                if ticker.lower() == "fake":  # Skip obviously invalid tickers
                    self.logger.warning(
                        f"✗ {ticker} (Unknown)  WARNING: ${ticker.upper()}: possibly delisted; no timezone found"
                    )
                    continue

                try:
                    stock = yf.Ticker(ticker)
                    if ticker.endswith(".L"):  # London securities
                        valid_tickers.append(ticker)
                        valid_currencies.add("GBp")
                    elif "=X" in ticker:  # Exchange rates
                        valid_tickers.append(ticker)
                    else:
                        currency = stock.info.get("currency")
                        if currency:
                            valid_tickers.append(ticker)
                            if currency != "GBP":
                                valid_currencies.add(currency)
                except Exception as e:
                    self.logger.warning(f"✗ {ticker} (Unknown)  WARNING: {str(e)}")
                    continue

            self.tickers = valid_tickers
            self.currencies = valid_currencies
            self.logger.info(f"Valid tickers: {self.tickers}")
            self.logger.info(f"Detected currencies: {self.currencies}")

        except Exception as e:
            self.logger.logger.error(f"Error setting up tickers and currencies: {e}")
            raise

    @log_operation
    def run(self) -> bool:
        """Execute main workflow ensuring complete output."""
        try:
            

            dates = self._setup_dates()
            self._print_header(dates)

            if not self.exchange_rates.fetch_rates(
                self.currencies, dates["start_iso"], dates["end_iso"]
            ):
                return False

            ticker_data = self.stock_prices.fetch_prices(
                dates["start_iso"], dates["end_iso"]
            )

            if not ticker_data:
                return False

            self.stock_prices._print_debug_info()

            if self._save_data(ticker_data.values()):
                self._print_summary(ticker_data)
                self._copy_output_to_clipboard()
                return self.quicken.handle_import()

            return False

        except Exception as e:
            self.logger.logger.error(f"Fatal error during application run: {e}")
            traceback.print_exc()
            return False

    def _setup_dates(self) -> dict:
        """Calculate start and end dates ensuring full market days."""
        period_years = self.config.config["collection"]["period_years"]
        end_date = datetime.now()
        days_back = int(period_years * 365)  # Convert years to days
        start_date = end_date - timedelta(days=days_back)

        # Generate all dates in the range
        all_dates = [
            start_date + timedelta(days=i)
            for i in range((end_date - start_date).days + 1)
        ]

        # Count business days (Monday to Friday)
        business_days = sum(1 for date in all_dates if date.weekday() < 5)

        return {
            "start_iso": start_date.strftime("%Y-%m-%d"),
            "end_iso": end_date.strftime("%Y-%m-%d"),
            "start_display": start_date.strftime("%d/%m/%Y"),
            "end_display": end_date.strftime("%d/%m/%Y"),
            "business_days": business_days,
        }

    def header(self, title, major=False):
        """Print a section header with optional emphasis."""
        print(f"{HEADING}{title}{Style.RESET_ALL}")
        if major:
            print(f"{HEADING}{'=' * len(title)}{Style.RESET_ALL}")
        else:
            print(f"{HEADING}{'-' * len(title)}{Style.RESET_ALL}")

    def _print_header(self, dates: dict) -> None:
        """Print application header with date range and business days."""
        print(f"{HEADING}Stock Price Update{Style.RESET_ALL}")
        print("=" * 18)

        business_days = len(
            pd.bdate_range(start=dates["start_iso"], end=dates["end_iso"])
        )
        print(
            f"Range {dates['start_display']} - {dates['end_display']}:\n{business_days} business days\n"
        )

    def format_path(self, full_path):
        """Format a file path to show the last parent directory and filename."""
        path_parts = full_path.split("\\")
        if len(path_parts) > 1:
            return "\\" + "\\".join(path_parts[-2:])  # Show parent dir and filename
        return "\\" + full_path  # Just in case there's no parent directory

    def _strip_ansi(self, text):
        """Remove ANSI escape sequences from text."""
        import re

        ansi_escape = re.compile(r"\x1B(?:[@-Z\\-_]|\[[0-?]*[ -/]*[@-~])")
        return ansi_escape.sub("", text)

    def _print_summary(self, ticker_data: Dict[str, pd.DataFrame]) -> None:
        """Print execution summary."""
        print(f"\n{HEADING}Summary{Style.RESET_ALL}")
        print(f"{HEADING}-------{Style.RESET_ALL}")
        
        total_tickers = len(self.tickers) + 2  # Add 2 for exchange rate tickers
        execution_time = time.time() - self.start_time

        print(f"Tickers Processed: {total_tickers}")
        
        # Print conversions by currency in sorted order
        for currency, count in sorted(self.stock_prices.currency_conversions.items()):
            if count > 0:
                print(f"Tickers adjusted from {currency} to GBP: {count}")

        success_rate = (self.stock_prices.successful_conversions / self.stock_prices.total_processed) * 100
        print(f"Success Rate: {success_rate:.1f}%")
        print(f"Execution Time: {execution_time:.2f}s")
        
        # Format paths with parent directory
        base_path = Path(self.config.paths["base"])
        print(f"Log file saved to \\{base_path.name}\\{self.config.paths['log_file']}")
        print(f"Data saved to: \\{base_path.name}\\{self.config.paths['data_file']}")
        print("Terminal output copied to clipboard")
        
    def _save_data(self, dataframes: List[pd.DataFrame]) -> bool:
        """Save processed data to file."""
        try:
            df = pd.concat(list(dataframes), ignore_index=True)
            df = self.data_manager.sort_data_for_export(df)

            output_path = (
                Path(self.config.paths["base"]) / self.config.paths["data_file"]
            )
            df.to_csv(output_path, index=False, header=False)

            return True

        except Exception as e:
            self.logger.logger.error(f"Error saving data: {e}")
            return False

    def _copy_output_to_clipboard(self) -> None:
        """Copy output data to clipboard."""
        try:
            output_path = (
                Path(self.config.paths["base"]) / self.config.paths["data_file"]
            )
            with open(output_path, "r") as f:
                data = f.read()
            pyperclip.copy(data)
            # Removed duplicate print statement
        except Exception as e:
            self.logger.logger.error(f"Error copying to clipboard: {e}")


#################################################################
# Main Function                                                 #
#################################################################
def main():
    """Application entry point."""
    try:
        app = Application()
        success = app.run()
        sys.exit(0 if success else 1)
    except KeyboardInterrupt:
        print("\nOperation cancelled by user")
        sys.exit(1)
    except Exception as e:
        sys.stderr.write(f"Fatal error: {e}\n")
        sys.exit(1)


if __name__ == "__main__":
    main()

#################################################################
# End                                                           #
#################################################################
