"""
QuickenPrice.py - Stock Price Fetcher and Quicken Data Import Tool

This script fetches stock prices from Yahoo Finance and prepares them for import into Quicken 2004.
It handles multiple currencies, caches data to minimize API calls, and manages automated data import.

Features:
- Fetches stock and index prices from Yahoo Finance
- Handles currency conversions (USD, GBP, EUR)
- Implements caching with validation
- Provides automated Quicken import
- Includes comprehensive error handling and logging

Usage:
    python QuickenPrice.py

Testing:
    The code includes a comprehensive test suite in the 'tests' directory.
    To run all tests from the QuickenPrices root directory:
        pytest tests/ -v

    To run specific test categories:
        pytest tests/test_yahoo_fetcher.py -v      # Yahoo Finance fetcher tests
        pytest tests/test_price_converter.py -v     # Price conversion tests
        pytest tests/test_rate_limiter.py -v       # Rate limiting tests
        pytest tests/test_cache_manager.py -v      # Cache management tests
        pytest tests/test_data_validator.py -v     # Data validation tests

Requirements:
    See requirements.txt for dependencies

Author: Wayne Gault
Last Modified: 13/11/2024
"""

##################################################
# Imports and configuration                      #
##################################################

# === Standard Library Imports ===
import ctypes  # Provides low-level operating system interfaces, used for system-level operations
import subprocess  # Enables spawning of new processes and interaction with their I/O
import sys  # Provides access to Python interpreter variables and functions
import time  # Offers time-related functions, used for delays and timing operations
import calendar  # Provides calendar-related functions, used for date calculations
import logging  # Implements a flexible event logging system for tracking program execution
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

# === Third-Party Imports ===
import pyautogui  # Provides cross-platform GUI automation tools
import yfinance as yf  # Yahoo Finance API interface for financial data
import pandas as pd  # Data manipulation and analysis library
import psutil  # Cross-platform utilities for retrieving information on running processes
import weakref  # Support for weak references and finalization
import win32api
import win32con
import winerror

print("Debugging: Starting program...")

# Initialize colorama to enable cross-platform colored terminal output
# - init: initializes colorama
# - Fore: foreground colors
# - Style: text styles (bright, dim, etc.)
try:
    from colorama import init, Fore, Style  # Colored terminal text library

    init(autoreset=True)
except ImportError:
    print("Colorama is not installed. Please install it using `pip install colorama`.")


class Config:
    """
    Configuration class that centralizes all application settings and parameters.

    This class serves as a single source of truth for configuration values,
    making it easier to modify settings without changing code throughout the application.

    Attributes:
        TICKERS (list): List of stock symbols to track and process
                       - Prefixed with '^' are index symbols (e.g., ^GSPC for S&P 500)
                       - Suffixed with '.L' are London Stock Exchange symbols
                       - Others are regular stock symbols

        DEBUG_TICKERS (list): Subset of tickers used for debugging purposes
                             Empty by default, can be populated for testing

        PERIOD (float): Time period in years for historical data collection
                       e.g., 0.08 represents approximately 1 month

        PATH (str): Base directory path for the application
                   Stores logs, cache, and output files

        CURRENCY_SYMBOLS (dict): Maps currency codes to their conversion rate symbols
                                - Keys are currency codes (USD, GBp, EUR)
                                - Values are Yahoo Finance symbols for conversion rates
                                - None value indicates special handling needed

        QUICKEN_PATH (str): File system path to the Quicken executable
                           Used for automation of data import
    """

    def __init__(self):
        # List of financial instruments to track: market indices (^), stocks, and international securities (.L)
        self.TICKERS = [
            # Market Indices
            "^GSPC",  # S&P 500
            "^OEX",  # S&P 100
            "^FTAS",  # FTSE All-Share
            "^FTSE",  # FTSE 100
            # London Stock Exchange Listings (.L suffix)
            "0P00013P6I.L",  # Custom index or fund
            "0P00018XAP.L",  # Custom index or fund
            "CUKX.L",  # iShares Core FTSE 100
            "CYSE.L",  # Custom security
            "ELIX.L",  # Custom security
            "GLTA.L",  # Custom security
            "HMWO.L",  # Custom security
            "IIND.L",  # Custom security
            "IUIT.L",  # Custom security
            "IHCU.L",  # Custom security
            "SEE.L",  # Custom security
            "VHVG.L",  # Custom security
            "VUKG.L",  # Custom security
            # US Stocks
            "AMGN",  # Amgen
            "MCHI",  # iShares MSCI China ETF
            "VNRX",  # VolitionRX
            # Currency Pairs
            "GBP=X",  # GBP/USD exchange rate
            "EURGBP=X",  # GB/EURO exchange rate
        ]

        # Debug mode tickers - empty by default. Can be populated for testing specific scenarios
        self.DEBUG_TICKERS = ["VHVG.L", "ELIX.L", "VNRX", "EURGBP=X"]

        # Historical data period in years. 0.08 years ≈ 1 month (0.08 * 365 ≈ 29.2 days)
        self.PERIOD = 0.08

        # Application base path - stores all data files, logs, and cache
        self.PATH = (
            "C:\\Users\\wayne\\OneDrive\\Documents\\Python\\Projects\\QuickenPrices\\"
        )
        self.data_file_name = "data.csv"
        self.log_file_name = "prices.log"

        # Currency conversion mapping (Keys: Currency codes, Values: Yahoo Finance symbols for exchange rates)
        self.CURRENCY_SYMBOLS = {
            "USD": "GBP=X",  # US Dollar to British Pound conversion
            "GBp": None,  # British Pence (requires division by 100)
            "EUR": "EURGBP=X",  # Euro to British Pound conversion
        }

        # Path to Quicken software executable. Used for automated data import
        self.QUICKEN_PATH = "C:\\Program Files (x86)\\Quicken\\qw.exe"

        # Cache settings
        self.cache_settings = {
            "max_age_hours": 168,  # 1 week (7 days * 24 hours)
            "cleanup_threshold": 200,  # Number of entries before cleanup
        }

        # Log rotation settings...
        self.log_rotation = {
            "max_bytes": 5 * 1024 * 1024,  # 5MB
            "backup_count": 5,  # Keep 5 backup files
        }


##################################################
# Formatting class                               #
##################################################


class DateManager:
    """
    Handles date calculations and formatting for the application.

    Features:
    - Date range calculations
    - Business day counting
    - Date formatting standardization
    """

    def __init__(self, config=None):
        self.config = config

        # Initialize StructuredLogger instead of standard logger
        self.logger = StructuredLogger("DateManager", config=config)

    @staticmethod
    def setup_dates(period_years):
        """
        Set up date range based on specified period.

        Args:
            period_years (float): Period in years to look back

        Returns:
            dict: Dictionary containing formatted dates:
                - start_iso: Start date in ISO format
                - end_iso: End date in ISO format
                - start_display: Start date in display format
                - end_display: End date in display format
        """
        end_date = datetime.now()
        start_date = end_date - timedelta(days=period_years * 365)

        return {
            "start_iso": start_date.strftime("%Y-%m-%d"),
            "end_iso": (end_date + timedelta(days=1)).strftime("%Y-%m-%d"),
            "start_display": start_date.strftime("%d/%m/%Y"),
            "end_display": end_date.strftime("%d/%m/%Y"),
        }

    @staticmethod
    def calculate_business_days(start_date: str, end_date: str) -> int:
        """
        Calculate number of business days between two dates.

        Args:
            start_date_str (str): Start date string
            end_date_str (str): End date string
            format (str): Date string format

        Returns:
            int: Number of business days
        """
        start = pd.to_datetime(start_date)
        end = pd.to_datetime(end_date)
        return len(pd.bdate_range(start=start, end=end))


class OutputFormatter:
    """
    Handles consistent formatting of output messages and status information.

    Features:
    - Formatted status messages
    - Path shortening
    - Number formatting
    - Output buffering
    """

    def __init__(
        self,
        config=None,
    ):
        """Initialize the formatter with an empty output buffer."""
        # Initialize StructuredLogger instead of standard logger
        self.config = config
        self.logger = StructuredLogger("OutputFormatter", config=config)
        self.output_buffer = []

    def format_number(self, number, decimals=4):
        """Format a number with specified decimal places."""
        return f"{number:.{decimals}f}"

    def format_path(self, full_path):
        """Format a file path to show the last parent directory and filename."""
        path_parts = full_path.split("\\")
        if len(path_parts) > 1:
            return "\\" + "\\".join(path_parts[-2:])  # Show parent dir and filename
        return "\\" + full_path  # Just in case there's no parent directory

    def format_status(self, status):
        """Format status indicators with colors."""
        status_map = {
            "new data": f"{Fore.GREEN}✓{Style.RESET_ALL}",
            "using cache": f"{Fore.BLUE}#{Style.RESET_ALL}",
            "failed": f"{Fore.RED}✗{Style.RESET_ALL}",
        }
        return status_map.get(status, f"{Fore.RED}?{Style.RESET_ALL}")

    def format_ticker_status(self, ticker, status, currency=None, width=30):
        """Format ticker status with currency information."""
        status_symbol = self.format_status(status)
        ticker_with_currency = f"{ticker} ({currency})" if currency else ticker
        return f"{status_symbol} {ticker_with_currency:<{width}}"

    def print_section(self, title, major=False):
        """Print a section header with optional emphasis."""
        self.capture_print(f"{Fore.WHITE}{Style.BRIGHT}{title}{Style.RESET_ALL}")
        if major:
            self.capture_print(
                f"{Fore.WHITE}{Style.BRIGHT}{'=' * len(title)}{Style.RESET_ALL}"
            )
        else:
            self.capture_print(
                f"{Fore.WHITE}{Style.BRIGHT}{'-' * len(title)}{Style.RESET_ALL}"
            )

    def _strip_ansi(self, text):
        """Remove ANSI escape sequences from text."""
        import re

        ansi_escape = re.compile(r"\x1B(?:[@-Z\\-_]|\[[0-?]*[ -/]*[@-~])")
        return ansi_escape.sub("", text)

    def capture_print(self, *args, **kwargs):
        """Print and store output in buffer."""
        output = " ".join(str(arg) for arg in args)
        # Store the raw output with ANSI codes for terminal display
        print(output, **kwargs)
        # Store the clean output without ANSI codes for clipboard
        self.output_buffer.append(self._strip_ansi(output))

    def get_clean_output(self):
        """Get the complete output buffer without ANSI codes."""
        return "\n".join(self.output_buffer)

    ##################################################
    # Logging and monitoring infrastructure classes  #
    ##################################################


class LogLevel(Enum):
    """
    Enumeration of logging levels used throughout the application.
    Provides standardized severity levels for log messages.

    Levels:
    - DEBUG: Detailed information for debugging
    - INFO: General operational messages
    - WARNING: Indicates potential issues
    - ERROR: Serious problems that need attention
    """

    DEBUG = "DEBUG"
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"


@dataclass
class MetricsContext:
    """
    Data class for storing metrics and context information for operations.
    Uses Python's @dataclass decorator to automatically generate special methods.

    Attributes:
        correlation_id (str): Unique identifier for tracking related log entries
        start_time (float): Timestamp when the operation started
        metrics (Dict): Collection of metrics for the operation
        parent_id (Optional[str]): ID of parent operation for nested operations
    """

    correlation_id: str
    start_time: float
    metrics: Dict[str, Any]
    parent_id: Optional[str] = None


class StructuredLogger:
    """
    Advanced logging system that provides structured logging with context tracking
    and performance metrics.

    Features:
    - Correlation IDs for tracking related log entries
    - Progress tracking and ETA calculation
    - Hierarchical operation tracking
    - Thread-safe logging

    Additional Methods:
    - error(message: str, **kwargs): Log an error message.
    - info(message: str, **kwargs): Log an info message.
    - warning(message: str, **kwargs): Log a warning message.
    - debug(message: str, **kwargs): Log a debug message.
    """

    def __init__(self, component_name: str, config=None):
        """
        Initialize the logger for a specific component.

        Args:
            component_name (str): Name of the component using this logger
            config: Configuration object containing logging settings
        """
        self.component_name = component_name
        self.config = config
        self._log_lock = threading.Lock()
        self.metrics_context = threading.local()
        # Define format_path method before setting up logging
        self.format_path = lambda p: str(p).replace(
            "\\", "/"
        )  # Make it a public method
        # use Python's standard logging module to log problems in this class rather than an instance of StructuredLogger to prevent infinite recursion.
        self.logger = self._setup_logging()

    def _setup_logging(self):
        """
        Configure the logging system with appropriate formatters and handlers.
        Ensures paths are logged as plain text.
        """
        logger = logging.getLogger(self.component_name)

        if not logger.handlers:
            logger.setLevel(logging.DEBUG)

            # File handler with plain text formatter
            if self.config and hasattr(self.config, "log_rotation"):
                max_bytes = self.config.log_rotation.get("max_bytes", 5 * 1024 * 1024)
                backup_count = self.config.log_rotation.get("backup_count", 5)
            else:
                max_bytes = 5 * 1024 * 1024
                backup_count = 5

            if self.config:
                log_path = Path(self.config.PATH) / self.config.log_file_name
            else:
                log_path = Path("prices.log")

            rotating_handler = RotatingFileHandler(
                log_path, maxBytes=max_bytes, backupCount=backup_count, encoding="utf-8"
            )
            rotating_formatter = logging.Formatter(
                "%(asctime)s - %(levelname)s - %(message)s"
            )
            rotating_handler.setFormatter(rotating_formatter)
            rotating_handler.setLevel(logging.DEBUG)
            logger.addHandler(rotating_handler)

            # Console handlers remain the same...
            error_console_handler = logging.StreamHandler(sys.stderr)
            error_console_formatter = logging.Formatter(
                f"{Style.BRIGHT}{Fore.RED}%(message)s{Style.RESET_ALL}"
            )
            error_console_handler.setFormatter(error_console_formatter)
            error_console_handler.setLevel(logging.ERROR)
            logger.addHandler(error_console_handler)

            console_handler = logging.StreamHandler()
            console_formatter = logging.Formatter(
                f"{Fore.GREEN}%(asctime)s [%(correlation_id)s] %(levelname)s: %(message)s{Style.RESET_ALL}"
            )
            console_handler.setFormatter(console_formatter)
            console_handler.setLevel(logging.ERROR)
            logger.addHandler(console_handler)

        return logger

    def _strip_ansi(self, text):
        """Remove ANSI escape sequences and underline formatting from text."""
        # Remove ANSI escape sequences
        ansi_escape = re.compile(r"\x1B(?:[@-Z\\-_]|\[[0-?]*[ -/]*[@-~])")
        # Remove underline formatting
        text = str(text).replace("\x1B[4m", "").replace("\x1B[24m", "")
        return ansi_escape.sub("", text)

    def start_operation(
        self, operation_name: str, parent_id: Optional[str] = None
    ) -> str:
        correlation_id = str(uuid.uuid4())
        self.metrics_context.context = MetricsContext(
            correlation_id=correlation_id,
            start_time=time.time(),
            metrics={
                "operation": operation_name,
                "status": "in_progress",
                "steps_completed": 0,
                "total_steps": 0,
                "errors": [],
            },
            parent_id=parent_id,
        )
        return correlation_id

    def log(self, level: LogLevel, message: str, **kwargs) -> None:
        """
        Log a message with the specified level and additional context.
        Thread-safe logging with comprehensive error handling and context tracking.

        Args:
            level (LogLevel): Severity level of the log message
            message (str): The message to log
            **kwargs: Additional context parameters to include in the log

        Example:
            logger.log(LogLevel.INFO, "Processing started",
                    process_id=123, ticker="AAPL")
        """
        with self._log_lock:  # Thread-safe logging
            try:
                # Get current operation context
                context = getattr(self.metrics_context, "context", None)

                # Format timestamp with microsecond precision
                timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]

                # Process and sanitize the message
                if isinstance(message, (str, Path)):
                    formatted_message = self._sanitize_message(str(message))
                else:
                    formatted_message = str(message)

                # Format all kwargs consistently
                formatted_kwargs = self._format_kwargs(kwargs)

                # Build extra context
                extra = {
                    "timestamp": timestamp,
                    "component": self.component_name,
                    "correlation_id": context.correlation_id if context else "",
                    "thread_id": threading.get_ident(),
                    **formatted_kwargs,
                }

                # Add execution context if available
                if context:
                    try:
                        with threading.Lock():
                            context.metrics.update(formatted_kwargs)
                            extra.update(
                                {
                                    "operation": context.metrics.get(
                                        "operation", "unknown"
                                    ),
                                    "parent_id": context.parent_id,
                                    "elapsed_time": f"{time.time() - context.start_time:.3f}s",
                                }
                            )
                    except Exception as context_error:
                        self._handle_context_error(context_error)

                # Add memory usage metrics for resource-intensive operations
                if level in [LogLevel.ERROR, LogLevel.WARNING]:
                    extra.update(self._get_memory_metrics())

                # Add stack trace for errors
                if level == LogLevel.ERROR:
                    extra["stack_trace"] = self._get_stack_trace()

                # Perform actual logging
                self._perform_log(level, formatted_message, extra)

                # Handle urgent notifications if needed
                if level == LogLevel.ERROR:
                    self._handle_urgent_notification(formatted_message, extra)

            except Exception as e:
                # Fallback error handling if logging fails
                self._handle_logging_failure(e, level, message)

    def _sanitize_message(self, message: str) -> str:
        """
        Sanitize log message to prevent injection and ensure proper formatting.

        Args:
            message (str): Raw message to sanitize

        Returns:
            str: Sanitized message
        """
        try:
            # Remove control characters
            message = "".join(char for char in message if char.isprintable())

            # Truncate extremely long messages
            MAX_MESSAGE_LENGTH = 10000
            if len(message) > MAX_MESSAGE_LENGTH:
                message = f"{message[:MAX_MESSAGE_LENGTH]}... (truncated)"

            return message
        except Exception as e:
            return f"Message sanitization failed: {str(e)}"

    def _format_kwargs(self, kwargs: Dict[str, Any]) -> Dict[str, str]:
        """
        Format kwargs consistently for logging.

        Args:
            kwargs: Dictionary of additional logging context

        Returns:
            Dict[str, str]: Formatted kwargs
        """
        formatted = {}
        for key, value in kwargs.items():
            try:
                if isinstance(value, (datetime, pd.Timestamp)):
                    formatted[key] = value.strftime("%Y-%m-%d %H:%M:%S")
                elif isinstance(value, Path):
                    formatted[key] = self.format_path(str(value))
                elif isinstance(value, (dict, list, tuple)):
                    formatted[key] = json.dumps(value, default=str)
                elif value is None:
                    formatted[key] = "null"
                else:
                    formatted[key] = str(value)
            except Exception as e:
                formatted[key] = f"Formatting error: {str(e)}"
        return formatted

    def _get_memory_metrics(self) -> Dict[str, float]:
        """
        Get current memory usage metrics.

        Returns:
            Dict[str, float]: Memory usage metrics
        """
        try:
            process = psutil.Process()
            memory_info = process.memory_info()
            return {
                "memory_rss_mb": memory_info.rss / (1024 * 1024),
                "memory_vms_mb": memory_info.vms / (1024 * 1024),
                "memory_percent": process.memory_percent(),
            }
        except Exception:
            return {}

    def _get_stack_trace(self) -> str:
        """
        Get formatted stack trace for error logging.

        Returns:
            str: Formatted stack trace
        """
        try:
            return "".join(traceback.format_stack())
        except Exception:
            return "Stack trace unavailable"

    def _perform_log(
        self, level: LogLevel, message: str, extra: Dict[str, Any]
    ) -> None:
        """
        Perform the actual logging operation.

        Args:
            level: Log level
            message: Formatted message
            extra: Additional context
        """
        if hasattr(logging, level.value.upper()):
            log_func = getattr(self.logger, level.value.lower())
            log_func(message, extra=extra)
        else:
            self.logger.info(message, extra=extra)

    def _handle_urgent_notification(
        self, message: str, context: Dict[str, Any]
    ) -> None:
        """
        Handle urgent notifications for critical errors.

        Args:
            message: Error message
            context: Error context
        """
        if level == LogLevel.ERROR:
            try:
                # Implement urgent notification logic here
                # e.g., send email, Slack message, etc.
                pass
            except Exception:
                pass  # Suppress notification errors

    def _handle_logging_failure(
        self, error: Exception, level: LogLevel, message: str
    ) -> None:
        """
        Handle cases where normal logging fails.

        Args:
            error: The exception that occurred
            level: Original log level
            message: Original message
        """
        try:
            # Attempt to log to stderr as last resort
            error_msg = (
                f"Logging failure: {str(error)}\n"
                f"Original level: {level}\n"
                f"Original message: {message}"
            )
            print(error_msg, file=sys.stderr)
        except Exception:
            pass  # If all else fails, suppress the error

    def _handle_context_error(self, error: Exception) -> None:
        """
        Handle errors in context processing.

        Args:
            error: The exception that occurred
        """
        try:
            print(f"Context processing error: {str(error)}", file=sys.stderr)
        except Exception:
            pass  # Suppress context handling errors

    # Convenience methods
    def error(self, message: str, **kwargs):
        """Log an error message without color codes in file."""
        self.log(LogLevel.ERROR, message, **kwargs)

    def info(self, message: str, **kwargs):
        """Log an info message without color codes in file."""
        self.log(LogLevel.INFO, message, **kwargs)

    def warning(self, message: str, **kwargs):
        """Log a warning message without color codes in file."""
        self.log(LogLevel.WARNING, message, **kwargs)

    def debug(self, message: str, **kwargs):
        """Log a debug message without color codes in file."""
        self.log(LogLevel.DEBUG, message, **kwargs)

    def update_progress(self, steps_completed: int, total_steps: int):
        """
        Update progress metrics for the current operation.

        Args:
            steps_completed (int): Number of steps completed
            total_steps (int): Total number of steps in the operation
        """
        if hasattr(self.metrics_context, "context"):
            # Calculate progress percentage and ETA
            self.metrics_context.context.metrics.update(
                {
                    "steps_completed": steps_completed,
                    "total_steps": total_steps,
                    "progress": f"{(steps_completed/total_steps)*100:.1f}%",
                    "eta": self._calculate_eta(steps_completed, total_steps),
                }
            )

    def _calculate_eta(self, steps_completed: int, total_steps: int) -> str:
        """
        Calculate estimated time remaining based on progress.

        Args:
            steps_completed (int): Number of steps completed
            total_steps (int): Total number of steps

        Returns:
            str: Formatted string with estimated time remaining
        """
        if not hasattr(self.metrics_context, "context") or steps_completed == 0:
            return "calculating..."

        context = self.metrics_context.context
        elapsed_time = time.time() - context.start_time
        time_per_step = elapsed_time / steps_completed
        remaining_steps = total_steps - steps_completed
        eta_seconds = time_per_step * remaining_steps

        return f"{eta_seconds:.1f}s"

    def end_operation(self, status: str = "completed"):
        """
        Mark the current operation as complete and log final metrics.

        Args:
            status (str): Final status of the operation
        """
        if hasattr(self.metrics_context, "context"):
            context = self.metrics_context.context
            elapsed_time = time.time() - context.start_time

            # Update final metrics
            context.metrics.update(
                {"status": status, "elapsed_time": f"{elapsed_time:.2f}s"}
            )

            # Log final metrics summary
            self.logger.info(
                f"Operation completed: {json.dumps(context.metrics, indent=2)}"
            )


##################################################
# Validation                                     #
##################################################


class DataValidator:
    """
    Validates financial data for consistency and correctness.

    Features:
    - DataFrame structure validation
    - Price data validation
    - Date continuity checking
    - Required column verification
    """

    def __init__(self, config=None):
        """
        Initialize the validator.

        Args:
            config: Configuration object containing validation settings
        """
        self.logger = StructuredLogger("DataValidator", config=config)
        self.required_columns = {"Close", "Open", "High", "Low", "Volume"}

    def validate_dataframe(
        self, df: pd.DataFrame, ticker: str
    ) -> tuple[bool, list[str]]:
        """
        Validate entire DataFrame for data quality and completeness.

        Args:
            df: DataFrame to validate
            ticker: Stock ticker symbol

        Returns:
            tuple: (is_valid: bool, error_messages: list[str])
        """
        errors = []

        try:
            # Basic DataFrame checks
            if df is None:
                return False, ["DataFrame is None"]

            if df.empty:
                return False, [f"Empty DataFrame for {ticker}"]

            # Column validation
            missing_columns = self.required_columns - set(df.columns)
            if missing_columns:
                errors.append(f"Missing required columns: {missing_columns}")
                return False, errors

            # Check for missing values
            null_counts = df[list(self.required_columns)].isnull().sum()
            if null_counts.any():
                for col, count in null_counts.items():
                    if count > 0:
                        errors.append(f"Found {count} missing values in {col}")

            # Price validation
            if (df["Close"] <= 0).any():
                errors.append("Found non-positive price values")

            if not np.issubdtype(df["Close"].dtype, np.number):
                errors.append("Price column is not numeric")

            # Volume validation
            if (df["Volume"] < 0).any():
                errors.append("Found negative volume values")

            # Date validation
            if isinstance(df.index, pd.DatetimeIndex):
                # Check for duplicate dates
                if df.index.duplicated().any():
                    errors.append("Found duplicate dates")

                # Check for date gaps
                date_gaps = self._check_date_continuity(df.index)
                if date_gaps:
                    errors.append(f"Found gaps in dates: {date_gaps}")

            return len(errors) == 0, errors

        except Exception as e:
            self.logger.error(f"Validation error for {ticker}: {str(e)}")
            return False, [f"Validation error: {str(e)}"]

    def _check_date_continuity(self, dates: pd.DatetimeIndex) -> list[str]:
        """
        Check for gaps in trading dates.

        Args:
            dates: DatetimeIndex to check

        Returns:
            list: Dates where gaps were found
        """
        try:
            # Convert to business day frequency
            bday = pd.tseries.offsets.BDay()
            expected_dates = pd.date_range(
                start=dates.min(), end=dates.max(), freq=bday
            )

            # Find missing business days
            missing_dates = set(expected_dates) - set(dates)

            if missing_dates:
                return [d.strftime("%Y-%m-%d") for d in sorted(missing_dates)]

            return []

        except Exception as e:
            self.logger.error(f"Error checking date continuity: {e}")
            return []

    def validate_ticker_data(self, hist_data: pd.DataFrame, ticker: str) -> bool:
        """
        Validate historical data for a specific ticker.

        Args:
            hist_data: Historical price data
            ticker: Stock ticker symbol

        Returns:
            bool: True if data is valid
        """
        is_valid, errors = self.validate_dataframe(hist_data, ticker)

        if not is_valid:
            self.logger.warning(
                f"Validation failed for {ticker}:\n"
                + "\n".join(f"- {error}" for error in errors)
            )

        return is_valid


##################################################
# Memory management and data processing          #
##################################################


class MemoryManager:
    """
    Manages memory usage and cleanup throughout the application.
    Monitors memory consumption and triggers cleanup when needed.

    Features:
    - Memory usage monitoring
    - Automatic cleanup triggers
    - Weak reference caching
    - Resource usage tracking
    """

    def __init__(self, max_memory_percent=75, chunk_size=1000, config=None):
        """
        Initialize the memory manager with specified thresholds.

        Args:
            max_memory_percent (int): Maximum memory usage percentage before cleanup
            chunk_size (int): Size of data chunks for processing
        """
        self.max_memory_percent = max_memory_percent
        self.chunk_size = chunk_size
        self.logger = StructuredLogger("MemoryManager")
        # Use weak references to allow garbage collection of unused data
        self._cached_data = weakref.WeakValueDictionary()
        self.logger = StructuredLogger("MemoryManager", config=config)

    def get_memory_usage(self) -> Dict[str, float]:
        """
        Get current memory usage statistics.

        Returns:
            Dict containing:
            - rss: Resident Set Size (actual memory used) in MB
            - vms: Virtual Memory Size in MB
            - percent: Percentage of system memory used
        """
        process = psutil.Process()
        memory_info = process.memory_info()

        try:
            process = psutil.Process()
            memory_info = process.memory_info()
            return {
                "rss": memory_info.rss / (1024 * 1024),
                "vms": memory_info.vms / (1024 * 1024),
                "percent": process.memory_percent(),
            }
        except psutil.Error as e:
            self.logger.error(f"Error getting memory usage: {e}")
            return {"rss": 0, "vms": 0, "percent": 0}

    @contextmanager
    def monitor_memory(self, operation_name: str):
        """
        Context manager to monitor memory usage during an operation.

        Args:
            operation_name (str): Name of the operation being monitored

        Usage:
            with memory_manager.monitor_memory("data_processing"):
                process_data()
        """
        initial_usage = self.get_memory_usage()
        yield
        final_usage = self.get_memory_usage()

        # Log memory usage delta
        self.logger.info(
            f"Memory delta for {operation_name}: "
            f"{final_usage['rss'] - initial_usage['rss']:.2f}MB",
        )

    def should_cleanup(self) -> bool:
        """
        Check if memory cleanup is needed based on usage threshold.

        Returns:
            bool: True if cleanup should be performed
        """
        return self.get_memory_usage()["percent"] > self.max_memory_percent

    def cleanup(self) -> int:
        """
        Perform memory cleanup operations.

        Returns:
            int: Number of items cleaned from cache
        """
        # Force garbage collection
        gc.collect()
        n_cleaned = len(self._cached_data)
        # Clear weak reference cache
        self._cached_data.clear()
        return n_cleaned


class StreamingDataProcessor:
    """
    Base class for processing data in streaming fashion to manage memory usage.
    Processes data in chunks to avoid loading entire dataset into memory.

    Features:
    - Chunk-based processing
    - Memory usage monitoring
    - Automatic cleanup triggers
    """

    def __init__(self, config=None, chunk_size=1000):
        """
        Initialize the streaming processor.

        Args:
            config: Configuration object containing settings.
            chunk_size (int): Number of items to process in each chunk.
        """
        self.config = config
        self.chunk_size = chunk_size
        self.memory_manager = MemoryManager(chunk_size=chunk_size)
        self.logger = StructuredLogger("DataProcessor", config=config)

    def process_data_stream(self, data_iterator: Iterator) -> Iterator:
        """
        Process a stream of data items in chunks.

        Args:
            data_iterator: Iterator yielding data items

        Yields:
            Processed data items one at a time
        """
        buffer = []
        processed_count = 0

        for item in data_iterator:
            buffer.append(item)

            # Process buffer when it reaches chunk size
            if len(buffer) >= self.chunk_size:
                yield from self._process_buffer(buffer)
                processed_count += len(buffer)
                buffer = []

                # Check if cleanup needed
                if self.memory_manager.should_cleanup():
                    cleaned = self.memory_manager.cleanup()
                    self.logger.info(
                        f"Memory cleanup triggered: {cleaned} items cleared",
                    )

            # Process remaining items in buffer
            if buffer:
                yield from self._process_buffer(buffer)

    def _process_buffer(self, buffer: list) -> Iterator:
        """
        Process a buffer of items while monitoring memory usage with batching.

        Args:
            buffer (list): List of items to process

        Yields:
            Processed items one at a time
        """
        with self.memory_manager.monitor_memory("buffer_processing"):
            # Process items in batches for better performance
            batch_size = 100
            for i in range(0, len(buffer), batch_size):
                batch = buffer[i : i + batch_size]

                # Process batch in parallel using ThreadPoolExecutor
                with ThreadPoolExecutor(max_workers=min(len(batch), 5)) as executor:
                    futures = [
                        executor.submit(self._process_single_item, item)
                        for item in batch
                    ]

                    for future in as_completed(futures):
                        try:
                            result = future.result()
                            if result is not None:
                                yield result
                        except Exception as e:
                            self.logger.error(f"Batch processing error: {e}")

    def _process_single_item(self, item):
        """
        Process a single data item. Override in subclasses.

        Args:
            item: Data item to process

        Returns:
            Processed item
        """
        return item


class StreamingTickerProcessor(StreamingDataProcessor):
    """
    Specialized streaming processor for handling stock ticker data.
    Inherits from StreamingDataProcessor to leverage chunk-based processing.
    """

    def __init__(self, config=None, chunk_size=50):
        """
        Initialize the ticker processor.

        Args:
            config: Configuration object containing ticker settings.
            chunk_size (int): Number of tickers to process in each chunk.
        """
        super().__init__(config, chunk_size)
        self.config = config
        self.logger = StructuredLogger("StreamingTickerProcessor", config=config)

    def _process_single_item(self, item):
        """
        Process a single ticker item.

        Args:
            item: Dictionary containing ticker information

        Returns:
            Processed ticker data
        """
        ticker = item["ticker"]
        raw_price = item["price"]
        currency = item["currency"]
        date = item["date"]

        with self.memory_manager.monitor_memory(f"process_{ticker}"):
            try:
                # Process  data here if it is memory intensive
                # position start = time.time() and end = time.time() depending on what you want to time

                start = time.time()

                # Step 1: Convert Price
                # converted_price = self.price_converter.convert_price(
                #     raw_price, currency, self.exchange_rates, date, ticker
                # )

                end = time.time()

                # Step 2: Clean Data
                # cleaned_data = self.data_processor.clean_data(converted_price)

                # Step 3: Calculate Indicators
                # indicators = self.data_processor.calculate_indicators(cleaned_data)

                # Step 4: Aggregate Data
                # final_data = self.data_processor.aggregate_data(indicators)

                # Optional: Cache the processed data
                # cache_path = self.cache_manager.get_cache_path(
                #     ticker, date.strftime("%Y-%m-%d"), date.strftime("%Y-%m-%d")
                # )
                # self.cache_manager.save_to_cache(cache_path, (final_data, currency))

                return final_data

            except Exception as e:
                self.logger.error(f"Error processing ticker {ticker}: {e}")
                # Depending on your application logic, you might return None or handle differently
                return None

            # Log metrics related to memory usage, processing time, etc., to identify bottlenecks or inefficiencies.
            self.logger.info(f"Processed {ticker} in {end - start:.2f} seconds")

        return item


##################################################
# Data fetching and caching infrastructure       #
##################################################


class CacheManager:
    """
    Manages caching of financial data to reduce API calls and improve performance.

    Features:
    - File-based caching with MD5 hashing
    - Cache validation and expiration
    - Metadata tracking
    - Automatic cleanup
    """

    def __init__(
        self, config_path, max_age_hours=168, cleanup_threshold=200, config=None
    ):
        """
        Initialize the cache manager.

        Args:
            config_path (str): Base path for cache storage
            max_age_hours (int): Maximum age of cache entries in hours (default 1 week)
            cleanup_threshold (int): Number of entries before triggering cleanup
        """
        self.config_path = Path(config_path).resolve()
        self.cache_dir = self.config_path / "cache"
        # Create cache directory if it doesn't exist
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.max_age = timedelta(hours=max_age_hours)
        self.cleanup_threshold = cleanup_threshold

        # Initialize StructuredLogger instead of standard logger
        self.logger = StructuredLogger("CacheManager", config=config)

        self.metadata_file = self.cache_dir / "cache_metadata.pkl"
        self.metadata = self._load_metadata()

    def _format_path_for_log(self, path):
        """Convert path to plain string without formatting."""
        return str(path).replace("\\", "/")

    def get_cache_path(self, ticker, start_date, end_date):
        """Generate a consistent and versioned cache file path."""
        CACHE_VERSION = "v1"  # Increment when cache format changes

        # Normalize dates to ensure consistent formatting
        start = pd.Timestamp(start_date).strftime("%Y-%m-%d")
        end = pd.Timestamp(end_date).strftime("%Y-%m-%d")

        # Create a unique and consistent cache key
        cache_key = f"{CACHE_VERSION}_{ticker}_{start}_{end}".encode("utf-8")
        hash_value = hashlib.sha256(cache_key).hexdigest()

        # Include ticker in filename for easier debugging
        return self.cache_dir / f"{ticker}_{hash_value[:12]}.pkl"

    def get_formatted_path(self, path):
        """Returns a formatted string version of the path for logging."""
        return str(path).replace("\\", "/")

    def load_from_cache(self, cache_path):
        """Load data from cache if valid."""
        try:
            if not self._is_cache_valid(cache_path):
                return None

            with open(cache_path, "rb") as f:
                data = pickle.load(f)
                if self._validate_cache_data(data):
                    self._update_metadata(cache_path, "access")
                    self.logger.info(
                        f"Successfully retrieved from cache: {self._format_path_for_log(cache_path.name)}"
                    )
                    return data
            return None

        except Exception as e:
            self.logger.error(
                f"Cache load error for {self._format_path_for_log(cache_path.name)}: {e}"
            )
            self._mark_invalid(cache_path)
            return None

    def save_to_cache(self, cache_path, data):
        """
        Save data to cache with metadata.

        Args:
            cache_path (Path): Path to cache file
            data: Data to cache

        Returns:
            bool: True if save successful, False otherwise
        """
        try:
            with open(cache_path, "wb") as f:
                pickle.dump(data, f)
            self._update_metadata(cache_path, "write")
            # Use str() for path
            self.logger.info(f"Successfully saved to cache: {str(cache_path.name)}")
            return True

        except Exception as e:
            # Use str() for path
            self.logger.error(f"Cache save error for {str(cache_path.name)}: {e}")
            return False

    def _load_metadata(self):
        """
        Load or initialize cache metadata.

        Returns:
            dict: Cache metadata
        """
        if self.metadata_file.exists():
            try:
                with open(self.metadata_file, "rb") as f:
                    return pickle.load(f)
            except Exception as e:
                self.logger.error(f"Error loading cache metadata: {e}")

        # Initialize new metadata if loading fails
        metadata = {"entries": {}, "last_cleanup": datetime.now()}

        try:
            with open(self.metadata_file, "wb") as f:
                pickle.dump(metadata, f)
        except Exception as e:
            self.logger.error(f"Error saving initial metadata: {e}")

        return metadata

    def _save_metadata(self):
        with threading.Lock():
            try:
                with open(self.metadata_file, "wb") as f:
                    pickle.dump(self.metadata, f)
            except Exception as e:
                self.logger.error(f"Error saving cache metadata: {e}")

    def _update_metadata(self, cache_path, action_type):
        """
        Update metadata for a cache entry.

        Args:
            cache_path (Path): Path to cache file
            action_type (str): Type of action ('write' or 'access')
        """
        now = datetime.now()
        str_path = str(cache_path)

        if action_type == "write":
            # New cache entry
            self.metadata["entries"][str_path] = {
                "created": now,
                "last_access": now,
                "valid": True,
            }
        elif action_type == "access":
            # Update access time for existing entry
            if str_path in self.metadata["entries"]:
                self.metadata["entries"][str_path]["last_access"] = now

        self._save_metadata()

    def _is_cache_valid(self, cache_path):
        """Check if a cache entry is valid and not expired."""
        if not cache_path.exists():
            self.logger.info(
                f"Cache file does not exist: {self.get_formatted_path(cache_path)}"
            )
            return False

        str_path = str(cache_path)
        entry = self.metadata["entries"].get(str_path)

        if not entry:
            self.logger.info(
                f"No metadata entry for: {self.get_formatted_path(cache_path)}"
            )
            return False

        if not entry.get("valid", False):
            self.logger.info(
                f"Cache entry marked invalid: {self._format_path_for_log(cache_path.name)}"
            )
            return False

        age = datetime.now() - entry["created"]
        if age > self.max_age:
            self.logger.info(
                f"Cache expired for {self._format_path_for_log(cache_path.name)}. Age: {age}"
            )
            return False

        return True

    def _validate_cache_data(self, data):
        """
        Validate cached data structure.

        Args:
            data: Cached data to validate

        Returns:
            bool: True if data structure is valid
        """
        try:
            # Check if data is a tuple with 2 elements
            if not isinstance(data, tuple) or len(data) != 2:
                return False

            # Unpack data and verify structure
            df, currency = data
            if not isinstance(df, pd.DataFrame):
                return False

            required_columns = {"Date", "price", "ticker"}
            if not required_columns.issubset(df.columns):
                return False

            # Validate data types
            if not df["price"].dtype.kind in "fc":  # float or complex
                return False

            # Check for invalid values
            if df["price"].isnull().any() or (df["price"] <= 0).any():
                return False

            return True

        except Exception as e:
            self.logger.error(f"Cache validation error: {e}")
            return False

    def _mark_invalid(self, cache_path):
        """
        Mark a cache entry as invalid in metadata.

        Args:
            cache_path (Path): Path to cache file
        """
        str_path = str(cache_path)
        if str_path in self.metadata["entries"]:
            self.metadata["entries"][str_path]["valid"] = False
            self._save_metadata()
            self.logger.info(f"Marked invalid: {cache_path.name}")

    def clear_cache(self):
        """
        Clear all cache entries and reset metadata.

        Returns:
            bool: True if cleanup successful
        """
        try:
            # Remove all cache files
            for path in self.cache_dir.glob("*.pkl"):
                path.unlink()

            # Reset metadata
            self.metadata = {"entries": {}, "last_cleanup": datetime.now()}
            self._save_metadata()

            self.logger.info("Cache cleared successfully")
            return True

        except Exception as e:
            self.logger.error(f"Error clearing cache: {e}")
            return False


##################################################
# Yahoo Finance data fetching implementation     #
##################################################


class RetryManager:
    """
    Manages retry attempts for API calls with exponential backoff.

    Features:
    - Exponential backoff between retries
    - Rate limiting with token bucket algorithm
    - Detailed attempt logging
    - Custom validation support
    """

    class RateLimit:
        """Internal class for rate limiting."""

        def __init__(self, max_requests: int, time_window: int):
            """
            Initialize rate limiter.

            Args:
                max_requests: Maximum requests allowed in time window
                time_window: Time window in seconds
            """
            self.max_requests = max_requests
            self.time_window = time_window
            self.requests = []
            self._lock = threading.Lock()

        def wait_if_needed(self) -> None:
            """Implement rate limiting with token bucket algorithm."""
            with self._lock:
                now = time.time()

                # Remove expired timestamps
                self.requests = [t for t in self.requests if now - t < self.time_window]

                if len(self.requests) >= self.max_requests:
                    # Calculate sleep time needed
                    sleep_time = self.requests[0] + self.time_window - now
                    if sleep_time > 0:
                        time.sleep(sleep_time)

                self.requests.append(now)

    def __init__(
        self,
        max_retries: int = 3,
        base_delay: int = 1,
        max_delay: int = 60,
        config=None,
    ):
        """
        Initialize the retry manager.

        Args:
            max_retries: Maximum number of retry attempts
            base_delay: Initial delay between retries in seconds
            max_delay: Maximum delay between retries in seconds
            config: Configuration object
        """
        self.max_retries = max_retries
        self.base_delay = base_delay
        self.max_delay = max_delay
        self.logger = StructuredLogger("RetryManager", config=config)
        # Create rate limiter instance
        self.rate_limit = self.RateLimit(max_requests=30, time_window=60)
        self.attempt_logs = []

    def execute_with_retry(self, operation, validation_func=None):
        """
        Execute an operation with automatic retries on failure.

        Args:
            operation: Function to execute
            validation_func: Optional function to validate results

        Returns:
            tuple: (result, number of attempts)

        Raises:
            RuntimeError: If all retry attempts fail
        """
        last_exception = None

        for attempt in range(self.max_retries):
            try:
                # Apply rate limiting
                self.rate_limit.wait_if_needed()

                # Attempt operation
                result = operation()

                # Validate result if function provided
                if validation_func and not validation_func(result):
                    raise ValueError("Validation failed")

                # Log success
                self.logger.info(
                    f"Operation succeeded on attempt {attempt + 1}/{self.max_retries}"
                )

                return result, attempt + 1

            except Exception as e:
                last_exception = e

                # Calculate delay with exponential backoff and jitter
                delay = min(
                    self.base_delay * (2**attempt) * (1 + random.random() * 0.1),
                    self.max_delay,
                )

                # Log attempt details
                attempt_info = {
                    "attempt": attempt + 1,
                    "error": str(e),
                    "error_type": type(e).__name__,
                    "delay": delay,
                    "timestamp": datetime.now().isoformat(),
                }
                self.attempt_logs.append(attempt_info)

                self.logger.warning(
                    f"Retry attempt {attempt + 1}/{self.max_retries} failed: {e}"
                )

                # Skip delay on last attempt
                if attempt < self.max_retries - 1:
                    time.sleep(delay)

        # If all attempts fail, raise error with attempt history
        raise RuntimeError(
            f"All retry attempts failed. Last error: {last_exception}\n"
            f"Attempt history: {json.dumps(self.attempt_logs, indent=2)}"
        )

    def reset(self):
        """Reset attempt logs and rate limiting data."""
        self.attempt_logs = []
        self.rate_limit = self.RateLimit(max_requests=30, time_window=60)


class YahooFinanceFetcher:
    """
    Handles fetching and processing financial data from Yahoo Finance API.

    Features:
    - Automatic retry handling
    - Data validation
    - Error recovery
    - Caching support
    - Backup management
    """

    def __init__(
        self,
        max_retries=3,
        retry_delay=2,
        validation_days=5,
        cache_dir=None,
        config=None,
    ):
        self.config = config
        self.validator = DataValidator(config=config)
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        self.validation_days = validation_days
        self.cache_dir = Path(cache_dir) if cache_dir else None
        self.retry_manager = RetryManager(
            max_retries=max_retries, base_delay=retry_delay
        )
        self.logger = StructuredLogger("YahooFinanceFetcher", config=config)

    def fetch_data(
        self, ticker: str, start_date: str, end_date: str
    ) -> tuple[pd.DataFrame, str]:
        """
        Main method to fetch data with retries, validation, and fallbacks.

        Args:
            ticker (str): Stock ticker symbol
            start_date (str): Start date for data range (ISO format)
            end_date (str): End date for data range (ISO format)

        Returns:
            tuple: (Historical data DataFrame, Currency string)

        Raises:
            RuntimeError: If all data fetching attempts fail
            ValueError: If input validation fails
        """
        # Input validation
        try:
            pd.Timestamp(start_date)
            pd.Timestamp(end_date)
        except ValueError as e:
            self.logger.error(f"Invalid date format for {ticker}: {e}")
            raise ValueError(f"Invalid date format: {e}")

        if not isinstance(ticker, str) or not ticker:
            raise ValueError("Invalid ticker symbol")

        errors = []
        retry_count = 0
        max_extended_retries = 2  # Number of times to try with extended dates

        # Initialize metrics tracking
        metrics = {
            "attempts": 0,
            "cache_hits": 0,
            "network_errors": 0,
            "validation_failures": 0,
            "start_time": time.time(),
        }

        try:
            while retry_count <= self.max_retries:
                try:
                    # Try primary fetch method
                    hist, currency = self._try_yahoo_finance(
                        ticker, start_date, end_date
                    )

                    # Validate the fetched data
                    if self._validate_data(hist, ticker):
                        # Save successful fetch to backup
                        self._save_to_backup(ticker, hist, currency)
                        self._log_success_metrics(metrics)
                        return hist, currency

                    metrics["validation_failures"] += 1
                    self.logger.warning(
                        f"Attempt {retry_count + 1}: Data validation failed for {ticker}. "
                        "Trying extended date range..."
                    )

                    # Try with extended date range if validation fails
                    for extended_try in range(max_extended_retries):
                        try:
                            # Calculate extended dates
                            extended_days = (
                                extended_try + 1
                            ) * 5  # Extend by 5, then 10 days
                            hist, currency = self._try_with_extended_dates(
                                ticker,
                                start_date,
                                end_date,
                                extended_days=extended_days,
                            )

                            if self._validate_data(hist, ticker):
                                # Trim to requested date range and save backup
                                trimmed_hist = self._trim_to_requested_dates(
                                    hist, start_date, end_date
                                )
                                self._save_to_backup(ticker, trimmed_hist, currency)
                                self._log_success_metrics(metrics)
                                return trimmed_hist, currency

                        except Exception as extended_e:
                            error_msg = (
                                f"Extended date attempt {extended_try + 1} failed for "
                                f"{ticker}: {str(extended_e)}"
                            )
                            self.logger.warning(error_msg)
                            errors.append(error_msg)

                except requests.RequestException as e:
                    metrics["network_errors"] += 1
                    error_msg = f"Network error on attempt {retry_count + 1} for {ticker}: {str(e)}"
                    self.logger.warning(error_msg)
                    errors.append(error_msg)

                except yf.YFinanceError as e:
                    error_msg = f"YFinance error on attempt {retry_count + 1} for {ticker}: {str(e)}"
                    self.logger.warning(error_msg)
                    errors.append(error_msg)

                except Exception as e:
                    error_msg = f"Unexpected error on attempt {retry_count + 1} for {ticker}: {str(e)}"
                    self.logger.warning(error_msg)
                    errors.append(error_msg)

                retry_count += 1
                metrics["attempts"] += 1

                if retry_count < self.max_retries:
                    # Calculate backoff time with jitter
                    backoff = min(
                        self.retry_delay
                        * (2**retry_count)
                        * (1 + random.random() * 0.1),
                        self.max_delay,
                    )
                    self.logger.info(f"Retrying {ticker} in {backoff:.2f}s...")
                    time.sleep(backoff)

            # All attempts failed, try to recover from backup
            try:
                self.logger.info(f"Attempting backup recovery for {ticker}")
                hist, currency = self._recover_from_backup(ticker, start_date, end_date)
                if hist is not None:
                    self.logger.info(f"Successfully recovered {ticker} from backup")
                    self._log_success_metrics(metrics, from_backup=True)
                    return hist, currency
            except Exception as backup_e:
                errors.append(f"Backup recovery failed: {str(backup_e)}")

            # Log detailed failure metrics
            self._log_failure_metrics(metrics, ticker)

            # If everything fails, raise error with detailed attempt history
            raise RuntimeError(
                f"All attempts to fetch {ticker} failed after {metrics['attempts']} attempts:\n"
                f"Network errors: {metrics['network_errors']}\n"
                f"Validation failures: {metrics['validation_failures']}\n"
                f"Total time: {time.time() - metrics['start_time']:.2f}s\n"
                f"Errors:\n" + "\n".join(f"- {e}" for e in errors)
            )

        finally:
            # Clean up any resources
            self._cleanup_resources()

    def _log_success_metrics(self, metrics: Dict[str, Any], from_backup: bool = False):
        """Log success metrics for monitoring and debugging."""
        duration = time.time() - metrics["start_time"]
        self.logger.info(
            f"Data fetch succeeded {'from backup' if from_backup else 'from API'} "
            f"after {metrics['attempts']} attempts in {duration:.2f}s"
        )

    def _log_failure_metrics(self, metrics: Dict[str, Any], ticker: str):
        """Log failure metrics for monitoring and debugging."""
        duration = time.time() - metrics["start_time"]
        self.logger.error(
            f"Data fetch failed for {ticker}:\n"
            f"Attempts: {metrics['attempts']}\n"
            f"Network errors: {metrics['network_errors']}\n"
            f"Validation failures: {metrics['validation_failures']}\n"
            f"Duration: {duration:.2f}s"
        )

    def _cleanup_resources(self):
        """Clean up any resources used during data fetching."""
        try:
            # Add any necessary cleanup code here
            gc.collect()  # Force garbage collection
        except Exception as e:
            self.logger.warning(f"Cleanup error: {e}")

    def _try_yahoo_finance(self, ticker, start_date, end_date):
        """
        Attempt to fetch data directly from Yahoo Finance.

        Args:
            ticker (str): Stock ticker symbol
            start_date (str): Start date
            end_date (str): End date

        Returns:
            tuple: (Historical data, Currency)
        """
        stock = yf.Ticker(ticker)
        hist = stock.history(
            start=start_date, end=end_date, interval="1d", auto_adjust=True
        )

        if hist.empty:
            raise ValueError(f"No data returned for {ticker}")

        # Localize timezone for the index
        if isinstance(hist.index, pd.DatetimeIndex):
            hist.index = hist.index.tz_localize(None)  # Remove timezone info

        return hist, stock.info.get("currency", "Unknown")

    def _validate_data(self, hist, ticker):
        """
        Validate the fetched data meets requirements.

        Args:
            hist (DataFrame): Historical data
            ticker (str): Stock ticker symbol

        Returns:
            bool: True if data is valid
        """
        try:
            if hist.empty:
                self.logger.warning(f"Empty data received for {ticker}")
                return False

            # Check required columns
            required_columns = {"Close", "Open", "High", "Low", "Volume"}
            if not all(col in hist.columns for col in required_columns):
                self.logger.warning(
                    f"Missing required columns for {ticker}. "
                    f"Required: {required_columns}, "
                    f"Got: {hist.columns.tolist()}"
                )
                return False

            # Check for missing values in required columns
            if hist[list(required_columns)].isnull().any().any():
                self.logger.warning(f"Missing values in required columns for {ticker}")
                return False

            # Check for price anomalies
            if (hist["Close"] <= 0).any():
                self.logger.warning(
                    f"Invalid prices (zero or negative) found for {ticker}"
                )
                return False

            return True

        except Exception as e:
            self.logger.error(f"Error validating data for {ticker}: {e}")
            return False

    def _try_with_extended_dates(self, ticker, start_date, end_date):
        """
        Try fetching with extended date range for more data.

        Args:
            ticker (str): Stock ticker symbol
            start_date (str): Original start date in ISO format
            end_date (str): Original end date in ISO format

        Returns:
            tuple: (Historical data, Currency)
        """
        # Convert to timestamps, assuming ISO input
        extended_start = pd.Timestamp(start_date) - pd.Timedelta(days=5)
        extended_end = pd.Timestamp(end_date) + pd.Timedelta(days=5)

        # Return to ISO format strings for API call
        return self._try_yahoo_finance(
            ticker,
            extended_start.strftime("%Y-%m-%d"),
            extended_end.strftime("%Y-%m-%d"),
        )

    def _trim_to_requested_dates(self, hist, start_date, end_date):
        """
        Trim data to originally requested date range.

        Args:
            hist (DataFrame): Historical data
            start_date (str): Start date
            end_date (str): End date

        Returns:
            DataFrame: Trimmed historical data
        """
        return hist[start_date:end_date]

    def _save_to_backup(self, ticker, hist, currency):
        """
        Save successful fetch to backup cache.

        Args:
            ticker (str): Stock ticker symbol
            hist (DataFrame): Historical data
            currency (str): Currency of the data
        """
        if not self.cache_dir:
            return

        try:
            backup_file = self.cache_dir / f"{ticker}_backup.pkl"
            data = {"timestamp": datetime.now(), "data": hist, "currency": currency}
            pd.to_pickle(data, backup_file)
        except Exception as e:
            self.logger.warning(f"Failed to save backup for {ticker}: {e}")

    def _recover_from_backup(
        self, ticker: str, start_date: str, end_date: str
    ) -> tuple[pd.DataFrame | None, str | None]:
        """Attempt to recover data from backup cache."""
        if not self.cache_dir:
            self.logger.info(f"No cache directory configured for {ticker}")
            return None, None

        try:
            backup_file = self.cache_dir / f"{ticker}_backup.pkl"
            if not backup_file.exists():
                self.logger.info(f"No backup file found for {ticker}")
                return None, None

            with open(backup_file, "rb") as f:
                data = pickle.load(f)

            # Check backup age
            if datetime.now() - data["timestamp"] > timedelta(hours=24):
                self.logger.info(f"Backup too old for {ticker}")
                return None, None

            hist = data["data"]

            # Validate data structure and content
            if not self._validate_data(hist, ticker):
                self.logger.info(f"Invalid backup data for {ticker}")
                return None, None

            # Validate date range
            if not self._validate_date_range(hist, start_date, end_date):
                return None, None

            trimmed_data = self._trim_to_requested_dates(hist, start_date, end_date)
            if trimmed_data is not None:
                self.logger.info(f"Successfully recovered {ticker} from backup")
                return trimmed_data, data["currency"]
            else:
                self.logger.info(f"Failed to trim backup data for {ticker}")

        except Exception as e:
            self.logger.error(f"Failed to recover backup for {ticker}: {e}")

        return None, None

    def _validate_backup_data(self, data):
        """Validate backup data structure and content"""
        required_keys = {"timestamp", "data", "currency"}
        if not isinstance(data, dict) or not all(k in data for k in required_keys):
            return False

        if not isinstance(data["data"], pd.DataFrame):
            return False

        required_columns = {"Close", "Open", "High", "Low", "Volume"}
        if not all(col in data["data"].columns for col in required_columns):
            return False

        return True

    def _validate_date_range(
        self, hist: pd.DataFrame, start_date: str, end_date: str
    ) -> bool:
        """Validate if the historical data covers the requested date range."""
        try:
            start = pd.Timestamp(start_date)
            end = pd.Timestamp(end_date)
            data_start = hist.index.min()
            data_end = hist.index.max()

            if start < data_start or end > data_end:
                self.logger.info(
                    f"Date range mismatch: requested {start} to {end}, "
                    f"but have {data_start} to {data_end}"
                )
                return False
            return True
        except Exception as e:
            self.logger.error(f"Error validating date range: {e}")
            return False


##################################################
# Price conversion and data processing           #
##################################################


class PriceConverter:
    """
    Handles currency conversions and price adjustments for financial data.

    Features:
    - Currency conversion to GBP
    - Special handling for British pence (GBp)
    - Detailed debugging information
    - Exchange rate management
    """

    def __init__(self, config=None):
        """
        Initialize the price converter.

        Args:
            config: Configuration object containing currency settings
        """
        self.config = config
        self.debug_info = []
        self._last_dates = {}
        self.logger = StructuredLogger("PriceConverter", config=config)

    def add_debug_info(
        self, ticker, date, price, currency, rate=None, converted_price=None
    ):
        """
        Add detailed debugging information for a price conversion.
        Uses ISO format dates consistently.
        """
        debug_entry = [
            "",  # Add empty line for readability
            f"Debug - {ticker}:",
            f"Date: {date.strftime('%d/%m/%Y')}",
            f"Currency: {currency}",
            f"Original price: {price:.4f} {currency}",
        ]

        if currency == "GBp":
            converted_price = price / 100
            debug_entry.extend(
                [
                    f"Converting GBp to GBP (divide by 100)",
                    f"Revised price = {price:.4f} / 100 = {converted_price:.4f} GBP",
                ]
            )
        elif currency == "GBP":
            debug_entry.extend(["No need for conversion"])
        elif rate is not None:
            debug_entry.extend(
                [
                    f"{date.strftime('%d/%m/%Y')} Exchange rate ({currency}/GBP): {rate:.4f}",
                    f"Revised price = {price:.4f} * {rate:.4f} = {converted_price:.4f} GBP",
                ]
            )

        self.debug_info.extend(debug_entry)

    def get_exchange_rate(self, rates, date):
        """
        Get exchange rate for a specific date, falling back to closest available.
        """
        if rates is None or not isinstance(rates, dict):
            logging.error(f"Invalid rates data: {rates}")
            return 1.0

        try:
            # Ensure input date is naive (no timezone)
            if isinstance(date, pd.Timestamp):
                if date.tz is not None:
                    date = date.tz_localize(None)
            else:
                date = pd.Timestamp(date).tz_localize(None)

            # Convert all rate dates to naive timestamps for comparison
            naive_rates = {
                k.tz_localize(None) if k.tz is not None else k: v
                for k, v in rates.items()
            }

            # If exact date exists, use it
            if date in naive_rates:
                return naive_rates[date]

            # Find closest available date
            available_dates = sorted(naive_rates.keys())
            closest_date = min(available_dates, key=lambda x: abs(x - date))
            return naive_rates[closest_date]

        except (KeyError, AttributeError) as e:
            logging.error(f"Exchange rate error: {e}")
            return 1.0

    def convert_price(self, price, currency, exchange_rates, date, ticker):
        """
        Convert price from source currency to GBP.
        """
        # First ensure the date is timezone-aware and in UTC
        if isinstance(date, pd.Timestamp):
            if date.tz is None:
                date = date.tz_localize("UTC")
            else:
                date = date.tz_convert("UTC")

        # Rest of the method stays the same
        if ticker in self.config.CURRENCY_SYMBOLS.values():
            return price  # Return price as-is for exchange rate tickers

        is_debug = ticker in self.config.DEBUG_TICKERS

        if is_debug:
            if ticker not in self._last_dates or date > self._last_dates[ticker]:
                self._last_dates[ticker] = date
                should_debug = True
            else:
                should_debug = False
        else:
            should_debug = False

        if currency == "GBp":
            converted = price / 100
            if should_debug:
                self.add_debug_info(ticker, date, price, currency, None, converted)
            return converted

        if (
            currency in exchange_rates
            and exchange_rates.get(currency)
            and exchange_rates[currency] is not None
        ):
            rate = self.get_exchange_rate(exchange_rates[currency], date)
            if rate is not None:
                converted = price * rate
                if should_debug:
                    self.add_debug_info(ticker, date, price, currency, rate, converted)
                return converted

        if should_debug:
            self.add_debug_info(ticker, date, price, currency)
        return price


class DataProcessor:
    """
    Processes and transforms financial data into required format.

    Features:
    - Data formatting and cleaning
    - Currency conversion handling
    - Date normalization
    - Error handling
    """

    def __init__(self, config=None):
        """
        Initialize the data processor.

        Args:
            config: Configuration object containing processing settings
        """
        self.config = config
        self.validator = DataValidator(config=config)
        self.logger = StructuredLogger("DataProcessor", config)
        self.price_converter = PriceConverter(config=config)

    def process_ticker_data(self, hist_data, ticker, currency, exchange_rates):
        """
        Process historical data for a ticker.

        Args:
            hist_data (DataFrame): Historical price data
            ticker (str): Stock ticker symbol
            currency (str): Currency of the data
            exchange_rates (dict): Exchange rates for conversion

        Returns:
            DataFrame: Processed data in standard format
        """
        try:
            # Create copy to avoid modifying original
            processed = hist_data.copy()

            # Convert prices if needed
            if currency in self.config.CURRENCY_SYMBOLS:
                conversion_rates = pd.Series(
                    [
                        self.price_converter.get_exchange_rate(
                            exchange_rates[currency], date
                        )
                        for date in processed.index
                    ]
                )
                processed["Close"] *= conversion_rates

            # Format data into standard structure
            formatted = self._format_data(processed, ticker)

            return formatted, self.price_converter.debug_info

        except Exception as e:
            self.logger.error(f"Error processing {ticker} data: {str(e)}")
            return None, []

    def _format_data(self, data, ticker):
        """
        Format data into standard structure with ISO dates.

        Args:
            data (DataFrame): Raw data to format
            ticker (str): Stock ticker symbol

        Returns:
            DataFrame: Formatted data
        """
        try:
            # Select and rename columns
            formatted = data[["Close"]].copy()

            # Add ISO format dates
            formatted["Date"] = data.index.strftime(
                "%Y-%m-%d"
            )  # Already datetime index from yfinance
            formatted["ticker"] = ticker

            # Rename price column
            formatted = formatted.rename(columns={"Close": "price"})

            # Reorder columns and sort by date descending
            formatted = formatted[["Date", "price", "ticker"]]

            return formatted

        except Exception as e:
            self.logger.error(f"Error formatting data for {ticker}: {str(e)}")
            return None

    def validate_formatted_data(self, data):
        """
        Validate formatted data meets requirements.

        Args:
            data (DataFrame): Formatted data to validate

        Returns:
            bool: True if data is valid
        """
        if data is None or data.empty:
            return False

        required_columns = ["Date", "price", "ticker"]

        # Check all required columns exist
        if not all(col in data.columns for col in required_columns):
            return False

        # Check for missing values
        if data[required_columns].isnull().any().any():
            return False

        # Check price values are valid
        if (data["price"] <= 0).any():
            return False

        return True


##################################################
# Stock data management & Quicken interface      #
##################################################


class DataFetcher:
    """
    Manages fetching and caching of financial data.

    Features:
    - Data retrieval from Yahoo Finance
    - Cache management
    - Error handling
    - Progress tracking
    """

    def __init__(self, config=None, cache_enabled=True):
        """
        Initialize the data fetcher.

        Args:
            config: Configuration object containing settings
            cache_enabled (bool): Whether to use caching
        """
        self.config = config
        self.cache_enabled = cache_enabled
        self.cache_manager = CacheManager(
            config_path=self.config.PATH,
            max_age_hours=self.config.cache_settings["max_age_hours"],
            cleanup_threshold=self.config.cache_settings["cleanup_threshold"],
        )
        self.price_converter = PriceConverter(config=config)
        self.yahoo_fetcher = YahooFinanceFetcher(config=config)  # Pass `config`
        self.formatter = OutputFormatter()
        self.logger = StructuredLogger("DataFetcher", config=config)
        self.data_processor = DataProcessor(config=config)

    def fetch_ticker_data(self, ticker, start_date, end_date, exchange_rates=None):
        """
        Fetch and process ticker data.

        Args:
            ticker (str): Stock ticker symbol
            start_date (str): Start date for data
            end_date (str): End date for data
            exchange_rates (dict, optional): Exchange rates for conversion

        Returns:
            tuple: (Processed data, Status, Debug info, Currency)
        """
        debug_info = []
        is_exchange_rate = ticker in self.config.CURRENCY_SYMBOLS.values()

        try:
            cache_key_date = self._normalize_dates(start_date, end_date)
            self.logger.info(f"\n\nProcessing {ticker}\n===================")
            self.logger.info(f"Normalized dates: {cache_key_date}")
            cache_path = self.cache_manager.get_cache_path(
                ticker, cache_key_date["start"], cache_key_date["end"]
            )
            self.logger.info(
                f"Generated cache path:\n{self.logger.format_path(cache_path)}"
            )

            if self._should_use_cache(ticker):
                cached_result = self._try_cache(
                    ticker, cache_key_date["start"], cache_key_date["end"]
                )
                if cached_result:
                    self.logger.info(f"Cache hit for {ticker}")
                    data, curr = cached_result
                    self.logger.info(f"Cached data summary for {ticker}:")
                    self.logger.info(f"  Data shape: {data.shape}")
                    self.logger.info(f"  Currency: {curr}")
                    self.logger.info(
                        f"  Date range: {data['Date'].iloc[-1]} to {data['Date'].iloc[0]}"
                    )

                    # Generate debug info even for cached data if it's a debug ticker
                    if ticker in self.config.DEBUG_TICKERS:
                        # Convert Date column to datetime with explicit formatting
                        data["Date_dt"] = pd.to_datetime(
                            data["Date"], format="%d/%m/%Y"
                        )

                        # Sort by the datetime column to ensure proper ordering
                        data = data.sort_values("Date_dt", ascending=False)

                        # Get most recent date's data
                        latest_row = data.iloc[0]
                        latest_date = data["Date_dt"].iloc[0]
                        latest_price = latest_row["price"]

                        # Debug print to verify dates
                        self.logger.info(
                            f"Dates available for {ticker}:\n{data['Date'].values}"
                        )
                        self.logger.info(f"Selected latest date: {latest_date}")

                        self.price_converter.convert_price(
                            latest_price, curr, exchange_rates, latest_date, ticker
                        )
                        debug_info = self.price_converter.debug_info
                        self.price_converter.debug_info = []  # Clear for next use

                        # Clean up temporary column
                        data = data.drop("Date_dt", axis=1)

                    return data, "using cache", debug_info, curr

            self.logger.info(f"Cache miss - fetching fresh data for {ticker}")
            hist, currency = self.yahoo_fetcher.fetch_data(ticker, start_date, end_date)

            return self._process_fresh_data(
                hist, ticker, currency, exchange_rates, is_exchange_rate, cache_key_date
            )

        except Exception as e:
            self.logger.error(f"Error fetching {ticker}: {e}")
            return None, "failed", debug_info, "Unknown"

    def _should_use_cache(self, ticker):
        """Determine if cache should be used for ticker."""
        if not self.cache_enabled:
            self.logger.info(f"Cache disabled for {ticker}")
            return False

        # if ticker in self.config.DEBUG_TICKERS:
        #     self.logger.info(f"Skip cache for debug ticker: {ticker}")
        #     return False

        self.logger.info(f"Cache enabled for {ticker}")
        return True

    def _try_cache(self, ticker, start_date, end_date):
        """Try to load data from cache."""
        cache_path = self.cache_manager.get_cache_path(ticker, start_date, end_date)
        self.logger.info(f"READ attempt - Cache path:\n{cache_path}")

        if cache_path.exists():
            self.logger.info(f"Cache file exists for {ticker}")
            cached_data = self.cache_manager.load_from_cache(cache_path)

            if cached_data:
                if isinstance(cached_data, tuple) and len(cached_data) == 2:
                    self.logger.info(f"Valid cache data found for {ticker}")
                    data, curr = cached_data

                    # Fix the data structure
                    if isinstance(data.index, pd.DatetimeIndex):
                        data = data.copy()
                        data["Date"] = data.index.strftime("%d/%m/%Y")
                        data = data.reset_index(drop=True)

                    return data, curr
                else:
                    self.logger.warning(
                        f"Invalid cache format for {ticker}: {type(cached_data)}"
                    )
        else:
            self.logger.info(f"No cache file found for {ticker}")

        return None

    def _normalize_dates(self, start_date, end_date):
        """Normalize dates to ensure consistent cache keys."""
        start_norm = pd.Timestamp(start_date).strftime("%Y-%m-%d")
        end_norm = pd.Timestamp(end_date).strftime("%Y-%m-%d")
        self.logger.info(
            f"Normalized dates: {start_date} -> {start_norm}, {end_date} -> {end_norm}"
        )
        return {"start": start_norm, "end": end_norm}

    def _process_fresh_data(
        self, hist, ticker, currency, exchange_rates, is_exchange_rate, cache_dates
    ):
        """Process new data from Yahoo Finance."""
        processed_hist = hist.copy()
        self.logger.info(f"Processing fresh data for {ticker}")
        self.logger.info(f"Input data shape: {hist.shape}")

        # Get the latest date's data only for debug info
        if ticker in self.config.DEBUG_TICKERS:
            latest_date = processed_hist.index.max()
            latest_row = processed_hist.loc[latest_date]
            self.price_converter.convert_price(
                latest_row["Close"], currency, exchange_rates, latest_date, ticker
            )

        # Process all rows normally
        processed_hist["Close"] = processed_hist.apply(
            lambda row: self.price_converter.convert_price(
                row["Close"], currency, exchange_rates, row.name, ticker
            ),
            axis=1,
        )

        processed_data = self._format_output_data(processed_hist, ticker)

        if self.cache_enabled and not is_exchange_rate:
            cache_path = self.cache_manager.get_cache_path(
                ticker, cache_dates["start"], cache_dates["end"]
            )
            self.logger.info(f"WRITE attempt - Cache path: {cache_path}")
            self.logger.info(f"Saving data shape: {processed_data.shape}")
            cache_success = self.cache_manager.save_to_cache(
                cache_path, (processed_data, currency)
            )
            if cache_success:
                self.logger.info(f"Successfully cached {ticker} data")
            else:
                self.logger.warning(f"Failed to cache {ticker} data")

        debug_info = self.price_converter.debug_info
        self.price_converter.debug_info = []  # Clear for next ticker
        return processed_data, "new data", debug_info, currency

    def _format_output_data(self, hist, ticker):
        """Format the output data frame."""
        formatted = (
            hist[["Close"]]
            .assign(Date=hist.index.strftime("%d/%m/%Y"), ticker=ticker)
            .rename(columns={"Close": "price"})[["Date", "price", "ticker"]]
        )
        self.logger.info(f"Formatted output shape for {ticker}: {formatted.shape}")
        return formatted


class Results:

    def __init__(self, config=None):
        self.config = config
        self.all_data = []
        self.failed_tickers = []
        self.debug_info = []
        self.ticker_count = 0
        self.current_line = []
        self.logger = StructuredLogger("Results", config=config)


class StockDataManager:
    """
    Central manager for stock data operations and coordination.

    Features:
    - Overall process coordination
    - Exchange rate management
    - Data processing pipeline
    - Error handling and logging
    - Performance monitoring
    """

    def __init__(self, config=None):
        self.config = config
        self.formatter = OutputFormatter()
        self.data_fetcher = DataFetcher(config)
        self.exchange_rates = {}
        self.processor = DataProcessor(config)
        self.streaming_processor = StreamingTickerProcessor(config)
        self.debug_info = []
        self.price_converter = PriceConverter(config=config)
        self.logger = StructuredLogger("StockDataManager", config=config)

    def fetch_exchange_rates(self, start_date_iso, end_date_iso):
        """
        Fetch and process exchange rates for all required currencies.
        """
        self.formatter.print_section("\nExchange Rates")
        self.formatter.capture_print(
            f"Status: {Fore.GREEN}✓{Style.RESET_ALL} = New Data  "
            f"{Fore.BLUE}#{Style.RESET_ALL} = Cached  "
            f"{Fore.RED}✗{Style.RESET_ALL} = Failed"
        )

        if not hasattr(self, "exchange_rates"):
            self.exchange_rates = {}

        for currency, rate_ticker in self.config.CURRENCY_SYMBOLS.items():
            if not rate_ticker:
                continue

            try:

                success = self._fetch_single_exchange_rate(
                    currency, rate_ticker, start_date_iso, end_date_iso
                )
                if not success:
                    return False
            except Exception as e:
                logging.error(f"Failed to fetch {currency} exchange rate: {e}")
                return False

        # Add blank line at the end of exchange rates section
        self.formatter.capture_print("")
        return bool(self.exchange_rates)

    def _fetch_single_exchange_rate(
        self, currency, rate_ticker, start_date_iso, end_date_iso
    ):
        try:

            original_cache_enabled = self.data_fetcher.cache_enabled
            self.data_fetcher.cache_enabled = True

            today = datetime.now().strftime("%Y-%m-%d")
            if end_date_iso >= today:
                self.data_fetcher.cache_enabled = False

            rate_data, status, _, _ = self.data_fetcher.fetch_ticker_data(
                rate_ticker, start_date_iso, end_date_iso
            )

            if rate_data is not None:
                debug_info = self._process_and_print_rates(
                    rate_data, currency, rate_ticker
                )
                if debug_info:  # Store debug info if generated
                    self.debug_info.extend(
                        debug_info
                    )  # Need to add debug_info list to class
                return True
            else:
                raise ValueError(f"No data returned for {currency}")

        finally:
            self.data_fetcher.cache_enabled = original_cache_enabled

    def _process_rate_data(self, rate_data, currency, rate_ticker):
        try:
            self._process_and_print_rates(rate_data, currency, rate_ticker)

        except Exception as e:
            self.logger.error(f"Error processing rate data for {currency}: {str(e)}")
            raise

    def _process_and_print_rates(self, rate_data, currency, rate_ticker):
        """
        Process and print exchange rate data with consistent formatting.
        Shows status for each individual rate and generates debug info if needed.
        """
        try:
            dates = pd.to_datetime(rate_data["Date"], dayfirst=True)

            # Store rates with consistent date format
            self.exchange_rates[currency] = {
                date.normalize(): price
                for date, price in zip(dates, rate_data["price"])
            }

            last_three_dates = sorted(self.exchange_rates[currency].keys())[-3:]
            today = pd.Timestamp.now().normalize()

            self.formatter.capture_print(
                f"\n{currency} exchange rates (using {rate_ticker}):"
            )

            # Generate debug info if this is a debug ticker
            debug_info = []
            if rate_ticker in self.config.DEBUG_TICKERS:
                latest_date = last_three_dates[-1]
                latest_rate = self.exchange_rates[currency][latest_date]
                self.price_converter.add_debug_info(
                    rate_ticker,
                    latest_date,
                    latest_rate,
                    "GBP",  # Changed to GBP since these are rates against GBP
                    None,
                    None,
                )
                debug_info = self.price_converter.debug_info
                self.price_converter.debug_info = []

            # Print rates for last three days with status indicators
            for date in reversed(last_three_dates):
                rate = self.exchange_rates[currency][date]
                status_symbol = self.formatter.format_status(
                    "new data" if date == today else "using cache"
                )
                self.formatter.capture_print(
                    f"  {status_symbol} {date.strftime('%d/%m/%Y')} rate: {self.formatter.format_number(rate)}"
                )

            # Return debug info to be collected
            return debug_info

        except Exception as e:
            self.logger.error(f"Error processing rate data for {currency}: {str(e)}")
            raise

    def process_tickers(self, start_date_iso, end_date_iso):
        """
        Process all tickers and collect results.
        """
        self.formatter.print_section("Downloading Prices")
        self._print_status_legend()

        results = Results()

        # Process regular tickers
        for ticker in self._get_tickers_to_process():
            self._process_single_ticker(ticker, start_date_iso, end_date_iso, results)

            # If we've processed 3 tickers, print the current line
            if results.ticker_count % 3 == 0:
                if results.current_line:
                    self.formatter.capture_print(" ".join(results.current_line))
                results.current_line = []

        # Add exchange rate tickers to be processed
        exchange_rate_tickers = [
            rate_ticker
            for rate_ticker in self.config.CURRENCY_SYMBOLS.values()
            if rate_ticker is not None
        ]

        for ticker in exchange_rate_tickers:
            self._process_single_ticker(ticker, start_date_iso, end_date_iso, results)

            if results.current_line:  # Print any remaining tickers
                self.formatter.capture_print(" ".join(results.current_line))
            results.current_line = []

        # Print any remaining tickers in the last line
        if results.current_line:
            self.formatter.capture_print(" ".join(results.current_line))
        self.formatter.capture_print("")

        # Print debug info if available
        if results.debug_info:
            self._print_debug_info(results.debug_info)

        return results.all_data, results.failed_tickers

    def _print_status_legend(self):
        """Print legend explaining status symbols."""
        self.formatter.capture_print(
            f"Status: {Fore.GREEN}✓{Style.RESET_ALL} = New Data  "
            f"{Fore.BLUE}#{Style.RESET_ALL} = Cached  "
            f"{Fore.RED}✗{Style.RESET_ALL} = Failed"
        )
        self.formatter.capture_print("")

    def _process_all_tickers(self, start_date_iso, end_date_iso):
        """
        Process all tickers using streaming processor.

        Args:
            start_date_iso (str): Start date
            end_date_iso (str): End date

        Returns:
            Results object containing processed data and status
        """

    def _get_tickers_to_process(self):
        """
        Get list of tickers to process, excluding currency symbols.

        Returns:
            list: Tickers to process
        """
        return [
            t
            for t in self.config.TICKERS
            if t not in self.config.CURRENCY_SYMBOLS.values()
        ]

    def _process_single_ticker(self, ticker, start_date_iso, end_date_iso, results):
        """Process a single ticker and update results."""
        try:
            cache_path = self.data_fetcher.cache_manager.get_cache_path(
                ticker, start_date_iso, end_date_iso
            )
            # Format path for logging
            formatted_path = self.data_fetcher.cache_manager.get_formatted_path(
                cache_path
            )
            self.logger.info(f"Generated cache path: {formatted_path}")

            data, status, ticker_debug, currency = self.data_fetcher.fetch_ticker_data(
                ticker,
                start_date_iso,
                end_date_iso,
                exchange_rates=self.exchange_rates,
            )

            # Add debug info if this is a debug ticker
            if ticker in self.config.DEBUG_TICKERS and ticker_debug:
                self.logger.info(f"Adding debug info for {ticker}")
                results.debug_info.extend(ticker_debug)

            # Format and append status to current line
            results.current_line.append(
                self.formatter.format_ticker_status(ticker, status, currency)
            )
            results.ticker_count += 1

            # Store results
            if data is not None:
                results.all_data.append(data)
            else:
                results.failed_tickers.append(ticker)

        except requests.RequestException as e:
            self.logger.error(f"Network error processing {ticker}: {e}")
            results.failed_tickers.append(ticker)
        except ValueError as e:
            self.logger.error(f"Data validation error for {ticker}: {e}")
            results.failed_tickers.append(ticker)
        except Exception as e:
            self.logger.error(f"Unexpected error processing {ticker}: {e}")
            results.failed_tickers.append(ticker)

    def _print_debug_info(self, debug_info):
        """
        Print debug information if available.

        Args:
            debug_info (list): Debug information to print
        """
        if debug_info:
            self.formatter.print_section("Debugging")
            current_ticker = None
            # Add exchange rate debug info
            if self.debug_info:  # Class-level debug info from exchange rates
                debug_info.extend(self.debug_info)

            for line in debug_info:
                if line.startswith("Debug - "):
                    ticker = line.split("Debug - ")[1].strip(":")
                    if current_ticker and ticker != current_ticker:
                        self.formatter.capture_print("")
                    current_ticker = ticker
                if line:
                    self.formatter.capture_print(line)
            self.formatter.capture_print("")

    def _prepare_data_frame(self, all_data):
        """
        Prepare and format the final DataFrame, including exchange rates.
        """
        try:
            # Combine all data frames
            df = pd.concat(all_data, ignore_index=True)

            # Create exchange rate records
            exchange_rate_records = []
            for currency, rates_dict in self.exchange_rates.items():
                # Get the most recent rate
                latest_date = max(rates_dict.keys())
                rate = rates_dict[latest_date]

                # Create record for each exchange rate
                rate_ticker = self.config.CURRENCY_SYMBOLS.get(currency)
                if rate_ticker:
                    exchange_rate_records.append(
                        {
                            "ticker": rate_ticker,
                            "price": rate,
                            "Date": latest_date.strftime("%d/%m/%Y"),
                        }
                    )

            # Convert exchange rate records to DataFrame
            if exchange_rate_records:
                exchange_df = pd.DataFrame(exchange_rate_records)
                # Concatenate with main data
                df = pd.concat([df, exchange_df], ignore_index=True)

            # Reorder columns
            df = df[["ticker", "price", "Date"]]

            return df

        except Exception as e:
            self.logger.error(f"Error preparing data frame: {str(e)}")
            raise

    def save_data(self, all_data):
        """
        Save processed data to CSV file.

        Args:
            all_data (list): List of processed data frames

        Returns:
            bool: True if save was successful
        """
        if not all_data:
            self.formatter.capture_print("\nData was NOT fetched successfully!")
            return False

        try:
            df = self._prepare_data_frame(all_data)

            # Add debug logging
            self.logger.debug(f"DataFrame before saving:\n{df.head()}")
            self.logger.debug(
                f"Exchange rate tickers present:\n{df['ticker'].unique()}"
            )

            # Confirm that 'Date' column is recognized as a datetime object in the format required for quicken
            df["Date"] = pd.to_datetime(df["Date"], dayfirst=True).dt.strftime(
                "%d/%m/%Y"
            )

            df.to_csv(
                f"{self.config.PATH}{self.config.data_file_name}",
                index=False,
                header=False,
            )

            self.formatter.capture_print(
                f"Log file saved to {self.formatter.format_path(self.config.PATH + self.config.log_file_name)}"
                f"\nData saved to: {self.formatter.format_path(self.config.PATH + self.config.data_file_name)}"
                "\nTerminal output copied to clipboard"
            )
            return True

        except Exception as e:
            self.logger.error(f"Error saving data: {e}")
            return False


##################################################
# Quicken interface & main execution flow        #
##################################################


class QuickenInterface:
    """
    Manages interaction with Quicken software for data import.

    Features:
    - Automated GUI interaction
    - Safety measures for automation
    - Error handling
    - Status reporting
    """

    def __init__(self, config=None):
        """
        Initialize Quicken interface.

        Args:
            config: Configuration object containing Quicken settings
        """
        self.config = config
        self.formatter = OutputFormatter()
        self._setup_pyautogui()
        self.logger = StructuredLogger("QuickenInterface", config=config)

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
        self.formatter.capture_print(
            "Quicken cannot be opened from here.\n\nInstead:\n"
        )
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


def get_currency_conversion_counts(all_data, currency_symbols):
    """
    Calculate the number of conversions performed for each currency.

    Args:
        all_data (list): List of processed data frames
        currency_symbols (dict): Dictionary of currency symbols

    Returns:
        dict: Count of conversions per currency
    """
    # Initialize counters for currencies that need conversion
    currencies = [
        currency for currency, rate_ticker in currency_symbols.items() if rate_ticker
    ]
    conversion_counts = {currency: 0 for currency in currencies}
    converted_tickers = {currency: set() for currency in currencies}

    # Count conversions for each data frame
    for data in all_data:
        try:
            ticker = data["ticker"].iloc[0]
            if _should_count_conversion(ticker, currency_symbols):
                currency = _get_ticker_currency(ticker)
                _update_conversion_counts(
                    ticker, currency, currencies, conversion_counts, converted_tickers
                )
        except Exception as e:
            logging.warning(f"Error counting conversions for ticker: {e}")
            continue

    # Add GBp conversions if applicable
    if "GBp" in currency_symbols:
        conversion_counts["GBp"] = _count_gbp_conversions(all_data, currency_symbols)

    return conversion_counts


def _should_count_conversion(ticker, currency_symbols):
    """
    Determine if a ticker should be counted for conversion.

    Args:
        ticker (str): Stock ticker symbol
        currency_symbols (dict): Dictionary of currency symbols

    Returns:
        bool: True if ticker should be counted
    """
    return not (ticker.startswith("^") or ticker in currency_symbols.values())


def _get_ticker_currency(ticker):
    """
    Get the currency for a ticker.

    Args:
        ticker (str): Stock ticker symbol

    Returns:
        str: Currency code or "Unknown"
    """
    try:
        stock = yf.Ticker(ticker)
        return stock.info.get("currency", "Unknown")
    except Exception:
        return "Unknown"


def _update_conversion_counts(ticker, currency, currencies, counts, converted):
    """
    Update conversion counts for a ticker.

    Args:
        ticker (str): Stock ticker symbol
        currency (str): Currency code
        currencies (list): List of currencies to track
        counts (dict): Conversion count dictionary
        converted (dict): Dictionary of converted tickers per currency
    """
    if currency in currencies and ticker not in converted[currency]:
        counts[currency] += 1
        converted[currency].add(ticker)


def _count_gbp_conversions(all_data, currency_symbols):
    """
    Count the number of GBP conversions.

    Args:
        all_data (list): List of processed data frames
        currency_symbols (dict): Dictionary of currency symbols

    Returns:
        int: Number of GBP conversions
    """
    gbp_count = 0
    gbp_tickers = set()

    for data in all_data:
        try:
            ticker = data["ticker"].iloc[0]
            if _should_count_conversion(ticker, currency_symbols):
                currency = _get_ticker_currency(ticker)
                if currency == "GBp" and ticker not in gbp_tickers:
                    gbp_count += 1
                    gbp_tickers.add(ticker)
        except Exception:
            continue

    return gbp_count


def _print_summary(
    manager, all_data, failed_tickers, start_time, end_time=time.time(), config=None
):
    """
    Print execution summary statistics.

    Args:
        manager: StockDataManager instance
        config: Configuration object
        all_data: List of processed data
        failed_tickers: List of failed tickers
        start_time: Process start timestamp
        end_time: Process end timestamp
    """

    success_count = len(config.TICKERS) - len(failed_tickers)
    total_count = len(config.TICKERS)
    execution_time = end_time - start_time

    conversion_counts = get_currency_conversion_counts(
        all_data, config.CURRENCY_SYMBOLS
    )

    manager.formatter.capture_print(f"Tickers Processed: {success_count}")
    for currency, count in conversion_counts.items():
        if count > 0:
            manager.formatter.capture_print(
                f"Tickers adjusted from {currency} to GBP: {count}"
            )

    manager.formatter.capture_print(
        f"\nSuccess Rate: {(success_count/total_count)*100:.1f}%"
    )
    manager.formatter.capture_print(f"Execution Time: {execution_time:.2f}s")


def _handle_quicken_import(manager, config=None):
    """
    Handle the Quicken import process.
    """
    quicken = QuickenInterface(config)
    return quicken.handle_import()


def copy_to_clipboard(text):
    """Copy clean text to system clipboard."""
    try:
        import pyperclip

        clean_text = (
            text.get_clean_output() if hasattr(text, "get_clean_output") else text
        )
        pyperclip.copy(clean_text)
    except ImportError:
        print("pyperclip not installed. Cannot copy to clipboard.")


def main():
    """
    Main program execution flow.
    Coordinates all components and handles the complete process.

    Returns:
        bool: True if entire process completed successfully
    """
    try:
        # Initialize components

        # global config # Makes config a global variable
        config = Config()
        manager = StockDataManager(config)
        date_manager = DateManager(config)
        start_time = time.time()

        # Setup dates
        dates = date_manager.setup_dates(config.PERIOD)
        business_days = date_manager.calculate_business_days(
            dates["start_iso"], dates["end_iso"]
        )

        # Clear screen for clean output
        os.system("cls" if os.name == "nt" else "clear")

        # Print header
        manager.formatter.print_section("Stock Price Update", major=True)
        manager.formatter.capture_print(
            f"Range {datetime.fromisoformat(dates['start_iso']).strftime('%d/%m/%Y')} - "
            f"{datetime.fromisoformat(dates['end_iso']).strftime('%d/%m/%Y')}:\n"
            f"{business_days} business days"
        )

        # Fetch exchange rates
        if not manager.fetch_exchange_rates(dates["start_iso"], dates["end_iso"]):
            return False

        # Process tickers
        all_data, failed_tickers = manager.process_tickers(
            dates["start_iso"], dates["end_iso"]
        )

        # Print summary
        manager.formatter.print_section("Summary")
        _print_summary(
            manager,
            all_data,
            failed_tickers,
            start_time,
            time.time(),
            config,
        )

        # Save and import data
        if manager.save_data(all_data):
            # Get clean output and copy to clipboard
            copy_to_clipboard(manager.formatter)
            return _handle_quicken_import(manager, config)

        return False

    except Exception as e:
        # Log the error using StructuredLogger
        main_logger = StructuredLogger("Main", config=config)
        main_logger.error(f"An error occurred: {e}")
        return False


if __name__ == "__main__":
    try:
        success = main()
        sys.exit(0)
    except Exception as e:
        sys.exit(1)

##################################################
# End                                            #
##################################################
