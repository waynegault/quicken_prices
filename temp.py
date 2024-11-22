import yaml
import logging
from logging.handlers import RotatingFileHandler
from functools import wraps
import time
from typing import Any


# Logging Decorator
def log_function(logger: logging.Logger = None):
    """
    A decorator to log function entry, exit, and execution details.
    """

    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            start_time = time.time()
            if logger:
                logger.debug(
                    f"Entering: {func.__name__} with args: {args}, kwargs: {kwargs}"
                )
            result = func(*args, **kwargs)
            elapsed_time = time.time() - start_time
            if logger:
                logger.debug(
                    f"Exiting: {func.__name__}, took {elapsed_time:.4f}s, result: {result}"
                )
            return result

        return wrapper

    return decorator


# Retry Decorator
def enhanced_retry(max_retries: int = 3, delay: int = 2, logger: logging.Logger = None):
    """
    A decorator to retry a function on failure with a delay.
    Logs each retry attempt and eventual success or failure.
    """

    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            attempts = 0
            last_exception = None
            while attempts < max_retries:
                try:
                    result = func(*args, **kwargs)
                    if logger:
                        logger.debug(
                            f"Success on attempt {attempts + 1} for {func.__name__}"
                        )
                    return result
                except Exception as e:
                    attempts += 1
                    last_exception = e
                    if logger:
                        logger.warning(
                            f"Attempt {attempts} failed for {func.__name__} with error: {e}",
                            exc_info=True,
                        )
                    if attempts < max_retries:
                        time.sleep(delay)
            if logger:
                logger.error(
                    f"All {max_retries} retries failed for {func.__name__}. Last error: {last_exception}",
                    exc_info=True,
                )
            raise last_exception  # Raise the last exception explicitly

        return wrapper

    return decorator


# Centralized Logger Class
class CentralizedLogger:
    """
    Centralized logging class to manage logging configurations dynamically.
    Extracts settings from YAML and ensures consistency across all classes.
    """

    def __init__(self, config: dict):
        self.logger = logging.getLogger("ApplicationLogger")
        self.logger.setLevel(logging.DEBUG)  # Set to capture all logs

        # Extract logging configuration from YAML or use defaults
        log_file = config.get("paths", {}).get("log_file", "application.log")
        max_bytes = config.get("logging", {}).get("max_bytes", 5 * 1024 * 1024)
        backup_count = config.get("logging", {}).get("backup_count", 5)
        terminal_level = (
            config.get("logging", {}).get("levels", {}).get("terminal", "INFO").upper()
        )
        file_level = (
            config.get("logging", {}).get("levels", {}).get("file", "DEBUG").upper()
        )

        # File Handler with Rotation
        self.file_handler = RotatingFileHandler(
            log_file, maxBytes=max_bytes, backupCount=backup_count
        )
        file_formatter = logging.Formatter(
            "%(asctime)s - %(levelname)s - %(funcName)s:%(lineno)d - %(message)s",
            "%Y-%m-%dT%H:%M:%S",
        )
        self.file_handler.setFormatter(file_formatter)
        self.file_handler.setLevel(getattr(logging, file_level))
        self.logger.addHandler(self.file_handler)

        # Console Handler
        self.console_handler = logging.StreamHandler()
        console_formatter = logging.Formatter(
            "%(asctime)s - %(levelname)s - %(funcName)s:%(lineno)d - %(message)s"
        )
        self.console_handler.setFormatter(console_formatter)
        self.console_handler.setLevel(getattr(logging, terminal_level))
        self.logger.addHandler(self.console_handler)

    def update_log_levels(self, terminal_level: str, file_level: str):
        """
        Updates the logging levels for terminal and file handlers dynamically.
        """
        self.console_handler.setLevel(getattr(logging, terminal_level.upper()))
        self.file_handler.setLevel(getattr(logging, file_level.upper()))
        self.logger.info(
            f"Updated logging levels: Terminal={terminal_level}, File={file_level}"
        )


# ConfigManager Class
class ConfigManager:
    """
    Manages application configuration with centralized logging, error handling, and retry logic.
    """

    def __init__(self, config_path: str, logger: CentralizedLogger):
        self.config_path = config_path
        self.config_data = None
        self.logger = logger

    @log_function(logger=None)  # Log function start/stop
    @enhanced_retry(max_retries=3, delay=2, logger=None)  # Retry on transient errors
    def load_config(self) -> None:
        """
        Loads the YAML configuration file and validates its contents.
        Updates the logger levels based on the configuration.
        """
        self.logger.logger.info(
            f"Attempting to load configuration from: {self.config_path}"
        )
        try:
            with open(self.config_path, "r") as file:
                self.config_data = yaml.safe_load(file)
            self.logger.logger.info("Configuration successfully loaded.")
            self.validate_config()

            # Update logger levels dynamically based on the YAML config
            logging_config = self.config_data.get("logging", {}).get("levels", {})
            terminal_level = logging_config.get("terminal", "INFO")
            file_level = logging_config.get("file", "DEBUG")
            self.logger.update_log_levels(terminal_level, file_level)

        except FileNotFoundError:
            self.logger.logger.error(
                f"Configuration file not found at {self.config_path}", exc_info=True
            )
            raise
        except yaml.YAMLError as e:
            self.logger.logger.error(f"Error parsing YAML file: {e}", exc_info=True)
            raise ValueError(f"Invalid YAML syntax in configuration file: {e}")

    @log_function(logger=None)  # Log function start/stop
    def validate_config(self) -> None:
        """
        Validates the essential sections of the configuration file.
        Raises errors if critical fields are missing or invalid.
        """
        self.logger.logger.debug("Starting validation of the configuration file.")
        required_sections = ["tickers", "paths", "collection", "logging"]
        for section in required_sections:
            if section not in self.config_data:
                self.logger.logger.error(
                    f"Missing required section in config: {section}"
                )
                raise ValueError(f"Missing required section in config: {section}")

        # Validate paths
        paths = self.config_data.get("paths", {})
        required_paths = ["base", "quicken", "data_file", "log_file", "cache"]
        for path in required_paths:
            if not paths.get(path):
                self.logger.logger.error(f"Missing required path: {path}")
                raise ValueError(f"Missing required path: {path}")

        # Validate tickers
        if not self.config_data.get("tickers", []):
            self.logger.logger.error("Ticker list cannot be empty.")
            raise ValueError("Ticker list cannot be empty.")

        # Validate logging configuration
        logging_config = self.config_data.get("logging", {})
        if (
            "levels" not in logging_config
            or not logging_config["levels"].get("file")
            or not logging_config["levels"].get("terminal")
        ):
            self.logger.logger.error(
                "Logging levels for 'file' and 'terminal' must be specified."
            )
            raise ValueError(
                "Logging levels for 'file' and 'terminal' must be specified."
            )

        self.logger.logger.debug(
            "Configuration file validation completed successfully."
        )
 
 
# Initialize CentralizedLogger from YAML
try:
    with open("config.yaml", "r") as file:
        config_yaml = yaml.safe_load(file)
    central_logger = CentralizedLogger(config_yaml)
except Exception as e:
    print(f"Failed to initialize logger: {e}")
    raise

# Reinitialize ConfigManager with CentralizedLogger
config_manager = ConfigManager(
    config_path="config.yaml", logger=central_logger
)

try:
    config_manager.load_config()
except Exception as e:
    central_logger.logger.error(f"Configuration loading failed: {e}")
