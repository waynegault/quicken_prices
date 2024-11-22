import logging
import yfinance as yf
import yaml

# ============================ Utility Functions ============================ #

def setup_logger(log_file):
    """
    Sets up a centralized custom logger for the application.
    Args:
        log_file (str): The log file path.
    Returns:
        logging.Logger: Configured logger instance.
    """
    logger = logging.getLogger("custom_logger")
    logger.setLevel(logging.DEBUG)

    # Create handlers
    file_handler = logging.FileHandler(log_file)
    terminal_handler = logging.StreamHandler()

    # Set log levels
    file_handler.setLevel(logging.DEBUG)
    terminal_handler.setLevel(logging.INFO)

    # Create formatters
    formatter = logging.Formatter(
        "%(asctime)s - %(levelname)s - %(message)s", datefmt="%Y-%m-%d %H:%M:%S"
    )
    file_handler.setFormatter(formatter)
    terminal_handler.setFormatter(formatter)

    # Add handlers to the logger
    logger.addHandler(file_handler)
    logger.addHandler(terminal_handler)

    return logger

# ============================ Core Classes ============================ #

class ConfigLoader:
    """Handles loading and validation of configuration data."""

    @staticmethod
    def load_config(config_file):
        """
        Loads configuration from a YAML file.

        Args:
            config_file (str): Path to the YAML configuration file.
        Returns:
            dict: Configuration dictionary.
        """
        try:
            custom_logger.info(f"Attempting to load configuration from: {config_file}")
            with open(config_file, "r") as file:
                config = yaml.safe_load(file)
            custom_logger.info("Configuration successfully loaded.\n")
            return config
        except Exception as e:
            custom_logger.error(f"Failed to load configuration: {str(e)}")
            raise

class TickerValidator:
    """Validates tickers and identifies their type."""

    def __init__(self, tickers, default_period="5d"):
        self.tickers = tickers
        self.default_period = default_period
        self.valid_tickers = []
        self.invalid_tickers = []
        self.managed_funds = []
        self.error_tickers = []

    def validate_tickers(self):
        """Validates tickers and categorizes them."""
        custom_logger.info("Starting ticker validation...\n")
        for ticker in self.tickers:
            custom_logger.info(f"Processing ticker: {ticker}")
            try:
                ticker_obj = yf.Ticker(ticker)
                data = ticker_obj.history(period=self.default_period)

                if data.empty:
                    raise ValueError(f"No data found for ticker: {ticker}")

                # Check for missing 'Close' values
                if data['Close'].isna().sum() > 0:
                    data = data.dropna(subset=['Close'])

                # Identify managed funds
                info = ticker_obj.info
                if "totalAssets" in info:
                    self.managed_funds.append(ticker)
                    custom_logger.info(f"Ticker {ticker} identified as a Managed Fund.\n")
                else:
                    self.valid_tickers.append(ticker)
                    custom_logger.info(f"Validation successful for ticker: {ticker}.\n")

            except Exception as e:
                if "Period '5d' is invalid" in str(e):
                    self.managed_funds.append(ticker)
                    custom_logger.warning(f"Ticker {ticker} is a Managed Fund; using '1mo'.\n")
                elif "No data found" in str(e):
                    self.invalid_tickers.append(ticker)
                    custom_logger.error(f"Validation failed for ticker {ticker}: {str(e)}")
                else:
                    self.error_tickers.append(ticker)
                    custom_logger.error(f"Error processing ticker {ticker}: {str(e)}")

        self._log_summary()

    def _log_summary(self):
        """Logs the validation summary."""
        custom_logger.info("\n=== Validation Summary ===")
        custom_logger.info(f"Valid tickers ({len(self.valid_tickers)}): {', '.join(self.valid_tickers)}")
        custom_logger.info(f"Managed funds ({len(self.managed_funds)}): {', '.join(self.managed_funds)}")
        custom_logger.info(f"Invalid tickers ({len(self.invalid_tickers)}): {', '.join(self.invalid_tickers)}")
        custom_logger.info(f"Error tickers ({len(self.error_tickers)}): {', '.join(self.error_tickers)}")
        if self.invalid_tickers or self.error_tickers:
            custom_logger.warning("Please review invalid or error tickers for possible typos or delisted symbols.\n")


class DataCollector:
    """Collects data for validated tickers."""

    def __init__(self, valid_tickers, managed_funds):
        self.valid_tickers = valid_tickers
        self.managed_funds = managed_funds
        self.failed_tickers = []

    def collect_data(self):
        """Collects price data for valid tickers."""
        custom_logger.info("Proceeding with data collection...\n")
        all_tickers = self.valid_tickers + self.managed_funds
        for ticker in all_tickers:
            try:
                period = "1mo" if ticker in self.managed_funds else "5d"
                data = yf.Ticker(ticker).history(period=period)

                if data.empty:
                    raise ValueError(f"No data found for ticker: {ticker}")

                custom_logger.debug(f"Data successfully collected for ticker: {ticker}.")
            except Exception as e:
                self.failed_tickers.append(ticker)
                custom_logger.error(f"Data collection failed for ticker {ticker}: {str(e)}")

        self._log_results(all_tickers)

    def _log_results(self, all_tickers):
        """Logs results of the data collection."""
        successful_tickers = set(all_tickers) - set(self.failed_tickers)
        if successful_tickers:
            custom_logger.info(f"Data successfully collected for: {', '.join(successful_tickers)}.\n")
        if self.failed_tickers:
            custom_logger.warning(f"Data not collected for: {', '.join(self.failed_tickers)}.\n")

# ============================ Main Execution ============================ #

if __name__ == "__main__":
    try:
        # Load configuration
        config_file = "config.yaml"
        config = ConfigLoader.load_config(config_file)

        # Setup logger with config values
        log_file = config.get("paths", {}).get("log_file", "app.log")
        custom_logger = setup_logger(log_file)

        # Extract tickers from the configuration
        tickers = config.get("tickers", [])
        if not tickers:
            custom_logger.error("No tickers found in configuration. Exiting.")
            exit(1)

        # Validate tickers
        validator = TickerValidator(tickers, default_period="5d")
        validator.validate_tickers()

        # Collect data for valid tickers
        collector = DataCollector(validator.valid_tickers, validator.managed_funds)
        collector.collect_data()

    except Exception as e:
        custom_logger.critical(f"An unexpected error occurred: {str(e)}")
        exit(1)
