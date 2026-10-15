#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Shebang and coding declaration.

"""
quicken_prices.py

This script fetches historical price data for a list of tickers specified in 'configuration.yaml' and saves the data to a CSV file.

It includes features such as configuration management, logging, error handling, data fetching, data processing, currency conversion, and more.

Wayne Gault
23/11/2024

Updated 17/7/25 with revised error handling and logging improvements and concurrent processing for faster ticker validation.

MIT Licence
"""

# Imports #####################################

# Standard Library Imports
import logging
import os
import sys
import tempfile
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import List

# Third-Party Data and Scientific Computing Imports
import pandas as pd
import yaml
import yfinance as yf

from settings import AppSettings
from market_models import TickerMetadata
from market_data import (
    fetch_historical_data,
    get_date_range,
    get_tickers,
)

from price_processing import convert_prices, process_converted_prices
from storage import (
    clean_cache,
    load_cache,
    save_cache,
    save_to_csv,
)
from quicken_automation import (
    ensure_project_python,
    quicken_import,
    run_as_admin,
    setup_pyautogui,
    validate_quicken_path,
)

# GUI and Automation Imports


# Configuration ###############################


def load_configuration(config_file: str = "configuration.yaml") -> AppSettings:
    """
    Load strict YAML settings, filling only model-defined defaults.
    """

    base_path = Path(__file__).resolve().parent
    config_path = base_path / config_file

    if not config_path.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")

    logging.info(f"Using configuration file: {config_path}")
    with config_path.open("r", encoding="utf-8") as config_stream:
        user_config = yaml.safe_load(config_stream)
    if not isinstance(user_config, dict):
        raise ValueError(f"Configuration in {config_path} must be a YAML mapping.")

    raw_paths = user_config.get("paths", {})
    if not isinstance(raw_paths, dict):
        raise ValueError("Configuration 'paths' must be a mapping.")
    user_config["paths"] = {
        **raw_paths,
        "base": raw_paths.get("base", str(base_path)),
    }
    settings = AppSettings.model_validate(user_config)

    # Validate Quicken path, but honour an explicit configuration override.
    configured_quicken_path = settings.paths.quicken
    if configured_quicken_path:
        configured_quicken_path = os.path.expanduser(
            os.path.expandvars(configured_quicken_path)
        )

    if configured_quicken_path:
        if not Path(configured_quicken_path).is_file():
            raise FileNotFoundError(
                f"Configured Quicken executable was not found: {configured_quicken_path}"
            )
        quicken_path = configured_quicken_path
        logging.info(f"Using Quicken path from configuration: {quicken_path}")
    else:
        quicken_path = validate_quicken_path()

    if quicken_path:
        logging.info(f"Using Quicken at: {quicken_path}")
    else:
        logging.warning(
            "Quicken executable not found. CSV generation will continue, but GUI import automation is unavailable."
        )

    resolved_settings = settings.model_copy(
        update={
            "paths": settings.paths.model_copy(
                update={"quicken": quicken_path or ""}
            )
        }
    )
    return AppSettings.model_validate(resolved_settings.model_dump())








# Note: yfinance now handles its own session internally, no custom session needed

# Coding utilities ############################




def setup_logging(config: AppSettings) -> None:
    """
    Sets up logging configuration.
    """

    base_path = Path(config.paths.base)
    file_log_level = config.logging_level.file
    terminal_log_level = config.logging_level.terminal

    log_file_path = base_path / config.paths.log_file
    log_file_path.parent.mkdir(parents=True, exist_ok=True)

    # Create a rotating file handler with a simpler format for readability
    file_handler = RotatingFileHandler(
        log_file_path,
        maxBytes=config.log_file.max_bytes,
        backupCount=config.log_file.backup_count,
        encoding="utf-8",
    )
    file_handler.setLevel(getattr(logging, file_log_level))
    file_formatter = logging.Formatter(
        "%(asctime)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s"
    )
    file_handler.setFormatter(file_formatter)

    # Create a console handler with the same format
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(getattr(logging, terminal_log_level))
    console_formatter = logging.Formatter(
        "%(asctime)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s"
    )
    console_handler.setFormatter(console_formatter)

    logging.getLogger().handlers = []
    logging.getLogger().addHandler(file_handler)
    logging.getLogger().addHandler(console_handler)
    logging.getLogger().setLevel(logging.DEBUG)

    logging.info(f"Log at: {log_file_path}")


def run_startup_self_check(config: AppSettings) -> None:
    """
    Run deterministic date and cache checks without touching user data.
    """

    logging.info("Running startup self-check...")
    start_date, end_date = get_date_range(config)
    if start_date >= end_date:
        raise ValueError(
            f"Startup self-check failed: invalid date range ({start_date} >= {end_date})."
        )

    with tempfile.TemporaryDirectory() as temp_cache_dir:
        sample_data = pd.DataFrame(
            [
                {
                    "Ticker": "SELFTEST",
                    "Old Price": 1.0,
                    "Date": pd.Timestamp("2026-01-01", tz="UTC"),
                },
                {
                    "Ticker": "SELFTEST",
                    "Old Price": 1.1,
                    "Date": pd.Timestamp("2026-01-02", tz="UTC"),
                },
            ]
        )
        save_cache("SELFTEST", sample_data, temp_cache_dir)
        loaded_data, status = load_cache("SELFTEST", temp_cache_dir)
        if status != "loaded" or loaded_data.empty:
            raise RuntimeError(
                f"Startup self-check failed: cache round-trip returned status={status}."
            )

    logging.info("Startup self-check passed.")


# Initialisation ##############################









# Helper functions ############################














# Validate tickers ############################






# Get data ####################################


















# Process Data ################################








# Quicken GUI #################################




















# Main ########################################


def main():
    """
    Main script execution.
    """
    exit_prompt = "\n\t Press Enter to exit...\n\t"
    should_pause_before_exit = True

    try:
        ensure_project_python()
        config = load_configuration("configuration.yaml")

        # 0) Initialise
        os.system("cls" if os.name == "nt" else "clear")
        print("Starting script\n")
        print("sys.executable", sys.executable)
        print("yfinance version:", yf.__version__, "\n")

        # 1) Configure logging from YAML/ini settings
        setup_logging(config=config)

        # 2) Quick startup checks to catch obvious regressions before network/UI work
        run_startup_self_check(config)

        # 3) Attempt to elevate privileges (if needed on Windows)
        if run_as_admin():
            should_pause_before_exit = False
            return

        # 4) Read ticker config from your YAML
        tickers = config.tickers

        # 5) Validate and acquire ticker metadata
        home_currency = config.home_currency
        valid_tickers: List[TickerMetadata] = get_tickers(
            tickers,
            home_currency=home_currency,
            retry_settings=config.retries.model_dump(),
        )

        # 6) Fetch historical data

        # Retrieve date range and cache_dir from config
        start_date, end_date = get_date_range(config)
        cache_dir = Path(config.paths.base) / config.paths.cache
        price_data = fetch_historical_data(
            valid_tickers, start_date, end_date, cache_dir
        )

        if price_data.empty:
            raise ValueError("No valid price data fetched.")

        # 7) Convert prices to the home currency
        processed_data = convert_prices(price_data, home_currency=home_currency)

        # 8) Build the final output (e.g., pivoting, filtering, etc.)
        output_csv = process_converted_prices(processed_data, start_date)

        # 9) Save to CSV or your preferred data store
        save_to_csv(output_csv, config=config)

        # 10) Clean up the cache if desired
        clean_cache(config=config)

        # 11) Optionally automate an import sequence
        setup_pyautogui()
        quicken_import(config=config)

        # End
        logging.info("Script completed successfully")

    except KeyboardInterrupt:
        logging.warning("Script interrupted by user.")
        raise
    except Exception:
        logging.exception("An unexpected error occurred.")
        raise
    finally:
        if should_pause_before_exit:
            pause_before_exit(exit_prompt)

def pause_before_exit(prompt: str = "\n\t Press Enter to exit...\n\t") -> None:
    """Keep the terminal open so the user can review the output."""
    try:
        input(prompt)
    except (EOFError, KeyboardInterrupt):
        logging.debug("No interactive terminal available for exit prompt.")


if __name__ == "__main__":

    main()
