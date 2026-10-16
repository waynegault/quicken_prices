#!/usr/bin/env python3

"""
Fetch market prices, write a Quicken CSV, and optionally automate import.
"""

import logging
import os
import sys
import tempfile
from logging.handlers import RotatingFileHandler
from pathlib import Path

try:
    import pandas as pd
    import yaml
    import yfinance as yf

    from market_data import (
        fetch_historical_data,
        get_date_range,
        get_tickers,
    )
    from market_models import TickerMetadata
    from price_processing import convert_prices, process_converted_prices
    from quicken_automation import (
        ensure_project_python,
        quicken_import,
        run_as_admin,
        validate_quicken_path,
    )
    from settings import AppSettings
    from storage import (
        clean_cache,
        load_cache,
        save_cache,
        save_to_csv,
    )
except ModuleNotFoundError as missing:
    # By far the most likely cause: the file was launched by double-click or by
    # a shortcut, so Windows ran it with the machine's default Python instead of
    # the repository venv - the only interpreter that has these packages. Say
    # which interpreter failed and how to run it properly, rather than leaving a
    # bare import traceback that says nothing about the venv.
    print(
        f"'{missing.name}' is not installed for this interpreter:\n"
        f"  {sys.executable}\n"
        "\nRun the script through the repository environment instead:\n"
        "  quicken_prices.bat\n"
        "  .\\.venv\\Scripts\\python.exe quicken_prices.py\n"
    )
    raise SystemExit(2) from missing


def load_configuration(config_file: str = "configuration.yaml") -> AppSettings:
    """Load strict YAML settings, filling only model-defined defaults."""
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
            "Quicken executable not found. CSV generation will continue, "
            "but GUI import automation is unavailable."
        )

    resolved_settings = settings.model_copy(
        update={
            "paths": settings.paths.model_copy(
                update={"quicken": quicken_path or ""}
            )
        }
    )
    return AppSettings.model_validate(resolved_settings.model_dump())


def setup_logging(config: AppSettings) -> None:
    """Configure rotating file and console logging."""
    base_path = Path(config.paths.base)
    file_log_level = config.logging_level.file
    terminal_log_level = config.logging_level.terminal

    log_file_path = base_path / config.paths.log_file
    log_file_path.parent.mkdir(parents=True, exist_ok=True)

    file_handler = RotatingFileHandler(
        log_file_path,
        maxBytes=config.log_file.max_bytes,
        backupCount=config.log_file.backup_count,
        encoding="utf-8",
    )
    file_handler.setLevel(getattr(logging, file_log_level))
    formatter = logging.Formatter(
        "%(asctime)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s"
    )
    file_handler.setFormatter(formatter)

    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(getattr(logging, terminal_log_level))
    console_handler.setFormatter(formatter)

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


def _bootstrap_logging() -> None:
    """Log to the console until the configured handlers are installed.

    main() checks the interpreter and loads the configuration before it can
    point the log file anywhere, so a failure in either step used to reach only
    the console - and the log, which is where anyone looks afterwards, stayed
    empty. This is deliberately replaced by setup_logging, which clears the
    root handlers before installing the configured ones.
    """
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s",
    )


def main() -> None:
    """Run the validated market-data, conversion, export, and import flow."""
    exit_prompt = "\n\t Press Enter to exit...\n\t"
    should_pause_before_exit = True

    try:
        _bootstrap_logging()
        ensure_project_python()
        config = load_configuration("configuration.yaml")

        os.system("cls" if os.name == "nt" else "clear")
        print("Starting script\n")
        print("sys.executable", sys.executable)
        print("yfinance version:", yf.__version__, "\n")

        setup_logging(config)
        run_startup_self_check(config)

        if run_as_admin():
            should_pause_before_exit = False
            return

        home_currency = config.home_currency
        valid_tickers: list[TickerMetadata] = get_tickers(
            config.tickers,
            home_currency=home_currency,
            retry_settings=config.retries,
        )

        start_date, end_date = get_date_range(config)
        cache_dir = Path(config.paths.base) / config.paths.cache
        price_data = fetch_historical_data(
            valid_tickers,
            start_date,
            end_date,
            cache_dir,
            retry_settings=config.retries,
        )

        if price_data.empty:
            raise ValueError("No valid price data fetched.")

        processed_data = convert_prices(price_data, home_currency=home_currency)
        output_csv = process_converted_prices(processed_data, start_date)
        save_to_csv(output_csv, config)
        clean_cache(config)
        if quicken_import(config):
            logging.info("Price data exported and imported successfully.")
        else:
            logging.info("Price data exported; Quicken GUI import was not performed.")

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
