"""Atomic CSV storage and cache operations."""

import logging
import tempfile
import time
from pathlib import Path
from typing import Tuple

import pandas as pd

from common import format_path, pluralise
from settings import AppSettings

SECONDS_IN_A_DAY = 86400
CACHE_COLUMNS = ("Ticker", "Old Price", "Date")


def normalize_cache_data(ticker: str, data: pd.DataFrame) -> pd.DataFrame:
    """Normalize one ticker's rows identically for load, merge, and save."""
    missing_columns = set(CACHE_COLUMNS).difference(data.columns)
    if missing_columns:
        raise ValueError(
            f"{ticker}: Price data is missing columns {sorted(missing_columns)}."
        )

    normalized = data.loc[:, CACHE_COLUMNS].copy()
    normalized = normalized.dropna(subset=["Ticker"])
    normalized["Ticker"] = normalized["Ticker"].astype(str).str.upper()
    normalized["Date"] = pd.to_datetime(normalized["Date"], errors="coerce", utc=True)
    normalized["Old Price"] = pd.to_numeric(normalized["Old Price"], errors="coerce")
    normalized = (
        normalized.dropna(subset=["Date", "Old Price"])
        .drop_duplicates(subset=["Date"], keep="last")
        .sort_values("Date")
        .reset_index(drop=True)
    )
    cached_tickers = set(normalized["Ticker"])
    if cached_tickers and cached_tickers != {ticker.upper()}:
        raise ValueError(
            f"{ticker}: Price data contains unexpected tickers "
            f"{sorted(cached_tickers)}."
        )
    return normalized


def load_cache(ticker: str, cache_dir: str) -> Tuple[pd.DataFrame, str]:
    """
    Loads schema-checked CSV data for a ticker cache.

    Args:
        ticker (str): The ticker symbol.
        cache_dir (str): Path to the cache directory.

    Returns:
        Tuple[pd.DataFrame, str]: A tuple containing:
            - A DataFrame with cached data, or an empty frame when not found.
            - A status of "not_found" or "loaded".
    """
    cache_file = Path(cache_dir) / f"{ticker}.csv"

    if not cache_file.exists():
        return pd.DataFrame(columns=CACHE_COLUMNS), "not_found"

    if cache_file.stat().st_size == 0:
        raise ValueError(f"{ticker}: Cache file is empty: {cache_file}")

    data = normalize_cache_data(ticker, pd.read_csv(cache_file))
    return data, "loaded"


def write_csv_atomically(
    data: pd.DataFrame, destination: Path, *, index: bool, header: bool
) -> None:
    """Write a CSV beside its destination and replace it only after success."""
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        newline="",
        prefix=f".{destination.name}.",
        suffix=".tmp",
        dir=destination.parent,
        delete=False,
    ) as temporary_file:
        temporary_path = Path(temporary_file.name)

    try:
        data.to_csv(temporary_path, index=index, header=header)
        temporary_path.replace(destination)
    except BaseException:
        temporary_path.unlink(missing_ok=True)
        raise


def save_cache(ticker: str, data: pd.DataFrame, cache_dir: str) -> None:
    """
    Save a cache as an atomic CSV replacement.

    Args:
        ticker (str): The ticker symbol.
        data (pd.DataFrame): DataFrame containing data to cache.
    """
    cache_path = Path(cache_dir)
    cache_file = cache_path / f"{ticker}.csv"

    missing_columns = set(CACHE_COLUMNS).difference(data.columns)
    if missing_columns:
        raise ValueError(
            f"Cannot save cache for {ticker}; missing columns "
            f"{sorted(missing_columns)}."
        )

    cache_data = normalize_cache_data(ticker, data)
    if cache_data.empty:
        raise ValueError(f"Cannot save an empty cache for {ticker}.")
    write_csv_atomically(cache_data, cache_file, index=False, header=True)

    logging.info(
        f"{ticker}: Cache saved from {cache_data['Date'].min().strftime('%d/%m/%Y')} "
        f"to {cache_data['Date'].max().strftime('%d/%m/%Y')} "
        f"({len(cache_data)} days)."
    )


def clean_cache(config: AppSettings) -> None:
    """
    Cleans up cache files older than the configured maximum age.

    Args:
        config (dict): Configuration dictionary containing cache path and max age settings.

    """

    cache_dir = Path(config.paths.base) / config.paths.cache
    max_age_seconds = config.cache.max_age_days * SECONDS_IN_A_DAY
    now = time.time()

    if not cache_dir.is_dir():
        logging.info(f"Cache cleaning: no cache directory at {cache_dir}")
        return

    deleted_files = 0
    for entry in cache_dir.iterdir():
        if entry.is_file() and (now - entry.stat().st_mtime) > max_age_seconds:
            entry.unlink()
            deleted_files += 1
            logging.info(
                f"Cache cleaning: {entry.name} deleted. Older than "
                f"{config.cache.max_age_days} "
                f"{pluralise('day', config.cache.max_age_days)}"
            )

    logging.info(
        f"Cache cleaning: {deleted_files} cache {pluralise('file', deleted_files)} deleted\n"
    )


def save_to_csv(df: pd.DataFrame, config: AppSettings) -> None:
    """
    Saves the DataFrame to a CSV file without headers.

    Args:
        df: DataFrame to save.
        config: Configuration dictionary.

    """

    file_path = Path(config.paths.base) / config.paths.data_file
    if df.empty:
        raise ValueError("No data to save.")
    required_columns = ["Ticker", "Price", "Date"]
    missing_columns = set(required_columns).difference(df.columns)
    if missing_columns:
        raise ValueError(
            f"Cannot save CSV; missing required columns {sorted(missing_columns)}."
        )

    write_csv_atomically(df[required_columns], file_path, index=False, header=False)
    logging.info(f"Data saved successfully to {format_path(file_path)}\n")
