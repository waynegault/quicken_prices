"""Yahoo Finance validation, downloads, and date-range cache planning."""

import calendar
import concurrent.futures
import logging
import time
from functools import wraps
from numbers import Real
from typing import Any, Dict, List, Optional, Tuple, Type, Union

import pandas as pd
import requests
import yfinance as yf

from common import native_currencies, pluralise
from market_models import TickerMetadata
from settings import RetrySettings
from storage import CACHE_COLUMNS, load_cache, normalize_cache_data, save_cache

PUBLIC_HOLIDAYS = {"01-01", "12-25", "12-26"}
DEFAULT_TRANSIENT_EXCEPTIONS = (
    requests.exceptions.ConnectionError,
    requests.exceptions.Timeout,
    requests.exceptions.HTTPError,
)


def retry(
    exceptions: Union[Type[BaseException], Tuple[Type[BaseException], ...]] = Exception,
):
    """
    Retry failures using explicit validated policy settings or model defaults.
    """

    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            retry_settings = kwargs.pop("retry_settings", None)
            policy = (
                RetrySettings.model_validate(retry_settings)
                if retry_settings is not None
                else RetrySettings()
            )
            delay = policy.retry_delay

            for attempt in range(1, policy.max_retries + 1):
                try:
                    return func(*args, **kwargs)
                except exceptions as e:
                    if attempt == policy.max_retries:
                        logging.error(
                            f"[{func.__name__}] Max attempts reached "
                            f"({policy.max_retries}). "
                            f"Last error: {e}"
                        )
                        raise
                    logging.warning(
                        f"[{func.__name__}] Attempt {attempt}/{policy.max_retries} failed "
                        f"with error: {e}. Retrying in {delay:.1f} seconds..."
                    )
                    time.sleep(delay)
                    delay *= policy.backoff

        return wrapper

    return decorator

def adjust_date(date: pd.Timestamp, direction: str) -> pd.Timestamp:
    """
    Adjusts a date to the nearest valid business day (avoiding weekends and public holidays).

    Args:
        date (pd.Timestamp): The initial date to adjust.
        direction (str): "start" to move forwards, "end" to move backwards.

    Returns:
        pd.Timestamp: The adjusted date.
    """
    if direction not in {"start", "end"}:
        raise ValueError("Direction must be 'start' or 'end'.")

    while True:
        # Adjust for weekends
        if date.weekday() >= 5:  # Weekend check (5=Saturday, 6=Sunday)
            if direction == "start":
                days_to_add = (calendar.MONDAY - date.weekday()) % 7
                date += pd.Timedelta(days=days_to_add)
            elif direction == "end":
                date -= pd.Timedelta(days=date.weekday() - calendar.FRIDAY)
        else:
            # Check for public holidays
            if date.strftime("%m-%d") not in PUBLIC_HOLIDAYS:
                break  # Valid date found
            # Move the date if it's a holiday
            if direction == "start":
                date += pd.Timedelta(days=1)
            elif direction == "end":
                date -= pd.Timedelta(days=1)

    return date.floor("D")  # Normalize to remove time component

def get_date_range(config: dict) -> Tuple[pd.Timestamp, pd.Timestamp]:
    """
    Calculates start and end dates, avoiding weekends and public holidays.

    Args:
        config (dict): Configuration containing 'collection_period_years'.

    Returns:
        Tuple[pd.Timestamp, pd.Timestamp]: The adjusted start and end dates, localized to UTC.
    """
    if "collection_period_years" not in config:
        raise KeyError("Config is missing 'collection_period_years' key.")

    collection_period_years = config["collection_period_years"]
    if (
        not isinstance(collection_period_years, (int, float))
        or collection_period_years <= 0
    ):
        raise ValueError("'collection_period_years' must be a positive number.")

    # Calculate the number of days in the collection period
    collection_period_days = int(round(collection_period_years * 365.25))

    # Get today's date in UTC
    today_utc = pd.Timestamp.now(tz="UTC").normalize()

    # Calculate initial start and end dates
    end_date = today_utc
    start_date = end_date - pd.Timedelta(days=collection_period_days)

    # Adjust for weekends and public holidays
    end_date = adjust_date(end_date.tz_localize(None), direction="end").tz_localize(
        "UTC"
    )
    start_date = adjust_date(
        start_date.tz_localize(None), direction="start"
    ).tz_localize("UTC")

    # Log the collection period details
    business_days_range = pd.bdate_range(start_date, end_date, tz="UTC")
    non_weekend_days = len(business_days_range)

    logging.info(
        f"Collection period: {collection_period_years} years "
        f"({collection_period_days} days, {non_weekend_days} business days)"
    )

    return start_date, end_date

def find_missing_ranges(
    start_date: pd.Timestamp,
    end_date: pd.Timestamp,
    first_cache: Optional[pd.Timestamp],
    last_cache: Optional[pd.Timestamp],
) -> List[Tuple[pd.Timestamp, pd.Timestamp]]:
    """Finds missing date ranges, no longer adjusting for weekends."""

    if start_date > end_date:
        logging.warning("start_date is after end_date; no valid range.")
        return []

    # If no cache data, just return the entire [start_date, end_date].
    if pd.isna(first_cache) or pd.isna(last_cache):
        return [(start_date, end_date)]

    if first_cache > last_cache:
        first_cache, last_cache = last_cache, first_cache

    # If entire requested range is already in cache, no missing
    if start_date >= first_cache and end_date <= last_cache:
        return []

    missing_ranges = []

    # If there's a portion before the cache
    if start_date < first_cache:
        missing_end = min(end_date, first_cache - pd.Timedelta(days=1))
        if start_date <= missing_end:
            missing_ranges.append((start_date, missing_end))

    # If there's a portion after the cache
    if end_date > last_cache:
        missing_start = max(start_date, last_cache + pd.Timedelta(days=1))
        if missing_start <= end_date:
            missing_ranges.append((missing_start, end_date))

    return missing_ranges

def get_tickers(
    tickers: List[str],
    max_show: int = 5,
    home_currency: str = "GBP",
    retry_settings: Optional[Dict[str, Any]] = None,
) -> List[TickerMetadata]:
    """
    Get and validate stock and FX tickers.

    Args:
        tickers: A list of ticker symbols to validate.
        max_show: Maximum number of tickers to display in logs.
        home_currency: The target currency for price conversion (e.g. "GBP").

    Returns:
        List of valid tickers with metadata.
    """

    if not tickers:
        raise ValueError("No tickers defined in configuration.")

    logging.info(
        f"Validating {len(tickers)} stock {pluralise('ticker', len(tickers))}..."
    )
    valid_tickers = validate_tickers(tickers, max_show, retry_settings)

    if not valid_tickers:
        raise ValueError("No valid tickers found; check symbols and network access.")

    # Generate FX tickers based on the currencies in valid_tickers
    currencies = {
        ticker.currency
        for ticker in valid_tickers
        if ticker.currency not in native_currencies(home_currency)
    }
    fx_tickers = {f"{currency}{home_currency}=X" for currency in currencies}

    # Exclude tickers from fx_tickers that might already be in valid_tickers
    valid_ticker_strings = {ticker.ticker for ticker in valid_tickers}
    fx_tickers = {ticker for ticker in fx_tickers if ticker not in valid_ticker_strings}

    if fx_tickers:
        logging.info(
            f"Validating {len(fx_tickers)} FX {pluralise('ticker', len(fx_tickers))}..."
        )
        valid_fx_tickers = validate_tickers(
            sorted(fx_tickers), max_show, retry_settings
        )
        all_valid_tickers = []
        seen_tickers = set()
        for ticker_info in valid_tickers + valid_fx_tickers:
            if ticker_info.ticker in seen_tickers:
                continue
            seen_tickers.add(ticker_info.ticker)
            all_valid_tickers.append(ticker_info)
    else:
        logging.info("No new FX tickers required.")
        all_valid_tickers = valid_tickers

    logging.info(
        f"Validated {len(all_valid_tickers)} {pluralise('ticker', len(all_valid_tickers))} in total\n"
    )
    return all_valid_tickers

def validate_tickers(
    tickers: List[str],
    max_show: int = 5,
    retry_settings: Optional[Dict[str, Any]] = None,
) -> List[TickerMetadata]:
    validated_by_symbol: Dict[str, TickerMetadata] = {}
    # Use a ThreadPoolExecutor for concurrent validation
    max_workers = max(1, min(10, len(tickers)))
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Map each ticker to the validate_ticker function
        future_to_ticker = {
            executor.submit(
                validate_ticker, ticker, retry_settings=retry_settings
            ): ticker
            for ticker in tickers
        }
        for future in concurrent.futures.as_completed(future_to_ticker):
            ticker_symbol = future_to_ticker[future]
            data = future.result()
            if (
                data
                and data.instrument_type != "Unknown"
                and data.currency != "Unknown"
            ):
                validated_by_symbol[data.ticker.upper()] = data

    valid_tickers = []
    for ticker_symbol in tickers:
        normalized_symbol = ticker_symbol.upper()
        if normalized_symbol in validated_by_symbol:
            valid_tickers.append(validated_by_symbol[normalized_symbol])

    logging.info(
        f"{len(valid_tickers)} {pluralise('ticker', len(valid_tickers))} validated: "
        f"{[ticker.ticker for ticker in valid_tickers[:max_show]]}"
        f"{'...' if len(valid_tickers) > max_show else ''}"
    )
    return valid_tickers

@retry(exceptions=DEFAULT_TRANSIENT_EXCEPTIONS)
def validate_ticker(
    ticker_symbol: str,
) -> Optional[TickerMetadata]:
    if not isinstance(ticker_symbol, str):
        raise TypeError("Ticker must be a string.")

    # Let yfinance handle its own session (no longer accepts custom sessions).
    ticker = yf.Ticker(ticker_symbol)
    info = ticker.get_info()

    if not info or info.get("symbol") is None:
        logging.error(
            f"Ticker '{ticker_symbol}' is invalid. Check if the symbol is correct or available on yfinance."
        )
        return None

    first_trade_date = info.get("firstTradeDateEpochUtc")
    if first_trade_date is None:
        earliest_date = None
    else:
        try:
            earliest_date = pd.Timestamp(first_trade_date, unit="s").tz_localize(
                "UTC"
            )
        except (ValueError, OverflowError) as e:
            logging.warning(
                f"Failed to parse firstTradeDateEpochUtc={first_trade_date} for '{ticker_symbol}': {e}"
            )
            earliest_date = None

    current_price = info.get("bid", float("nan"))
    if not isinstance(current_price, Real) or pd.isna(current_price):
        current_price = float("nan")

    return TickerMetadata(
        ticker=info.get("symbol", ticker_symbol).upper(),
        earliest_date=earliest_date,
        instrument_type=info.get("quoteType", "Unknown"),
        currency=info.get("currency", "Unknown"),
        current_price=float(current_price),
    )

def fetch_historical_data(
    tickers: List[TickerMetadata],
    start_date: pd.Timestamp,
    end_date: pd.Timestamp,
    cache_dir: str,
) -> pd.DataFrame:
    """
    Fetch historical data for a list of tickers using daily yfinance data
    and caching logic (handled in fetch_ticker_data).

    Args:
        tickers: A list of tuples containing:
            (ticker_symbol, earliest_date, ticker_type, currency, current_price).
        start_date: Start of the requested data period (inclusive).
        end_date: End of the requested data period (inclusive).
        cache_dir: Directory used for per-ticker cache files.

    Returns:
        A combined DataFrame with columns in CACHE_COLUMNS plus 'Type'
        and 'Original Currency'. Returns empty DataFrame if no data.
    """

    if not tickers:
        logging.info("No tickers provided. Returning empty DataFrame.")
        return pd.DataFrame(columns=list(CACHE_COLUMNS) + ["Type", "Original Currency"])

    logging.info(
        f"Seeking data from {start_date.strftime('%d/%m/%Y')} "
        f"to {end_date.strftime('%d/%m/%Y')}\n"
    )
    records = []

    # For each ticker, fetch data
    for ticker_info in tickers:
        ticker = ticker_info.ticker
        earliest_date = (
            pd.Timestamp(ticker_info.earliest_date)
            if ticker_info.earliest_date is not None
            else None
        )
        normalized_earliest_date = (
            earliest_date if earliest_date is not None and not pd.isna(earliest_date) else None
        )

        # Ensure earliest_date and end_date are directly comparable
        if normalized_earliest_date is not None and getattr(normalized_earliest_date, "tzinfo", None):
            if normalized_earliest_date.tzinfo != end_date.tzinfo:
                normalized_earliest_date = normalized_earliest_date.tz_convert(
                    end_date.tzinfo
                )  # Convert to the same timezone as end_date

        # Skip tickers that began trading after our end_date
        if normalized_earliest_date is not None and end_date < normalized_earliest_date:
            logging.warning(
                f"Skipping {ticker}: Requested period ends before ticker existed."
            )
            continue

        # Only fetch from max(start_date, earliest_date)
        data_start_date = max(start_date, normalized_earliest_date) if normalized_earliest_date is not None else start_date

        df = fetch_ticker_data(
            ticker, data_start_date, end_date, ticker_info.current_price, cache_dir
        )
        if df.empty:
            logging.warning(f"No data found for ticker {ticker}. Skipping.")
            continue

        if "Type" not in df.columns:
            df["Type"] = ticker_info.instrument_type
        if "Original Currency" not in df.columns:
            df["Original Currency"] = ticker_info.currency
        records.append(df)

    # Combine records
    valid_records = [df for df in records if not df.empty and df is not None]
    if valid_records:
        combined_df = pd.concat(valid_records, ignore_index=True)
        logging.info(
            f"Finished retrieving {len(combined_df)} {pluralise('record', len(combined_df))} "
            f"for {len(valid_records)} {pluralise('ticker', len(valid_records))}\n"
        )
        return combined_df

    logging.warning("No valid data fetched for any tickers.")
    return pd.DataFrame(columns=list(CACHE_COLUMNS) + ["Type", "Original Currency"])

def fetch_ticker_data(
    ticker: str,
    start_date: pd.Timestamp,
    end_date: pd.Timestamp,
    current_price: float,
    cache_dir: str,
) -> pd.DataFrame:
    """
    Fetch historical data (daily) for a ticker using local cache + yfinance,
    only calling yfinance for missing date ranges.
    Forward-fills missing dates if this is a CURRENCY ticker.
    """

    # 1) Load cache
    cache_data, cache_status = load_cache(ticker, cache_dir)

    # 2) Identify missing date ranges in cache
    if cache_status == "not_found":
        logging.info(f"{ticker}: No cache file found. Will fetch entire range.")
        first_cache, last_cache = None, None
    else:
        first_cache = cache_data["Date"].min()
        last_cache = cache_data["Date"].max()
        if cache_data.empty or pd.isna(first_cache) or pd.isna(last_cache):
            logging.info(f"{ticker}: Cache contained no usable rows. Will fetch range.")
            first_cache, last_cache = None, None
        else:
            logging.info(
                f"{ticker}: Cache loaded from {first_cache.strftime('%d/%m/%Y')} to "
                f"{last_cache.strftime('%d/%m/%Y')} "
                f"({len(cache_data)} {pluralise('day', len(cache_data))})"
            )

    # 3) Determine missing date ranges
    missing_ranges = find_missing_ranges(start_date, end_date, first_cache, last_cache)

    # 4) Download missing data via yfinance
    downloaded_frames = []
    if missing_ranges:
        for rng_start, rng_end in missing_ranges:
            partial_data = download_data(ticker, rng_start, rng_end)
            if not partial_data.empty:
                downloaded_frames.append(partial_data)
    downloaded_data = (
        normalize_cache_data(ticker, pd.concat(downloaded_frames, ignore_index=True))
        if downloaded_frames
        else pd.DataFrame(columns=CACHE_COLUMNS)
    )

    # 5) Merge newly downloaded data with cache
    if cache_data.empty:
        combined_data = downloaded_data
    elif downloaded_data.empty:
        combined_data = cache_data
    else:
        combined_data = pd.concat([cache_data, downloaded_data], ignore_index=True)

    combined_data = normalize_cache_data(ticker, combined_data)
    if combined_data.empty:
        logging.warning(f"{ticker}: No price data available after cache and download checks.")
        return combined_data

    # 6) Save cache only if changes occurred (and before forwards-fill)
    if not downloaded_data.empty:
        save_cache(ticker, combined_data, cache_dir)

    # 7) Forward-fill logic (for FX tickers)
    if "=X" in ticker.upper():
        min_date = combined_data["Date"].min().normalize()
        max_date = combined_data["Date"].max().normalize()
        # Create full range of dates for every calendar day even if there are gaps in the original data
        all_days = pd.date_range(min_date, max_date, freq="D")

        # Create a full date range for each ticker where all calendar days are present, and missing data is covered by forward-filling the last known value.
        # Get a list of all unique tickers in the dataset
        all_tickers = combined_data["Ticker"].unique()
        # create a MultiIndex that combines all dates and all tickers
        all_ticker_dates = pd.MultiIndex.from_product(
            [all_days, all_tickers], names=["Date", "Ticker"]
        )
        # Create a df where all calendar days are present for every ticker, and missing data is covered by forward-filling the last known value.
        forward_filled_data = (
            # Temporarily use Date and Ticker as a MultiIndex
            combined_data.set_index(["Date", "Ticker"])
            # Reindex to include every day in all_days, (adding rows missing in combined_data) filling any missing rows with the last available data (forward-filling).
            .reindex(all_ticker_dates, method="ffill")
            # Convert the MultiIndex back into regular columns.
            .reset_index()
        )

        if not combined_data.equals(forward_filled_data):
            logging.info(f"{ticker}: Forward-filled FX data ({len(all_days)} days)")
            combined_data = forward_filled_data

    # 8) Append latest price if today's closing not available
    max_date = combined_data["Date"].max()
    today = pd.Timestamp.now(tz="UTC").normalize()

    if (
        pd.notna(max_date)
        and max_date.normalize() < today
        and isinstance(current_price, Real)
        and not pd.isna(current_price)
        and current_price > 0
    ):
        new_row = pd.DataFrame(
            [
                {
                    "Ticker": ticker,
                    "Old Price": current_price,
                    "Date": today,
                }
            ]
        )
        combined_data = pd.concat([combined_data, new_row], ignore_index=True)
        combined_data = normalize_cache_data(ticker, combined_data)
        logging.info(
            f"{ticker}: Appended today's {today.strftime('%d/%m/%Y')} latest price = {current_price:.3f}"
        )

    # 9) Final log
    logging.info(
        f"{ticker}: Got {len(combined_data)} {pluralise('day', len(combined_data))} data "
        f"from {combined_data['Date'].min().strftime('%d/%m/%Y')} to {combined_data['Date'].max().strftime('%d/%m/%Y')}\n"
    )
    return combined_data

def download_data(
    ticker_symbol: str,
    rng_start: pd.Timestamp,
    rng_end: pd.Timestamp,
) -> pd.DataFrame:
    """
    Downloads daily data for the specified date range [rng_start, rng_end] from yfinance.
    Returns a DataFrame with columns including "Date".
    """
    today = pd.Timestamp.now(tz="UTC").normalize()
    if today == rng_start:
        return pd.DataFrame()

    inclusive_end = rng_end + pd.Timedelta(days=1)
    ticker = yf.Ticker(ticker_symbol)
    raw_df = ticker.history(
        start=rng_start,
        end=inclusive_end,
        interval="1d",
    )

    if raw_df is None or raw_df.empty:
        logging.error(
            f"{ticker_symbol}: No data from {rng_start.strftime('%d/%m/%Y')} to "
            f"{inclusive_end.strftime('%d/%m/%Y')}."
        )
        return pd.DataFrame()

    raw_df = raw_df.reset_index().rename(columns={"Close": "Old Price"})
    raw_df["Ticker"] = ticker_symbol
    raw_df["Date"] = pd.to_datetime(raw_df["Date"], utc=True).dt.normalize()
    raw_df["Old Price"] = pd.to_numeric(raw_df["Old Price"], errors="coerce")
    raw_df = raw_df.dropna(subset=["Date", "Old Price"]).loc[:, CACHE_COLUMNS]

    if raw_df.empty:
        logging.error(f"{ticker_symbol}: Download contained no valid price rows.")
        return raw_df

    logging.info(
        f"{ticker_symbol}: Downloaded from {raw_df['Date'].min().strftime('%d/%m/%Y')} "
        f"to {raw_df['Date'].max().strftime('%d/%m/%Y')} ({len(raw_df)} days.)"
    )
    return raw_df
