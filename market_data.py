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
from settings import AppSettings, RetrySettings
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

def get_date_range(config: AppSettings) -> Tuple[pd.Timestamp, pd.Timestamp]:
    """
    Calculates start and end dates, avoiding weekends and public holidays.

    Args:
        config: Strictly validated application settings.

    Returns:
        Tuple[pd.Timestamp, pd.Timestamp]: The adjusted start and end dates, localized to UTC.
    """
    collection_period_years = config.collection_period_years

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
    try:
        info = ticker.get_info()
    except ValueError as error:
        if "Expecting value" not in str(error):
            raise
        raise requests.exceptions.HTTPError(
            f"Yahoo Finance returned an invalid response for {ticker_symbol}."
        ) from error

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
        tickers: Validated ticker metadata.
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


def _get_cache_bounds(
    ticker: str, cache_data: pd.DataFrame, cache_status: str
) -> Tuple[Optional[pd.Timestamp], Optional[pd.Timestamp]]:
    if cache_status == "not_found":
        logging.info(f"{ticker}: No cache file found. Will fetch entire range.")
        return None, None

    first_cache = cache_data["Date"].min()
    last_cache = cache_data["Date"].max()
    if cache_data.empty or pd.isna(first_cache) or pd.isna(last_cache):
        logging.info(f"{ticker}: Cache contained no usable rows. Will fetch range.")
        return None, None

    logging.info(
        f"{ticker}: Cache loaded from {first_cache.strftime('%d/%m/%Y')} to "
        f"{last_cache.strftime('%d/%m/%Y')} "
        f"({len(cache_data)} {pluralise('day', len(cache_data))})"
    )
    return first_cache, last_cache


def _download_missing_data(
    ticker: str, missing_ranges: List[Tuple[pd.Timestamp, pd.Timestamp]]
) -> pd.DataFrame:
    downloaded_frames = []
    for range_start, range_end in missing_ranges:
        partial_data = download_data(ticker, range_start, range_end)
        if not partial_data.empty:
            downloaded_frames.append(partial_data)

    if not downloaded_frames:
        return pd.DataFrame(columns=CACHE_COLUMNS)
    return normalize_cache_data(
        ticker, pd.concat(downloaded_frames, ignore_index=True)
    )


def _merge_and_save_cache(
    ticker: str,
    cache_data: pd.DataFrame,
    downloaded_data: pd.DataFrame,
    cache_dir: str,
) -> pd.DataFrame:
    combined_data = normalize_cache_data(
        ticker, pd.concat([cache_data, downloaded_data], ignore_index=True)
    )
    if not combined_data.empty and not downloaded_data.empty:
        save_cache(ticker, combined_data, cache_dir)
    return combined_data


def _forward_fill_fx_data(
    ticker: str, combined_data: pd.DataFrame
) -> pd.DataFrame:
    if "=X" not in ticker.upper() or combined_data.empty:
        return combined_data

    min_date = combined_data["Date"].min().normalize()
    max_date = combined_data["Date"].max().normalize()
    all_days = pd.date_range(min_date, max_date, freq="D")
    filled_data = (
        combined_data.set_index("Date")
        .reindex(all_days, method="ffill")
        .rename_axis("Date")
        .reset_index()
    )
    if len(filled_data) != len(combined_data):
        logging.info(f"{ticker}: Forward-filled FX data ({len(all_days)} days)")
    return filled_data


def _append_current_price(
    ticker: str,
    combined_data: pd.DataFrame,
    current_price: float,
    end_date: pd.Timestamp,
) -> pd.DataFrame:
    today = pd.Timestamp.now(tz="UTC").normalize()
    if (
        not combined_data.empty
        and end_date.normalize() >= today
        and combined_data["Date"].max().normalize() < today
        and isinstance(current_price, Real)
        and not pd.isna(current_price)
        and current_price > 0
    ):
        new_row = pd.DataFrame(
            [{"Ticker": ticker, "Old Price": current_price, "Date": today}]
        )
        combined_data = normalize_cache_data(
            ticker, pd.concat([combined_data, new_row], ignore_index=True)
        )
        logging.info(
            f"{ticker}: Appended today's {today.strftime('%d/%m/%Y')} "
            f"latest price = {current_price:.3f}"
        )
    return combined_data


def _select_requested_data(
    ticker: str,
    combined_data: pd.DataFrame,
    start_date: pd.Timestamp,
    end_date: pd.Timestamp,
) -> pd.DataFrame:
    in_range = combined_data[
        combined_data["Date"].between(start_date, end_date, inclusive="both")
    ]
    if "=X" not in ticker.upper():
        return in_range.reset_index(drop=True)

    prior_context = combined_data[combined_data["Date"] < start_date].tail(1)
    return pd.concat([prior_context, in_range], ignore_index=True)


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

    if start_date > end_date:
        raise ValueError("start_date must not be after end_date.")

    cache_data, cache_status = load_cache(ticker, cache_dir)
    first_cache, last_cache = _get_cache_bounds(ticker, cache_data, cache_status)
    missing_ranges = find_missing_ranges(start_date, end_date, first_cache, last_cache)
    downloaded_data = _download_missing_data(ticker, missing_ranges)
    combined_data = _merge_and_save_cache(
        ticker, cache_data, downloaded_data, cache_dir
    )
    if combined_data.empty:
        logging.warning(f"{ticker}: No price data available after cache and download checks.")
        return combined_data

    combined_data = _forward_fill_fx_data(ticker, combined_data)
    combined_data = _append_current_price(
        ticker, combined_data, current_price, end_date
    )
    combined_data = _select_requested_data(
        ticker, combined_data, start_date, end_date
    )
    if combined_data.empty:
        logging.warning(f"{ticker}: No data falls within the requested date range.")
        return combined_data

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
