"""Price conversion and Quicken output preparation."""

import logging

import pandas as pd

from common import native_currencies, pluralise

PRICE_COLUMNS = ("Ticker", "Price", "Date")
CONVERSION_COLUMNS = ("Date", "Type", "Ticker", "Old Price", "Original Currency")


def _prepare_price_rows(data: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    if data.empty:
        return data.copy(), data.copy()

    missing_columns = set(CONVERSION_COLUMNS).difference(data.columns)
    if missing_columns:
        raise ValueError(
            f"Missing required columns for conversion: {sorted(missing_columns)}"
        )

    normalized = data.drop(columns=["Price"], errors="ignore").copy()
    normalized["Date"] = pd.to_datetime(
        normalized["Date"], utc=True, errors="coerce"
    ).dt.normalize()
    normalized["Old Price"] = pd.to_numeric(
        normalized["Old Price"], errors="coerce"
    )
    normalized = normalized.dropna(
        subset=["Date", "Ticker", "Type", "Old Price", "Original Currency"]
    )
    equity_rows = normalized[normalized["Type"] != "CURRENCY"].copy()
    fx_rows = normalized[normalized["Type"] == "CURRENCY"].copy()
    return equity_rows, fx_rows


def _log_date_coverage(
    equity_rows: pd.DataFrame, fx_rows: pd.DataFrame
) -> None:
    equity_days = set(equity_rows["Date"])
    fx_days = set(fx_rows["Date"])
    common_days = equity_days.intersection(fx_days)
    equity_only_days = equity_days - fx_days
    fx_only_days = fx_days - equity_days
    logging.info(
        f"Total equity {pluralise('day', len(equity_days))}: {len(equity_days)}"
    )
    logging.info(f"Total FX {pluralise('day', len(fx_days))}: {len(fx_days)}")
    logging.info(f"Common {pluralise('day', len(common_days))}: {len(common_days)}")
    logging.info(
        f"Equity-only {pluralise('day', len(equity_only_days))}: "
        f"{len(equity_only_days)}"
    )
    logging.info(
        f"FX-only {pluralise('day', len(fx_only_days))}: {len(fx_only_days)}\n"
    )


def _make_exchange_rates(fx_rows: pd.DataFrame) -> pd.DataFrame:
    return (
        fx_rows.groupby(["Date", "Ticker"])["Old Price"]
        .first()
        .rename("FX Rate")
        .reset_index()
        .rename(columns={"Ticker": "FX Ticker"})
    )


def _attach_historical_fx_rates(
    equity_rows: pd.DataFrame,
    exchange_rates: pd.DataFrame,
    home_currency: str,
) -> pd.DataFrame:
    equity_rows = equity_rows.copy()
    equity_rows["FX Ticker"] = "-"
    currencies = equity_rows["Original Currency"].astype(str)
    native = native_currencies(home_currency)
    non_home_mask = ~currencies.isin(native)
    equity_rows.loc[non_home_mask, "FX Ticker"] = (
        currencies[non_home_mask] + "GBP=X"
    )
    equity_rows["FX Ticker"] = equity_rows["FX Ticker"].astype(str)

    equity_rows["_row_order"] = range(len(equity_rows))
    merged_groups = []
    for fx_ticker, equity_group in equity_rows.groupby("FX Ticker", sort=False):
        equity_group = equity_group.sort_values("Date").copy()
        fx_group = exchange_rates.loc[
            exchange_rates["FX Ticker"] == fx_ticker, ["Date", "FX Rate"]
        ].sort_values("Date")

        if fx_group.empty:
            equity_group["FX Rate"] = pd.NA
        else:
            equity_group = pd.merge_asof(
                equity_group,
                fx_group,
                on="Date",
                direction="backward",
            )
        merged_groups.append(equity_group)

    return (
        pd.concat(merged_groups, ignore_index=True)
        .sort_values("_row_order")
        .drop(columns="_row_order")
        .reset_index(drop=True)
    )


def _convert_equity_prices(
    equity_rows: pd.DataFrame, home_currency: str
) -> pd.DataFrame:
    if equity_rows.empty:
        return equity_rows

    native = native_currencies(home_currency)
    home_mask = equity_rows["Original Currency"] == home_currency
    non_home_mask = ~equity_rows["Original Currency"].isin(native)
    equity_rows.loc[home_mask, "FX Rate"] = 1.0
    if home_currency == "GBP":
        equity_rows.loc[equity_rows["Original Currency"] == "GBp", "FX Rate"] = 0.01

    missing_fx_mask = equity_rows["FX Rate"].isna() & non_home_mask
    if missing_fx_mask.any():
        missing_rows = equity_rows[missing_fx_mask]
        logging.error("Missing historical FX rates for equity rows:\n%s", missing_rows)
        raise ValueError(
            f"{len(missing_rows)} equity rows have no FX rate on or before "
            "their price date."
        )

    equity_rows["Price"] = equity_rows["Old Price"] * equity_rows["FX Rate"]
    converted_count = int((equity_rows["FX Ticker"] != "-").sum())
    logging.info(
        f"{converted_count} prices successfully converted to {home_currency}\n"
    )
    return equity_rows


def _combine_price_rows(
    equity_rows: pd.DataFrame, fx_rows: pd.DataFrame
) -> pd.DataFrame:
    if equity_rows.empty:
        combined = fx_rows.rename(columns={"Old Price": "Price"})
        return combined.loc[:, list(PRICE_COLUMNS)].sort_values(
            by=["Date", "Ticker"]
        ).reset_index(drop=True)

    equity_output = equity_rows.loc[:, list(PRICE_COLUMNS)]
    fx_output = fx_rows.rename(columns={"Old Price": "Price"}).loc[
        :, list(PRICE_COLUMNS)
    ]
    return (
        pd.concat([equity_output, fx_output], ignore_index=True)
        .sort_values(by=["Date", "Ticker"])
        .reset_index(drop=True)
    )


def convert_prices(df: pd.DataFrame, home_currency: str = "GBP") -> pd.DataFrame:
    """Convert equity rows using the same-day-or-prior FX rate.

    Input rows must contain ``Date``, ``Type``, ``Ticker``, ``Old Price``, and
    ``Original Currency``. FX tickers are emitted unchanged alongside equities.
    """
    if df.empty:
        logging.warning("Empty DataFrame received for conversion.")
        return pd.DataFrame(columns=PRICE_COLUMNS)

    equity_rows, fx_rows = _prepare_price_rows(df)
    _log_date_coverage(equity_rows, fx_rows)
    if equity_rows.empty:
        return _combine_price_rows(equity_rows, fx_rows)

    exchange_rates = _make_exchange_rates(fx_rows)
    equity_rows = _attach_historical_fx_rates(
        equity_rows, exchange_rates, home_currency
    )
    equity_rows = _convert_equity_prices(equity_rows, home_currency)
    return _combine_price_rows(equity_rows, fx_rows)


def process_converted_prices(
    converted_df: pd.DataFrame, start_date: pd.Timestamp
) -> pd.DataFrame:
    """Filter and format converted prices for headerless Quicken CSV export."""
    if converted_df.empty:
        logging.warning("Input DataFrame is empty. Returning empty output.")
        return pd.DataFrame(columns=PRICE_COLUMNS)

    missing_columns = set(PRICE_COLUMNS).difference(converted_df.columns)
    if missing_columns:
        raise ValueError(
            f"Converted prices are missing columns {sorted(missing_columns)}."
        )

    output = converted_df.loc[:, PRICE_COLUMNS].copy()
    output["Date"] = pd.to_datetime(
        output["Date"], format="mixed", dayfirst=True, errors="raise", utc=True
    )
    output = output.sort_values(by="Date", ascending=False)
    output = output[output["Date"] >= start_date]
    output["Date"] = output["Date"].dt.strftime("%d/%m/%Y")
    return output.reset_index(drop=True)
