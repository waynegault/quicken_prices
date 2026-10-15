"""Pure price conversion and Quicken output preparation."""

import logging

import pandas as pd

from common import native_currencies, pluralise

def convert_prices(df: pd.DataFrame, home_currency: str = "GBP") -> pd.DataFrame:
    """
    Converts prices to the home currency based on ticker type and currency
    (daily data), while also logging mismatched days between equity and FX.

    Logic Overview:
      1) Separate equity rows (Type != "CURRENCY") from FX rows (Type == "CURRENCY").
      2) Log how many dates appear in equity vs. FX vs. both (mismatched days).
      3) For currency rows, group by (Date, Ticker) to build an 'FX Rate'.
      4) Merge 'FX Rate' into equity rows on (Date, FX Ticker) using merge_asof.
      5) Set FX Rate = 1.0 (or 0.01 for the GBP pence variant).
      6) Raise an error if non-home rows have missing FX rates.
      7) Compute final Price in the home currency and return the result.

    Args:
        df: DataFrame containing historical data. Must include columns:
            'Date', 'Type', 'Ticker', 'Old Price', 'Original Currency'.
        home_currency: The target currency for conversion (e.g. "GBP").

    Returns:
        DataFrame with a new 'Price' column converted to the home currency.
        Raises ValueError if missing FX rates are found for non-home instruments.
        Returns empty DataFrame if 'df' is empty or errors occur.
    """
    # Quick check
    if df.empty:
        logging.error("Empty DataFrame received.")
        return pd.DataFrame()

    required_columns = {"Date", "Type", "Ticker", "Old Price", "Original Currency"}
    missing_columns = required_columns.difference(df.columns)
    if missing_columns:
        logging.error(f"Missing required columns for conversion: {sorted(missing_columns)}")
        return pd.DataFrame()

    # 1) Separate equity rows vs. FX rows
    equity_df = df[df["Type"] != "CURRENCY"].copy()
    fx_df = df[df["Type"] == "CURRENCY"].copy()
    equity_df["Date"] = pd.to_datetime(equity_df["Date"], utc=True, errors="coerce").dt.normalize()
    fx_df["Date"] = pd.to_datetime(fx_df["Date"], utc=True, errors="coerce").dt.normalize()
    equity_df["Old Price"] = pd.to_numeric(equity_df["Old Price"], errors="coerce")
    fx_df["Old Price"] = pd.to_numeric(fx_df["Old Price"], errors="coerce")
    equity_df.dropna(subset=["Date", "Ticker", "Old Price"], inplace=True)
    fx_df.dropna(subset=["Date", "Ticker", "Old Price"], inplace=True)

    # 2) Log day coverage mismatches
    equity_days = set(equity_df["Date"])
    fx_days = set(fx_df["Date"])
    common_days = equity_days.intersection(fx_days)
    unmatched_equity_days = equity_days - fx_days
    unmatched_fx_days = fx_days - equity_days
    logging.info(
        f"Total equity {pluralise('day', len(equity_days))}: {len(equity_days)}"
    )
    logging.info(f"Total FX {pluralise('day', len(fx_days))}: {len(fx_days)}")
    logging.info(f"Common {pluralise('day', len(common_days))}: {len(common_days)}")
    logging.info(
        f"Equity-only {pluralise('day', len(unmatched_equity_days))}: {len(unmatched_equity_days)}"
    )
    logging.info(
        f"FX-only {pluralise('day', len(unmatched_fx_days))}: {len(unmatched_fx_days)}\n"
    )

    # 3) Build a minimal FX DataFrame -> 'FX Rate'
    exchange_rate_df = (
        fx_df.groupby(["Date", "Ticker"])["Old Price"]
        .first()  # one price per date+ticker
        .rename("FX Rate")
        .reset_index()
        .rename(columns={"Ticker": "FX Ticker"})
    )

    # 4) Work on a copy of the equity data to avoid mutating the original
    equity_df = equity_df.copy()

    # We'll rename 'Price' -> 'Old Price' so we can store a new 'Price'
    if "Price" in equity_df.columns:
        equity_df = equity_df.rename(columns={"Price": "Old Price"})

    if equity_df.empty:
        fx_only_df = fx_df.rename(columns={"Old Price": "Price"})[["Ticker", "Price", "Date"]]
        return fx_only_df.sort_values(by=["Date", "Ticker"]).reset_index(drop=True)

    # Prepare an 'FX Ticker' column for non-home currencies
    equity_df["FX Ticker"] = "-"
    equity_df["Original Currency"] = equity_df["Original Currency"].astype(str)
    native = native_currencies(home_currency)
    non_home_mask = ~equity_df["Original Currency"].isin(native)
    equity_df.loc[non_home_mask, "FX Ticker"] = (
        equity_df.loc[non_home_mask, "Original Currency"] + f"{home_currency}=X"
    )

    # Ensure data types match for merging
    equity_df["FX Ticker"] = equity_df["FX Ticker"].astype(str)
    exchange_rate_df["FX Ticker"] = exchange_rate_df["FX Ticker"].astype(str)

    # 5) Merge the 'FX Rate' onto the equity rows using merge_asof.
    # Pandas requires the merge key to be sorted, so do the asof merge
    # ticker-by-ticker instead of one global multi-key merge.
    equity_df = equity_df.reset_index().rename(columns={"index": "_row_order"})
    merged_groups = []
    for fx_ticker, equity_group in equity_df.groupby("FX Ticker", sort=False):
        equity_group = equity_group.sort_values("Date").copy()
        fx_group = (
            exchange_rate_df.loc[
                exchange_rate_df["FX Ticker"] == fx_ticker, ["Date", "FX Rate"]
            ]
            .sort_values("Date")
            .reset_index(drop=True)
        )

        if fx_group.empty:
            equity_group["FX Rate"] = pd.NA
        else:
            equity_group = pd.merge_asof(
                equity_group,
                fx_group,
                on="Date",
                direction="backward",  # Use the last known FX rate
            )

        merged_groups.append(equity_group)

    equity_df = (
        pd.concat(merged_groups, ignore_index=True)
        .sort_values("_row_order")
        .drop(columns="_row_order")
        .reset_index(drop=True)
    )

    # 6) Hard-code FX Rate for the home currency and its pence variant
    home_mask = equity_df["Original Currency"] == home_currency
    non_home_mask = ~equity_df["Original Currency"].isin(native)
    equity_df.loc[home_mask, "FX Rate"] = 1.0
    if home_currency == "GBP":
        equity_df.loc[equity_df["Original Currency"] == "GBp", "FX Rate"] = 0.01

    # 7) Refuse to apply rates from future dates to earlier equity prices.
    missing_fx_mask = equity_df["FX Rate"].isna() & non_home_mask
    if missing_fx_mask.any():
        missing_rows = equity_df[missing_fx_mask]
        logging.error(f"Missing historical FX rates for equity rows:\n{missing_rows}")
        raise ValueError(
            f"{len(missing_rows)} equity rows have no FX rate on or before "
            "their price date."
        )

    # 8) Calculate final Price in the home currency
    equity_df["Price"] = equity_df["Old Price"] * equity_df["FX Rate"]

    # Log how many conversions happened (where 'FX Ticker' != '-')
    converted_count = int((equity_df["FX Ticker"] != "-").sum())
    logging.info(f"{converted_count} prices successfully converted to {home_currency}\n")

    # 9) Validate floating-point multiplication
    conversion_error_mask = (
        equity_df["Price"].notna()
        & (equity_df["FX Rate"] * equity_df["Old Price"] - equity_df["Price"]).abs().gt(1e-6)
    )
    incorrect_rows = equity_df[conversion_error_mask]
    if not incorrect_rows.empty:
        logging.warning(f"The following rows have conversion errors:\n{incorrect_rows}")

    # 10) merge equity and fx df's with "Ticker", "Price", "Date" cols
    equity_df = equity_df[["Ticker", "Price", "Date"]]
    fx_df = fx_df.rename(columns={"Old Price": "Price"})
    fx_df = fx_df[["Ticker", "Price", "Date"]]
    combined_df = pd.concat([equity_df, fx_df])
    # Sort by Date then Ticker and reset index
    combined_df = combined_df.sort_values(by=["Date", "Ticker"]).reset_index(drop=True)

    return combined_df

def process_converted_prices(
    converted_df: pd.DataFrame, start_date: pd.Timestamp
) -> pd.DataFrame:
    """
    Prepares the final DataFrame for CSV export.

    Args:
        converted_df: DataFrame with "Ticker", "Date", "Price"

    Returns:
        Processed DataFrame ready to be saved as CSV, or an empty DataFrame if input is empty.
        Raises if a critical error occurs.
    """

    if converted_df.empty:
        logging.warning("Input DataFrame is empty. Returning empty DataFrame.")
        return pd.DataFrame(columns=["Ticker", "Price", "Date"])

    try:
        output_csv = converted_df[["Ticker", "Price", "Date"]].copy()

        # Ensure 'Date' is a proper datetime type
        output_csv["Date"] = pd.to_datetime(output_csv["Date"], errors="coerce")

        # Sort by descending date (in place)
        output_csv.sort_values(by="Date", ascending=False, inplace=True)

        # Filter rows to include only those with 'Date' >= start_date
        output_csv = output_csv[output_csv["Date"] >= start_date]

        # Format date as string using .dt.strftime() -> DD/MM/YYYY
        output_csv["Date"] = output_csv["Date"].dt.strftime("%d/%m/%Y")

        return output_csv

    except Exception as e:
        logging.exception(f"A critical error occurred: {e}")
        raise
