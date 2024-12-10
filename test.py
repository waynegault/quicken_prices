import yfinance as yf
import time
from datetime import datetime, timedelta, timezone
import pandas as pd
import numpy as np


def download_data(ticker_symbol, start, end):
    # assume start and end are datetime objects
    ticker = yf.Ticker(ticker_symbol)
    time.sleep(0.1)  # Rate limiting
    df = ticker.history(
        start=start, 
        end=end,
        interval="1d",
    )
    if df.empty:
        # Create an empty DataFrame with the desired columns and data types
        empty_df = pd.DataFrame(columns=["Date", "Old Price"])
        empty_df["Date"] = pd.to_datetime(empty_df["Date"])  # Convert to datetime64
        empty_df["Old Price"] = np.nan
        return empty_df
    df = df.rename(columns={"Close": "Old Price"})
    df.reset_index(inplace=True)  # Reset the index to make Date a column
    # Convert the 'Date' column to UTC timezone
    df["Date"] = df["Date"].dt.tz_convert("UTC")
    df = df[["Date", "Old Price"]]  # Select only the "Old Price" column
    return df


data = download_data("AMGN", start="2024-12-10", end="2024-12-11")

print(data)
print(data.dtypes)
