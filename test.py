import yfinance as yf
import time
from datetime import datetime, timedelta, date
import pandas as pd


def download_data(ticker_symbol, start, end):
    # assume start and end are datetime objects
    ticker = yf.Ticker(ticker_symbol)
    time.sleep(0.1)  # Rate limiting

    df = ticker.history(
        start=start,
        end=end ,
        interval="1d",
    )
    if df.empty:
        # Create an empty DataFrame with the desired columns and data types
        empty_df = pd.DataFrame(columns=["Date", "Old Price"])
        empty_df["Date"] = pd.to_datetime(empty_df["Date"], utc=True)
        empty_df["Old Price"] = np.nan
        return empty_df
    df = df.rename(columns={"Close": "Old Price"})
    df.reset_index(inplace=True)  # Reset the index to make Date a column
    # Convert the 'Date' column to UTC timezone
    df["Date"] = pd.to_datetime(df["Date"], utc=True).dt.date
    df = df[["Date", "Old Price"]]  # Select only the "Old Price" column
    return df

start = "2024-12-03"
end = "2024-12-10"

start = date.fromisoformat(start)
end = date.fromisoformat(end)

print(f"Start: {start} type {type(start)}")

data = download_data("AMGN", start=start, end=end)

print(data)
