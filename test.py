import yfinance as yf
import time
from datetime import datetime, timedelta


def download_data(ticker_symbol, start, end):
    # Convert start and end to datetime objects
    start_date = datetime.strptime(start, "%Y-%m-%d")
    end_date = datetime.strptime(end, "%Y-%m-%d") + timedelta(
        days=1
    )  # Add 1 day to end date

    ticker = yf.Ticker(ticker_symbol)
    time.sleep(0.1)  # Rate limiting
    df = ticker.history(
        start=start_date.strftime("%Y-%m-%d"),  # Convert back to string for yfinance
        end=end_date.strftime("%Y-%m-%d"),
        interval="1d",
    )
    df = df.rename(columns={"Close": "Old Price"})
    df.reset_index(inplace=True)  # Reset the index to make Date a column
    df["Date"] = df["Date"].dt.strftime("%Y-%m-%d")
    df = df[["Date", "Old Price"]]  # Select only the "Old Price" column
    return df


data = download_data("MSFT", start="2024-12-02", end="2024-12-09")

print(data)
print(data.dtypes)
