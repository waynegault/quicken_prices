import yfinance as yf
import json
import pandas as pd

# Define a diverse list of tickers across asset classes
tickers = [
    "AAPL",  # Stock
    "^GSPC",  # Index
    "CL=F",  # Commodity (Crude Oil)
    "GBP=X",  # Exchange Rate (GBP/USD)
    "SPY",  # ETF
    "VTSAX",  # Mutual Fund
    "FAKE",  # Invalid Ticker
]


# Convert Timestamps to strings recursively
def serialize_timestamps(obj):
    if isinstance(obj, pd.Timestamp):
        return obj.isoformat()  # Convert to ISO format string
    elif isinstance(obj, dict):
        return {k: serialize_timestamps(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [serialize_timestamps(i) for i in obj]
    else:
        return obj


# Fetch essential historical data (Date/Close with timezone)
def fetch_historical_data(ticker, start="2023-01-01", end="2023-01-31", interval="1d"):
    try:
        stock = yf.Ticker(ticker)
        hist = stock.history(start=start, end=end, interval=interval)
        return serialize_timestamps(
            hist.reset_index()[["Date", "Close"]].to_dict(orient="records")
        )
    except Exception as e:
        return {"error": str(e)}


# Fetch essential metadata (quoteType, currency)
def fetch_metadata(ticker):
    try:
        stock = yf.Ticker(ticker)
        info = stock.info
        return {
            "quoteType": info.get("quoteType", None),
            "currency": info.get("currency", None),
        }
    except Exception as e:
        return {"error": str(e)}


# Main execution
output = {}

for ticker in tickers:
    output[ticker] = {
        "metadata": fetch_metadata(ticker),
        "historical": fetch_historical_data(ticker),
    }

# Save results to a JSON file
with open("yfinance_essential_data_diverse.json", "w") as f:
    json.dump(output, f, indent=4)

print("Diverse essential data saved to yfinance_essential_data_diverse.json")
