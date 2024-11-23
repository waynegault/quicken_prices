
import yfinance as yf
import json
import pandas as pd

# Define a list of tickers for testing
tickers = ["AAPL", "^GSPC", "CL=F", "GBP=X", "SPY", "VTSAX", "FAKE"]

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
        return serialize_timestamps(hist.reset_index()[["Date", "Close"]].to_dict(orient="records"))
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
with open("yfinance_updated_test_output.json", "w") as f:
    json.dump(output, f, indent=4)

print("Updated test data saved to yfinance_updated_test_output.json")
