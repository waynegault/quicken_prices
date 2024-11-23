
import json

# Load the mock simulator data
with open("yfinance_essential_data_diverse.json", "r") as f:
    mock_data = json.load(f)

# Simulator class
class YFinanceSimulator:
    def __init__(self, data):
        self.data = data

    def Ticker(self, ticker):
        if ticker not in self.data:
            raise ValueError(f"Ticker '{ticker}' not found in simulator data.")
        return MockTicker(self.data[ticker])

class MockTicker:
    def __init__(self, ticker_data):
        self.info = ticker_data.get("metadata", {})
        self.historical_data = ticker_data.get("historical", [])

    def history(self, start=None, end=None, interval="1d"):
        # Simply return the historical data as-is from the dataset
        return self.historical_data

# Instantiate the simulator
simulator = YFinanceSimulator(mock_data)

# Save a simulated output for validation
output = {}

tickers_to_test = ["AAPL", "^GSPC", "CL=F", "GBP=X", "SPY", "VTSAX", "FAKE"]

for ticker in tickers_to_test:
    try:
        ticker_obj = simulator.Ticker(ticker)
        output[ticker] = {
            "metadata": ticker_obj.info,
            "historical": ticker_obj.history()
        }
    except ValueError as e:
        output[ticker] = {"error": str(e)}

# Save simulator output to a file
with open("simulator_test_output.json", "w") as f:
    json.dump(output, f, indent=4)

print("Simulator output saved to simulator_test_output.json")
