import json
import yaml
from datetime import datetime, timedelta
import random


class YFinanceSimulator:
    def __init__(self, mock_data_file):
        """
        Initialize the simulator with mock data from a JSON file.
        """
        with open(mock_data_file, "r") as file:
            self.mock_data = json.load(file)

    def get_metadata(self, ticker):
        """
        Get metadata for a given ticker.
        """
        if ticker not in self.mock_data:
            raise ValueError(f"Ticker {ticker} not found in mock data.")
        return self.mock_data[ticker].get("metadata", {})

    def get_historical_data(self, ticker, start_date=None, end_date=None):
        """
        Get historical data for a given ticker.
        """
        if ticker not in self.mock_data:
            raise ValueError(f"Ticker {ticker} not found in mock data.")
        historical = self.mock_data[ticker].get("historical", [])

        # Filter by date range if provided
        if start_date or end_date:
            start_date = (
                datetime.strptime(start_date, "%Y-%m-%d")
                if start_date
                else datetime.min
            )
            end_date = (
                datetime.strptime(end_date, "%Y-%m-%d") if end_date else datetime.max
            )
            historical = [
                entry
                for entry in historical
                if start_date
                <= datetime.strptime(entry["Date"], "%Y-%m-%d")
                <= end_date
            ]
        return historical

    @staticmethod
    def generate_fake_data(ticker, quote_type="EQUITY", currency="USD"):
        """
        Generate fake data for a given ticker.
        """
        metadata = {
            "currency": currency,
            "quoteType": quote_type,
            "symbol": ticker,
            "shortName": f"{ticker} Mock",
            "longName": f"{ticker} Mock Data",
            "exchange": "SIM",
            "timeZoneFullName": "Europe/London",
            "timeZoneShortName": "GMT",
        }

        historical = []
        start_date = datetime.now() - timedelta(days=30)
        for i in range(30):
            date = start_date + timedelta(days=i)
            if date.weekday() >= 5:  # Skip weekends
                continue
            close_price = round(random.uniform(50, 500), 2)
            historical.append({"Date": date.strftime("%Y-%m-%d"), "Close": close_price})

        return {"metadata": metadata, "historical": historical}


def load_config(config_path):
    """
    Load the configuration from a YAML file.
    """
    with open(config_path, "r") as file:
        config = yaml.safe_load(file)

    if not config.get("tickers"):
        raise KeyError("The 'tickers' key is missing in the configuration file.")
    return config


def populate_missing_tickers(config, mock_data_file):
    """
    Populate missing tickers in the mock data file using the simulator's fake data generator.
    """
    simulator = YFinanceSimulator(mock_data_file)
    tickers = config["tickers"]
    updated_mock_data = simulator.mock_data

    for ticker in tickers:
        if ticker not in updated_mock_data:
            # Use default values for missing tickers
            updated_mock_data[ticker] = simulator.generate_fake_data(ticker)

    # Save the updated mock data back to the file
    with open(mock_data_file, "w") as file:
        json.dump(updated_mock_data, file, indent=4)

    print("Mock data updated with missing tickers.")


if __name__ == "__main__":
    # Load the configuration
    config = load_config("test.yaml")

    # Update the mock data with missing tickers
    populate_missing_tickers(config, "simulator data.json")

    # Instantiate the simulator and test its functionality
    simulator = YFinanceSimulator("simulator data.json")
    print("Metadata for first ticker:", simulator.get_metadata(config["tickers"][0]))
    print(
        "Historical data for first ticker:",
        simulator.get_historical_data(config["tickers"][0]),
    )
