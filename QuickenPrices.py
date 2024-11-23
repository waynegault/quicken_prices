import yaml
import logging
from simulator import (YFinanceSimulator as API)
# import yfinance as API



def setup_logging(log_file="quicken_prices.log"):
    """
    Set up logging to both a file and the terminal.
    """
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[logging.FileHandler(log_file), logging.StreamHandler()],
    )


def load_config(config_path="config.yaml"):
    """
    Load the configuration from a YAML file.
    """
    with open(config_path, "r") as file:
        config = yaml.safe_load(file)

    # Validate required keys
    required_keys = ["tickers", "exchange_rates"]
    for key in required_keys:
        if key not in config:
            raise KeyError(f"Missing required config key: {key}")

    # Optional keys with defaults
    config.setdefault("log_file", "quicken_prices.log")
    config.setdefault("mock_data", "yfinance_essential_data_diverse.json")
    config.setdefault("output_file", "quicken_prices.csv")

    return config


def validate_exchange_rates(tickers, exchange_rates, simulator):
    """
    Validate that all necessary exchange rate tickers are available in the data.
    """
    for ticker in tickers:
        metadata = simulator.get_metadata(ticker)
        currency = metadata.get("currency")
        if currency != "GBP" and currency not in exchange_rates:
            logging.warning(
                f"Missing exchange rate for {currency}. Adding {currency}GBP=X."
            )
            exchange_rates.append(f"{currency}GBP=X")
    return exchange_rates


def process_tickers(simulator, tickers, exchange_rates):
    """
    Process each ticker to fetch metadata, historical prices, and perform currency conversions if needed.
    """
    results = []

    for ticker in tickers:
        try:
            metadata = simulator.get_metadata(ticker)
            historical = simulator.get_historical_data(ticker)

            currency = metadata.get("currency")
            quote_type = metadata.get("quoteType")

            for entry in historical:
                price = entry["Close"]
                date = entry["Date"]

                # Perform currency conversion if necessary
                if currency != "GBP" and ticker not in exchange_rates:
                    logging.warning(
                        f"Skipping {ticker} due to missing exchange rate for {currency}."
                    )
                    continue
                elif currency == "GBp":  # Convert pence to pounds
                    price = price / 100
                    currency = "GBP"

                results.append(
                    {
                        "Ticker": ticker,
                        "Date": date,
                        "Close": price,
                        "Currency": currency,
                        "QuoteType": quote_type,
                    }
                )

        except Exception as e:
            logging.error(f"Error processing ticker {ticker}: {e}")

    return results


def write_csv(results, output_file):
    """
    Write the processed data to a CSV file in Quicken-compatible format.
    """
    with open(output_file, "w") as file:
        for row in results:
            date_ddmmyyyy = row["Date"][:10].replace("-", "/")
            file.write(f"{row['Ticker']},{row['Close']:.5f},{date_ddmmyyyy}\n")


def main():
    """
    Main execution function.
    """
    config = load_config()
    setup_logging(config["log_file"])

    simulator = API(config["mock_data"])
    tickers = config["tickers"]
    exchange_rates = config["exchange_rates"]

    # Validate exchange rate tickers
    exchange_rates = validate_exchange_rates(tickers, exchange_rates, simulator)

    # Process tickers and generate results
    results = process_tickers(simulator, tickers, exchange_rates)

    # Write results to a CSV file
    write_csv(results, config["output_file"])

    logging.info(f"Processing complete. Results saved to {config['output_file']}.")


if __name__ == "__main__":
    main()
