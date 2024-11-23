# QuickenPrices Testing Environment Setup

This guide outlines how to set up and use the QuickenPrices testing environment, including the necessary files, configurations, and steps to ensure smooth functionality.

## 1. Overview
The testing environment is designed to simulate the behavior of the `yfinance` API using a custom-built simulator. This allows for local testing without relying on external API calls, ensuring a controlled and reproducible environment.

## 2. Prerequisites
- Python 3.10 or later
- The following Python libraries installed via pip:
  - `PyYAML`
  - `pandas`
  - `logging` (standard library)
  - `datetime` (standard library)
  - `json` (standard library)

## 3. Required Files
The following files are required to set up the testing environment:

1. **`config.yaml`**  
   This file contains the configuration parameters for QuickenPrices. Ensure this file is placed in the same directory as the main script. Example keys:

   ```yaml
   base_directory: ./
   mock_data: yfinance_essential_data_diverse.json
   output_file: quicken_prices.csv
   log_file: quicken_prices.log
   exchange_rates: GBP=X
   tickers:
     - AAPL
     - GBP=X
     - ^GSPC
2. **`simulator.py`**

    A custom simulator script that mimics the behavior of the yfinance API. This file generates mock data in a controlled format for testing purposes.

3. **`yfinance_essential_data_diverse.json`**

    A JSON dataset containing mock responses for a diverse range of tickers. This file is used by the simulator to return appropriate data.

4. **`QuickenPrices.py`**

    The main script being tested. This script processes financial data, performs currency conversions, and outputs results to a CSV file.     

## 4. Setting Up the Environment
1. Install Dependencies
Install the required Python libraries by running:

```bash
pip install pyyaml pandas
```

2. Place Files in the Working Directory

    Ensure all required files (config.yaml, simulator.py, yfinance_essential_data_diverse.json, QuickenPrices.py) are in the same working directory.

3. Configure the Main Script

    Update the import statements in QuickenPrices.py to use the simulator for testing:

    ```python
    # Uncomment the line below for testing with the simulator
    from simulator import YFinanceSimulator as API

    # Comment out the line below to avoid actual API calls
    # import yfinance as API

4. Verify the Simulator Data

    Ensure the JSON file contains mock data for all tickers specified in config.yaml. 
    Example structure:

    ```json
    {
    "AAPL": {
        "metadata": { "currency": "USD", "quoteType": "EQUITY" },
        "historical": [
            { "Date": "2023-01-01", "Close": 150.0 },
            { "Date": "2023-01-02", "Close": 152.5 }
        ]
    },
    "GBP=X": {
        "metadata": { "currency": "GBP", "quoteType": "CURRENCY" },
        "historical": [
            { "Date": "2023-01-01", "Close": 0.75 },
            { "Date": "2023-01-02", "Close": 0.76 }
        ]
    }
}

5. Run the Script

    Execute the main script:

    ```bash
    python QuickenPrices.py

6.  Using the Environment

- Testing with Mock Data

    Ensure mock_data is set to yfinance_essential_data_diverse.json in config.yaml. The simulator will use this data for all operations.

- Switching to Live Data

    To test with live yfinance data, update the import statement in QuickenPrices.py:

    ```python
    # Uncomment this line for live testing
    import yfinance as API
    # Comment out this line to avoid the simulator
    # from simulator import YFinanceSimulator as API
- Log Verification

    Review quicken_prices.log for detailed logs of the script execution, including errors and warnings.

6. Known Issues

- Mismatch in JSON Data

    If a ticker is not found in the JSON dataset, add its mock data to yfinance_essential_data_diverse.json.

- Exchange Rate Tickers

    Ensure exchange rate tickers are correctly defined in both config.yaml and the JSON dataset.

7. Additional Notes

- The environment supports iterative testing. Modify the JSON or YAML as needed to simulate edge cases.
- Always validate terminal output and CSV structure against the project specification.
- Ensure all required data (mock or real) matches the expected schema.

8. Troubleshooting
 
- Script Errors: Check if all dependencies are installed and files are correctly placed.
- Invalid Output: Review quicken_prices.log for detailed execution logs.