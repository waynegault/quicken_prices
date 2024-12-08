# QuickenPrices

**QuickenPrices** is a Python application designed to fetch, process, and validate stock prices, exchange rates, and related data, ultimately generating a CSV file compatible with **Quicken XG 2004 UK Edition**. It automates many tasks, such as fetching historical data, currency conversion, and even the GUI automation for importing data into Quicken. This application ensures robust logging, error handling, and configuration management, making it reliable and user-friendly.

---

## Features

### 1. **Data Fetching**
- Fetches stock prices, exchange rates, and metadata from Yahoo Finance using the `yfinance` library.
- Differentiates tickers by type (`Equity`, `ETF`, `Currency`, etc.).
- Retrieves and caches historical data to reduce redundant API calls.
- Automatically adjusts for non-trading days.

### 2. **Currency Conversion**
- Converts all non-GBP prices into GBP.
- Uses exchange rates fetched alongside stock prices for accurate conversion.
- Handles edge cases like unknown currencies or missing exchange rates gracefully.

### 3. **Validation**
- Validates tickers against metadata from Yahoo Finance.
- Logs and skips invalid or misspelled tickers.
- Filters out historical data outside the range of a ticker’s availability.

### 4. **Outputs**
- **Terminal Output**: Clear, readable summaries with details about fetched data, conversion rates, and debugging info.
- **Log File**: Detailed logs, including debug-level information, written to a rotating log file.
- **CSV Output**: A CSV file formatted specifically for importing prices into Quicken.

### 5. **Quicken Integration**
- Detects and validates the Quicken installation path via the Windows Registry.
- Automates the GUI import process for Quicken using `pyautogui`.
- Gracefully handles scenarios where Quicken cannot be opened or elevated permissions are unavailable.

### 6. **Configuration**
- Fully configurable via `configuration.yaml`.
- Includes default values and fallback mechanisms for missing or invalid settings.
- Centralized settings for paths, tickers, logging levels, and operational parameters.

### 7. **Robust Logging and Error Handling**
- Centralized logging for terminal and file outputs, with separate formats for each.
- Comprehensive error handling with retry mechanisms for transient issues.
- Clear notifications for failed operations or invalid inputs.

---

## Installation

### Prerequisites
- Python 3.8 or later
- Windows (required for Quicken XG 2004 integration)
- Administrator privileges (for Quicken import automation)

### Dependencies
The required Python libraries are listed in the `requirements.txt` file. Install them using:

```bash
pip install -r requirements.txt
git clone https://github.com/waynegault/quicken_prices.git
cd quicken_prices
```

### Setting Up
1. Clone the repository:
   ```bash
   git clone https://github.com/waynegault/quicken_prices.git
   cd quicken_prices
   ```

2. Create a `configuration.yaml` file in the same directory as the script. If the file does not exist, a default one will be generated upon running the app.

3. Ensure Quicken XG 2004 is installed and its path is correctly set in the `configuration.yaml` file. If not, the application will attempt to detect it automatically.

---

## Usage

### Running the App
Run the script with Python:

```bash
python QuickenPrices.py
```

### Configuration
Modify `configuration.yaml` to customize the app's behaviour. Key sections include:

#### `tickers`
Specify the stock and currency tickers you want to track.

#### `paths`
Define paths for:
- Data output file
- Log file
- Cache directory
- Quicken executable

#### `collection`
Set the data collection period, retry behaviour, and other operational parameters.

#### `logging`
Control log levels, file sizes, and retention policies.

#### Example Configuration
```yaml
tickers:
  - "^FTSE"
  - "AAPL"
paths:
  data_file: "data.csv"
  log_file: "prices.log"
  cache: "cache"
collection:
  period_years: 0.08
  max_retries: 3
  retry_delay: 2
cache:
  max_age_days: 30
  cleanup_threshold: 200
logging:
  levels:
    file: DEBUG
    terminal: INFO
  max_bytes: 5242880
  backup_count: 5
```

---

## Key Functionalities

### Fetching Data
The app collects:
- Stock prices: Historical and live data.
- Exchange rates: Required for currency conversion.

### Currency Conversion
Handles:
- GBP conversion using exchange rates.
- Special cases like prices in GBp (pence).

### Output Files
- **CSV File**: Contains columns for `Ticker`, `Price`, and `Date`.
- **Log File**: Captures detailed execution and error information.

### Quicken Integration
Automates the import process:
1. Opens Quicken.
2. Navigates to the portfolio view.
3. Opens the price import dialogue.
4. Inputs the generated CSV file path.

---

## Logs and Debugging

### Log Levels
- **ERROR**: Critical issues affecting functionality.
- **WARNING**: Recoverable issues or skipped items.
- **INFO**: General status updates and summaries.
- **DEBUG**: Detailed data for troubleshooting.

### Log File
Stored in the `logs` directory, with rotation to prevent excessive file sizes.

---
## Practical Usage

### Automating the Script with a Batch File and Shortcut

To simplify running the script, you can use a batch (`.bat`) file and create a shortcut to execute it conveniently with elevated privileges. This approach ensures the script runs smoothly, even if administrative permissions are required.

### Step 1: Create the Batch File
1. Open a text editor, such as Notepad.
2. Paste the following code into the editor:
   ```bat
   @echo off
   cd C:\Users\wayne\OneDrive\Documents\GitHub\Python\Projects\quicken_prices\
   python QuickenPrices.py
   exit
   ```
3. Save the file with a `.bat` extension, such as `quickenprice.bat`. Choose a location that is easy to manage, such as: ```C:\Programs\bat Files\quickenprice.bat```


### Step 2: Create a Shortcut
1. Right-click the `.bat` file and select **Create Shortcut**.
2. Move the shortcut to your Desktop or another convenient location.

### Step 3: Configure the Shortcut
1. Right-click the shortcut and select **Properties**.
2. Under the **Shortcut** tab:
- Set the **Target** to the location of your `.bat` file.
- Set the **Start in** field to the directory containing the batch file (e.g., `C:\Programs\bat Files`).
3. Under the **Advanced** button:
- Check **Run as administrator** to ensure the script runs with elevated privileges.
4. Under the **Change Icon** button:
- Select a Quicken-related icon. If you don’t have one, locate the Quicken icon file (usually a `.ico` file) and assign it to the shortcut.

### Step 4: Run the Script
To execute the script:
1. Double-click the shortcut.
2. Accept the User Account Control (UAC) prompt to allow the script to run with elevated privileges.

### Summary
With this setup, you can run the script by simply clicking the shortcut, which:
- Navigates to the correct directory.
- Executes the Python script.
- Ensures administrative permissions for Quicken integration.
- Displays a Quicken icon for quick identification.

---

## Troubleshooting

### Common Issues
1. **Quicken Path Not Found**:
   - Ensure Quicken is installed in a standard directory or specify its path in `configuration.yaml`.
2. **Invalid Tickers**:
   - Check for typos or unsupported symbols.
3. **Missing Exchange Rates**:
   - Confirm the currency pairs are available in Yahoo Finance.

### Debugging
Enable `DEBUG` level logging in `configuration.yaml` to see detailed execution logs.

---

## Development Notes

### Design Philosophy
- Modular functions with clear separation of concerns.
- Centralized logging, configuration, and error handling.
- Reusable utilities and decorators.

---

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

---

## Contribution

Feel free to fork the repository, create issues, or submit pull requests to contribute to the project.

---

For further assistance, please contact [Wayne Gault](https://github.com/waynegault).
