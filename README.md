# QuickenPrices Documentation

## Overview

QuickenPrices is a Python automation script that fetches historical stock price data from Yahoo Finance and automatically imports it into Quicken 2004. The script handles multi-currency portfolios by converting all prices to a home currency (GBP by default) and provides a seamless end-to-end workflow from data collection to Quicken import.

**Key Features:**
- 📈 Fetches historical price data for stocks, ETFs, indices, and commodities
- 💱 Multi-currency support with automatic FX rate conversion
- 🚀 Concurrent processing for fast ticker validation
- 💾 Intelligent caching system to minimize API calls
- 🔧 Full Quicken automation via GUI interaction
- 🛡️ Windows UAC elevation handling for proper Quicken automation
- 📝 Comprehensive logging and error handling
- ⚙️ Configurable via YAML file

## Purpose

This script was developed to automate the tedious process of manually updating stock prices in Quicken 2004. It's particularly useful for:

- **Portfolio Management**: Automatically update prices for large investment portfolios
- **Multi-Currency Investments**: Handle international stocks with automatic currency conversion
- **Regular Updates**: Set up scheduled runs to keep Quicken data current
- **Data Consistency**: Ensure all prices come from a single, reliable source (Yahoo Finance)

## Project Files

**Core Files:**
- `QuickenPrices.py` - Main automation script (1750+ lines)
- `configuration.yaml` - User configuration file with tickers and settings
- `requirements.txt` - Python package dependencies for easy installation

**Generated Files:**
- `data.csv` - Price data output file for Quicken import
- `prices.log` - Detailed execution logs with rotation
- `cache/` - Directory containing cached price data (`.pkl` files)

## User Instructions

### Prerequisites

1. **Windows OS** (required for Quicken automation)
2. **Python 3.8+** installed
3. **Quicken 2004** installed and configured
4. **Administrator privileges** (required for GUI automation)

### Installation

1. **Clone or download** the project to your local machine
2. **Install required Python packages**:
   ```bash
   pip install -r requirements.txt
   ```
   
   **Alternative**: Install packages individually:
   ```bash
   pip install pandas yfinance pyautogui pygetwindow pyyaml requests numpy
   ```

### Configuration

1. **Edit `configuration.yaml`** to customize your settings:

   ```yaml
   # Add your tickers
   tickers:
     - AAPL      # US stocks
     - MSFT.L    # London Stock Exchange
     - ^FTSE     # Indices
     - GBP=X     # FX rates
   
   # Set your home currency
   home_currency: GBP
   
   # Data collection period (in years)
   collection_period_years: 1
   ```

2. **Ticker Format Examples**:
   - **US Stocks**: `AAPL`, `MSFT`, `GOOGL`
   - **London SE**: `VODL.L`, `BARC.L`, `BP.L`
   - **Indices**: `^FTSE`, `^GSPC`, `^DJI`
   - **FX Rates**: `GBP=X`, `EUR=X`, `JPY=X`
   - **Commodities**: `GC=F` (Gold), `CL=F` (Oil)

### Running the Script

#### Method 1: Direct Execution (Recommended)
1. **Double-click** `QuickenPrices.py` in Windows Explorer
2. **Allow UAC elevation** when prompted
3. **Wait for completion** - the script will:
   - Fetch and validate all tickers
   - Download historical price data
   - Convert currencies if needed
   - Open Quicken automatically
   - Import the data
   - Close Quicken and exit after 5 seconds

#### Method 2: Command Line
```bash
python QuickenPrices.py
```

#### Method 3: From VS Code
- Open the project in VS Code
- Run the script (F5 or Ctrl+F5)

### What Happens During Execution

1. **Initialization**: Script checks admin privileges and elevates if needed
2. **Ticker Validation**: Validates all tickers concurrently via Yahoo Finance API
3. **Data Collection**: Downloads historical price data (uses cache when available)
4. **Currency Conversion**: Converts non-home currency prices using FX rates
5. **Data Export**: Saves processed data to `data.csv`
6. **Quicken Automation**: 
   - Opens Quicken Premier
   - Navigates to Portfolio view
   - Opens import dialog
   - Imports the CSV file
   - Closes Quicken
7. **Auto-exit**: Script closes automatically after 5 seconds (press Enter to cancel)

### Output Files

- **`data.csv`**: Final processed price data ready for Quicken import
- **`prices.log`**: Detailed execution log with timestamps
- **`cache/`**: Cached price data to speed up subsequent runs

## Developer Insights

### Architecture Overview

The script follows a modular design with clear separation of concerns:

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Configuration │    │   Data Fetching │    │   Data Processing│
│   Management    │────│   & Validation  │────│   & Currency    │
│   (YAML)        │    │   (Yahoo Finance)│    │   Conversion    │
└─────────────────┘    └─────────────────┘    └─────────────────┘
         │                       │                       │
         └───────────────────────┼───────────────────────┘
                                 │
         ┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
         │    Logging &    │    │   Caching       │    │   Quicken       │
         │    Error        │────│   System        │────│   Automation    │
         │    Handling     │    │   (Pickle)      │    │   (PyAutoGUI)   │
         └─────────────────┘    └─────────────────┘    └─────────────────┘
```

### Key Functions

#### Core Data Functions

**`validate_ticker(ticker_symbol: str)`**
- Validates individual ticker symbols via Yahoo Finance API
- Returns metadata including currency, type, and trading dates
- Includes retry logic for rate limiting

**`get_tickers(tickers: List[str])`**
- Concurrently validates multiple tickers for performance
- Separates stock tickers from FX tickers
- Returns validated ticker list with metadata

**`fetch_ticker_data(ticker_info: Tuple)`**
- Downloads historical price data for individual tickers
- Implements intelligent caching to avoid redundant API calls
- Handles various date ranges and data gaps

**`convert_prices(df: pd.DataFrame)`**
- Converts multi-currency price data to home currency
- Merges equity data with FX rates using time-aware joins
- Implements robust error handling for missing FX rates

#### Automation Functions

**`run_as_admin()`**
- Detects current elevation status
- Requests UAC elevation if needed
- Handles elevation failures gracefully

**`quicken_import(config)`**
- Automates complete Quicken import workflow
- Uses PyAutoGUI for GUI interaction
- Includes window detection and error recovery

**`close_quicken()`**
- Safely terminates Quicken processes
- Uses Windows taskkill for reliable closure

**`countdown_exit(seconds: int = 5)`**
- Provides user-friendly auto-exit with cancellation option
- Uses msvcrt for real-time keyboard input detection

#### Utility Functions

**`setup_logging(config)`**
- Configures dual logging (file + console)
- Implements log rotation to manage file sizes
- Supports configurable log levels

**`load_config(config_path: str)`**
- Loads and validates YAML configuration
- Provides default values for missing settings
- Handles configuration errors gracefully

### Dependencies

#### Core Dependencies
```python
# Data Processing
pandas>=1.5.0          # DataFrame operations and data manipulation
numpy>=1.21.0           # Numerical operations
yfinance>=0.2.18        # Yahoo Finance API client

# GUI Automation  
pyautogui>=0.9.54       # GUI automation for Quicken interaction
pygetwindow>=0.0.9      # Window management and detection

# Configuration & Utilities
pyyaml>=6.0             # YAML configuration file parsing
requests>=2.28.0        # HTTP requests and API calls
```

#### Standard Library
```python
# System & OS
os, sys, subprocess     # Operating system interface
ctypes                  # Windows API access for UAC
winreg                  # Windows registry access
msvcrt                  # Microsoft Visual C Runtime

# Data & Time
datetime, time          # Date/time operations
calendar                # Calendar utilities

# Concurrency & Error Handling  
concurrent.futures      # Concurrent ticker validation
logging                 # Comprehensive logging system
traceback              # Error traceback handling

# Data Storage
pickle                  # Cache serialization
pathlib                # Modern path handling
```

### Caching System

The script implements a sophisticated caching mechanism:

**Cache Structure:**
```
cache/
├── AAPL.pkl           # Individual ticker cache files
├── MSFT.L.pkl         # Stored as pickle for fast I/O
├── GBP=X.pkl          # FX rates cached separately
└── ...
```

**Cache Logic:**
- **Age-based expiry**: Configurable max age (default: 15 days)
- **Smart updates**: Only fetches missing date ranges
- **FX rate handling**: Special logic for forward-filling FX data
- **Automatic cleanup**: Removes expired cache files

### Error Handling & Resilience

**Multi-level Error Handling:**
1. **Network Level**: Retry logic for API timeouts and rate limiting
2. **Data Level**: Validation of ticker symbols and date ranges  
3. **Processing Level**: Robust currency conversion with fallbacks
4. **Automation Level**: Window detection and GUI interaction recovery
5. **System Level**: UAC elevation and process management

**Retry Mechanisms:**
```python
@retry(exceptions=(requests.exceptions.HTTPError,))
def validate_ticker(ticker_symbol: str):
    # Automatic retry for transient network errors
```

**Graceful Degradation:**
- Missing FX rates: Forward/backward fill from available data
- Invalid tickers: Skip and continue with valid ones
- Quicken automation failure: Continue without import
- Elevation failure: Run without admin (limited functionality)

### Logging Strategy

**Dual Logging System:**
- **File Logging**: Persistent detailed logs with rotation
- **Console Logging**: Real-time feedback for user interaction

**Log Levels:**
- **DEBUG**: Detailed technical information for troubleshooting
- **INFO**: Normal operation progress and results
- **WARNING**: Recoverable issues (missing data, retries)
- **ERROR**: Serious problems that may affect functionality

**Sample Log Output:**
```
2025-07-17 17:11:33,369 - INFO - [QuickenPrices.py:344] - Current elevation status: Not elevated
2025-07-17 17:11:33,369 - INFO - [QuickenPrices.py:349] - Attempting to re-run with administrative privileges...
2025-07-17 17:11:38,142 - INFO - [QuickenPrices.py:669] - 24 tickers validated: ['AAPL', 'MSFT.L', ...]
2025-07-17 17:11:38,750 - INFO - [QuickenPrices.py:1317] - 2016 prices successfully converted to GBP
```

### Performance Optimizations

**Concurrent Processing:**
- Ticker validation uses ThreadPoolExecutor for parallel API calls
- Reduces validation time from ~30s to ~5s for 25 tickers

**Intelligent Caching:**
- Avoid redundant API calls for recent data
- Smart date range calculations for partial updates
- Pickle serialization for fast cache I/O

**Memory Management:**
- Processes data in chunks to handle large datasets
- Explicit cleanup of temporary DataFrames
- Efficient pandas operations with vectorization

### Maintenance Guidelines

#### Regular Maintenance Tasks

1. **Update Dependencies** (Monthly):
   ```bash
   pip install --upgrade -r requirements.txt
   ```
   
   **Alternative**: Update specific packages:
   ```bash
   pip install --upgrade yfinance pandas pyautogui
   ```

2. **Clean Cache** (When needed):
   - Delete `cache/` folder to force fresh data download
   - Useful after extended periods or data inconsistencies

3. **Review Logs** (Weekly):
   - Check `prices.log` for warnings or errors
   - Monitor API rate limiting issues

4. **Validate Tickers** (Quarterly):
   - Remove delisted securities from `configuration.yaml`
   - Add new holdings as needed

#### Troubleshooting Common Issues

**Problem: "Ticker validation failed"**
- **Solution**: Check ticker symbol format for specific exchanges
- **Example**: UK stocks need `.L` suffix (e.g., `VODL.L`)

**Problem: "Missing FX rates" error**
- **Solution**: Ensure FX tickers are included (e.g., `EUR=X` for EUR data)
- **Fallback**: Script now includes automatic forward/backward fill

**Problem: "Quicken automation failed"**
- **Solution**: Ensure script runs with administrator privileges
- **Check**: Quicken is not already running when script starts

**Problem: "Permission denied" errors**
- **Solution**: Run script as administrator
- **Alternative**: Right-click script → "Run as administrator"

#### Extending the Script

**Adding New Data Sources:**
1. Implement new fetcher function following `validate_ticker()` pattern
2. Add retry logic and error handling
3. Integrate with existing caching system

**Supporting Additional Currencies:**
1. Add currency to `configuration.yaml`
2. Ensure corresponding FX rate ticker exists (e.g., `CAD=X`)
3. Test currency conversion logic

**Enhancing Quicken Automation:**
1. Study PyAutoGUI documentation for new GUI interactions
2. Add window detection for different Quicken versions
3. Implement more robust error recovery

### Security Considerations

**Administrator Privileges:**
- Required for reliable GUI automation
- Script requests elevation automatically
- UAC prompt ensures user consent

**Data Privacy:**
- No personal financial data is transmitted
- Only public market data is accessed
- Local file storage with standard permissions

**Network Security:**
- HTTPS connections to Yahoo Finance API
- No credentials or API keys required
- Standard HTTP retry and timeout mechanisms

### Future Enhancements

**Potential Improvements:**
1. **Multiple Quicken Versions**: Detect and adapt to different Quicken UI layouts
2. **Additional Data Sources**: Integration with other financial APIs
3. **Portfolio Analytics**: Calculate returns, volatility, and other metrics
4. **Scheduling**: Built-in task scheduler for automated daily updates
5. **GUI Interface**: Optional desktop GUI for easier configuration
6. **Docker Support**: Containerized deployment for consistent environments

**Performance Enhancements:**
1. **Async/Await**: Replace threading with async operations
2. **Batch Processing**: Group API calls for better efficiency
3. **Incremental Updates**: Only fetch changed data since last run
4. **Compressed Caching**: Reduce cache file sizes

---

## License

MIT License - See project files for full license text.

## Author

**Wayne Gault**  
*Initial Development: November 23, 2024*  
*Latest Update: July 17, 2025*

---

*This documentation provides comprehensive coverage of the QuickenPrices automation script. For additional support or feature requests, please refer to the project repository or contact the developer.*
