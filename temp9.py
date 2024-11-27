import pandas as pd
import logging
from typing import Any

# -------------------- Logger Configuration --------------------

# Configure the logger to display warnings and errors with timestamps
logging.basicConfig(
    level=logging.WARNING,  # Set to DEBUG for more detailed logs
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler()],  # Logs will be output to the console
)
logger = logging.getLogger(__name__)

# -------------------- Helper Function --------------------


def safe_get_date(date: Any) -> str:
    """
    Safely retrieve the date in ISO format or return the original string.

    Parameters:
    - date (Any): The date value to process.

    Returns:
    - str: The date in ISO format if it's a Timestamp, the original string if it's a string, or 'Unknown Date'.
    """
    if isinstance(date, pd.Timestamp):
        return date.date().isoformat()
    elif isinstance(date, str):
        return date
    else:
        return "Unknown Date"


# -------------------- Date Normalization Function --------------------


def normalize_dates(ticker_dataframe: pd.DataFrame) -> pd.DataFrame:
    """
    Normalizes the 'Date' column in the dataframe by converting it to datetime objects.
    Rows with invalid or missing dates are excluded from the returned dataframe.

    Parameters:
    - ticker_dataframe (pd.DataFrame): The input dataframe containing ticker information.

    Returns:
    - pd.DataFrame: A new dataframe with the 'Date' column normalized and invalid rows excluded.
    """
    # Step 1: Ensure 'Date' column exists
    if "Date" not in ticker_dataframe.columns:
        logger.error("The dataframe does not contain a 'Date' column.")
        return pd.DataFrame()  # Return empty dataframe

    # Step 2: Convert 'Date' column to datetime, coercing errors to NaT
    ticker_dataframe["Date"] = pd.to_datetime(ticker_dataframe["Date"], errors="coerce")

    # Step 3: Identify rows with invalid dates (NaT)
    invalid_dates = ticker_dataframe[ticker_dataframe["Date"].isna()]
    if not invalid_dates.empty:
        logger.error(
            "Some dates could not be parsed and are set to NaT. These rows will be excluded from the final dataframe."
        )
        print("\n=== Invalid Dates ===")
        print(invalid_dates)

    # Step 4: Exclude rows with invalid dates
    normalized_df = ticker_dataframe[ticker_dataframe["Date"].notna()].copy()

    # Optional: Reset index for cleanliness
    normalized_df.reset_index(drop=True, inplace=True)

    return normalized_df


# -------------------- Test Data Creation --------------------


def create_test_data() -> pd.DataFrame:
    """
    Creates a test dataframe with various edge cases to test the normalize_dates function.

    Returns:
    - pd.DataFrame: The test dataframe.
    """
    # Define test data as a list of dictionaries
    test_data = [
        # Valid rows
        {
            "Ticker": "^OEX",
            "Price": 2894.939941,
            "Date": "2024-11-26",
            "QuoteType": "INDEX",
            "Currency": "USD",
        },
        {
            "Ticker": "AUDGBP=X",
            "Price": 0.515710,
            "Date": "2024-11-26",
            "QuoteType": "CURRENCY",
            "Currency": "GBP",
        },
        {
            "Ticker": "IHCU.L",
            "Price": 912.125000,
            "Date": "2024-11-26",
            "QuoteType": "ETF",
            "Currency": "GBp",
        },
        {
            "Ticker": "VUKG.L",
            "Price": 41.895000,
            "Date": "2024-11-26",
            "QuoteType": "ETF",
            "Currency": "GBP",
        },
        {
            "Ticker": "SEE.L",
            "Price": 3.200000,
            "Date": "2024-11-26",
            "QuoteType": "EQUITY",
            "Currency": "GBp",
        },
        {
            "Ticker": "SQX.CN",
            "Price": 0.005000,
            "Date": "2024-10-29",
            "QuoteType": "EQUITY",
            "Currency": "CAD",
        },
        {
            "Ticker": "HMWO.L",
            "Price": 2865.625000,
            "Date": "2024-10-29",
            "QuoteType": "ETF",
            "Currency": "GBp",
        },
        {
            "Ticker": "IHCU.L",
            "Price": 900.875000,
            "Date": "2024-10-29",
            "QuoteType": "ETF",
            "Currency": "GBp",
        },
        {
            "Ticker": "GL1S.MU",
            "Price": 36.029999,
            "Date": "2024-10-28",
            "QuoteType": "ETF",
            "Currency": "EUR",
        },
        {
            "Ticker": "LAU.AX",
            "Price": 0.885000,
            "Date": "2024-10-28",
            "QuoteType": "EQUITY",
            "Currency": "AUD",
        },
        {
            "Ticker": "EURGBP=X",
            "Price": 0.85,
            "Date": "2024-10-28",
            "QuoteType": "CURRENCY",
            "Currency": "GBP",
        },
        {
            "Ticker": "GBPUSD=X",
            "Price": 1.30,
            "Date": "2024-10-28",
            "QuoteType": "CURRENCY",
            "Currency": "GBP",
        },
        {
            "Ticker": "UNKNOWN.TICKER",
            "Price": 100,
            "Date": "2024-10-28",
            "QuoteType": "ETF",
            "Currency": "XYZ",
        },
        {
            "Ticker": "MISSINGRATE.L",
            "Price": 50,
            "Date": "2024-10-27",
            "QuoteType": "EQUITY",
            "Currency": "JPY",
        },
        {
            "Ticker": "JPYGBP=X",
            "Price": 0.0065,
            "Date": "2024-10-27",
            "QuoteType": "CURRENCY",
            "Currency": "GBP",
        },
        {
            "Ticker": "NOPRICE.L",
            "Price": None,
            "Date": "2024-10-26",
            "QuoteType": "EQUITY",
            "Currency": "GBP",
        },
        {
            "Ticker": "INDEX.CURRENCY",
            "Price": 1000,
            "Date": "2024-10-26",
            "QuoteType": "CURRENCY",
            "Currency": "USD",
        },
        # Edge Cases
        {
            "Ticker": "GL1S.MU",
            "Price": 35.849998,
            "Date": "2024-11-25",
            "QuoteType": "ETF",
            "Currency": "EUR",
        },
        {
            "Ticker": "LAU.AX",
            "Price": 0.890000,
            "Date": "2024-11-25",
            "QuoteType": "EQUITY",
            "Currency": "AUD",
        },
        {
            "Ticker": "GL1S.MU",
            "Price": 35.884998,
            "Date": "2024-11-24",
            "QuoteType": "ETF",
            "Currency": "EUR",
        },
        {
            "Ticker": "LAU.AX",
            "Price": 0.880000,
            "Date": "2024-11-24",
            "QuoteType": "EQUITY",
            "Currency": "AUD",
        },
        {
            "Ticker": "GL1S.MU",
            "Price": 35.884998,
            "Date": "2024-11-17",
            "QuoteType": "ETF",
            "Currency": "EUR",
        },
        {
            "Ticker": "LAU.AX",
            "Price": 0.875000,
            "Date": "2024-11-17",
            "QuoteType": "EQUITY",
            "Currency": "AUD",
        },
        {
            "Ticker": "GL1S.MU",
            "Price": 35.685001,
            "Date": "2024-11-10",
            "QuoteType": "ETF",
            "Currency": "EUR",
        },
        {
            "Ticker": "LAU.AX",
            "Price": 0.870000,
            "Date": "2024-11-10",
            "QuoteType": "EQUITY",
            "Currency": "AUD",
        },
        {
            "Ticker": "GL1S.MU",
            "Price": 35.599998,
            "Date": "2024-11-03",
            "QuoteType": "ETF",
            "Currency": "EUR",
        },
        {
            "Ticker": "LAU.AX",
            "Price": 0.900000,
            "Date": "2024-11-03",
            "QuoteType": "EQUITY",
            "Currency": "AUD",
        },
        {
            "Ticker": "GL1S.MU",
            "Price": 35.544998,
            "Date": "2024-11-24",
            "QuoteType": "ETF",
            "Currency": "EUR",
        },
        {
            "Ticker": "LAU.AX",
            "Price": 0.855000,
            "Date": "2024-11-07",
            "QuoteType": "EQUITY",
            "Currency": "AUD",
        },
        {
            "Ticker": "GL1S.MU",
            "Price": 35.419998,
            "Date": "2024-11-17",
            "QuoteType": "ETF",
            "Currency": "EUR",
        },
        {
            "Ticker": "LAU.AX",
            "Price": 0.880000,
            "Date": "2024-11-17",
            "QuoteType": "EQUITY",
            "Currency": "AUD",
        },
        {
            "Ticker": "GL1S.MU",
            "Price": 35.075001,
            "Date": "2024-11-04",
            "QuoteType": "ETF",
            "Currency": "EUR",
        },
        {
            "Ticker": "LAU.AX",
            "Price": 0.820000,
            "Date": "2024-11-03",
            "QuoteType": "EQUITY",
            "Currency": "AUD",
        },
        {
            "Ticker": "GL1S.MU",
            "Price": 35.160000,
            "Date": "2024-11-03",
            "QuoteType": "ETF",
            "Currency": "EUR",
        },
        {
            "Ticker": "GL1S.MU",
            "Price": 35.360001,
            "Date": "2024-10-31",
            "QuoteType": "ETF",
            "Currency": "EUR",
        },
        {
            "Ticker": "LAU.AX",
            "Price": 0.850000,
            "Date": "2024-10-31",
            "QuoteType": "EQUITY",
            "Currency": "AUD",
        },
        {
            "Ticker": "GL1S.MU",
            "Price": 35.130001,
            "Date": "2024-10-30",
            "QuoteType": "ETF",
            "Currency": "EUR",
        },
        {
            "Ticker": "LAU.AX",
            "Price": 0.850000,
            "Date": "2024-10-30",
            "QuoteType": "EQUITY",
            "Currency": "AUD",
        },
        {
            "Ticker": "GL1S.MU",
            "Price": 35.705002,
            "Date": "2024-10-29",
            "QuoteType": "ETF",
            "Currency": "EUR",
        },
        {
            "Ticker": "LAU.AX",
            "Price": 0.865000,
            "Date": "2024-10-29",
            "QuoteType": "EQUITY",
            "Currency": "AUD",
        },
        {
            "Ticker": "LAU.AX",
            "Price": 0.885000,
            "Date": "2024-10-28",
            "QuoteType": "EQUITY",
            "Currency": "AUD",
        },
        {
            "Ticker": "GL1S.MU",
            "Price": 36.029999,
            "Date": "2024-10-28",
            "QuoteType": "ETF",
            "Currency": "EUR",
        },
        # Malformed Date Rows (Edge Cases)
        {
            "Ticker": "BADDATE1.L",
            "Price": 10.0,
            "Date": "2024/13/40",
            "QuoteType": "ETF",
            "Currency": "EUR",
        },  # Invalid date
        {
            "Ticker": "BADDATE2.L",
            "Price": 20.0,
            "Date": "NotADate",
            "QuoteType": "EQUITY",
            "Currency": "USD",
        },  # Invalid date
        {
            "Ticker": "BADDATE3.L",
            "Price": 30.0,
            "Date": "",
            "QuoteType": "ETF",
            "Currency": "GBP",
        },  # Missing date
    ]

    # Create DataFrame
    test_df = pd.DataFrame(test_data)

    return test_df


# -------------------- Execution and Testing --------------------


def main():
    """
    Main function to execute the normalize_dates function with test data.
    """
    # Create test data
    ticker_dataframe: pd.DataFrame = create_test_data()

    # Display initial DataFrame dtypes
    print("=== Initial DataFrame Dtypes ===")
    print(ticker_dataframe.dtypes)

    # Display initial DataFrame (optional)
    print("\n=== Initial DataFrame (First 10 Rows) ===")
    print(ticker_dataframe.head(10))

    # Normalize dates
    normalized_df: pd.DataFrame = normalize_dates(ticker_dataframe)

    # Display final DataFrame dtypes
    print("\n=== Final DataFrame Dtypes ===")
    print(normalized_df.dtypes)

    # Display a summary of the normalization
    print("\n=== Summary of Date Normalization ===")
    total_rows = len(ticker_dataframe)
    valid_rows = len(normalized_df)
    invalid_rows = total_rows - valid_rows
    print(f"Total Rows in Original DataFrame: {total_rows}")
    print(f"Total Rows After Normalization: {valid_rows}")
    print(f"Total Rows Excluded Due to Invalid Dates: {invalid_rows}")

    # Display the first few rows of the normalized DataFrame
    print("\n=== Normalized DataFrame (First 20 Rows) ===")
    print(normalized_df.head(20))

    # Display remaining issues (if any)
    print("\n=== Remaining Issues (if any) ===")
    # Check for any remaining 'NaN' dates
    nan_dates = normalized_df[normalized_df["Date"].isna()]
    if not nan_dates.empty:
        print("Rows with NaN Dates:")
        print(nan_dates)
    else:
        print("No rows with NaN Dates.")


if __name__ == "__main__":
    main()
