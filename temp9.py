import pandas as pd
import logging
from typing import Any

# -------------------- Logger Configuration --------------------

# Configure the logger to display debug, info, warnings, and errors with timestamps
logging.basicConfig(
    level=logging.DEBUG,  # Set to DEBUG to capture all levels of logs
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
        logger.debug(f"Converting pd.Timestamp to ISO format: {date}")
        return date.date().isoformat()
    elif isinstance(date, str):
        logger.debug(f"Date is a string: {date}")
        return date
    else:
        logger.debug(f"Date type is unrecognized: {type(date)}")
        return "Unknown Date"


# -------------------- Date Normalization Function --------------------



def normalise_dates(df, date_column="Date"):
    """
    Normalizes all date values in the specified column of the DataFrame to UTC and returns
    them in 'yyyy-mm-dd' format. Invalid dates are handled gracefully and returned as a separate DataFrame.

    Parameters:
    - df: pandas DataFrame containing the date column.
    - date_column: The name of the date column. Defaults to 'Date'.

    Returns:
    - normalized_df: DataFrame with normalized dates in 'yyyy-mm-dd' format.
    - invalid_dates_df: DataFrame containing rows with invalid or unparseable dates.
    """

    normalized_dates = []
    invalid_rows = []

    for index, row in df.iterrows():
        date_value = row[date_column]

        try:
            # Parse date, coercing invalid formats to NaT
            parsed_date = pd.to_datetime(date_value, errors="coerce")

            if pd.isna(parsed_date):
                # If parsing fails, log the invalid date
                invalid_rows.append(row)
                normalized_dates.append(None)
            else:
                # If timezone-naive, assume it's in UTC and localize it
                if parsed_date.tzinfo is None or parsed_date.tz is None:
                    localized_date = parsed_date.tz_localize(
                        "UTC", ambiguous="NaT", nonexistent="NaT"
                    )
                else:
                    # If timezone-aware, convert it to UTC
                    localized_date = parsed_date.tz_convert("UTC")

                # Append the date in 'yyyy-mm-dd' format
                normalized_dates.append(localized_date.strftime("%Y-%m-%d"))

        except Exception as e:
            # If any unexpected error occurs, log the invalid date
            invalid_rows.append(row)
            normalized_dates.append(None)

    # Add the normalized dates to the DataFrame
    df[date_column] = normalized_dates

    # Convert the 'Date' column explicitly to datetime (in case some values are still objects)
    df[date_column] = pd.to_datetime(
        df[date_column], errors="coerce", format="%Y-%m-%d"
    )

    # Create a DataFrame of invalid rows
    invalid_dates_df = pd.DataFrame(invalid_rows, columns=df.columns)

    # Print column types after transformation
    print("\nColumn Types After Transformation:")
    print(df.dtypes)

    # Print rows that were not successfully transformed
    if not invalid_dates_df.empty:
        print("\nRows with Invalid or Untransformed Dates:")
        print(invalid_dates_df)
    else:
        print("\nAll rows have been successfully transformed.")

    return df


# Usage example:
# normalized_df, invalid_dates_df = normalize_dates_to_utc(df)
# You can then inspect normalized_df for the transformed dates and invalid_dates_df for problematic entries.
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
            "Date": "2024-11-27 20:24:50,880",
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

    logger.debug("Test data created with edge cases.")
    return test_df


# -------------------- Execution and Testing --------------------


def main():
    """
    Main function to execute the normalize_dates function with test data.
    """
    logger.debug("Program started.")

    # Create test data
    ticker_dataframe: pd.DataFrame = create_test_data()

    # Display initial DataFrame dtypes
    logger.info("=== Initial DataFrame Dtypes ===")
    logger.info("\n" + str(ticker_dataframe.dtypes))

    # Display initial DataFrame (optional)
    logger.info("\n=== Initial DataFrame (First 10 Rows) ===")
    logger.debug("\n" + str(ticker_dataframe.head(10)))

    # Normalize dates
    normalized_df: pd.DataFrame = normalise_dates(ticker_dataframe, "Date")

    logger.debug("Program completed.")


if __name__ == "__main__":
    main()
