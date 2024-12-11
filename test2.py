from datetime import date, timedelta
import pandas as pd
from typing import Dict, Any, Tuple
import pytest

def get_date_range(config: Dict[str, Any]):
    """
    Calculates the start and end dates for data collection, and the number of business days between them,
    considering weekends.

    Args:
        config (Dict[str, Any]): Configuration dictionary containing "collection_period_years" key.

    Returns:
        Tuple[datetime, datetime, int]: Tuple containing start date (datetime64[ns]), end date (datetime64[ns]), and number of business days.
    """

    period_year = config["collection_period_years"]
    period_days = period_year * 365

    # Start with tomorrow's date
    potential_start_date = date.today() + timedelta(days=1)

    # Adjust for weekends (Monday as start of week)
    if potential_start_date.weekday() == 5:  # Friday
        start_date = potential_start_date + timedelta(days=3)  # Following Monday
    elif potential_start_date.weekday() == 6:  # Saturday
        start_date = potential_start_date + timedelta(days=2)  # Following Monday
    else:
        start_date = potential_start_date

    end_date = start_date + timedelta(
        days=int(period_days) - 1
    )  # Adjust for inclusive start, exclusive end

    # Convert start and end dates to datetime64[ns] type directly using pandas
    start_datetime = pd.to_datetime(start_date, utc=True)
    end_datetime = pd.to_datetime(end_date, utc=True)

    # Calculate business days
    business_days = len(pd.bdate_range(start_datetime, end_datetime))

    return start_datetime, end_datetime, business_days


def test_weekday_start():
    config = {"collection_period_years": 1}
    start_date, end_date, _ = get_date_range(config)

    # Assuming today is a Monday
    today = date.today()
    expected_start_date = today + timedelta(days=1)
    expected_end_date = expected_start_date + timedelta(days=364)

    assert start_date == pd.to_datetime(expected_start_date, utc=True)
    assert end_date == pd.to_datetime(expected_end_date, utc=True)


def test_friday_start():
    config = {"collection_period_years": 1}
    today = date(2023, 11, 24)  # A Friday
    expected_start_date = today + timedelta(days=3)
    expected_end_date = expected_start_date + timedelta(days=364)

    start_date, end_date, _ = get_date_range(config)

    assert start_date == pd.to_datetime(expected_start_date, utc=True)
    assert end_date == pd.to_datetime(expected_end_date, utc=True)


def test_saturday_start():
    config = {"collection_period_years": 1}
    today = date(2023, 11, 25)  # A Saturday
    expected_start_date = today + timedelta(days=2)
    expected_end_date = expected_start_date + timedelta(days=364)

    start_date, end_date, _ = get_date_range(config)

    assert start_date == pd.to_datetime(expected_start_date, utc=True)
    assert end_date == pd.to_datetime(expected_end_date, utc=True)


if __name__ == "__main__":
    pytest.main()
