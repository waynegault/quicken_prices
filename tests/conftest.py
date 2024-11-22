# tests/conftest.py
import sys
from pathlib import Path

# Add the root directory to Python's path
root_dir = Path(__file__).parent.parent
sys.path.append(str(root_dir))

# Keep your existing fixtures
import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta


@pytest.fixture
def mock_config():
    """Shared config fixture"""

    class MockConfig:
        def __init__(self):
            self.PATH = str(Path.cwd() / "test_data")
            self.TICKERS = ["AAPL", "MSFT", "GOOGL"]
            self.CURRENCY_SYMBOLS = {"USD": "USDGBP=X", "GBp": None, "EUR": "EURGBP=X"}
            self.cache_settings = {
                "max_age_hours": 168,
                "cleanup_threshold": 200,
            }

    return MockConfig()


@pytest.fixture
def sample_price_data():
    """Sample price data fixture"""
    dates = pd.date_range(start="2024-01-01", end="2024-01-10", freq="D")
    return pd.DataFrame(
        {"Close": np.random.uniform(10, 100, size=len(dates)), "Date": dates}
    )


@pytest.fixture
def exchange_rates():
    """Sample exchange rates fixture"""
    dates = pd.date_range(start="2024-01-01", end="2024-01-10", freq="D")
    return {
        "USD": {date: 0.78 + np.random.uniform(-0.02, 0.02) for date in dates},
        "EUR": {date: 0.85 + np.random.uniform(-0.02, 0.02) for date in dates},
    }
