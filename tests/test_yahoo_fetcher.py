# run with:
#
# pytest test_yahoo_fetcher.py -v

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from pathlib import Path
import pickle
import logging
import psutil
from QuickenPrice3 import YahooFinanceFetcher, Config


# Mock Config class for testing
class Config:
    """Mock configuration class for testing."""

    def __init__(self):
        self.PATH = str(Path.cwd() / "test_data")
        self.TICKERS = ["AAPL", "MSFT", "GOOGL"]
        self.CURRENCY_SYMBOLS = {"USD": "USDGBP=X", "GBp": None, "EUR": "EURGBP=X"}
        self.cache_settings = {
            "max_age_hours": 168,
            "cleanup_threshold": 200,
        }
        self.data_file_name = "test_data.csv"
        self.log_file_name = "test_log.log"
        self.log_rotation = {
            "max_bytes": 5 * 1024 * 1024,
            "backup_count": 5,
        }


class TestYahooFinanceFetcher:
    @pytest.fixture
    def setup_fetcher(self, tmp_path):
        """Setup a YahooFinanceFetcher instance with a temporary cache directory"""
        config = Config()  # Your config class
        
        # Ensure logging is configured
        logging.basicConfig(level=logging.INFO)
        
        fetcher = YahooFinanceFetcher(config=config)
        fetcher.cache_dir = tmp_path  # Use pytest's temporary directory
        
        # Explicitly configure the logger
        fetcher.logger = logging.getLogger("YahooFinanceFetcher")
        fetcher.logger.setLevel(logging.INFO)
        
        return fetcher

    @pytest.fixture
    def sample_data(self):
        """Create sample historical data"""
        dates = pd.date_range(start="2024-01-01", end="2024-01-10", freq="D")
        data = pd.DataFrame(
            {
                "Close": np.random.uniform(10, 100, size=len(dates)),
                "Open": np.random.uniform(10, 100, size=len(dates)),
                "High": np.random.uniform(10, 100, size=len(dates)),
                "Low": np.random.uniform(10, 100, size=len(dates)),
                "Volume": np.random.randint(1000, 10000, size=len(dates)),
            },
            index=dates,
        )
        return data

    def create_backup_file(
        self, tmp_path, ticker, hist_data, currency="USD", age_hours=0
    ):
        """Helper to create a backup file"""
        backup_data = {
            "timestamp": datetime.now() - timedelta(hours=age_hours),
            "data": hist_data,
            "currency": currency,
        }
        backup_file = tmp_path / f"{ticker}_backup.pkl"
        with open(backup_file, "wb") as f:
            pickle.dump(backup_data, f)
        return backup_file

    def test_successful_recovery(self, setup_fetcher, sample_data):
        """Test successful recovery from a valid backup"""
        fetcher = setup_fetcher
        ticker = "AAPL"
        start_date = "2024-01-02"
        end_date = "2024-01-08"

        # Create backup file
        backup_file = self.create_backup_file(fetcher.cache_dir, ticker, sample_data)

        # Test recovery
        hist, currency = fetcher._recover_from_backup(ticker, start_date, end_date)

        assert hist is not None
        assert currency == "USD"
        assert not hist.empty
        assert len(hist) == 7  # Should include dates from 01-02 to 01-08
        assert hist.index.min().strftime("%Y-%m-%d") == start_date
        assert hist.index.max().strftime("%Y-%m-%d") == end_date

    def test_expired_backup(self, setup_fetcher, sample_data):
        """Test recovery from an expired backup (> 24 hours old)"""
        fetcher = setup_fetcher
        ticker = "AAPL"

        # Create expired backup (25 hours old)
        self.create_backup_file(fetcher.cache_dir, ticker, sample_data, age_hours=25)

        # Test recovery
        hist, currency = fetcher._recover_from_backup(
            ticker, "2024-01-02", "2024-01-08"
        )

        assert hist is None
        assert currency is None

    def test_invalid_backup_data(self, setup_fetcher, tmp_path):
        """Test recovery with invalid backup data structure"""
        fetcher = setup_fetcher
        ticker = "AAPL"

        # Create invalid backup data
        invalid_data = {"invalid": "data"}
        backup_file = tmp_path / f"{ticker}_backup.pkl"
        with open(backup_file, "wb") as f:
            pickle.dump(invalid_data, f)

        # Test recovery
        hist, currency = fetcher._recover_from_backup(
            ticker, "2024-01-02", "2024-01-08"
        )

        assert hist is None
        assert currency is None

    def test_missing_backup(self, setup_fetcher):
        """Test recovery when backup file doesn't exist"""
        fetcher = setup_fetcher
        ticker = "NONEXISTENT"

        hist, currency = fetcher._recover_from_backup(
            ticker, "2024-01-02", "2024-01-08"
        )

        assert hist is None
        assert currency is None

    def test_corrupted_backup(self, setup_fetcher, tmp_path):
        """Test recovery with corrupted backup file"""
        fetcher = setup_fetcher
        ticker = "AAPL"

        # Create corrupted backup file
        backup_file = tmp_path / f"{ticker}_backup.pkl"
        with open(backup_file, "wb") as f:
            f.write(b"corrupted data")

        # Test recovery
        hist, currency = fetcher._recover_from_backup(
            ticker, "2024-01-02", "2024-01-08"
        )

        assert hist is None
        assert currency is None

    def test_date_range_mismatch(self, setup_fetcher, sample_data):
        """Test recovery when backup doesn't cover requested date range"""
        fetcher = setup_fetcher
        ticker = "AAPL"
        
        # Create backup with limited date range (2024-01-01 to 2024-01-10)
        self.create_backup_file(fetcher.cache_dir, ticker, sample_data)

        # Test recovery with dates outside backup range
        hist, currency = fetcher._recover_from_backup(
            ticker,
            "2023-12-01",  # Date before our backup range
            "2024-02-01"   # Date after our backup range
        )
        
        # Add this debug print
        if hist is not None:
            print(f"Backup date range: {hist.index.min()} to {hist.index.max()}")
        
        assert hist is None, "Should return None for date range mismatch"
        assert currency is None, "Should return None for date range mismatch"

    def test_backup_with_missing_columns(self, setup_fetcher):
        """Test recovery with backup missing required columns"""
        fetcher = setup_fetcher
        ticker = "AAPL"
        
        # Create data with missing columns
        dates = pd.date_range(start='2024-01-02', end='2024-01-08', freq='D')
        incomplete_data = pd.DataFrame({
            'Close': np.random.uniform(10, 100, size=len(dates))
        }, index=dates)
        
        # The validation should fail because required columns are missing
        self.create_backup_file(fetcher.cache_dir, ticker, incomplete_data)

        # Test recovery
        hist, currency = fetcher._recover_from_backup(
            ticker, "2024-01-02", "2024-01-08"
        )
        
        # Add this debug print
        if hist is not None:
            print(f"Available columns: {hist.columns.tolist()}")
        
        assert hist is None, "Should return None for missing required columns"
        assert currency is None, "Should return None for missing required columns"

    @pytest.mark.parametrize("test_currency", ["USD", "GBP", "EUR"])
    def test_different_currencies(self, setup_fetcher, sample_data, test_currency):
        """Test recovery with different currencies"""
        fetcher = setup_fetcher
        ticker = "AAPL"

        self.create_backup_file(
            fetcher.cache_dir, ticker, sample_data, currency=test_currency
        )

        hist, currency = fetcher._recover_from_backup(
            ticker, "2024-01-02", "2024-01-08"
        )

        assert hist is not None
        assert currency == test_currency

    def test_recovery_memory_usage(self, setup_fetcher, sample_data):
        """Test memory usage during recovery"""
        import psutil
        
        fetcher = setup_fetcher
        ticker = "AAPL"
        self.create_backup_file(fetcher.cache_dir, ticker, sample_data)
        
        process = psutil.Process()
        initial_memory = process.memory_info().rss
        
        hist, currency = fetcher._recover_from_backup(
            ticker, "2024-01-02", "2024-01-08"
        )
        
        final_memory = process.memory_info().rss
        memory_difference = final_memory - initial_memory
        
        # Check that memory usage doesn't exceed reasonable limits
        assert memory_difference < 1024 * 1024 * 100  # 100MB limit
        
    def test_recovery_logging(self, setup_fetcher, sample_data, caplog):
        """Test that recovery operations are properly logged"""
        with caplog.at_level(logging.INFO):
            fetcher = setup_fetcher
            ticker = "AAPL"
            
            # Create and verify backup file
            backup_file = self.create_backup_file(fetcher.cache_dir, ticker, sample_data)
            assert backup_file.exists(), "Backup file was not created"
            
            # Attempt recovery
            hist, currency = fetcher._recover_from_backup(
                ticker, "2024-01-02", "2024-01-08"
            )
            
            # Check the logs
            print("Captured logs:", caplog.text)  # Debug print
            
            # Check both the success message and the result
            assert hist is not None, "Recovery should succeed"
            assert currency is not None, "Currency should be returned"
            assert "Successfully recovered" in caplog.text, "Success message not logged"
            assert ticker in caplog.text, f"Ticker {ticker} not found in logs" 
            
    def test_failed_recovery_logging(self, setup_fetcher, caplog):
        """Test logging when recovery fails"""
        with caplog.at_level(logging.INFO):
            fetcher = setup_fetcher
            ticker = "NONEXISTENT"
            
            # Attempt recovery of non-existent backup
            hist, currency = fetcher._recover_from_backup(
                ticker, "2024-01-02", "2024-01-08"
            )
            
            # Check the logs
            assert hist is None, "Recovery should fail"
            assert currency is None, "Currency should be None"
            assert "No backup file found" in caplog.text, "Failure message not logged"
            assert ticker in caplog.text, f"Ticker {ticker} not found in logs"
