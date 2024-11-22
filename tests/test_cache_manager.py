# tests/test_cache_manager.py
import pytest
import time
from pathlib import Path
from QuickenPrice3 import CacheManager


class TestCacheManager:
    @pytest.fixture
    def cache_manager(self, tmp_path, mock_config):
        return CacheManager(
            config_path=tmp_path,
            max_age_hours=1,
            cleanup_threshold=5,
            config=mock_config,
        )

    def test_save_and_load(self, cache_manager, sample_price_data):
        """Test basic save and load functionality"""
        cache_path = cache_manager.get_cache_path("TEST", "2024-01-01", "2024-01-10")

        # Save data
        assert cache_manager.save_to_cache(cache_path, (sample_price_data, "USD"))

        # Load data
        loaded_data = cache_manager.load_from_cache(cache_path)
        assert loaded_data is not None
        df, currency = loaded_data
        assert df.equals(sample_price_data)
        assert currency == "USD"

    def test_cache_expiration(self, cache_manager, sample_price_data):
        """Test cache expiration"""
        cache_path = cache_manager.get_cache_path("TEST", "2024-01-01", "2024-01-10")

        # Save data
        cache_manager.save_to_cache(cache_path, (sample_price_data, "USD"))

        # Modify metadata to make cache expired
        cache_manager.metadata["entries"][str(cache_path)][
            "created"
        ] = datetime.now() - timedelta(hours=2)

        # Load should return None for expired cache
        assert cache_manager.load_from_cache(cache_path) is None

    def test_cache_cleanup(self, cache_manager, sample_price_data):
        """Test cache cleanup functionality"""
        # Create multiple cache entries
        for i in range(10):
            cache_path = cache_manager.get_cache_path(
                f"TEST{i}", "2024-01-01", "2024-01-10"
            )
            cache_manager.save_to_cache(cache_path, (sample_price_data, "USD"))

        # Trigger cleanup
        initial_count = len(list(cache_manager.cache_dir.glob("*.pkl")))
        cache_manager._cleanup()
        final_count = len(list(cache_manager.cache_dir.glob("*.pkl")))

        assert final_count < initial_count
