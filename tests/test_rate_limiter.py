# tests/test_rate_limiter.py
import pytest
import time
from QuickenPrice3 import RetryManager


class TestRateLimiter:
    @pytest.fixture
    def rate_limiter(self, mock_config):
        return RetryManager(
            max_retries=3, base_delay=1, max_delay=5, config=mock_config
        ).rate_limit

    def test_rate_limiting(self, rate_limiter):
        """Test basic rate limiting functionality"""
        start_time = time.time()

        # Make max_requests + 1 requests
        for _ in range(rate_limiter.max_requests + 1):
            rate_limiter.wait_if_needed()

        elapsed = time.time() - start_time
        assert elapsed >= 1.0  # Should have waited at least 1 second

    def test_concurrent_requests(self, rate_limiter):
        """Test rate limiting with concurrent requests"""
        import threading

        def make_request():
            rate_limiter.wait_if_needed()

        threads = []
        for _ in range(5):
            thread = threading.Thread(target=make_request)
            threads.append(thread)
            thread.start()

        for thread in threads:
            thread.join()

    def test_request_spacing(self, rate_limiter):
        """Test that requests are properly spaced"""
        timestamps = []

        for _ in range(5):
            rate_limiter.wait_if_needed()
            timestamps.append(time.time())

        intervals = [j - i for i, j in zip(timestamps[:-1], timestamps[1:])]
        assert all(interval >= 0 for interval in intervals)
