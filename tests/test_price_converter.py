# tests/test_price_converter.py
import pytest
from datetime import datetime
import pandas as pd
import numpy as np
from QuickenPrice3 import PriceConverter


class TestPriceConverter:
    @pytest.fixture
    def converter(self, mock_config):
        return PriceConverter(config=mock_config)

    def test_gbp_conversion(self, converter):
        """Test conversion from GBp to GBP"""
        price = 1234.56
        result = converter.convert_price(price, "GBp", None, datetime.now(), "TEST")
        assert result == pytest.approx(12.3456)

    def test_usd_conversion(self, converter, exchange_rates):
        """Test USD to GBP conversion"""
        price = 100.00
        date = pd.Timestamp("2024-01-05")
        result = converter.convert_price(price, "USD", exchange_rates, date, "TEST")
        expected_rate = exchange_rates["USD"][date]
        assert result == pytest.approx(price * expected_rate)

    def test_invalid_currency(self, converter, exchange_rates):
        """Test handling of invalid currency"""
        price = 100.00
        result = converter.convert_price(
            price, "INVALID", exchange_rates, datetime.now(), "TEST"
        )
        assert result == price

    def test_missing_exchange_rate(self, converter, exchange_rates):
        """Test handling of missing exchange rate"""
        price = 100.00
        future_date = pd.Timestamp("2025-01-01")
        result = converter.convert_price(
            price, "USD", exchange_rates, future_date, "TEST"
        )
        assert result is not None

    @pytest.mark.parametrize(
        "price,currency", [(0, "USD"), (-100, "GBp"), (float("inf"), "EUR")]
    )
    def test_edge_cases(self, converter, exchange_rates, price, currency):
        """Test edge cases with various prices"""
        result = converter.convert_price(
            price, currency, exchange_rates, datetime.now(), "TEST"
        )
        assert isinstance(result, (int, float))
