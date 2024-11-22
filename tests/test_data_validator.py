# tests/test_data_validator.py
import pytest
import pandas as pd
import numpy as np
from QuickenPrice3 import DataValidator


class TestDataValidator:
    @pytest.fixture
    def validator(self, mock_config):
        return DataValidator(config=mock_config)

    def test_valid_data(self, validator, sample_price_data):
        """Test validation of valid data"""
        is_valid, errors = validator.validate_dataframe(sample_price_data, "TEST")
        assert is_valid
        assert not errors

    def test_missing_columns(self, validator):
        """Test validation with missing columns"""
        df = pd.DataFrame(
            {
                "Close": [1.0, 2.0],
                # Missing other required columns
            }
        )
        is_valid, errors = validator.validate_dataframe(df, "TEST")
        assert not is_valid
        assert any("Missing columns" in error for error in errors)

    def test_invalid_prices(self, validator, sample_price_data):
        """Test validation with invalid prices"""
        df = sample_price_data.copy()
        df.loc[df.index[0], "Close"] = -1.0

        is_valid, errors = validator.validate_dataframe(df, "TEST")
        assert not is_valid
        assert any("Invalid price" in error for error in errors)

    def test_missing_values(self, validator, sample_price_data):
        """Test validation with missing values"""
        df = sample_price_data.copy()
        df.loc[df.index[0], "Close"] = np.nan

        is_valid, errors = validator.validate_dataframe(df, "TEST")
        assert not is_valid
        assert any("Missing values" in error for error in errors)

    @pytest.mark.parametrize(
        "test_input,expected_valid",
        [
            (pd.DataFrame(), False),  # Empty DataFrame
            (None, False),  # None input
            (pd.DataFrame({"Close": []}), False),  # Empty columns
        ],
    )
    def test_edge_cases(self, validator, test_input, expected_valid):
        """Test validation edge cases"""
        is_valid, errors = validator.validate_dataframe(test_input, "TEST")
        assert is_valid == expected_valid
