"""Validated market-data records."""

from datetime import datetime

from pydantic import Field, field_validator

from settings import StrictModel


class TickerMetadata(StrictModel):
    ticker: str = Field(min_length=1)
    earliest_date: datetime | None
    instrument_type: str = Field(min_length=1)
    currency: str = Field(min_length=1)
    current_price: float

    @field_validator("ticker")
    @classmethod
    def uppercase_ticker(cls, ticker: str) -> str:
        return ticker.upper()

    @field_validator("currency")
    @classmethod
    def validate_currency(cls, currency: str) -> str:
        if currency != "Unknown" and len(currency) != 3:
            raise ValueError("currency must be a three-character code")
        return currency
