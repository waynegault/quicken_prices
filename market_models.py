"""Validated market-data records."""

from datetime import datetime, timezone

from pydantic import Field, field_validator

from settings import StrictModel, TickerSymbol


class TickerMetadata(StrictModel):
    ticker: TickerSymbol
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

    @field_validator("earliest_date")
    @classmethod
    def require_utc_aware_earliest_date(
        cls, earliest_date: datetime | None
    ) -> datetime | None:
        if earliest_date is None:
            return None
        if earliest_date.tzinfo is None or earliest_date.utcoffset() is None:
            raise ValueError("earliest_date must include a timezone")
        return earliest_date.astimezone(timezone.utc)
