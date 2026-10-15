"""Strict configuration models for QuickenPrices."""

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator


class StrictModel(BaseModel):
    model_config = ConfigDict(strict=True, extra="forbid")


class RetrySettings(StrictModel):
    max_retries: int = Field(default=3, ge=1)
    retry_delay: float = Field(default=2, ge=0)
    backoff: float = Field(default=2, ge=1)


class LoggingLevels(StrictModel):
    file: Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"] = "DEBUG"
    terminal: Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"] = "DEBUG"

    @field_validator("file", "terminal", mode="before")
    @classmethod
    def uppercase_level(cls, value: str) -> str:
        if isinstance(value, str):
            return value.upper()
        return value


class LogFileSettings(StrictModel):
    max_bytes: int = Field(default=5_242_880, gt=0)
    backup_count: int = Field(default=5, ge=0)


class PathSettings(StrictModel):
    data_file: str = "data.csv"
    log_file: str = "prices.log"
    cache: str = "cache"
    base: str
    quicken: str = ""

    @field_validator("data_file", "log_file", "cache", "base")
    @classmethod
    def non_empty_path(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("path values must not be empty")
        return value


class CacheSettings(StrictModel):
    max_age_days: int = Field(default=30, ge=1)


class AppSettings(StrictModel):
    tickers: list[str] = Field(default_factory=lambda: ["^FTSE"], min_length=1)
    home_currency: str = Field(default="GBP", min_length=3, max_length=3)
    collection_period_years: float = Field(default=0.08, gt=0)
    retries: RetrySettings = Field(default_factory=RetrySettings)
    logging_level: LoggingLevels = Field(default_factory=LoggingLevels)
    log_file: LogFileSettings = Field(default_factory=LogFileSettings)
    paths: PathSettings
    cache: CacheSettings = Field(default_factory=CacheSettings)

    @field_validator("tickers")
    @classmethod
    def validate_tickers(cls, tickers: list[str]) -> list[str]:
        if any(not ticker.strip() for ticker in tickers):
            raise ValueError("tickers must not contain empty values")
        normalized_tickers = [ticker.upper() for ticker in tickers]
        if len(set(normalized_tickers)) != len(normalized_tickers):
            raise ValueError("tickers must be unique, ignoring case")
        return tickers

    @field_validator("home_currency")
    @classmethod
    def uppercase_currency(cls, currency: str) -> str:
        return currency.upper()
