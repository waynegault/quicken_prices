"""Provider validation, download, retry and cache-planning tests."""

import tempfile
import unittest
from unittest import mock

import pandas as pd
from pydantic import ValidationError

import market_data
import storage
from market_models import TickerMetadata
from settings import AppSettings


class MarketDataTests(unittest.TestCase):
    def test_get_date_range_returns_ordered_utc_dates(self):
        config = AppSettings(
            paths={"base": "."},
            collection_period_years=0.1,
        )
        start_date, end_date = market_data.get_date_range(config)

        self.assertLess(start_date, end_date)
        self.assertEqual(str(start_date.tz), "UTC")
        self.assertEqual(str(end_date.tz), "UTC")

    def test_find_missing_ranges_handles_empty_cache(self):
        start_date = pd.Timestamp("2026-01-01", tz="UTC")
        end_date = pd.Timestamp("2026-01-10", tz="UTC")

        missing = market_data.find_missing_ranges(start_date, end_date, None, None)

        self.assertEqual(missing, [(start_date, end_date)])

    def test_find_missing_ranges_handles_left_and_right_gaps(self):
        start_date = pd.Timestamp("2026-01-01", tz="UTC")
        end_date = pd.Timestamp("2026-01-10", tz="UTC")
        first_cache = pd.Timestamp("2026-01-03", tz="UTC")
        last_cache = pd.Timestamp("2026-01-08", tz="UTC")

        missing = market_data.find_missing_ranges(start_date, end_date, first_cache, last_cache)

        self.assertEqual(
            missing,
            [
                (
                    pd.Timestamp("2026-01-01", tz="UTC"),
                    pd.Timestamp("2026-01-02", tz="UTC"),
                ),
                (
                    pd.Timestamp("2026-01-09", tz="UTC"),
                    pd.Timestamp("2026-01-10", tz="UTC"),
                ),
            ],
        )

    def test_find_missing_ranges_returns_empty_when_fully_covered(self):
        start = pd.Timestamp("2026-01-05", tz="UTC")
        end = pd.Timestamp("2026-01-10", tz="UTC")
        first = pd.Timestamp("2026-01-01", tz="UTC")
        last = pd.Timestamp("2026-01-15", tz="UTC")

        self.assertEqual(market_data.find_missing_ranges(start, end, first, last), [])

    def test_find_missing_ranges_right_gap_only(self):
        start = pd.Timestamp("2026-01-01", tz="UTC")
        end = pd.Timestamp("2026-01-10", tz="UTC")
        first = pd.Timestamp("2025-12-20", tz="UTC")
        last = pd.Timestamp("2026-01-05", tz="UTC")

        missing = market_data.find_missing_ranges(start, end, first, last)

        self.assertEqual(
            missing, [(pd.Timestamp("2026-01-06", tz="UTC"), end)]
        )

    def test_find_missing_ranges_left_gap_only(self):
        start = pd.Timestamp("2026-01-01", tz="UTC")
        end = pd.Timestamp("2026-01-10", tz="UTC")
        first = pd.Timestamp("2026-01-05", tz="UTC")
        last = pd.Timestamp("2026-01-15", tz="UTC")

        missing = market_data.find_missing_ranges(start, end, first, last)

        self.assertEqual(
            missing, [(start, pd.Timestamp("2026-01-04", tz="UTC"))]
        )

    def test_fetch_ticker_data_downloads_when_cache_missing(self):
        ticker = "TEST"
        start = pd.Timestamp("2026-01-01", tz="UTC")
        end = pd.Timestamp("2026-01-03", tz="UTC")
        downloaded = pd.DataFrame(
            [
                {"Ticker": ticker, "Old Price": 1.0, "Date": pd.Timestamp("2026-01-01", tz="UTC")},
                {"Ticker": ticker, "Old Price": 2.0, "Date": pd.Timestamp("2026-01-02", tz="UTC")},
                {"Ticker": ticker, "Old Price": 3.0, "Date": pd.Timestamp("2026-01-03", tz="UTC")},
            ]
        )

        with tempfile.TemporaryDirectory() as cache_dir:
            with mock.patch.object(
                market_data, "download_data", return_value=downloaded
            ) as m:
                result = market_data.fetch_ticker_data(
                    ticker, start, end, float("nan"), cache_dir
                )
                m.assert_called_once()

        self.assertEqual(len(result), 3)
        self.assertEqual(set(result.columns), set(storage.CACHE_COLUMNS))

    def test_fetch_ticker_data_uses_cache_without_download(self):
        ticker = "TEST"
        start = pd.Timestamp("2026-01-01", tz="UTC")
        end = pd.Timestamp("2026-01-03", tz="UTC")
        cached = pd.DataFrame(
            [
                {"Ticker": ticker, "Old Price": 1.0, "Date": pd.Timestamp("2026-01-01", tz="UTC")},
                {"Ticker": ticker, "Old Price": 2.0, "Date": pd.Timestamp("2026-01-02", tz="UTC")},
                {"Ticker": ticker, "Old Price": 3.0, "Date": pd.Timestamp("2026-01-03", tz="UTC")},
            ]
        )

        with tempfile.TemporaryDirectory() as cache_dir:
            storage.save_cache(ticker, cached, cache_dir)
            with mock.patch.object(market_data, "download_data") as m:
                result = market_data.fetch_ticker_data(
                    ticker, start, end, float("nan"), cache_dir
                )
                m.assert_not_called()

        self.assertEqual(len(result), 3)

    def test_fetch_ticker_data_merges_cache_and_download(self):
        ticker = "TEST"
        start = pd.Timestamp("2026-01-01", tz="UTC")
        end = pd.Timestamp("2026-01-05", tz="UTC")
        cached = pd.DataFrame(
            [
                {"Ticker": ticker, "Old Price": 1.0, "Date": pd.Timestamp("2026-01-01", tz="UTC")},
                {"Ticker": ticker, "Old Price": 2.0, "Date": pd.Timestamp("2026-01-02", tz="UTC")},
            ]
        )
        downloaded = pd.DataFrame(
            [
                {"Ticker": ticker, "Old Price": 3.0, "Date": pd.Timestamp("2026-01-03", tz="UTC")},
                {"Ticker": ticker, "Old Price": 4.0, "Date": pd.Timestamp("2026-01-04", tz="UTC")},
                {"Ticker": ticker, "Old Price": 5.0, "Date": pd.Timestamp("2026-01-05", tz="UTC")},
            ]
        )

        with tempfile.TemporaryDirectory() as cache_dir:
            storage.save_cache(ticker, cached, cache_dir)
            with mock.patch.object(
                market_data, "download_data", return_value=downloaded
            ) as m:
                result = market_data.fetch_ticker_data(
                    ticker, start, end, float("nan"), cache_dir
                )
                m.assert_called_once()

        self.assertEqual(len(result), 5)
        self.assertEqual(result["Old Price"].min(), 1.0)
        self.assertEqual(result["Old Price"].max(), 5.0)

    def test_fetch_ticker_data_limits_rows_and_current_price_to_requested_end(self):
        ticker = "TEST"
        today = pd.Timestamp.now(tz="UTC").normalize()
        start = today - pd.Timedelta(days=5)
        end = today - pd.Timedelta(days=2)
        cached = pd.DataFrame(
            [
                {"Ticker": ticker, "Old Price": 1.0, "Date": today - pd.Timedelta(days=7)},
                {"Ticker": ticker, "Old Price": 2.0, "Date": today - pd.Timedelta(days=4)},
                {"Ticker": ticker, "Old Price": 3.0, "Date": today - pd.Timedelta(days=1)},
            ]
        )

        with tempfile.TemporaryDirectory() as cache_dir:
            storage.save_cache(ticker, cached, cache_dir)
            with mock.patch.object(market_data, "download_data") as download:
                result = market_data.fetch_ticker_data(
                    ticker, start, end, 99.0, cache_dir
                )
                download.assert_not_called()

        self.assertEqual(result["Date"].tolist(), [today - pd.Timedelta(days=4)])
        self.assertNotIn(99.0, result["Old Price"].tolist())

    def test_fx_fetch_keeps_one_prior_context_row_without_leaking_future_rows(self):
        ticker = "USDGBP=X"
        today = pd.Timestamp.now(tz="UTC").normalize()
        start = today - pd.Timedelta(days=5)
        end = today - pd.Timedelta(days=2)
        cached = pd.DataFrame(
            [
                {"Ticker": ticker, "Old Price": 0.7, "Date": today - pd.Timedelta(days=7)},
                {"Ticker": ticker, "Old Price": 0.8, "Date": today - pd.Timedelta(days=4)},
                {"Ticker": ticker, "Old Price": 0.9, "Date": today - pd.Timedelta(days=1)},
            ]
        )

        with tempfile.TemporaryDirectory() as cache_dir:
            storage.save_cache(ticker, cached, cache_dir)
            with mock.patch.object(market_data, "download_data") as download:
                result = market_data.fetch_ticker_data(
                    ticker, start, end, float("nan"), cache_dir
                )
                download.assert_not_called()

        self.assertEqual(result["Date"].min(), start - pd.Timedelta(days=1))
        self.assertEqual(result["Date"].max(), end)
        self.assertEqual(result.iloc[0]["Old Price"], 0.7)
        self.assertNotIn(today - pd.Timedelta(days=1), result["Date"].tolist())

    def test_fetch_ticker_data_appends_live_price_only_when_requested_through_today(self):
        ticker = "TEST"
        today = pd.Timestamp.now(tz="UTC").normalize()
        yesterday = today - pd.Timedelta(days=1)
        cached = pd.DataFrame(
            [{"Ticker": ticker, "Old Price": 1.0, "Date": yesterday}]
        )

        with tempfile.TemporaryDirectory() as cache_dir:
            storage.save_cache(ticker, cached, cache_dir)
            with mock.patch.object(market_data, "download_data", return_value=pd.DataFrame()):
                result = market_data.fetch_ticker_data(
                    ticker, yesterday, today, 2.0, cache_dir
                )
            persisted, _ = storage.load_cache(ticker, cache_dir)

        self.assertIn(today, result["Date"].tolist())
        self.assertEqual(
            result.loc[result["Date"] == today, "Old Price"].iloc[0], 2.0
        )
        self.assertNotIn(today, persisted["Date"].tolist())

    def test_fetch_historical_data_propagates_ticker_failures(self):
        ticker_info = TickerMetadata(
            ticker="TEST",
            earliest_date=None,
            instrument_type="EQUITY",
            currency="GBP",
            current_price=float("nan"),
        )

        retry_settings = AppSettings(paths={"base": "."}).retries
        with mock.patch.object(
            market_data,
            "fetch_ticker_data",
            side_effect=RuntimeError("cache read failed"),
        ) as fetch:
            with self.assertRaisesRegex(RuntimeError, "cache read failed"):
                market_data.fetch_historical_data(
                    [ticker_info],
                    pd.Timestamp("2026-01-01", tz="UTC"),
                    pd.Timestamp("2026-01-03", tz="UTC"),
                    "cache",
                    retry_settings=retry_settings,
                )
        self.assertEqual(fetch.call_count, 1)
        call_args = fetch.call_args.args
        self.assertEqual(
            call_args[:3],
            (
                "TEST",
                pd.Timestamp("2026-01-01", tz="UTC"),
                pd.Timestamp("2026-01-03", tz="UTC"),
            ),
        )
        self.assertTrue(pd.isna(call_args[3]))
        self.assertEqual(call_args[4:], ("cache", retry_settings))

    def test_fetch_historical_data_requires_timezone_aware_bounds(self):
        with self.assertRaisesRegex(ValueError, "start_date must include a timezone"):
            market_data.fetch_historical_data(
                [],
                pd.Timestamp("2026-01-01"),
                pd.Timestamp("2026-01-03", tz="UTC"),
                "cache",
            )

    def test_download_data_propagates_provider_failures(self):
        ticker = mock.Mock()
        ticker.history.side_effect = RuntimeError("provider unavailable")

        with mock.patch.object(market_data.yf, "Ticker", return_value=ticker):
            with self.assertRaisesRegex(RuntimeError, "provider unavailable"):
                market_data.download_data(
                    "TEST",
                    pd.Timestamp("2026-01-01", tz="UTC"),
                    pd.Timestamp("2026-01-03", tz="UTC"),
                )

    def test_download_data_retries_transient_provider_failures(self):
        ticker = mock.Mock()
        ticker.history.side_effect = [
            market_data.requests.exceptions.ConnectionError("temporary"),
            pd.DataFrame(
                {"Close": [12.5]},
                index=pd.DatetimeIndex(
                    [pd.Timestamp("2026-01-02", tz="UTC")], name="Date"
                ),
            ),
        ]
        retry_settings = {"max_retries": 2, "retry_delay": 0.25, "backoff": 2.0}

        with (
            mock.patch.object(market_data.yf, "Ticker", return_value=ticker),
            mock.patch.object(market_data.time, "sleep") as sleep,
            self.assertLogs(level="WARNING") as captured,
        ):
            result = market_data.download_data(
                "TEST",
                pd.Timestamp("2026-01-01", tz="UTC"),
                pd.Timestamp("2026-01-03", tz="UTC"),
                retry_settings=retry_settings,
            )

        self.assertEqual(result["Old Price"].tolist(), [12.5])
        self.assertEqual(ticker.history.call_count, 2)
        sleep.assert_called_once_with(0.25)
        self.assertIn("Attempt 1/2 failed", captured.output[0])

    def test_download_data_warns_when_invalid_provider_rows_are_discarded(self):
        ticker = mock.Mock()
        ticker.history.return_value = pd.DataFrame(
            {"Close": [12.5, "not-a-price"]},
            index=pd.Index(["2026-01-02", "not-a-date"], name="Date"),
        )

        with (
            mock.patch.object(market_data.yf, "Ticker", return_value=ticker),
            self.assertLogs(level="WARNING") as captured,
        ):
            result = market_data.download_data(
                "TEST",
                pd.Timestamp("2026-01-01", tz="UTC"),
                pd.Timestamp("2026-01-03", tz="UTC"),
            )

        self.assertEqual(result["Old Price"].tolist(), [12.5])
        self.assertIn("Discarded 1 malformed download rows", captured.output[0])

    def test_download_data_returns_empty_when_the_provider_has_no_rows(self):
        ticker = mock.Mock()
        ticker.history.return_value = pd.DataFrame()

        with (
            mock.patch.object(market_data.yf, "Ticker", return_value=ticker),
            self.assertLogs(level="ERROR") as captured,
        ):
            result = market_data.download_data(
                "TEST",
                pd.Timestamp("2026-01-01", tz="UTC"),
                pd.Timestamp("2026-01-03", tz="UTC"),
            )

        self.assertTrue(result.empty)
        self.assertIn("No data from", captured.output[0])

    def test_download_data_treats_a_missing_frame_as_no_data(self):
        ticker = mock.Mock()
        ticker.history.return_value = None

        with (
            mock.patch.object(market_data.yf, "Ticker", return_value=ticker),
            self.assertLogs(level="ERROR") as captured,
        ):
            result = market_data.download_data(
                "TEST",
                pd.Timestamp("2026-01-01", tz="UTC"),
                pd.Timestamp("2026-01-03", tz="UTC"),
            )

        self.assertTrue(result.empty)
        self.assertIn("No data from", captured.output[0])

    def test_download_data_reports_when_every_row_is_unusable(self):
        ticker = mock.Mock()
        ticker.history.return_value = pd.DataFrame(
            {"Close": ["not-a-price"]},
            index=pd.Index(["not-a-date"], name="Date"),
        )

        with (
            mock.patch.object(market_data.yf, "Ticker", return_value=ticker),
            self.assertLogs(level="ERROR") as captured,
        ):
            result = market_data.download_data(
                "TEST",
                pd.Timestamp("2026-01-01", tz="UTC"),
                pd.Timestamp("2026-01-03", tz="UTC"),
            )

        self.assertTrue(result.empty)
        self.assertIn("no valid price rows", captured.output[0])

    def test_download_data_keeps_the_cache_columns_from_reordered_output(self):
        ticker = mock.Mock()
        ticker.history.return_value = pd.DataFrame(
            {
                "Volume": [1000],
                "Close": [12.5],
                "Open": [11.0],
            },
            index=pd.DatetimeIndex(
                [pd.Timestamp("2026-01-02", tz="UTC")], name="Date"
            ),
        )

        with mock.patch.object(market_data.yf, "Ticker", return_value=ticker):
            result = market_data.download_data(
                "TEST",
                pd.Timestamp("2026-01-01", tz="UTC"),
                pd.Timestamp("2026-01-03", tz="UTC"),
            )

        self.assertEqual(tuple(result.columns), storage.CACHE_COLUMNS)
        self.assertEqual(result["Old Price"].tolist(), [12.5])

    def test_download_data_skips_the_provider_for_a_same_day_range(self):
        today = pd.Timestamp.now(tz="UTC").normalize()
        ticker = mock.Mock()

        with mock.patch.object(market_data.yf, "Ticker", return_value=ticker):
            result = market_data.download_data("TEST", today, today)

        self.assertTrue(result.empty)
        ticker.history.assert_not_called()

    def test_download_data_retries_a_rate_limit_and_reports_exhaustion(self):
        ticker = mock.Mock()
        ticker.history.side_effect = market_data.yf.exceptions.YFRateLimitError()
        retry_settings = {"max_retries": 2, "retry_delay": 0.25, "backoff": 2.0}

        with (
            mock.patch.object(market_data.yf, "Ticker", return_value=ticker),
            mock.patch.object(market_data.time, "sleep") as sleep,
            self.assertLogs(level="ERROR") as captured,
        ):
            with self.assertRaises(market_data.yf.exceptions.YFRateLimitError):
                market_data.download_data(
                    "TEST",
                    pd.Timestamp("2026-01-01", tz="UTC"),
                    pd.Timestamp("2026-01-03", tz="UTC"),
                    retry_settings=retry_settings,
                )

        self.assertEqual(ticker.history.call_count, 2)
        sleep.assert_called_once_with(0.25)
        self.assertIn("Max attempts reached", captured.output[0])

    def test_validate_ticker_returns_strict_metadata_model(self):
        info = {
            "symbol": "TEST",
            "firstTradeDateEpochUtc": int(
                pd.Timestamp("2026-01-01", tz="UTC").timestamp()
            ),
            "quoteType": "EQUITY",
            "currency": "USD",
            "bid": 42.5,
        }
        with mock.patch.object(market_data.yf, "Ticker") as ticker_factory:
            ticker_factory.return_value.get_info.return_value = info
            metadata = market_data.validate_ticker(
                "TEST", retry_settings=AppSettings(paths={"base": "."}).retries
            )

        self.assertEqual(metadata.ticker, "TEST")
        self.assertEqual(metadata.instrument_type, "EQUITY")
        self.assertEqual(metadata.currency, "USD")
        self.assertEqual(metadata.current_price, 42.5)
        self.assertEqual(metadata.earliest_date, pd.Timestamp("2026-01-01", tz="UTC"))

        with self.assertRaises(ValidationError):
            TickerMetadata(
                ticker="TEST",
                earliest_date=None,
                instrument_type="EQUITY",
                currency="USD",
                current_price="42.5",
            )
        with self.assertRaisesRegex(ValidationError, "timezone"):
            TickerMetadata(
                ticker="TEST",
                earliest_date=pd.Timestamp("2026-01-01"),
                instrument_type="EQUITY",
                currency="USD",
                current_price=42.5,
            )
        with self.assertRaises(ValidationError):
            TickerMetadata(
                ticker="../outside",
                earliest_date=None,
                instrument_type="EQUITY",
                currency="USD",
                current_price=42.5,
            )

    def test_validate_ticker_retries_yahoo_invalid_json_response(self):
        info = {
            "symbol": "TEST",
            "quoteType": "EQUITY",
            "currency": "USD",
            "bid": 42.5,
        }
        retry_settings = {"max_retries": 2, "retry_delay": 0, "backoff": 1.0}
        with (
            mock.patch.object(market_data.yf, "Ticker") as ticker_factory,
            mock.patch.object(market_data.time, "sleep"),
            self.assertLogs(level="WARNING") as captured,
        ):
            ticker_factory.return_value.get_info.side_effect = [
                ValueError("Expecting value: line 1 column 1"),
                info,
            ]
            metadata = market_data.validate_ticker(
                "TEST", retry_settings=retry_settings
            )

        self.assertEqual(metadata.ticker, "TEST")
        self.assertIn("Attempt 1/2 failed", captured.output[0])

    def test_validate_tickers_preserves_configured_order(self):
        metadata = {
            symbol: TickerMetadata(
                ticker=symbol,
                earliest_date=None,
                instrument_type="EQUITY",
                currency="GBP",
                current_price=float("nan"),
            )
            for symbol in ("SECOND", "FIRST")
        }
        with mock.patch.object(
            market_data,
            "validate_ticker",
            side_effect=lambda symbol, retry_settings=None: metadata.get(symbol),
        ):
            validated = market_data.validate_tickers(["SECOND", "FIRST"])

        self.assertEqual([ticker.ticker for ticker in validated], ["SECOND", "FIRST"])

    def test_retry_uses_explicit_strict_policy(self):
        attempts = []

        @market_data.retry(exceptions=ValueError)
        def flaky_operation():
            attempts.append("attempt")
            if len(attempts) == 1:
                raise ValueError("transient")
            return "ok"

        retry_settings = {
            "max_retries": 2,
            "retry_delay": 0.25,
            "backoff": 2.0,
        }
        with (
            mock.patch.object(market_data.time, "sleep") as sleep,
            self.assertLogs(level="WARNING") as captured,
        ):
            self.assertEqual(flaky_operation(retry_settings=retry_settings), "ok")

        self.assertEqual(len(attempts), 2)
        sleep.assert_called_once_with(0.25)
        self.assertIn("Attempt 1/2 failed", captured.output[0])
        with self.assertRaises(ValidationError):
            flaky_operation(retry_settings={**retry_settings, "max_retries": "2"})

    def test_retry_defaults_only_to_transient_provider_errors(self):
        attempts = []

        @market_data.retry()
        def flaky_provider():
            attempts.append(1)
            if len(attempts) == 1:
                raise market_data.requests.exceptions.Timeout("temporary")
            return "ok"

        retry_settings = {"max_retries": 2, "retry_delay": 0, "backoff": 1.0}
        with (
            mock.patch.object(market_data.time, "sleep"),
            self.assertLogs(level="WARNING"),
        ):
            self.assertEqual(
                flaky_provider(retry_settings=retry_settings),
                "ok",
            )
        self.assertEqual(len(attempts), 2)

        attempts.clear()

        @market_data.retry()
        def invalid_operation():
            attempts.append(1)
            raise ValueError("not transient")

        with self.assertRaisesRegex(ValueError, "not transient"):
            invalid_operation(retry_settings=retry_settings)
        self.assertEqual(len(attempts), 1)

    def test_retry_reports_exhaustion_after_the_configured_attempts(self):
        attempts = []

        @market_data.retry(exceptions=ValueError)
        def always_fails():
            attempts.append(1)
            raise ValueError("still broken")

        retry_settings = {"max_retries": 3, "retry_delay": 0.25, "backoff": 2.0}
        with (
            mock.patch.object(market_data.time, "sleep") as sleep,
            self.assertLogs(level="ERROR") as captured,
        ):
            with self.assertRaisesRegex(ValueError, "still broken"):
                always_fails(retry_settings=retry_settings)

        self.assertEqual(len(attempts), 3)
        self.assertEqual([entry.args[0] for entry in sleep.call_args_list], [0.25, 0.5])
        self.assertIn("Max attempts reached (3)", captured.output[0])


if __name__ == "__main__":
    unittest.main()
