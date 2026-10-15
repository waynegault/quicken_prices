import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import pandas as pd
from pydantic import ValidationError

import quicken_prices as qp
import price_processing as processing
import market_data
import storage
from market_models import TickerMetadata
from settings import AppSettings


class TestQuickenPricesPureData(unittest.TestCase):
    def test_load_configuration_applies_model_defaults_without_writing(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            config_path = Path(temp_dir) / "configuration.yaml"
            config_path.write_text("tickers:\n  - TEST\npaths: {}\n", encoding="utf-8")
            with (
                mock.patch.object(qp, "validate_quicken_path", return_value=None),
                self.assertLogs(level="WARNING"),
            ):
                config = qp.load_configuration(str(config_path))

            self.assertEqual(config["tickers"], ["TEST"])
            self.assertEqual(config["retries"]["max_retries"], 3)
            self.assertEqual(config["paths"]["base"], str(Path(qp.__file__).parent))
            self.assertEqual(config["paths"]["quicken"], "")
            self.assertTrue(config_path.is_file())

    def test_load_configuration_rejects_unknown_and_coerced_values(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            config_path = Path(temp_dir) / "configuration.yaml"
            config_path.write_text(
                "tickers:\n  - TEST\npaths: {}\ncollection_period_years: '0.2'\n",
                encoding="utf-8",
            )

            with self.assertRaises(ValidationError):
                qp.load_configuration(str(config_path))

            config_path.write_text(
                "tickers:\n  - TEST\npaths: {}\nlegacy_option: true\n",
                encoding="utf-8",
            )
            with self.assertRaises(ValidationError):
                qp.load_configuration(str(config_path))

            config_path.write_text(
                "tickers:\n  - TEST\npaths: {}\n"
                "retries:\n  max_retries: '3'\n",
                encoding="utf-8",
            )
            with self.assertRaises(ValidationError):
                qp.load_configuration(str(config_path))

            config_path.write_text(
                "tickers:\n  - TEST\n  - test\npaths: {}\n",
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValidationError, "unique"):
                qp.load_configuration(str(config_path))

    def test_missing_configuration_is_not_created(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            config_path = Path(temp_dir) / "missing.yaml"

            with self.assertRaises(FileNotFoundError):
                qp.load_configuration(str(config_path))

            self.assertFalse(config_path.exists())

    def test_get_date_range_returns_ordered_utc_dates(self):
        config = {"collection_period_years": 0.1}
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

    def test_convert_prices_converts_usd_via_usdgbp_and_preserves_gbp(self):
        data = pd.DataFrame(
            [
                {
                    "Ticker": "AAPL",
                    "Old Price": 100.0,
                    "Date": pd.Timestamp("2026-01-02", tz="UTC"),
                    "Type": "EQUITY",
                    "Original Currency": "USD",
                },
                {
                    "Ticker": "AAPL",
                    "Old Price": 110.0,
                    "Date": pd.Timestamp("2026-01-03", tz="UTC"),
                    "Type": "EQUITY",
                    "Original Currency": "USD",
                },
                {
                    "Ticker": "VUKG.L",
                    "Old Price": 200.0,
                    "Date": pd.Timestamp("2026-01-02", tz="UTC"),
                    "Type": "ETF",
                    "Original Currency": "GBP",
                },
                # USD -> GBP rate (GBP per USD): 100 USD * 0.8 = 80 GBP.
                {
                    "Ticker": "USDGBP=X",
                    "Old Price": 0.8,
                    "Date": pd.Timestamp("2026-01-02", tz="UTC"),
                    "Type": "CURRENCY",
                    "Original Currency": "USD",
                },
                {
                    "Ticker": "USDGBP=X",
                    "Old Price": 0.79,
                    "Date": pd.Timestamp("2026-01-03", tz="UTC"),
                    "Type": "CURRENCY",
                    "Original Currency": "USD",
                },
                # GBP -> USD rate (USD per GBP): must NOT be applied to USD rows.
                {
                    "Ticker": "GBP=X",
                    "Old Price": 1.27,
                    "Date": pd.Timestamp("2026-01-02", tz="UTC"),
                    "Type": "CURRENCY",
                    "Original Currency": "GBP",
                },
            ]
        )

        converted = processing.convert_prices(data)

        aapl_day_1 = converted[
            (converted["Ticker"] == "AAPL")
            & (converted["Date"] == pd.Timestamp("2026-01-02", tz="UTC"))
        ]["Price"].iloc[0]
        aapl_day_2 = converted[
            (converted["Ticker"] == "AAPL")
            & (converted["Date"] == pd.Timestamp("2026-01-03", tz="UTC"))
        ]["Price"].iloc[0]
        vukg_price = converted[
            (converted["Ticker"] == "VUKG.L")
            & (converted["Date"] == pd.Timestamp("2026-01-02", tz="UTC"))
        ]["Price"].iloc[0]

        # USD must be converted via USDGBP=X (0.8), never GBP=X (1.27).
        self.assertAlmostEqual(aapl_day_1, 80.0, places=6)
        self.assertAlmostEqual(aapl_day_2, 86.9, places=6)
        self.assertAlmostEqual(vukg_price, 200.0, places=6)

    def test_convert_prices_never_uses_a_future_fx_rate(self):
        data = pd.DataFrame(
            [
                {
                    "Ticker": "AAPL",
                    "Old Price": 100.0,
                    "Date": pd.Timestamp("2026-01-01", tz="UTC"),
                    "Type": "EQUITY",
                    "Original Currency": "USD",
                },
                {
                    "Ticker": "USDGBP=X",
                    "Old Price": 0.8,
                    "Date": pd.Timestamp("2026-01-02", tz="UTC"),
                    "Type": "CURRENCY",
                    "Original Currency": "USD",
                },
            ]
        )

        with self.assertLogs(level="ERROR") as captured:
            with self.assertRaisesRegex(ValueError, "on or before"):
                processing.convert_prices(data)
        self.assertIn("Missing historical FX rates", captured.output[0])

    def test_cache_helpers_round_trip(self):
        sample = pd.DataFrame(
            [
                {
                    "Ticker": "TEST",
                    "Old Price": 1.23,
                    "Date": pd.Timestamp("2026-01-01", tz="UTC"),
                },
                {
                    "Ticker": "TEST",
                    "Old Price": 1.24,
                    "Date": pd.Timestamp("2026-01-02", tz="UTC"),
                },
            ]
        )

        with tempfile.TemporaryDirectory() as temp_dir:
            storage.save_cache("TEST", sample, temp_dir)
            loaded, status = storage.load_cache("TEST", temp_dir)
            self.assertTrue((Path(temp_dir) / "TEST.csv").is_file())
            self.assertFalse((Path(temp_dir) / "TEST.pkl").exists())

        self.assertEqual(status, "loaded")
        self.assertFalse(loaded.empty)
        self.assertEqual(len(loaded), 2)
        self.assertEqual(set(loaded.columns), set(storage.CACHE_COLUMNS))

    def test_cache_load_rejects_empty_or_invalid_files(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            cache_file = Path(temp_dir) / "TEST.csv"
            cache_file.write_text("", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "empty"):
                storage.load_cache("TEST", temp_dir)

            cache_file.write_text("Ticker,Unrelated\nTEST,1\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "missing columns"):
                storage.load_cache("TEST", temp_dir)

    def test_save_cache_rejects_empty_or_mismatched_data(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            with self.assertRaisesRegex(ValueError, "missing columns"):
                storage.save_cache("TEST", pd.DataFrame(), temp_dir)

            wrong_ticker = pd.DataFrame(
                [
                    {
                        "Ticker": "OTHER",
                        "Old Price": 1.23,
                        "Date": pd.Timestamp("2026-01-01", tz="UTC"),
                    }
                ]
            )
            with self.assertRaisesRegex(ValueError, "unexpected tickers"):
                storage.save_cache("TEST", wrong_ticker, temp_dir)

            self.assertFalse((Path(temp_dir) / "TEST.csv").exists())

    def test_cache_normalization_is_idempotent_and_does_not_mutate_input(self):
        source = pd.DataFrame(
            [
                {
                    "Ticker": "test",
                    "Old Price": "1.0",
                    "Date": "2026-01-01",
                },
                {
                    "Ticker": "TEST",
                    "Old Price": "1.1",
                    "Date": "2026-01-01",
                },
                {
                    "Ticker": "TEST",
                    "Old Price": "2.0",
                    "Date": "2026-01-02",
                },
            ]
        )
        original = source.copy(deep=True)

        normalized = storage.normalize_cache_data("TEST", source)

        pd.testing.assert_frame_equal(source, original)
        pd.testing.assert_frame_equal(
            normalized, storage.normalize_cache_data("TEST", normalized)
        )
        self.assertEqual(normalized["Old Price"].tolist(), [1.1, 2.0])


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

    def test_process_converted_prices_filters_and_formats(self):
        converted = pd.DataFrame(
            [
                {
                    "Ticker": "AAPL",
                    "Price": 100.0,
                    "Date": pd.Timestamp("2026-01-05", tz="UTC"),
                },
                {
                    "Ticker": "AAPL",
                    "Price": 90.0,
                    "Date": pd.Timestamp("2026-01-01", tz="UTC"),
                },
                {
                    "Ticker": "MSFT",
                    "Price": 50.0,
                    "Date": pd.Timestamp("2026-01-03", tz="UTC"),
                },
                {
                    "Ticker": "OLD",
                    "Price": 5.0,
                    "Date": pd.Timestamp("2025-12-31", tz="UTC"),
                },
            ]
        )
        start_date = pd.Timestamp("2026-01-01", tz="UTC")

        result = processing.process_converted_prices(converted, start_date)

        self.assertEqual(set(result.columns), {"Ticker", "Price", "Date"})
        self.assertNotIn("OLD", set(result["Ticker"]))
        self.assertEqual(len(result), 3)
        self.assertIsInstance(result["Date"].iloc[0], str)
        self.assertEqual(
            result["Date"].tolist(),
            ["05/01/2026", "03/01/2026", "01/01/2026"],
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

    def test_fetch_historical_data_propagates_ticker_failures(self):
        ticker_info = TickerMetadata(
            ticker="TEST",
            earliest_date=None,
            instrument_type="EQUITY",
            currency="GBP",
            current_price=float("nan"),
        )

        with mock.patch.object(
            market_data,
            "fetch_ticker_data",
            side_effect=RuntimeError("cache read failed"),
        ):
            with self.assertRaisesRegex(RuntimeError, "cache read failed"):
                market_data.fetch_historical_data(
                    [ticker_info],
                    pd.Timestamp("2026-01-01", tz="UTC"),
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
            metadata = market_data.validate_ticker("TEST")

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

    def test_save_to_csv_is_repeatable_and_surfaces_write_errors(self):
        data = pd.DataFrame(
            [
                {"Ticker": "TEST", "Price": 1.23, "Date": "01/01/2026"},
                {"Ticker": "TEST", "Price": 1.24, "Date": "02/01/2026"},
            ]
        )

        with tempfile.TemporaryDirectory() as temp_dir:
            config = {"paths": {"base": temp_dir, "data_file": "data.csv"}}
            storage.save_to_csv(data, config)
            output_path = Path(temp_dir) / "data.csv"
            original_output = output_path.read_bytes()

            storage.save_to_csv(data, config)
            self.assertEqual(output_path.read_bytes(), original_output)

            with mock.patch.object(pd.DataFrame, "to_csv", side_effect=OSError("disk full")):
                with self.assertRaisesRegex(OSError, "disk full"):
                    storage.save_to_csv(data, config)

            self.assertEqual(output_path.read_bytes(), original_output)
            self.assertEqual(list(Path(temp_dir).glob("*.tmp")), [])

    def test_clean_cache_only_removes_expired_files_and_surfaces_errors(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            cache_dir = Path(temp_dir) / "cache"
            cache_dir.mkdir()
            expired_file = cache_dir / "expired.csv"
            recent_file = cache_dir / "recent.csv"
            expired_file.write_text("old", encoding="utf-8")
            recent_file.write_text("new", encoding="utf-8")
            os.utime(expired_file, (0, 0))
            config = {
                "paths": {"base": temp_dir, "cache": "cache"},
                "cache": {"max_age_days": 1},
            }

            storage.clean_cache(config)

            self.assertFalse(expired_file.exists())
            self.assertTrue(recent_file.exists())

            expired_file.write_text("old again", encoding="utf-8")
            os.utime(expired_file, (0, 0))
            with mock.patch.object(Path, "unlink", side_effect=PermissionError("denied")):
                with self.assertRaisesRegex(PermissionError, "denied"):
                    storage.clean_cache(config)

    def test_main_propagates_failures(self):
        with (
            mock.patch.object(qp, "load_configuration", side_effect=RuntimeError("bad config")),
            mock.patch.object(qp, "pause_before_exit"),
        ):
            with self.assertLogs(level="ERROR") as captured:
                with self.assertRaisesRegex(RuntimeError, "bad config"):
                    qp.main()
        self.assertIn("An unexpected error occurred", captured.output[0])

    def test_startup_self_check_does_not_touch_user_cache(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            config = AppSettings(paths={"base": temp_dir}).model_dump()
            config["collection_period_years"] = 0.1

            qp.run_startup_self_check(config)
            self.assertFalse((Path(temp_dir) / "cache").exists())


if __name__ == "__main__":
    unittest.main()
