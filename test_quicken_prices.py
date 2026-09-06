import tempfile
import unittest
from unittest import mock

import pandas as pd

import QuickenPrices as qp


class TestQuickenPricesPureData(unittest.TestCase):
    def test_get_date_range_returns_ordered_utc_dates(self):
        config = {"collection_period_years": 0.1}
        start_date, end_date = qp.get_date_range(config)

        self.assertLess(start_date, end_date)
        self.assertEqual(str(start_date.tz), "UTC")
        self.assertEqual(str(end_date.tz), "UTC")

    def test_find_missing_ranges_handles_empty_cache(self):
        start_date = pd.Timestamp("2026-01-01", tz="UTC")
        end_date = pd.Timestamp("2026-01-10", tz="UTC")

        missing = qp.find_missing_ranges(start_date, end_date, None, None)

        self.assertEqual(missing, [(start_date, end_date)])

    def test_find_missing_ranges_handles_left_and_right_gaps(self):
        start_date = pd.Timestamp("2026-01-01", tz="UTC")
        end_date = pd.Timestamp("2026-01-10", tz="UTC")
        first_cache = pd.Timestamp("2026-01-03", tz="UTC")
        last_cache = pd.Timestamp("2026-01-08", tz="UTC")

        missing = qp.find_missing_ranges(start_date, end_date, first_cache, last_cache)

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

        converted = qp.convert_prices(data)

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
            qp.save_cache("TEST", sample, temp_dir)
            loaded, status = qp.load_cache("TEST", temp_dir)

        self.assertEqual(status, "loaded")
        self.assertFalse(loaded.empty)
        self.assertEqual(len(loaded), 2)
        self.assertEqual(set(loaded.columns), set(qp.CACHE_COLUMNS))


    def test_find_missing_ranges_returns_empty_when_fully_covered(self):
        start = pd.Timestamp("2026-01-05", tz="UTC")
        end = pd.Timestamp("2026-01-10", tz="UTC")
        first = pd.Timestamp("2026-01-01", tz="UTC")
        last = pd.Timestamp("2026-01-15", tz="UTC")

        self.assertEqual(qp.find_missing_ranges(start, end, first, last), [])

    def test_find_missing_ranges_right_gap_only(self):
        start = pd.Timestamp("2026-01-01", tz="UTC")
        end = pd.Timestamp("2026-01-10", tz="UTC")
        first = pd.Timestamp("2025-12-20", tz="UTC")
        last = pd.Timestamp("2026-01-05", tz="UTC")

        missing = qp.find_missing_ranges(start, end, first, last)

        self.assertEqual(
            missing, [(pd.Timestamp("2026-01-06", tz="UTC"), end)]
        )

    def test_find_missing_ranges_left_gap_only(self):
        start = pd.Timestamp("2026-01-01", tz="UTC")
        end = pd.Timestamp("2026-01-10", tz="UTC")
        first = pd.Timestamp("2026-01-05", tz="UTC")
        last = pd.Timestamp("2026-01-15", tz="UTC")

        missing = qp.find_missing_ranges(start, end, first, last)

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

        result = qp.process_converted_prices(converted, start_date)

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
            with mock.patch.object(qp, "download_data", return_value=downloaded) as m:
                result = qp.fetch_ticker_data(
                    ticker, start, end, float("nan"), cache_dir
                )
                m.assert_called_once()

        self.assertEqual(len(result), 3)
        self.assertEqual(set(result.columns), set(qp.CACHE_COLUMNS))

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
            qp.save_cache(ticker, cached, cache_dir)
            with mock.patch.object(qp, "download_data") as m:
                result = qp.fetch_ticker_data(
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
            qp.save_cache(ticker, cached, cache_dir)
            with mock.patch.object(qp, "download_data", return_value=downloaded) as m:
                result = qp.fetch_ticker_data(
                    ticker, start, end, float("nan"), cache_dir
                )
                m.assert_called_once()

        self.assertEqual(len(result), 5)
        self.assertEqual(result["Old Price"].min(), 1.0)
        self.assertEqual(result["Old Price"].max(), 5.0)


if __name__ == "__main__":
    unittest.main()
