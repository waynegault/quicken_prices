import tempfile
import unittest

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

    def test_convert_prices_converts_usd_and_preserves_gbp(self):
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
                {
                    "Ticker": "GBP=X",
                    "Old Price": 0.8,
                    "Date": pd.Timestamp("2026-01-02", tz="UTC"),
                    "Type": "CURRENCY",
                    "Original Currency": "GBP",
                },
                {
                    "Ticker": "GBP=X",
                    "Old Price": 0.79,
                    "Date": pd.Timestamp("2026-01-03", tz="UTC"),
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


if __name__ == "__main__":
    unittest.main()
