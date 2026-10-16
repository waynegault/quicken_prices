"""FX conversion and Quicken output formatting tests."""

import unittest

import pandas as pd

import price_processing as processing

CONVERSION_SCENARIO_COLUMNS = (
    "Ticker",
    "Old Price",
    "Date",
    "Type",
    "Original Currency",
)
CONVERSION_SCENARIO = (
    ("AAPL", 100.0, "2026-01-02", "EQUITY", "USD"),
    ("AAPL", 110.0, "2026-01-03", "EQUITY", "USD"),
    ("VUKG.L", 200.0, "2026-01-02", "ETF", "GBP"),
    ("PENCE.L", 12500.0, "2026-01-02", "EQUITY", "GBp"),
    ("USDGBP=X", 0.8, "2026-01-02", "CURRENCY", "USD"),
    ("USDGBP=X", 0.79, "2026-01-03", "CURRENCY", "USD"),
    # Inverse quote must not be used for USD-denominated securities.
    ("GBP=X", 1.27, "2026-01-02", "CURRENCY", "GBP"),
)


def conversion_row(
    ticker: str,
    price: float,
    date: str,
    row_type: str = "EQUITY",
    currency: str = "USD",
) -> dict:
    """One conversion input row; provider dates arrive UTC-normalized."""
    return {
        "Ticker": ticker,
        "Old Price": price,
        "Date": pd.Timestamp(date, tz="UTC"),
        "Type": row_type,
        "Original Currency": currency,
    }


class ConversionTests(unittest.TestCase):
    def test_convert_prices_converts_usd_via_usdgbp_and_preserves_gbp(self):
        data = pd.DataFrame(
            CONVERSION_SCENARIO, columns=CONVERSION_SCENARIO_COLUMNS
        )
        converted = processing.convert_prices(data).set_index(["Ticker", "Date"])

        self.assertAlmostEqual(
            converted.loc[("AAPL", pd.Timestamp("2026-01-02", tz="UTC")), "Price"],
            80.0,
        )
        self.assertAlmostEqual(
            converted.loc[("AAPL", pd.Timestamp("2026-01-03", tz="UTC")), "Price"],
            86.9,
        )
        self.assertAlmostEqual(
            converted.loc[("VUKG.L", pd.Timestamp("2026-01-02", tz="UTC")), "Price"],
            200.0,
        )
        self.assertAlmostEqual(
            converted.loc[("PENCE.L", pd.Timestamp("2026-01-02", tz="UTC")), "Price"],
            125.0,
        )

    def test_convert_prices_uses_latest_prior_fx_rate(self):
        data = pd.DataFrame(
            [
                {
                    "Ticker": "AAPL",
                    "Old Price": 100.0,
                    "Date": pd.Timestamp("2026-01-05", tz="UTC"),
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
                {
                    "Ticker": "USDGBP=X",
                    "Old Price": 0.9,
                    "Date": pd.Timestamp("2026-01-06", tz="UTC"),
                    "Type": "CURRENCY",
                    "Original Currency": "USD",
                },
            ]
        )

        converted = processing.convert_prices(data)

        aapl = converted.loc[converted["Ticker"] == "AAPL", "Price"].iloc[0]
        self.assertEqual(aapl, 80.0)

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

    def test_convert_prices_rejects_missing_input_columns(self):
        with self.assertRaisesRegex(ValueError, "Missing required columns"):
            processing.convert_prices(pd.DataFrame({"Ticker": ["TEST"]}))

    def test_convert_prices_uses_old_price_when_output_price_is_present(self):
        data = pd.DataFrame(
            [
                {
                    "Ticker": "AAPL",
                    "Old Price": 100.0,
                    "Price": 999.0,
                    "Date": pd.Timestamp("2026-01-02", tz="UTC"),
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
        original = data.copy(deep=True)

        result = processing.convert_prices(data)

        self.assertEqual(
            result.loc[result["Ticker"] == "AAPL", "Price"].iloc[0], 80.0
        )
        pd.testing.assert_frame_equal(data, original)

    def test_convert_prices_preserves_currency_rows_without_equities(self):
        fx_only = pd.DataFrame(
            [
                {
                    "Ticker": "USDGBP=X",
                    "Old Price": 0.8,
                    "Date": pd.Timestamp("2026-01-02", tz="UTC"),
                    "Type": "CURRENCY",
                    "Original Currency": "USD",
                }
            ]
        )

        result = processing.convert_prices(fx_only)

        self.assertEqual(result.to_dict("records"), [
            {
                "Ticker": "USDGBP=X",
                "Price": 0.8,
                "Date": pd.Timestamp("2026-01-02", tz="UTC"),
            }
        ])

    def test_convert_prices_keeps_the_first_of_duplicate_fx_rates(self):
        data = pd.DataFrame(
            [
                conversion_row("AAPL", 100.0, "2026-01-02"),
                conversion_row("USDGBP=X", 0.8, "2026-01-02", "CURRENCY"),
                conversion_row("USDGBP=X", 0.9, "2026-01-02", "CURRENCY"),
            ]
        )

        with self.assertLogs(level="WARNING") as captured:
            converted = processing.convert_prices(data)

        self.assertIn("Duplicate FX rows", captured.output[0])
        self.assertEqual(
            converted.loc[converted["Ticker"] == "AAPL", "Price"].iloc[0], 80.0
        )

    def test_convert_prices_warns_when_the_fx_rate_is_stale(self):
        data = pd.DataFrame(
            [
                conversion_row("AAPL", 100.0, "2026-01-20"),
                conversion_row("USDGBP=X", 0.8, "2026-01-02", "CURRENCY"),
            ]
        )

        with self.assertLogs(level="WARNING") as captured:
            converted = processing.convert_prices(data)

        self.assertIn("more than 7 days old", captured.output[0])
        self.assertIn("oldest 18 days", captured.output[0])
        self.assertEqual(
            converted.loc[converted["Ticker"] == "AAPL", "Price"].iloc[0], 80.0
        )

    def test_convert_prices_accepts_a_weekend_gap_without_warning(self):
        data = pd.DataFrame(
            [
                conversion_row("AAPL", 100.0, "2026-01-05"),
                conversion_row("USDGBP=X", 0.8, "2026-01-02", "CURRENCY"),
            ]
        )

        with self.assertNoLogs(level="WARNING"):
            converted = processing.convert_prices(data)

        self.assertEqual(
            converted.loc[converted["Ticker"] == "AAPL", "Price"].iloc[0], 80.0
        )

    def test_convert_prices_reports_a_currency_with_no_fx_series(self):
        data = pd.DataFrame(
            [
                conversion_row("AAPL", 100.0, "2026-01-02"),
                # The inverse quote is present but is not a USD rate.
                conversion_row("GBP=X", 1.27, "2026-01-02", "CURRENCY", "GBP"),
            ]
        )

        with self.assertLogs(level="WARNING") as captured:
            with self.assertRaisesRegex(ValueError, "no FX rate on or before"):
                processing.convert_prices(data)

        self.assertIn("No USDGBP=X rates", captured.output[0])

    def test_convert_prices_uses_the_pair_for_a_non_gbp_home_currency(self):
        data = pd.DataFrame(
            [
                conversion_row("VOD.L", 200.0, "2026-01-02", currency="GBP"),
                conversion_row("GBPUSD=X", 1.25, "2026-01-02", "CURRENCY", "GBP"),
            ]
        )

        converted = processing.convert_prices(data, home_currency="USD")

        self.assertEqual(
            converted.loc[converted["Ticker"] == "VOD.L", "Price"].iloc[0], 250.0
        )

    def test_price_conversion_warns_when_malformed_rows_are_dropped(self):
        data = pd.DataFrame(
            [
                {
                    "Ticker": "AAPL",
                    "Old Price": 100.0,
                    "Date": "2026-01-02",
                    "Type": "EQUITY",
                    "Original Currency": "GBP",
                },
                {
                    "Ticker": "BROKEN",
                    "Old Price": "not-a-price",
                    "Date": "2026-01-02",
                    "Type": "EQUITY",
                    "Original Currency": "GBP",
                },
            ]
        )

        with self.assertLogs(level="WARNING") as captured:
            result = processing.convert_prices(data)

        self.assertEqual(result["Ticker"].tolist(), ["AAPL"])
        self.assertIn("Discarded 1 malformed price rows", captured.output[0])

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
        pd.testing.assert_frame_equal(
            result,
            processing.process_converted_prices(result, start_date),
        )

    def test_process_converted_prices_rejects_missing_columns(self):
        with self.assertRaisesRegex(ValueError, "missing columns"):
            processing.process_converted_prices(
                pd.DataFrame({"Ticker": ["TEST"]}),
                pd.Timestamp("2026-01-01", tz="UTC"),
            )


if __name__ == "__main__":
    unittest.main()
