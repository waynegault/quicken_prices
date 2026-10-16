"""Cache and CSV persistence tests."""

import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import pandas as pd
from pydantic import ValidationError

import storage
from settings import AppSettings


class StorageTests(unittest.TestCase):
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

            with self.assertRaises(ValidationError):
                storage.load_cache("../outside", temp_dir)

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

        with self.assertLogs(level="WARNING") as captured:
            normalized = storage.normalize_cache_data("TEST", source)

        pd.testing.assert_frame_equal(source, original)
        pd.testing.assert_frame_equal(
            normalized, storage.normalize_cache_data("TEST", normalized)
        )
        self.assertEqual(normalized["Old Price"].tolist(), [1.1, 2.0])
        self.assertIn("Discarded 1 invalid or duplicate cache rows", captured.output[0])

    def test_cache_load_returns_no_rows_for_a_truncated_file(self):
        # A cache cut off mid-row must read as "nothing usable", not as a price,
        # so the caller refetches the range instead of trusting the fragment.
        with tempfile.TemporaryDirectory() as temp_dir:
            cache_file = Path(temp_dir) / "TEST.csv"
            cache_file.write_text("Ticker,Old Price,Date\nTEST,", encoding="utf-8")

            with self.assertLogs(level="WARNING") as captured:
                data, status = storage.load_cache("TEST", temp_dir)

        self.assertEqual(status, "loaded")
        self.assertTrue(data.empty)
        self.assertIn("Discarded 1 invalid or duplicate cache rows", captured.output[0])

    def test_cache_load_rejects_an_unreadable_file(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            cache_file = Path(temp_dir) / "TEST.csv"
            cache_file.write_bytes(b"\x00\x01\x02\xff\xfe")

            with self.assertRaises(ValueError):
                storage.load_cache("TEST", temp_dir)

    def test_cache_load_ignores_columns_it_does_not_need(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            cache_file = Path(temp_dir) / "TEST.csv"
            cache_file.write_text(
                "Ticker,Old Price,Date,Volume\nTEST,1.23,01/01/2026,1000\n",
                encoding="utf-8",
            )

            data, status = storage.load_cache("TEST", temp_dir)

        self.assertEqual(status, "loaded")
        self.assertEqual(tuple(data.columns), storage.CACHE_COLUMNS)
        self.assertEqual(data["Old Price"].tolist(), [1.23])

    def test_cache_write_leaves_the_previous_file_intact_when_it_fails(self):
        sample = pd.DataFrame(
            [
                {
                    "Ticker": "TEST",
                    "Old Price": 2.0,
                    "Date": pd.Timestamp("2026-01-02", tz="UTC"),
                }
            ]
        )

        with tempfile.TemporaryDirectory() as temp_dir:
            cache_file = Path(temp_dir) / "TEST.csv"
            cache_file.write_text(
                "Ticker,Old Price,Date\nTEST,1.0,01/01/2026\n", encoding="utf-8"
            )
            original = cache_file.read_bytes()

            with mock.patch.object(pd.DataFrame, "to_csv", side_effect=OSError("disk full")):
                with self.assertRaisesRegex(OSError, "disk full"):
                    storage.save_cache("TEST", sample, temp_dir)

            # The interrupted write must not damage the cache already on disk,
            # and must not leave its half-written replacement behind.
            self.assertEqual(cache_file.read_bytes(), original)
            self.assertEqual([entry.name for entry in Path(temp_dir).iterdir()], ["TEST.csv"])

    def test_cache_write_leaves_no_temporary_file_behind(self):
        sample = pd.DataFrame(
            [
                {
                    "Ticker": "TEST",
                    "Old Price": 1.23,
                    "Date": pd.Timestamp("2026-01-01", tz="UTC"),
                }
            ]
        )

        with tempfile.TemporaryDirectory() as temp_dir:
            storage.save_cache("TEST", sample, temp_dir)

            self.assertEqual([entry.name for entry in Path(temp_dir).iterdir()], ["TEST.csv"])

    def test_save_to_csv_is_repeatable_and_surfaces_write_errors(self):
        data = pd.DataFrame(
            [
                {"Ticker": "TEST", "Price": 1.23, "Date": "01/01/2026"},
                {"Ticker": "TEST", "Price": 1.24, "Date": "02/01/2026"},
            ]
        )

        with tempfile.TemporaryDirectory() as temp_dir:
            config = AppSettings(
                paths={"base": temp_dir, "data_file": "data.csv"}
            )
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
            unrelated_file = cache_dir / "user-notes.txt"
            expired_file.write_text("old", encoding="utf-8")
            recent_file.write_text("new", encoding="utf-8")
            unrelated_file.write_text("keep", encoding="utf-8")
            os.utime(expired_file, (0, 0))
            os.utime(unrelated_file, (0, 0))
            config = AppSettings(
                paths={"base": temp_dir, "cache": "cache"},
                cache={"max_age_days": 1},
            )

            storage.clean_cache(config)

            self.assertFalse(expired_file.exists())
            self.assertTrue(recent_file.exists())
            self.assertTrue(unrelated_file.exists())

            expired_file.write_text("old again", encoding="utf-8")
            os.utime(expired_file, (0, 0))
            with mock.patch.object(Path, "unlink", side_effect=PermissionError("denied")):
                with self.assertRaisesRegex(PermissionError, "denied"):
                    storage.clean_cache(config)


if __name__ == "__main__":
    unittest.main()
