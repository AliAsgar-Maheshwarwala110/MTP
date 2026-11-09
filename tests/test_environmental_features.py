import pandas as pd
import unittest

from scripts.build_environmental_features import aggregate_daily
from src.data.s1_s2_weather_dataset import (
    DAILY_COLUMNS,
    SEQUENCE_LENGTH,
    prepare_weather_sequences,
)


class EnvironmentalFeatureTests(unittest.TestCase):
    def test_features_use_only_days_before_event(self):
        samples = pd.DataFrame({"sample_id": ["a"], "event_date": ["2020-01-10"]})
        daily = pd.DataFrame({
            "sample_id": ["a", "a", "a"], "event_date": ["2020-01-10"] * 3,
            "date": ["2020-01-09", "2020-01-10", "2019-12-11"],
            "prectotcorr": [2.0, 99.0, 3.0], "t2m": [10, 10, 10],
            "t2m_min": [5, 5, 5], "t2m_max": [15, 15, 15], "rh2m": [50, 50, 50],
        })
        result = aggregate_daily(daily, samples).iloc[0]
        self.assertEqual(result["precipitation_1d_mm"], 2.0)
        self.assertEqual(result["precipitation_30d_mm"], 5.0)

    def test_missing_power_values_and_samples_are_explicit(self):
        samples = pd.DataFrame({
            "sample_id": ["a", "b"],
            "event_date": ["2020-01-10", "2020-01-10"],
        })
        daily = pd.DataFrame({
            "sample_id": ["a"],
            "event_date": ["2020-01-10"],
            "date": ["2020-01-09"],
            "prectotcorr": [-999.0],
            "t2m": [10.0],
            "t2m_min": [5.0],
            "t2m_max": [15.0],
            "rh2m": [50.0],
        })
        result = aggregate_daily(daily, samples).set_index("sample_id")
        self.assertTrue(result.loc["a", "data_available"])
        self.assertTrue(pd.isna(result.loc["a", "precipitation_1d_mm"]))
        self.assertEqual(result.loc["a", "missing_precipitation_1d"], 1)
        self.assertFalse(result.loc["b", "data_available"])

    def test_daily_weather_sequence_excludes_event_day_and_preserves_order(self):
        daily = pd.DataFrame({
            "sample_id": ["a", "a", "a"],
            "date": ["2020-01-01", "2020-01-02", "2020-01-03"],
            "prectotcorr": [1.0, 2.0, 999.0],
            "t2m": [10.0, 20.0, 999.0],
            "t2m_min": [5.0, 15.0, 999.0],
            "t2m_max": [15.0, 25.0, 999.0],
            "rh2m": [50.0, 60.0, 999.0],
        })
        sequences, stats = prepare_weather_sequences(
            daily, {"a": "2020-01-03"}, {"a"}
        )
        sequence = sequences["a"]
        self.assertEqual(sequence.shape, (SEQUENCE_LENGTH, len(DAILY_COLUMNS) * 2 + 1))
        self.assertEqual(stats["training_sample_count"], 1)
        self.assertAlmostEqual(float(sequence[-1, -1]), 0.0)
        self.assertAlmostEqual(float(sequence[-2, -1]), -1 / 29)
        self.assertEqual(float(sequence[-1, len(DAILY_COLUMNS)]), 0.0)
        self.assertEqual(float(sequence[-2, len(DAILY_COLUMNS)]), 0.0)
        self.assertEqual(float(sequence[-3, len(DAILY_COLUMNS)]), 1.0)
