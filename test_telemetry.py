import unittest

import numpy as np
import pandas as pd

from telemetry import ALERT_COLUMNS, SENSORS, alerts_from_flags, assess, forest_features


class TelemetryTests(unittest.TestCase):
    def setUp(self):
        self.times = pd.Series(pd.to_datetime([
            "2020-01-01 12:00:00", "2020-01-01 12:00:01", "2020-01-01 12:00:02",
            "2020-01-01 12:00:15", "2020-01-01 12:00:16", "2020-01-01 12:00:17",
        ]))

    def test_consecutive_flags_form_one_alert_per_segment(self):
        alerts = alerts_from_flags(self.times, [True] * 6, consecutive=3)
        self.assertEqual(list(alerts["start"]), list(self.times.iloc[[0, 3]]))
        self.assertEqual(list(alerts["confirmed_at"]), list(self.times.iloc[[2, 5]]))
        self.assertEqual(list(alerts["observations"]), [3, 3])

    def test_isolated_points_leave_a_header_only_output(self):
        alerts = alerts_from_flags(self.times, [True, False] * 3, consecutive=2)
        self.assertTrue(alerts.empty)
        self.assertEqual(list(alerts.columns), ALERT_COLUMNS)

    def test_alert_uses_confirmation_and_counts_duplicates(self):
        data = pd.DataFrame({"datetime": self.times, "anomaly": [0, 1, 1, 1, 1, 1]})
        alerts = alerts_from_flags(self.times, [True] * 6, consecutive=2)
        result = assess(data, alerts)
        self.assertEqual((result["matched"], result["extra"], result["delays"]), (1, 1, [0.0]))

    def test_early_alert_is_extra_even_if_it_overlaps(self):
        data = pd.DataFrame({"datetime": self.times, "anomaly": [0, 1, 1, 0, 0, 0]})
        alerts = alerts_from_flags(self.times, [True, True, True, False, False, False], consecutive=1)
        result = assess(data, alerts)
        self.assertEqual((result["matched"], result["extra"]), (0, 1))

    def test_rolling_features_use_only_observed_history_in_segment(self):
        data = pd.DataFrame({sensor: [1, 2, 3, 10, 11, 12] for sensor in SENSORS})
        data["datetime"] = self.times
        features = forest_features(data, window=3)
        np.testing.assert_allclose(features[:, len(SENSORS)], [1, 1.5, 2, 10, 10.5, 11])
        np.testing.assert_allclose(features[[0, 3], 2 * len(SENSORS)], [0, 0])


if __name__ == "__main__":
    unittest.main()
