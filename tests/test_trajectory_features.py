import math
import unittest

from uav_data.trajectory_features import TrajectoryPoint, extract_trajectory_features


class TrajectoryFeatureTests(unittest.TestCase):
    def test_extracts_hand_checked_motion_and_stop_features(self):
        points = (
            TrajectoryPoint(0.0, 0.0, 0.0, 0.0),
            TrajectoryPoint(1.0, 1.0, 0.0, 1.0),
            TrajectoryPoint(2.0, 2.0, 0.0, 0.0),
        )

        features = extract_trajectory_features(points, stop_speed_mps=0.2)

        self.assertEqual(features.duration_s, 2.0)
        self.assertEqual(features.displacement_metres, 2.0)
        self.assertEqual(features.path_length_metres, 2.0)
        self.assertEqual(features.median_speed_mps, 0.0)
        self.assertAlmostEqual(features.speed_p95_mps, 0.9)
        self.assertAlmostEqual(features.stop_fraction, 2 / 3)
        self.assertTrue(features.starts_stopped)
        self.assertTrue(features.ends_stopped)
        self.assertTrue(all(math.isfinite(value) for value in features.numeric_values()))

    def test_duplicate_timestamp_is_rejected(self):
        points = (
            TrajectoryPoint(0.0, 0.0, 0.0, 0.0),
            TrajectoryPoint(0.0, 1.0, 0.0, 1.0),
        )

        with self.assertRaisesRegex(ValueError, "strictly increase"):
            extract_trajectory_features(points)

    def test_empty_trajectory_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "at least one"):
            extract_trajectory_features(())


if __name__ == "__main__":
    unittest.main()
