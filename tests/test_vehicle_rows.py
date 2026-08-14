from __future__ import annotations

import unittest

import numpy as np

from parking_map.vehicle_rows import build_vehicle_rows


def detection(center_x: float, center_y: float, width: float = 12, height: float = 24) -> dict:
    return {
        "bbox": [
            center_x - width / 2,
            center_y - height / 2,
            center_x + width / 2,
            center_y + height / 2,
        ]
    }


class VehicleRowTests(unittest.TestCase):
    def setUp(self) -> None:
        self.facility = np.full((180, 220), 255, dtype=np.uint8)
        self.exclusion = np.zeros_like(self.facility)

    def test_two_parallel_rows_are_not_chained_together(self) -> None:
        detections = [
            *(detection(x, 60) for x in (40, 70, 100, 130)),
            *(detection(x, 96) for x in (40, 70, 100, 130)),
        ]

        result = build_vehicle_rows(detections, self.facility, self.exclusion)

        self.assertEqual(len(result.rows), 2)
        self.assertEqual(sorted(len(row.detection_indices) for row in result.rows), [4, 4])
        self.assertEqual(result.unassigned_detection_indices, ())

    def test_large_gap_splits_collinear_groups(self) -> None:
        detections = [
            *(detection(x, 70) for x in (25, 50, 75)),
            *(detection(x, 70) for x in (155, 180, 205)),
        ]

        result = build_vehicle_rows(detections, self.facility, self.exclusion)

        self.assertEqual(len(result.rows), 2)
        self.assertEqual(sorted(len(row.detection_indices) for row in result.rows), [3, 3])

    def test_public_road_exclusion_prevents_a_row(self) -> None:
        detections = [detection(x, 70) for x in (40, 70, 100, 130)]
        self.exclusion[45:95, 20:150] = 255

        result = build_vehicle_rows(detections, self.facility, self.exclusion)

        self.assertEqual(result.rows, ())
        self.assertEqual(result.unassigned_detection_indices, (0, 1, 2, 3))

    def test_input_order_does_not_change_row_geometry(self) -> None:
        detections = [detection(x, 70) for x in (35, 65, 95, 125)]

        forward = build_vehicle_rows(detections, self.facility, self.exclusion)
        reverse = build_vehicle_rows(list(reversed(detections)), self.facility, self.exclusion)

        self.assertEqual(len(forward.rows), 1)
        self.assertEqual(len(reverse.rows), 1)
        np.testing.assert_array_equal(forward.rows[0].mask, reverse.rows[0].mask)


if __name__ == "__main__":
    unittest.main()
