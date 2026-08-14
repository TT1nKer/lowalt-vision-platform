from __future__ import annotations

import unittest

import numpy as np

from parking.parking_vehicle_screen import detections_inside_mask


class ParkingVehicleScreenTests(unittest.TestCase):
    def test_keeps_only_vehicle_centres_inside_segformer_candidate(self) -> None:
        mask = np.zeros((100, 100), dtype=np.uint8)
        mask[20:80, 20:80] = 255
        detections = [
            {"bbox": [30, 30, 40, 40], "confidence": 0.7, "class_name": "car"},
            {"bbox": [0, 0, 10, 10], "confidence": 0.8, "class_name": "car"},
            {"bbox": [25, 25, 35], "confidence": 0.9, "class_name": "car"},
        ]

        selected = detections_inside_mask(detections, mask)

        self.assertEqual(selected, [detections[0]])

    def test_accepts_opencv_singleton_channel_masks(self) -> None:
        mask = np.zeros((20, 20, 1), dtype=np.uint8)
        mask[5:15, 5:15, 0] = 255
        detection = {"bbox": [7, 7, 11, 11], "confidence": 0.7, "class_name": "car"}

        self.assertEqual(detections_inside_mask([detection], mask), [detection])


if __name__ == "__main__":
    unittest.main()
