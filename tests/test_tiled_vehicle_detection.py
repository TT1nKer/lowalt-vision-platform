from __future__ import annotations

import unittest

from parking_map.tiled_detection import merge_overlapping_detections, tile_windows


class TiledVehicleDetectionTest(unittest.TestCase):
    def test_overlapping_windows_cover_image_edges(self) -> None:
        windows = tile_windows(1024, 1024, tile_size=512, overlap=128)

        self.assertEqual(len(windows), 9)
        self.assertIn((0, 0, 512, 512), windows)
        self.assertIn((512, 512, 1024, 1024), windows)

    def test_duplicate_boxes_from_neighboring_tiles_keep_higher_confidence(self) -> None:
        detections = [
            {"bbox": [100.0, 100.0, 130.0, 120.0], "confidence": 0.8, "class_name": "car"},
            {"bbox": [101.0, 100.0, 131.0, 120.0], "confidence": 0.9, "class_name": "car"},
            {"bbox": [300.0, 300.0, 330.0, 325.0], "confidence": 0.7, "class_name": "truck"},
        ]

        merged = merge_overlapping_detections(detections, maximum_iou=0.5)

        self.assertEqual(len(merged), 2)
        self.assertEqual(merged[0]["confidence"], 0.9)
        self.assertEqual(merged[1]["class_name"], "truck")

    def test_invalid_overlap_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "overlap"):
            tile_windows(1024, 1024, tile_size=512, overlap=512)


if __name__ == "__main__":
    unittest.main()
