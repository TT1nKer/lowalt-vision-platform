import unittest

import cv2
import numpy as np

from parking_map.image_evidence import (
    geographic_geometry_to_mask,
    match_vehicle_detections,
    measure_marking_evidence,
    measure_vehicle_arrangement,
)
from parking_map.tile_georeference import pixel_to_lonlat


def pixel_polygon(filename, x1, y1, x2, y2):
    zoom, row, col = 19, 174475, 875337
    ring = [
        pixel_to_lonlat(zoom, row, col, x1, y1),
        pixel_to_lonlat(zoom, row, col, x2, y1),
        pixel_to_lonlat(zoom, row, col, x2, y2),
        pixel_to_lonlat(zoom, row, col, x1, y2),
        pixel_to_lonlat(zoom, row, col, x1, y1),
    ]
    return {"type": "Polygon", "coordinates": [[list(point) for point in ring]]}


class ParkingImageEvidenceTests(unittest.TestCase):
    def test_geographic_polygon_rasterization_preserves_hole_and_components(self):
        filename = "block_z19_br0_bc0_r174475_c875337.png"
        outer = pixel_polygon(filename, 10, 10, 90, 90)["coordinates"]
        hole = pixel_polygon(filename, 30, 30, 70, 70)["coordinates"][0]
        second = pixel_polygon(filename, 120, 20, 160, 60)["coordinates"]
        geometry = {
            "type": "MultiPolygon",
            "coordinates": [[outer[0], hole], second],
        }

        mask = geographic_geometry_to_mask(geometry, filename, (200, 200))

        self.assertEqual(mask[20, 20], 255)
        self.assertEqual(mask[50, 50], 0)
        self.assertEqual(mask[40, 140], 255)
        self.assertEqual(mask[180, 180], 0)

    def test_parallel_repeated_markings_score_above_blank_candidate(self):
        image = np.zeros((256, 256, 3), dtype=np.uint8)
        for x in range(40, 201, 32):
            cv2.line(image, (x, 70), (x, 120), (255, 255, 255), 3)
        candidate_mask = np.zeros((256, 256), dtype=np.uint8)
        candidate_mask[40:150, 20:230] = 255

        marked = measure_marking_evidence(image, candidate_mask)
        blank = measure_marking_evidence(np.zeros_like(image), candidate_mask)

        self.assertGreaterEqual(marked.segment_count, 5)
        self.assertGreater(marked.strength, 0.6)
        self.assertEqual(blank.segment_count, 0)
        self.assertEqual(blank.strength, 0.0)

    def test_candidate_mask_boundary_is_not_counted_as_a_parking_marking(self):
        uniform = np.full((256, 256, 3), 180, dtype=np.uint8)
        candidate_mask = np.zeros((256, 256), dtype=np.uint8)
        candidate_mask[40:150, 20:230] = 255

        measured = measure_marking_evidence(uniform, candidate_mask)

        self.assertEqual(measured.segment_count, 0)
        self.assertEqual(measured.strength, 0.0)

    def test_vehicle_arrangement_requires_multiple_agreed_instances(self):
        aligned_boxes = [
            [20, 40, 40, 55],
            [50, 40, 70, 55],
            [80, 40, 100, 55],
            [110, 40, 130, 55],
        ]

        aligned = measure_vehicle_arrangement(aligned_boxes)
        isolated = measure_vehicle_arrangement(aligned_boxes[:1])

        self.assertGreater(aligned.strength, 0.6)
        self.assertGreater(aligned.alignment_ratio, 3.0)
        self.assertEqual(isolated.strength, 0.0)

    def test_vehicle_agreement_is_one_to_one_and_drops_unmatched_boxes(self):
        first = [
            {"bbox": [0, 0, 10, 10], "confidence": 0.8, "class_name": "car"},
            {"bbox": [1, 0, 11, 10], "confidence": 0.7, "class_name": "car"},
            {"bbox": [50, 50, 60, 60], "confidence": 0.9, "class_name": "truck"},
        ]
        second = [
            {"bbox": [0, 0, 10, 10], "confidence": 0.75, "class_name": "car"},
            {"bbox": [80, 80, 90, 90], "confidence": 0.9, "class_name": "car"},
        ]

        matches = match_vehicle_detections(first, second, minimum_iou=0.5)

        self.assertEqual(len(matches), 1)
        self.assertEqual(matches[0].bbox, (0.0, 0.0, 10.0, 10.0))
        self.assertAlmostEqual(matches[0].agreement_iou, 1.0)


if __name__ == "__main__":
    unittest.main()
