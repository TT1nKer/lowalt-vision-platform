import unittest

import cv2
import numpy as np

from parking_map.e1_extraction import build_aoi_image_evidence
from parking_map.evidence_gate import ParkingEvidenceKind
from parking_map.tile_georeference import pixel_to_lonlat


def candidate_feature(filename, target_id, x1, y1, x2, y2):
    zoom, row, col = 19, 174475, 875337
    points = [
        pixel_to_lonlat(zoom, row, col, x1, y1),
        pixel_to_lonlat(zoom, row, col, x2, y1),
        pixel_to_lonlat(zoom, row, col, x2, y2),
        pixel_to_lonlat(zoom, row, col, x1, y2),
        pixel_to_lonlat(zoom, row, col, x1, y1),
    ]
    return {
        "type": "Feature",
        "geometry": {"type": "Polygon", "coordinates": [[list(point) for point in points]]},
        "properties": {"target_id": target_id},
    }


def detection(bbox, confidence=0.8):
    return {"bbox": bbox, "confidence": confidence, "class_name": "car"}


class ParkingE1ExtractionTests(unittest.TestCase):
    def test_builds_marking_and_agreed_vehicle_evidence_without_claiming_exclusion_coverage(self):
        filename = "block_z19_br0_bc0_r174475_c875337.png"
        image = np.zeros((256, 256, 3), dtype=np.uint8)
        for x in range(40, 201, 32):
            cv2.line(image, (x, 70), (x, 120), (255, 255, 255), 3)
        candidates = {
            "type": "FeatureCollection",
            "features": [candidate_feature(filename, "candidate-1", 20, 30, 230, 150)],
        }
        boxes = [
            [30, 80, 45, 95],
            [65, 80, 80, 95],
            [100, 80, 115, 95],
            [135, 80, 150, 95],
        ]

        evidence = build_aoi_image_evidence(
            "aoi-1",
            filename,
            image,
            candidates,
            [detection(box) for box in boxes],
            [detection(box, 0.75) for box in boxes],
        )

        signals = evidence["signals"]["candidate-1"]
        by_kind = {signal["kind"]: signal for signal in signals}
        self.assertGreater(by_kind[ParkingEvidenceKind.PARKING_MARKING.value]["strength"], 0.6)
        self.assertGreater(by_kind[ParkingEvidenceKind.VEHICLE_ARRANGEMENT.value]["strength"], 0.6)
        self.assertEqual(evidence["exclusions"], [])
        self.assertEqual(evidence["exclusion_coverage_status"], "unavailable")
        self.assertEqual(evidence["truth_status"], "evidence_only_not_ground_truth")


if __name__ == "__main__":
    unittest.main()
