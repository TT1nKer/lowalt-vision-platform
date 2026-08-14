import unittest

import numpy as np

from parking_map.surface_hypothesis import select_anchor_candidate_ids, select_anchored_surfaces


def mask(y1, x1, y2, x2):
    result = np.zeros((100, 100), dtype=np.uint8)
    result[y1:y2, x1:x2] = 255
    return result


class ParkingSurfaceHypothesisTests(unittest.TestCase):
    def test_anchor_gate_rejects_large_weak_candidate_and_keeps_strong_local_evidence(self):
        selected = select_anchor_candidate_ids(
            [
                {"candidate_id": "large-weak", "mask_area_px": 260000, "marking_strength": 0.28, "agreed_vehicle_count": 0},
                {"candidate_id": "marked", "mask_area_px": 20000, "marking_strength": 0.8, "agreed_vehicle_count": 0},
                {"candidate_id": "two-vehicles", "mask_area_px": 30000, "marking_strength": 0.1, "agreed_vehicle_count": 2},
                {"candidate_id": "large-marked", "mask_area_px": 60000, "marking_strength": 0.9, "agreed_vehicle_count": 0},
            ],
            image_area_px=1_000_000,
        )

        self.assertEqual(selected, {"marked", "two-vehicles"})

    def test_unanchored_paved_surface_is_rejected(self):
        anchors = mask(10, 10, 20, 20)

        result = select_anchored_surfaces(
            [
                {"target_id": "anchored", "confidence": 0.6, "mask": mask(5, 5, 40, 40)},
                {"target_id": "unrelated", "confidence": 0.9, "mask": mask(60, 60, 95, 95)},
            ],
            anchors,
            np.zeros_like(anchors),
            minimum_surface_area_px=100,
            minimum_anchor_overlap_px=25,
        )

        self.assertEqual(result.accepted_target_ids, ("anchored",))
        self.assertEqual(result.component_count, 1)
        self.assertEqual(result.union_mask[70, 70], 0)

    def test_exclusion_splits_surface_without_reconnecting_components(self):
        anchors = mask(20, 10, 30, 90)
        surface = mask(10, 5, 60, 95)
        exclusion = mask(0, 45, 100, 55)

        result = select_anchored_surfaces(
            [{"target_id": "surface", "confidence": 0.8, "mask": surface}],
            anchors,
            exclusion,
            minimum_surface_area_px=100,
            minimum_anchor_overlap_px=25,
        )

        self.assertEqual(result.component_count, 2)
        self.assertEqual(result.union_mask[30, 50], 0)
        self.assertEqual(result.union_mask[30, 20], 255)
        self.assertEqual(result.union_mask[30, 80], 255)

    def test_low_confidence_and_fully_excluded_surfaces_are_not_used(self):
        anchors = mask(20, 20, 40, 40)
        result = select_anchored_surfaces(
            [
                {"target_id": "low", "confidence": 0.39, "mask": mask(10, 10, 50, 50)},
                {"target_id": "excluded", "confidence": 0.9, "mask": mask(10, 10, 50, 50)},
            ],
            anchors,
            mask(0, 0, 60, 60),
            minimum_surface_area_px=100,
            minimum_anchor_overlap_px=25,
        )

        self.assertEqual(result.accepted_target_ids, ())
        self.assertEqual(result.component_count, 0)
        self.assertFalse(np.any(result.union_mask))


if __name__ == "__main__":
    unittest.main()
