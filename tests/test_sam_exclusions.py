import unittest

import numpy as np

from parking_map.sam_exclusions import measure_exclusion_effects, select_exclusion_masks


def mask(y1, x1, y2, x2):
    result = np.zeros((100, 100), dtype=np.uint8)
    result[y1:y2, x1:x2] = 255
    return result


class SamExclusionTests(unittest.TestCase):
    def test_building_requires_high_confidence_and_material_area(self):
        accepted = select_exclusion_masks(
            "building",
            [
                {"target_id": "high", "confidence": 0.8, "mask": mask(20, 20, 60, 60)},
                {"target_id": "low", "confidence": 0.69, "mask": mask(10, 10, 50, 50)},
                {"target_id": "tiny", "confidence": 0.95, "mask": mask(10, 10, 12, 12)},
            ],
        )

        self.assertEqual(accepted.accepted_target_ids, ("high",))
        self.assertEqual(int(np.count_nonzero(accepted.union_mask)), 1600)

    def test_public_road_requires_border_contact_and_stricter_semantics(self):
        accepted = select_exclusion_masks(
            "public road",
            [
                {"target_id": "edge", "confidence": 0.68, "mask": mask(0, 20, 90, 50)},
                {"target_id": "internal", "confidence": 0.9, "mask": mask(20, 20, 80, 50)},
                {"target_id": "low", "confidence": 0.64, "mask": mask(0, 50, 80, 80)},
            ],
        )

        self.assertEqual(accepted.accepted_target_ids, ("edge",))
        self.assertEqual(int(np.count_nonzero(accepted.union_mask)), 2700)

    def test_unknown_prompt_is_rejected_instead_of_guessing_policy(self):
        with self.assertRaisesRegex(ValueError, "unsupported exclusion prompt"):
            select_exclusion_masks("road", [])

    def test_effects_distinguish_existing_multipolygon_from_new_split(self):
        effects = measure_exclusion_effects(
            {"already-multi": 3, "newly-split": 1, "trimmed": 1},
            [
                {"candidate_id": "already-multi", "component_count": 3, "removed_fraction": 0.0},
                {"candidate_id": "newly-split", "component_count": 2, "removed_fraction": 0.2},
                {"candidate_id": "trimmed", "component_count": 1, "removed_fraction": 0.005},
            ],
        )

        self.assertEqual(effects["source_already_multicomponent"], 1)
        self.assertEqual(effects["newly_split_after_exclusion"], 1)
        self.assertEqual(effects["candidates_with_any_removed_area"], 2)
        self.assertEqual(effects["candidates_with_at_least_1pct_removed"], 1)
        self.assertEqual(effects["candidates_with_at_least_10pct_removed"], 1)


if __name__ == "__main__":
    unittest.main()
