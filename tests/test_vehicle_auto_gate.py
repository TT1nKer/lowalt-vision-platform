import unittest

from audits.vehicle_auto_gate import classify_vehicle_candidate


class VehicleAutoGateTests(unittest.TestCase):
    def test_accepts_only_multi_evidence_high_purity_candidate(self):
        target = {
            "bbox": [10, 10, 30, 50],
            "confidence": 0.82,
            "source_prompts": ["parked car", "vehicle in parking lot"],
            "prompt_hits": 2,
            "audit_strata": {"edge": "interior"},
        }
        geometry = {
            "empty_mask": False,
            "material_components": 1,
            "largest_component_fraction": 0.99,
            "mask_bbox_iou": 0.6,
        }

        decision = classify_vehicle_candidate(target, geometry, 0.7, 0.65)

        self.assertEqual(decision["decision"], "auto_accept")

    def test_empty_mask_is_rejected(self):
        decision = classify_vehicle_candidate({}, {"empty_mask": True}, 0.0, 0.0)

        self.assertEqual(decision["decision"], "auto_reject")
        self.assertIn("empty_mask", decision["reasons"])

    def test_missing_independent_agreement_abstains_instead_of_becoming_background(self):
        target = {
            "bbox": [10, 10, 30, 50],
            "confidence": 0.9,
            "source_prompts": ["parked car", "vehicle in parking lot"],
            "prompt_hits": 2,
            "audit_strata": {"edge": "interior"},
        }
        geometry = {
            "empty_mask": False,
            "material_components": 1,
            "largest_component_fraction": 0.99,
            "mask_bbox_iou": 0.8,
        }

        decision = classify_vehicle_candidate(target, geometry, 0.0, 0.0)

        self.assertEqual(decision["decision"], "abstain")
        self.assertIn("independent_model_disagreement", decision["reasons"])


if __name__ == "__main__":
    unittest.main()
