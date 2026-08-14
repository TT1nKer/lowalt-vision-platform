import unittest

from audits.vehicle_sample_manifest import select_stratified_targets


class VehicleSampleManifestTests(unittest.TestCase):
    def test_selection_preserves_rare_size_and_review_strata(self):
        records = []
        for index in range(20):
            records.append({
                "target_id": f"common-{index}",
                "image": "2012-01-01_frame.jpg",
                "bbox": [10, 10, 30, 40],
                "confidence": 0.8,
                "class_name": "parked car",
                "source_prompts": ["parked car"],
                "prompt_hits": 1,
                "mask_file": f"common-{index}.png",
            })
        records.append({
            "target_id": "rare",
            "image": "2012-02-01_frame.jpg",
            "bbox": [0, 0, 8, 8],
            "confidence": 0.2,
            "class_name": "parked vehicle",
            "source_prompts": ["parked vehicle"],
            "prompt_hits": 1,
            "mask_file": "rare.png",
        })

        selected = select_stratified_targets(
            records, {"rare": {"label": "reject"}}, max_samples=4, image_size=(640, 640)
        )

        self.assertIn("rare", {target["target_id"] for target in selected})
        rare = next(target for target in selected if target["target_id"] == "rare")
        self.assertEqual(rare["audit_strata"]["size"], "<16")
        self.assertEqual(rare["audit_strata"]["review"], "reject")


if __name__ == "__main__":
    unittest.main()
