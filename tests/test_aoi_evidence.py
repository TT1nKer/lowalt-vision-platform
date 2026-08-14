import unittest
from pathlib import Path
import tempfile

from PIL import Image

from parking_map.aoi_evidence import (
    build_aoi_manifest,
    derive_aoi_candidates,
    write_evidence_packages,
)


class AoiEvidenceTests(unittest.TestCase):
    def test_manifest_is_deterministic_unique_and_explicitly_not_truth(self):
        candidates = []
        for group in ("multi_component", "inflated_obb", "single_component", "negative", "edge", "geo_spread"):
            for index in range(12):
                candidates.append({
                    "aoi_id": f"{group}-{index}",
                    "image": f"block_z19_br{index}_bc{index}_r{174475 + index * 4}_c{875337 + index * 4}.png",
                    "strata": [group],
                    "evidence": {"source": "existing_audit"},
                })
        quotas = {
            "multi_component": 8,
            "inflated_obb": 8,
            "single_component": 8,
            "negative": 8,
            "edge": 8,
            "geo_spread": 8,
        }

        first = build_aoi_manifest(candidates, quotas, dataset_version="imagery-proxy-v1")
        second = build_aoi_manifest(list(reversed(candidates)), quotas, dataset_version="imagery-proxy-v1")

        self.assertEqual(first, second)
        self.assertEqual(first["count"], 48)
        self.assertEqual(len({item["aoi_id"] for item in first["aois"]}), 48)
        self.assertEqual(first["truth_status"], "unverified_proxy_evidence")
        self.assertEqual(first["gsd_status"], "pending_confirmation")

    def test_candidates_preserve_geometry_risk_state_and_geographic_origin(self):
        image = "block_z19_br12_bc34_r174523_c875473.png"
        index_records = [{
            "target_id": f"{image}::0",
            "image": image,
            "bbox": [0, 40, 120, 200],
            "mask_file": "mask.png",
            "class_name": "parking lot",
            "confidence": 0.7,
        }]
        states = {f"{image}::0": {"label": "hard_negative"}}
        e0_records = [{
            "target_id": f"{image}::0",
            "metrics": {"valid_components": 3, "global_obb_area_ratio": 2.5},
        }]

        candidates = derive_aoi_candidates(index_records, states, e0_records)

        self.assertEqual(len(candidates), 1)
        candidate = candidates[0]
        self.assertEqual(candidate["tile_origin"], {"zoom": 19, "row": 174523, "col": 875473})
        self.assertEqual(candidate["block_grid"], {"row": 12, "col": 34})
        self.assertEqual(
            set(candidate["strata"]),
            {"multi_component", "inflated_obb", "negative", "edge", "geo_spread"},
        )
        self.assertEqual(candidate["evidence"]["targets"][0]["label"], "hard_negative")

    def test_writer_creates_one_record_and_preview_per_selected_aoi(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            images = root / "images"
            output = root / "evidence"
            images.mkdir()
            image_name = "block_z19_br0_bc0_r174475_c875337.png"
            Image.new("RGB", (1024, 1024), "white").save(images / image_name)
            manifest = {
                "schema_version": 1,
                "count": 1,
                "aois": [{
                    "aoi_id": "aoi-1",
                    "image": image_name,
                    "strata": ["geo_spread"],
                    "evidence": {"targets": []},
                }],
            }

            write_evidence_packages(manifest, images, output)

            self.assertTrue((output / "manifest.json").is_file())
            self.assertTrue((output / "aoi-1" / "evidence.json").is_file())
            with Image.open(output / "aoi-1" / "source_preview.jpg") as preview:
                self.assertEqual(preview.size, (512, 512))

    def test_geo_spread_quota_prefers_distinct_geographic_cells(self):
        candidates = [
            {
                "aoi_id": f"cell-{cell}-copy-{copy}",
                "image": f"image-{cell}-{copy}.png",
                "geo_cell": str(cell),
                "strata": ["geo_spread"],
                "evidence": {},
            }
            for cell in range(5)
            for copy in range(2)
        ]

        manifest = build_aoi_manifest(
            candidates,
            {"geo_spread": 5},
            dataset_version="test",
        )

        self.assertEqual({item["geo_cell"] for item in manifest["aois"]}, {"0", "1", "2", "3", "4"})


if __name__ == "__main__":
    unittest.main()
