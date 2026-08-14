from __future__ import annotations

import json
import math
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from lowalt_platform.services.catalog import ParkingCatalog, _coordinate_pairs


class ParkingCatalogTests(unittest.TestCase):
    def test_queries_support_and_bounds_without_changing_geometry(self) -> None:
        with TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            images, masks = root / "images", root / "masks"
            images.mkdir(); masks.mkdir()
            features = []
            for index, support in enumerate(("segformer_only", "vehicle_detected", "vehicle_row_supported")):
                aoi_id = f"aoi-{index}"
                (images / f"{aoi_id}.png").write_bytes(b"image")
                (masks / f"{aoi_id}_mask.png").write_bytes(b"mask")
                features.append({"type": "Feature", "geometry": {"type": "Polygon", "coordinates": [[[index, 0], [index + .5, 0], [index + .5, .5], [index, .5], [index, 0]]]}, "properties": {"aoi_id": aoi_id, "support_level": support}})
            candidates = root / "candidates.geojson"
            candidates.write_text(json.dumps({"type": "FeatureCollection", "features": features}), encoding="utf-8")
            summary = root / "summary.json"
            summary.write_text(json.dumps({"total_candidates": 3, "segformer_only": 1, "vehicle_detected": 1, "vehicle_row_supported": 1}), encoding="utf-8")

            catalog = ParkingCatalog.from_paths(candidates, summary, images, masks)
            selected = catalog.query(bounds=(0.8, -1, 1.8, 1), support_levels={"vehicle_detected"}, limit=10)

            self.assertEqual([item["properties"]["aoi_id"] for item in selected], ["aoi-1"])
            self.assertEqual(catalog.detail("aoi-1")["image_name"], "aoi-1.png")
            with self.assertRaises(KeyError):
                catalog.detail("../secret")

            with self.assertRaisesRegex(ValueError, "west.*east"):
                catalog.query(bounds=(2, 0, 1, 1))

    def test_rejects_non_finite_geometry(self) -> None:
        with self.assertRaisesRegex(ValueError, "finite"):
            list(_coordinate_pairs([math.nan, 30]))


if __name__ == "__main__":
    unittest.main()
