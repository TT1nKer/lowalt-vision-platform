from __future__ import annotations

import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

import cv2
import numpy as np

from parking.parking_segformer_candidates import export_positive_masks


class SegformerCandidateExportTests(unittest.TestCase):
    def test_exports_only_positive_masks_and_preserves_disconnected_polygons(self) -> None:
        with TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            images = root / "images"
            masks = root / "masks"
            output = root / "output"
            images.mkdir()
            masks.mkdir()
            positive_name = "block_z19_br1_bc2_r174535_c875365.png"
            empty_name = "block_z19_br2_bc3_r174539_c875369.png"
            image = np.zeros((100, 100, 3), dtype=np.uint8)
            cv2.imwrite(str(images / positive_name), image)
            cv2.imwrite(str(images / empty_name), image)
            positive = np.zeros((100, 100), dtype=np.uint8)
            positive[10:30, 10:30] = 255
            positive[60:90, 60:90] = 255
            cv2.imwrite(str(masks / f"{Path(positive_name).stem}_mask.png"), positive)
            cv2.imwrite(str(masks / f"{Path(empty_name).stem}_mask.png"), np.zeros_like(positive))

            summary = export_positive_masks(
                images=images,
                masks=masks,
                output_dir=output,
                minimum_component_area_px=100,
            )

            self.assertEqual(summary["positive_aois"], 1)
            manifest = json.loads((output / "manifest.json").read_text())
            self.assertEqual(manifest["count"], 1)
            self.assertEqual(manifest["aois"][0]["image"], positive_name)
            candidates = json.loads(
                (output / Path(positive_name).stem / "mask_geometries.geojson").read_text()
            )
            feature = candidates["features"][0]
            self.assertEqual(feature["geometry"]["type"], "MultiPolygon")
            self.assertEqual(feature["properties"]["component_count"], 2)
            surface = json.loads(
                (output / Path(positive_name).stem / "surface_hypothesis.geojson").read_text()
            )
            self.assertEqual(surface["features"][0]["geometry"]["type"], "MultiPolygon")
            self.assertEqual(
                surface["features"][0]["properties"]["qualified_anchor_candidate_ids"],
                [feature["properties"]["target_id"]],
            )
            exclusions = json.loads(
                (output / Path(positive_name).stem / "exclusion_layers.geojson").read_text()
            )
            self.assertEqual(exclusions["features"], [])
            self.assertEqual(exclusions["coverage_status"], "not_available")
            self.assertFalse((output / positive_name).exists())


if __name__ == "__main__":
    unittest.main()
