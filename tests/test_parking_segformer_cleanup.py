from __future__ import annotations

import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
import json

import numpy as np

from parking_map.tile_georeference import mask_to_geographic_geometry
from parking.parking_segformer_cleanup import (
    clean_parking_mask,
    merge_exclusion_layers,
    resolve_image_paths,
)


IMAGE_NAME = "block_z19_br1_bc2_r174535_c875365.png"


def exclusion_feature(mask: np.ndarray, kind: str) -> dict:
    geometry = mask_to_geographic_geometry(
        mask,
        IMAGE_NAME,
        minimum_component_area=1,
        simplify_fraction=0,
    ).geometry
    return {
        "type": "Feature",
        "geometry": geometry,
        "properties": {"exclusion_kind": kind},
    }


class SegformerParkingCleanupTest(unittest.TestCase):
    def test_additional_exclusions_are_combined_without_mutating_inputs(self) -> None:
        original = {"type": "FeatureCollection", "features": [{"id": "building"}]}
        additional = {"type": "FeatureCollection", "features": [{"id": "road"}]}

        merged = merge_exclusion_layers(original, additional)

        self.assertEqual([feature["id"] for feature in merged["features"]], ["building", "road"])
        self.assertEqual(len(original["features"]), 1)
        self.assertEqual(len(additional["features"]), 1)

    def test_manifest_limits_images_without_copying_them(self) -> None:
        with TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            images = root / "images"
            images.mkdir()
            selected = images / "selected.png"
            unselected = images / "unselected.png"
            selected.write_bytes(b"selected")
            unselected.write_bytes(b"unselected")
            manifest = root / "manifest.json"
            manifest.write_text(json.dumps({"aois": [{"image": selected.name}]}))

            self.assertEqual(resolve_image_paths(images, manifest), [selected])

    def test_building_and_public_road_are_removed_without_merging_components(self) -> None:
        raw_mask = np.zeros((100, 100), dtype=np.uint8)
        raw_mask[10:90, 10:90] = 255
        building = np.zeros_like(raw_mask)
        building[30:60, 30:60] = 255
        road = np.zeros_like(raw_mask)
        road[10:20, 10:90] = 255

        result = clean_parking_mask(
            raw_mask,
            {
                "type": "FeatureCollection",
                "features": [
                    exclusion_feature(building, "building"),
                    exclusion_feature(road, "public_road"),
                ],
            },
            IMAGE_NAME,
            minimum_component_area_px=20,
        )

        self.assertEqual(int(result.cleaned_mask[40, 40]), 0)
        self.assertEqual(int(result.cleaned_mask[15, 50]), 0)
        self.assertEqual(int(result.cleaned_mask[70, 70]), 255)
        self.assertGreater(result.removed_pixels, 0)
        self.assertEqual(result.component_count, 1)


if __name__ == "__main__":
    unittest.main()
