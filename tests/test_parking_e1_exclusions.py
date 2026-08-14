import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np

from parking.parking_e1_exclusions import _resolve_prompts, _write_overlay
from parking_map.tile_georeference import pixel_to_lonlat


class ParkingE1ExclusionOverlayTests(unittest.TestCase):
    def test_prompt_selection_defaults_to_both_and_rejects_unknown_prompts(self):
        self.assertEqual(_resolve_prompts(None), ("building", "public road"))
        self.assertEqual(_resolve_prompts(["building", "building"]), ("building",))
        with self.assertRaisesRegex(ValueError, "unsupported exclusion prompt"):
            _resolve_prompts(["vegetation"])

    def test_public_road_exclusion_can_be_rendered(self):
        filename = "block_z19_br0_bc0_r174475_c875337.png"
        points = [
            pixel_to_lonlat(19, 174475, 875337, x, y)
            for x, y in ((10, 10), (90, 10), (90, 90), (10, 90), (10, 10))
        ]
        exclusions = [{
            "type": "Feature",
            "geometry": {"type": "Polygon", "coordinates": [[list(point) for point in points]]},
            "properties": {"source_prompt": "public road"},
        }]
        evaluated = {"features": []}

        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "overlay.jpg"
            _write_overlay(
                np.zeros((100, 100, 3), dtype=np.uint8),
                filename,
                exclusions,
                evaluated,
                output,
            )

            self.assertTrue(output.is_file())
            self.assertIsNotNone(cv2.imread(str(output)))


if __name__ == "__main__":
    unittest.main()
