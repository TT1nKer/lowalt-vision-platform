from __future__ import annotations

import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from PIL import Image

from lowalt_platform.services.overview_builder import build_overview


class OverviewBuilderTests(unittest.TestCase):
    def test_builds_deterministic_grid_with_geographic_bounds(self) -> None:
        with TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            images = root / "images"; images.mkdir()
            Image.new("RGB", (8, 8), "red").save(images / "block_z19_br0_bc0_r100_c200.png")
            Image.new("RGB", (8, 8), "blue").save(images / "block_z19_br1_bc2_r104_c208.png")
            output, manifest_path = root / "overview.jpg", root / "overview.json"

            first = build_overview(images, output, manifest_path, thumbnail_size=4)
            first_bytes = output.read_bytes()
            second = build_overview(images, output, manifest_path, thumbnail_size=4)

            self.assertEqual(first, second)
            self.assertEqual(first_bytes, output.read_bytes())
            with Image.open(output) as overview:
                self.assertEqual(overview.size, (12, 8))
            self.assertEqual(json.loads(manifest_path.read_text())["image_count"], 2)
            self.assertEqual(len(first["bounds"]), 4)


if __name__ == "__main__":
    unittest.main()
