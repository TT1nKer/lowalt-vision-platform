from __future__ import annotations

import unittest

from PIL import Image

from lowalt_platform.services.parking_refinement import refine_parking_candidate


class ParkingRefinementTests(unittest.TestCase):
    def test_subtracts_all_evidence_and_never_expands_candidate(self) -> None:
        candidate = Image.new("L", (4, 1), 255)
        building = Image.new("L", (4, 1), 0); building.putpixel((0, 0), 255)
        vegetation = Image.new("L", (4, 1), 0); vegetation.putpixel((1, 0), 255)
        aisle = Image.new("L", (4, 1), 0); aisle.putpixel((2, 0), 255)

        refined = refine_parking_candidate(candidate, building, vegetation, aisle)

        self.assertEqual(list(refined.getdata()), [0, 0, 0, 255])

    def test_rejects_mismatched_sizes(self) -> None:
        with self.assertRaises(ValueError):
            refine_parking_candidate(
                Image.new("L", (2, 2)), Image.new("L", (1, 1)),
                Image.new("L", (2, 2)), Image.new("L", (2, 2)),
            )
