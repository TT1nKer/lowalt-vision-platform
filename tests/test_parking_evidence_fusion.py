import unittest

from parking_map.evidence_fusion import subtract_exclusions
from parking_map.topology import validate_geometry


def polygon(x1, y1, x2, y2):
    return {
        "type": "Polygon",
        "coordinates": [[[x1, y1], [x2, y1], [x2, y2], [x1, y2], [x1, y1]]],
    }


class ParkingEvidenceFusionTests(unittest.TestCase):
    def test_internal_building_is_preserved_as_exclusion_hole(self):
        result = subtract_exclusions(
            polygon(0, 0, 10, 10),
            [polygon(3, 3, 7, 7)],
        )

        self.assertIsNotNone(result)
        self.assertEqual(result.geometry["type"], "Polygon")
        self.assertEqual(len(result.geometry["coordinates"]), 2)
        self.assertAlmostEqual(result.removed_fraction, 0.16)
        self.assertEqual(validate_geometry(result.geometry), ())

    def test_road_cut_splits_candidate_instead_of_bridging_it(self):
        result = subtract_exclusions(
            polygon(0, 0, 10, 10),
            [polygon(4, -1, 6, 11)],
        )

        self.assertIsNotNone(result)
        self.assertEqual(result.geometry["type"], "MultiPolygon")
        self.assertEqual(result.component_count, 2)
        self.assertEqual(validate_geometry(result.geometry), ())

    def test_fully_excluded_candidate_returns_no_geometry(self):
        result = subtract_exclusions(
            polygon(0, 0, 10, 10),
            [polygon(-1, -1, 11, 11)],
        )

        self.assertIsNone(result)


if __name__ == "__main__":
    unittest.main()
