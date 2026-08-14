from __future__ import annotations

import math
import unittest

import numpy as np

from parking_map.osm_road_exclusions import build_aoi_road_exclusion
from parking_map.tile_georeference import pixel_to_lonlat


IMAGE_NAME = "block_z19_br1_bc2_r174535_c875365.png"


def osm_point(pixel_x: float, pixel_y: float) -> dict[str, float]:
    longitude, latitude = pixel_to_lonlat(19, 174535, 875365, pixel_x, pixel_y)
    return {"lon": longitude, "lat": latitude}


class OsmRoadExclusionTest(unittest.TestCase):
    def test_drivable_road_is_rasterized_with_source_attribution(self) -> None:
        result = build_aoi_road_exclusion(
            {
                "elements": [{
                    "type": "way",
                    "id": 11,
                    "tags": {"highway": "residential", "name": "Test Road"},
                    "geometry": [osm_point(10, 40), osm_point(90, 40)],
                }],
            },
            IMAGE_NAME,
            (100, 100),
        )

        self.assertGreater(np.count_nonzero(result.mask), 0)
        self.assertEqual(result.used_way_ids, (11,))
        feature = result.feature_collection["features"][0]
        self.assertEqual(feature["properties"]["exclusion_kind"], "public_road")
        self.assertEqual(feature["properties"]["source"], "OpenStreetMap")
        self.assertIn("OpenStreetMap contributors", feature["properties"]["attribution"])

    def test_disconnected_roads_remain_disconnected(self) -> None:
        result = build_aoi_road_exclusion(
            {
                "elements": [
                    {
                        "type": "way",
                        "id": 2,
                        "tags": {"highway": "service"},
                        "geometry": [osm_point(10, 20), osm_point(90, 20)],
                    },
                    {
                        "type": "way",
                        "id": 1,
                        "tags": {"highway": "service"},
                        "geometry": [osm_point(10, 80), osm_point(90, 80)],
                    },
                ],
            },
            IMAGE_NAME,
            (100, 100),
        )

        self.assertEqual(result.used_way_ids, (1, 2))
        self.assertEqual(result.component_count, 2)
        self.assertEqual(result.feature_collection["features"][0]["geometry"]["type"], "MultiPolygon")

    def test_internal_private_and_nonfinite_ways_are_not_exclusions(self) -> None:
        result = build_aoi_road_exclusion(
            {
                "elements": [
                    {
                        "type": "way",
                        "id": 1,
                        "tags": {"highway": "service", "service": "parking_aisle"},
                        "geometry": [osm_point(10, 20), osm_point(90, 20)],
                    },
                    {
                        "type": "way",
                        "id": 2,
                        "tags": {"highway": "residential", "access": "private"},
                        "geometry": [osm_point(10, 50), osm_point(90, 50)],
                    },
                    {
                        "type": "way",
                        "id": 3,
                        "tags": {"highway": "primary"},
                        "geometry": [{"lon": math.nan, "lat": 30.0}, osm_point(90, 80)],
                    },
                ],
            },
            IMAGE_NAME,
            (100, 100),
        )

        self.assertEqual(np.count_nonzero(result.mask), 0)
        self.assertEqual(result.used_way_ids, ())
        self.assertEqual(result.feature_collection["features"], [])
        self.assertEqual(result.coverage_status, "no_matching_public_roads")


if __name__ == "__main__":
    unittest.main()
