from __future__ import annotations

import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from parking.parking_map_aggregate import aggregate_map_layers


def feature(object_type: str, x: float) -> dict:
    return {
        "type": "Feature",
        "geometry": {
            "type": "Polygon",
            "coordinates": [[[x, 0], [x + 1, 0], [x + 1, 1], [x, 1], [x, 0]]],
        },
        "properties": {"object_type": object_type, "review_state": "abstain"},
    }


class ParkingMapAggregateTests(unittest.TestCase):
    def test_separates_supported_and_candidate_only_facilities(self) -> None:
        with TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            maps = root / "maps"
            output = root / "layers"
            for aoi_id, features in (
                ("supported", [feature("parking_facility", 0), feature("parking_band", 0)]),
                ("candidate", [feature("parking_facility", 2), feature("unknown_region", 2)]),
            ):
                directory = maps / aoi_id
                directory.mkdir(parents=True)
                (directory / "parking_map.geojson").write_text(json.dumps({
                    "type": "FeatureCollection",
                    "aoi_id": aoi_id,
                    "features": features,
                }))

            summary = aggregate_map_layers(maps, output, simplify_tolerance=0)

            self.assertEqual(summary["supported_parking_facilities"], 1)
            self.assertEqual(summary["candidate_only_facilities"], 1)
            supported = json.loads((output / "parking_facilities.geojson").read_text())
            candidates = json.loads((output / "facility_candidates_abstain.geojson").read_text())
            self.assertEqual(supported["features"][0]["properties"]["aoi_id"], "supported")
            self.assertEqual(candidates["features"][0]["properties"]["aoi_id"], "candidate")


if __name__ == "__main__":
    unittest.main()
