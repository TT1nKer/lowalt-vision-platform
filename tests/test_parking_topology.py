import unittest

from parking_map.schema import MapFeature, MapObjectType, ReviewState
from parking_map.topology import gate_map, validate_geometry


def polygon(x1, y1, x2, y2):
    return {
        "type": "Polygon",
        "coordinates": [[[x1, y1], [x2, y1], [x2, y2], [x1, y2], [x1, y1]]],
    }


class ParkingTopologyTests(unittest.TestCase):
    def test_self_intersection_is_rejected(self):
        bow_tie = {
            "type": "Polygon",
            "coordinates": [[[0, 0], [2, 2], [0, 2], [2, 0], [0, 0]]],
        }

        self.assertIn("self_intersection", validate_geometry(bow_tie))

    def test_child_outside_parent_fails_topology_gate(self):
        facility = MapFeature(
            object_id="facility-1",
            object_type=MapObjectType.PARKING_FACILITY,
            geometry=polygon(0, 0, 10, 10),
            review_state=ReviewState.AUTO_ACCEPT,
            source_version="test",
        )
        zone = MapFeature(
            object_id="zone-1",
            object_type=MapObjectType.PARKING_ZONE,
            geometry=polygon(9, 9, 12, 12),
            parent_id="facility-1",
            parent_type=MapObjectType.PARKING_FACILITY,
            review_state=ReviewState.AUTO_ACCEPT,
            source_version="test",
        )

        result = gate_map([facility, zone])

        self.assertEqual(result.decision, ReviewState.AUTO_REJECT)
        self.assertIn("child_outside_parent:zone-1", result.issues)

    def test_valid_multipolygon_is_not_collapsed_or_rejected(self):
        geometry = {
            "type": "MultiPolygon",
            "coordinates": [
                polygon(0, 0, 2, 2)["coordinates"],
                polygon(8, 8, 10, 10)["coordinates"],
            ],
        }
        feature = MapFeature(
            object_id="facility-1",
            object_type=MapObjectType.PARKING_FACILITY,
            geometry=geometry,
            review_state=ReviewState.ABSTAIN,
            source_version="test",
        )

        result = gate_map([feature])

        self.assertEqual(result.decision, ReviewState.AUTO_ACCEPT)
        self.assertEqual(len(feature.geometry["coordinates"]), 2)


if __name__ == "__main__":
    unittest.main()
