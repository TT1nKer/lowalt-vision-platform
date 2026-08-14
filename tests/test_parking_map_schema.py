import unittest

from parking_map.schema import EvidenceRef, MapFeature, MapObjectType, ReviewState


SQUARE = {
    "type": "Polygon",
    "coordinates": [[
        [120.0, 30.0], [120.001, 30.0], [120.001, 30.001],
        [120.0, 30.001], [120.0, 30.0],
    ]],
}


class ParkingMapSchemaTests(unittest.TestCase):
    def test_feature_exports_hierarchy_crs_evidence_and_uncertainty(self):
        feature = MapFeature(
            object_id="facility-1",
            object_type=MapObjectType.PARKING_FACILITY,
            geometry=SQUARE,
            review_state=ReviewState.ABSTAIN,
            evidence=(EvidenceRef("wmts", "block-z19-1", 0.8),),
            source_version="imagery-v1",
        )

        exported = feature.to_geojson()

        self.assertEqual(exported["type"], "Feature")
        self.assertEqual(exported["geometry"], SQUARE)
        self.assertEqual(exported["properties"]["object_type"], "parking_facility")
        self.assertEqual(exported["properties"]["review_state"], "abstain")
        self.assertEqual(exported["properties"]["crs"], "OGC:CRS84")
        self.assertEqual(exported["properties"]["evidence"][0]["source_id"], "block-z19-1")

    def test_polygon_objects_reject_bbox_as_truth_geometry(self):
        with self.assertRaisesRegex(ValueError, "Polygon or MultiPolygon"):
            MapFeature(
                object_id="zone-1",
                object_type=MapObjectType.PARKING_ZONE,
                geometry={"type": "BBox", "coordinates": [0, 0, 1, 1]},
                review_state=ReviewState.AUTO_REJECT,
                source_version="test",
            )

    def test_parking_space_requires_parking_band_parent(self):
        with self.assertRaisesRegex(ValueError, "parent_type"):
            MapFeature(
                object_id="space-1",
                object_type=MapObjectType.PARKING_SPACE,
                geometry=SQUARE,
                parent_id="facility-1",
                parent_type=MapObjectType.PARKING_FACILITY,
                review_state=ReviewState.ABSTAIN,
                source_version="test",
            )


if __name__ == "__main__":
    unittest.main()
