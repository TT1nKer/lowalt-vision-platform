import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
import json

import cv2
import numpy as np
from shapely.geometry import shape

from parking_map.e2_runner import RevalidatedBandSeed, select_revalidated_band_seeds
from parking_map.e2_map_builder import build_aoi_functional_map
from parking_map.e2_batch import run_functional_zoning_batch
from parking_map.tile_georeference import mask_to_geographic_geometry


def rectangle(y1: int, x1: int, y2: int, x2: int) -> np.ndarray:
    result = np.zeros((120, 120), dtype=np.uint8)
    result[y1:y2, x1:x2] = 255
    return result


class ParkingE2RunnerTests(unittest.TestCase):
    def test_direct_vehicle_row_seed_bypasses_broad_historic_candidate(self):
        image_name = "block_z19_br1_bc2_r174535_c875365.png"
        image = np.zeros((100, 100, 3), dtype=np.uint8)
        facility_mask = rectangle(5, 5, 95, 95)[:100, :100]
        row_mask = rectangle(40, 15, 60, 85)[:100, :100]

        geometry = mask_to_geographic_geometry(
            facility_mask,
            image_name,
            minimum_component_area=20,
            simplify_fraction=0,
        ).geometry
        result = build_aoi_functional_map(
            aoi_id="vehicle-row-aoi",
            image_name=image_name,
            image=image,
            surface_hypotheses={
                "type": "FeatureCollection",
                "features": [{"type": "Feature", "geometry": geometry, "properties": {}}],
            },
            parking_candidates={"type": "FeatureCollection", "features": []},
            evidence_layers={"agreed_vehicle_detections": []},
            exclusion_layers={"type": "FeatureCollection", "features": []},
            direct_band_seeds=(RevalidatedBandSeed(
                candidate_id="vehicle-row-001",
                mask=row_mask,
                support_kinds=("agreed_vehicle_alignment",),
            ),),
            minimum_component_area_px=20,
        )

        np.testing.assert_array_equal(result.masks.parking_band_mask, row_mask)
        self.assertEqual(result.aggregate_anchor_gate_status, "passed")
        self.assertEqual(result.revalidation_measurements, ())

    def test_markings_outside_clipped_facility_do_not_validate_a_band(self):
        image = np.zeros((120, 120, 3), dtype=np.uint8)
        for x in range(70, 111, 8):
            cv2.line(image, (x, 10), (x, 55), (255, 255, 255), 2)
        facility = rectangle(0, 0, 60, 60)
        candidate = rectangle(0, 0, 60, 115)

        result = select_revalidated_band_seeds(
            image,
            facility,
            [{"candidate_id": "historic-strong", "mask": candidate}],
            [],
            minimum_marking_strength=0.6,
            minimum_agreed_vehicles=2,
            minimum_seed_area_px=50,
        )

        self.assertEqual(result.accepted, ())
        self.assertEqual(result.measurements[0].marking_strength, 0.0)

    def test_two_agreed_vehicles_inside_clipped_candidate_validate_a_band_seed(self):
        image = np.zeros((120, 120, 3), dtype=np.uint8)
        facility = rectangle(10, 10, 100, 100)
        candidate = rectangle(20, 20, 80, 80)

        result = select_revalidated_band_seeds(
            image,
            facility,
            [{"candidate_id": "vehicle-supported", "mask": candidate}],
            [
                {"bbox": [25, 25, 35, 35]},
                {"bbox": [50, 50, 60, 60]},
                {"bbox": [105, 105, 115, 115]},
            ],
            minimum_marking_strength=0.6,
            minimum_agreed_vehicles=2,
            minimum_seed_area_px=50,
        )

        self.assertEqual(len(result.accepted), 1)
        self.assertEqual(result.accepted[0].candidate_id, "vehicle-supported")
        self.assertEqual(result.accepted[0].support_kinds, ("agreed_vehicles",))
        self.assertEqual(result.measurements[0].agreed_vehicle_count, 2)
        self.assertIsInstance(result.measurements[0].agreed_vehicle_count, int)

    def test_marking_only_union_covering_most_of_facility_is_downgraded_to_unknown(self):
        image = np.zeros((120, 120, 3), dtype=np.uint8)
        for x in range(12, 109, 8):
            cv2.line(image, (x, 15), (x, 105), (255, 255, 255), 2)
        facility = rectangle(10, 10, 110, 110)
        first_candidate = rectangle(10, 10, 110, 60)
        second_candidate = rectangle(10, 60, 110, 108)

        result = select_revalidated_band_seeds(
            image,
            facility,
            [
                {"candidate_id": "marking-left", "mask": first_candidate},
                {"candidate_id": "marking-right", "mask": second_candidate},
            ],
            [],
            minimum_marking_strength=0.6,
            minimum_agreed_vehicles=2,
            minimum_seed_area_px=50,
            maximum_marking_only_union_fraction=0.8,
        )

        self.assertEqual(result.accepted, ())
        self.assertEqual(result.aggregate_gate_status, "rejected_marking_only_union_too_large")
        self.assertGreater(result.accepted_union_facility_fraction, 0.8)
        self.assertTrue(all(not measurement.accepted for measurement in result.measurements))
        self.assertTrue(all(
            "marking_only_union_too_large" in measurement.rejection_reasons
            for measurement in result.measurements
        ))

    def test_aoi_builder_exports_abstaining_hierarchy_and_numeric_zone_features(self):
        image_name = "block_z19_br1_bc2_r174535_c875365.png"
        image = np.zeros((100, 100, 3), dtype=np.uint8)
        facility_mask = rectangle(5, 5, 95, 95)[:100, :100]
        first_band_mask = rectangle(20, 20, 80, 30)[:100, :100]
        second_band_mask = rectangle(20, 60, 80, 70)[:100, :100]
        road_mask = rectangle(0, 35, 8, 55)[:100, :100]

        def geometry(mask):
            return mask_to_geographic_geometry(
                mask,
                image_name,
                minimum_component_area=20,
                simplify_fraction=0,
            ).geometry

        result = build_aoi_functional_map(
            aoi_id="synthetic-aoi",
            image_name=image_name,
            image=image,
            surface_hypotheses={
                "type": "FeatureCollection",
                "features": [{
                    "type": "Feature",
                    "geometry": geometry(facility_mask),
                    "properties": {
                        "qualified_anchor_candidate_ids": ["band-1", "band-2"]
                    },
                }],
            },
            parking_candidates={
                "type": "FeatureCollection",
                "features": [
                    {
                        "type": "Feature",
                        "geometry": geometry(first_band_mask),
                        "properties": {"candidate_id": "band-1"},
                    },
                    {
                        "type": "Feature",
                        "geometry": geometry(second_band_mask),
                        "properties": {"candidate_id": "band-2"},
                    },
                ],
            },
            evidence_layers={
                "agreed_vehicle_detections": [
                    {"bbox": [21, 25, 25, 29]},
                    {"bbox": [24, 45, 28, 49]},
                    {"bbox": [61, 25, 65, 29]},
                    {"bbox": [64, 45, 68, 49]},
                ]
            },
            exclusion_layers={
                "type": "FeatureCollection",
                "features": [{
                    "type": "Feature",
                    "geometry": geometry(road_mask),
                    "properties": {"exclusion_kind": "public_road"},
                }],
            },
            maximum_between_band_distance_px=35,
            entrance_proximity_px=8,
            minimum_component_area_px=20,
        )

        features = result.feature_collection["features"]
        object_types = {feature["properties"]["object_type"] for feature in features}
        self.assertEqual(
            object_types,
            {
                "parking_facility",
                "parking_zone",
                "parking_band",
                "internal_aisle",
                "entrance_exit",
                "unknown_region",
            },
        )
        self.assertTrue(all(f["properties"]["review_state"] == "abstain" for f in features))
        self.assertTrue(all(f["geometry"]["type"] in {"Polygon", "MultiPolygon"} for f in features))
        self.assertEqual(len(result.zone_feature_records), 4)
        self.assertTrue(all(record["truth_status"] == "evidence_only_not_ground_truth" for record in result.zone_feature_records))
        self.assertTrue(all(isinstance(record["area_px"], int) for record in result.zone_feature_records))

    def test_batch_writes_map_overlay_revalidation_features_and_summary(self):
        image_name = "block_z19_br1_bc2_r174535_c875365.png"
        aoi_id = "synthetic-aoi"
        image = np.zeros((100, 100, 3), dtype=np.uint8)
        facility_mask = rectangle(5, 5, 95, 95)[:100, :100]
        first_band_mask = rectangle(20, 20, 80, 30)[:100, :100]
        second_band_mask = rectangle(20, 60, 80, 70)[:100, :100]
        road_mask = rectangle(0, 35, 8, 55)[:100, :100]

        def geometry(mask):
            return mask_to_geographic_geometry(
                mask,
                image_name,
                minimum_component_area=20,
                simplify_fraction=0,
            ).geometry

        surface = {
            "type": "FeatureCollection",
            "features": [{
                "type": "Feature",
                "geometry": geometry(facility_mask),
                "properties": {"qualified_anchor_candidate_ids": ["band-1", "band-2"]},
            }],
        }
        candidates = {
            "type": "FeatureCollection",
            "features": [
                {"type": "Feature", "geometry": geometry(first_band_mask), "properties": {"candidate_id": "band-1"}},
                {"type": "Feature", "geometry": geometry(second_band_mask), "properties": {"candidate_id": "band-2"}},
            ],
        }
        evidence = {
            "image": image_name,
            "agreed_vehicle_detections": [
                {"bbox": [21, 25, 25, 29]},
                {"bbox": [24, 45, 28, 49]},
                {"bbox": [61, 25, 65, 29]},
                {"bbox": [64, 45, 68, 49]},
            ],
        }
        exclusions = {
            "type": "FeatureCollection",
            "features": [{
                "type": "Feature",
                "geometry": geometry(road_mask),
                "properties": {"exclusion_kind": "public_road"},
            }],
        }

        with TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            images = root / "images"
            surface_root = root / "surfaces"
            evidence_root = root / "evidence"
            output = root / "output"
            images.mkdir()
            (surface_root / aoi_id).mkdir(parents=True)
            (evidence_root / aoi_id).mkdir(parents=True)
            cv2.imwrite(str(images / image_name), image)
            (surface_root / aoi_id / "surface_hypothesis.geojson").write_text(json.dumps(surface))
            (evidence_root / aoi_id / "parking_candidates.geojson").write_text(json.dumps(candidates))
            (evidence_root / aoi_id / "evidence_layers.json").write_text(json.dumps(evidence))
            (evidence_root / aoi_id / "exclusion_layers.geojson").write_text(json.dumps(exclusions))

            summary = run_functional_zoning_batch(
                images=images,
                surface_root=surface_root,
                evidence_root=evidence_root,
                output_dir=output,
                maximum_between_band_distance_px=35,
                entrance_proximity_px=8,
                minimum_component_area_px=20,
                east_west_metres_per_pixel=0.1,
                north_south_metres_per_pixel=0.2,
            )

            self.assertEqual(summary["totals"]["aois"], 1)
            self.assertEqual(summary["totals"]["parking_band_aois"], 1)
            self.assertEqual(summary["totals"].get("aggregate_marking_only_rejected_aois", 0), 0)
            self.assertTrue((output / aoi_id / "functional_zones.geojson").is_file())
            self.assertTrue((output / aoi_id / "functional_zones_overlay.jpg").is_file())
            self.assertTrue((output / aoi_id / "anchor_revalidation.json").is_file())
            revalidation = json.loads(
                (output / aoi_id / "anchor_revalidation.json").read_text()
            )
            self.assertEqual(revalidation["aggregate_gate_status"], "passed")
            self.assertGreater(revalidation["accepted_union_facility_fraction"], 0)
            self.assertEqual(
                summary["policy"]["maximum_between_band_distance_metres"],
                {"east_west": 3.5, "north_south": 7.0},
            )
            self.assertEqual(summary["policy"]["minimum_component_area_square_metres"], 0.4)
            feature_lines = (output / "zone_features.jsonl").read_text().splitlines()
            self.assertEqual(len(feature_lines), 4)
            self.assertTrue((output / "summary.json").is_file())

    def test_subthreshold_unknown_fragments_remain_measured_without_aborting_geojson(self):
        image_name = "block_z19_br1_bc2_r174535_c875365.png"
        image = np.zeros((100, 100, 3), dtype=np.uint8)
        facility_mask = rectangle(5, 5, 95, 95)[:100, :100]
        nearly_complete_band = facility_mask.copy()
        nearly_complete_band[50:52, 50:52] = 0

        def geometry(mask):
            return mask_to_geographic_geometry(
                mask,
                image_name,
                minimum_component_area=1,
                simplify_fraction=0,
            ).geometry

        result = build_aoi_functional_map(
            aoi_id="tiny-residual-aoi",
            image_name=image_name,
            image=image,
            surface_hypotheses={
                "type": "FeatureCollection",
                "features": [{
                    "type": "Feature",
                    "geometry": geometry(facility_mask),
                    "properties": {"qualified_anchor_candidate_ids": ["band"]},
                }],
            },
            parking_candidates={
                "type": "FeatureCollection",
                "features": [{
                    "type": "Feature",
                    "geometry": geometry(nearly_complete_band),
                    "properties": {"candidate_id": "band"},
                }],
            },
            evidence_layers={
                "agreed_vehicle_detections": [
                    {"bbox": [20, 20, 24, 24]},
                    {"bbox": [30, 30, 34, 34]},
                ]
            },
            exclusion_layers={"type": "FeatureCollection", "features": []},
            minimum_component_area_px=20,
        )

        object_types = {
            feature["properties"]["object_type"]
            for feature in result.feature_collection["features"]
        }
        self.assertNotIn("unknown_region", object_types)
        unknown_record = next(
            record for record in result.zone_feature_records
            if record["object_type"] == "unknown_region"
        )
        self.assertGreater(unknown_record["area_px"], 0)
        self.assertLess(unknown_record["area_px"], 20)
        self.assertEqual(unknown_record["geojson_export_status"], "below_minimum_component_area")

    def test_exported_child_geometries_remain_inside_ragged_facility_boundary(self):
        image_name = "block_z19_br1_bc2_r174535_c875365.png"
        image = np.zeros((1024, 1024, 3), dtype=np.uint8)
        facility_mask = np.zeros((1024, 1024), dtype=np.uint8)
        boundary = [(80, 80)]
        boundary.extend(
            (80 + ((y // 20) % 2) * 8, y)
            for y in range(80, 945, 20)
        )
        boundary.extend([(944, 944), (944, 80)])
        cv2.fillPoly(facility_mask, [np.asarray(boundary, dtype=np.int32)], 255)
        band_mask = facility_mask.copy()
        band_mask[:, 600:] = 0

        def geometry(mask):
            return mask_to_geographic_geometry(
                mask,
                image_name,
                minimum_component_area=100,
                simplify_fraction=0,
            ).geometry

        result = build_aoi_functional_map(
            aoi_id="ragged-aoi",
            image_name=image_name,
            image=image,
            surface_hypotheses={
                "type": "FeatureCollection",
                "features": [{
                    "type": "Feature",
                    "geometry": geometry(facility_mask),
                    "properties": {"qualified_anchor_candidate_ids": ["band"]},
                }],
            },
            parking_candidates={
                "type": "FeatureCollection",
                "features": [{
                    "type": "Feature",
                    "geometry": geometry(band_mask),
                    "properties": {"candidate_id": "band"},
                }],
            },
            evidence_layers={
                "agreed_vehicle_detections": [
                    {"bbox": [200, 200, 220, 220]},
                    {"bbox": [300, 300, 320, 320]},
                ]
            },
            exclusion_layers={"type": "FeatureCollection", "features": []},
            minimum_component_area_px=100,
        )

        features = result.feature_collection["features"]
        by_id = {feature["properties"]["object_id"]: feature for feature in features}
        for feature in features:
            parent_id = feature["properties"].get("parent_id")
            if parent_id:
                self.assertTrue(
                    shape(by_id[parent_id]["geometry"]).covers(shape(feature["geometry"])),
                    feature["properties"]["object_type"],
                )


if __name__ == "__main__":
    unittest.main()
