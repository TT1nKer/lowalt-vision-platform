import unittest

import numpy as np

from parking_map.functional_zoning import build_functional_zones, measure_zone_features


def rectangle(y1: int, x1: int, y2: int, x2: int) -> np.ndarray:
    result = np.zeros((100, 100), dtype=np.uint8)
    result[y1:y2, x1:x2] = 255
    return result


class ParkingFunctionalZoningTests(unittest.TestCase):
    def test_two_parking_bands_create_only_the_between_band_aisle(self):
        facility = rectangle(5, 5, 95, 95)
        first_band = rectangle(20, 20, 80, 30)
        second_band = rectangle(20, 60, 80, 70)

        result = build_functional_zones(
            facility,
            [first_band, second_band],
            np.zeros_like(facility),
            maximum_between_band_distance_px=35,
            minimum_component_area_px=20,
        )

        self.assertEqual(result.parking_band_mask[50, 25], 255)
        self.assertEqual(result.internal_aisle_mask[50, 45], 255)
        self.assertEqual(result.internal_aisle_mask[10, 10], 0)
        self.assertEqual(result.unknown_region_mask[10, 10], 255)

    def test_one_parking_band_cannot_self_validate_an_internal_aisle(self):
        facility = rectangle(5, 5, 95, 95)

        result = build_functional_zones(
            facility,
            [rectangle(20, 20, 80, 30)],
            np.zeros_like(facility),
            maximum_between_band_distance_px=35,
            minimum_component_area_px=20,
        )

        self.assertFalse(np.any(result.internal_aisle_mask))
        self.assertFalse(np.any(result.entrance_exit_mask))

    def test_entrance_requires_both_an_aisle_and_nearby_public_road(self):
        facility = rectangle(5, 5, 95, 95)
        bands = [rectangle(20, 20, 80, 30), rectangle(20, 60, 80, 70)]
        public_road = rectangle(0, 35, 8, 55)

        with_road = build_functional_zones(
            facility,
            bands,
            public_road,
            maximum_between_band_distance_px=35,
            entrance_proximity_px=8,
            minimum_component_area_px=20,
        )
        without_road = build_functional_zones(
            facility,
            bands,
            np.zeros_like(public_road),
            maximum_between_band_distance_px=35,
            entrance_proximity_px=8,
            minimum_component_area_px=20,
        )

        self.assertTrue(np.any(with_road.entrance_exit_mask))
        self.assertFalse(np.any(without_road.entrance_exit_mask))

    def test_zones_are_disjoint_and_exactly_partition_the_facility(self):
        facility = rectangle(5, 5, 95, 95)
        bands = [rectangle(20, 20, 80, 30), rectangle(20, 60, 80, 70)]
        public_road = rectangle(0, 35, 8, 55)

        result = build_functional_zones(
            facility,
            bands,
            public_road,
            maximum_between_band_distance_px=35,
            entrance_proximity_px=8,
            minimum_component_area_px=20,
        )

        masks = [
            result.parking_band_mask,
            result.internal_aisle_mask,
            result.entrance_exit_mask,
            result.unknown_region_mask,
        ]
        membership_count = sum((mask != 0).astype(np.uint8) for mask in masks)
        self.assertTrue(np.array_equal(membership_count != 0, facility != 0))
        self.assertLessEqual(int(membership_count.max()), 1)

    def test_mismatched_masks_are_rejected(self):
        with self.assertRaisesRegex(ValueError, "same-sized grayscale masks"):
            build_functional_zones(
                rectangle(5, 5, 95, 95),
                [np.zeros((50, 50), dtype=np.uint8)],
                np.zeros((100, 100), dtype=np.uint8),
            )

    def test_zone_features_record_area_shape_and_independent_adjacency(self):
        facility = rectangle(10, 10, 50, 50)
        zone = rectangle(20, 20, 30, 30)
        parking_bands = rectangle(20, 10, 30, 18)
        parking_bands |= rectangle(20, 32, 30, 40)
        public_road = rectangle(10, 20, 18, 30)

        result = measure_zone_features(
            zone,
            facility,
            public_road,
            parking_bands,
            adjacency_distance_px=3,
        )

        self.assertEqual(result.area_px, 100)
        self.assertEqual(result.component_count, 1)
        self.assertAlmostEqual(result.facility_area_fraction, 0.0625)
        self.assertAlmostEqual(result.bounding_box_extent, 1.0)
        self.assertEqual(result.adjacent_parking_band_components, 2)
        self.assertGreater(result.public_road_proximity_fraction, 0)
        self.assertFalse(result.touches_image_boundary)


if __name__ == "__main__":
    unittest.main()
