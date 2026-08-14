import unittest

from uav_data.physical_scale import pixel_area_to_square_metres, pixels_to_metres
from uav_data.schema import GsdStatus, SpatialReference


class UavPhysicalScaleTests(unittest.TestCase):
    def test_directional_gsd_is_not_averaged(self):
        reference = SpatialReference(
            crs="OGC:CRS84",
            gsd_status=GsdStatus.DERIVED,
            gsd_x_metres=0.13,
            gsd_y_metres=0.15,
        )

        self.assertEqual(pixels_to_metres(reference, 10, 20), (1.3, 3.0))

    def test_pixel_area_uses_both_directional_scales(self):
        reference = SpatialReference(
            crs="OGC:CRS84",
            gsd_status=GsdStatus.OBSERVED,
            gsd_x_metres=0.1,
            gsd_y_metres=0.2,
        )

        self.assertAlmostEqual(pixel_area_to_square_metres(reference, 100), 2.0)

    def test_unknown_gsd_blocks_conversion(self):
        reference = SpatialReference(
            crs="LOCAL:X",
            gsd_status=GsdStatus.UNKNOWN,
        )

        with self.assertRaisesRegex(ValueError, "GSD is unknown"):
            pixel_area_to_square_metres(reference, 100)

    def test_negative_pixel_measurement_is_rejected(self):
        reference = SpatialReference(
            crs="LOCAL:X",
            gsd_status=GsdStatus.OBSERVED,
            gsd_x_metres=0.1,
            gsd_y_metres=0.1,
        )

        with self.assertRaisesRegex(ValueError, "non-negative"):
            pixels_to_metres(reference, -1, 3)


if __name__ == "__main__":
    unittest.main()
