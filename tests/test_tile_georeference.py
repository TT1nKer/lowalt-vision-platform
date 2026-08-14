import unittest

import numpy as np

from parking_map.topology import validate_geometry
from parking_map.tile_georeference import (
    mask_to_geographic_geometry,
    parse_block_filename,
    pixel_to_lonlat,
    lonlat_to_pixel,
)


class TileGeoreferenceTests(unittest.TestCase):
    def test_pixel_lonlat_round_trip_preserves_subpixel_position(self):
        longitude, latitude = pixel_to_lonlat(19, 174475, 875337, 321.25, 654.75)

        pixel_x, pixel_y = lonlat_to_pixel(19, 174475, 875337, longitude, latitude)

        self.assertAlmostEqual(pixel_x, 321.25, places=6)
        self.assertAlmostEqual(pixel_y, 654.75, places=6)

    def test_epsg4326w_zoom_zero_has_two_columns_and_one_row(self):
        self.assertEqual(pixel_to_lonlat(0, 0, 0, 0, 0), (-180.0, 90.0))
        self.assertEqual(pixel_to_lonlat(0, 0, 0, 256, 256), (0.0, -90.0))

    def test_adjacent_tile_edges_have_identical_coordinates(self):
        left_edge = pixel_to_lonlat(19, 174475, 875337, 256, 128)
        right_edge = pixel_to_lonlat(19, 174475, 875338, 0, 128)
        self.assertEqual(left_edge, right_edge)

    def test_block_filename_keeps_zoom_and_absolute_tile_origin(self):
        block = parse_block_filename("block_z19_br0_bc20_r174475_c875337.png")
        self.assertEqual((block.zoom, block.block_row, block.block_col), (19, 0, 20))
        self.assertEqual((block.tile_row, block.tile_col), (174475, 875337))

    def test_disconnected_mask_becomes_two_geographic_polygons(self):
        mask = np.zeros((64, 64), dtype=np.uint8)
        mask[5:15, 5:20] = 255
        mask[40:55, 42:60] = 255

        result = mask_to_geographic_geometry(
            mask,
            "block_z19_br0_bc20_r174475_c875337.png",
            minimum_component_area=10,
        )

        self.assertEqual(result.component_count, 2)
        self.assertEqual(result.geometry["type"], "MultiPolygon")
        self.assertEqual(len(result.geometry["coordinates"]), 2)
        self.assertEqual(result.crs, "OGC:CRS84")

    def test_mask_hole_is_preserved_as_an_inner_ring(self):
        mask = np.zeros((64, 64), dtype=np.uint8)
        mask[5:55, 5:55] = 255
        mask[20:40, 20:40] = 0

        result = mask_to_geographic_geometry(
            mask,
            "block_z19_br0_bc20_r174475_c875337.png",
            minimum_component_area=10,
        )

        self.assertEqual(result.component_count, 1)
        self.assertEqual(result.geometry["type"], "Polygon")
        self.assertEqual(len(result.geometry["coordinates"]), 2)

    def test_point_touching_raster_is_split_into_valid_polygons(self):
        mask = np.zeros((30, 30), dtype=np.uint8)
        mask[2:12, 2:12] = 255
        mask[12:22, 12:22] = 255

        result = mask_to_geographic_geometry(
            mask,
            "block_z19_br0_bc0_r174475_c875337.png",
            minimum_component_area=10,
            simplify_fraction=0,
        )

        self.assertEqual(result.source_component_count, 1)
        self.assertEqual(result.component_count, 2)
        self.assertEqual(result.geometry["type"], "MultiPolygon")
        self.assertEqual(validate_geometry(result.geometry), ())


if __name__ == "__main__":
    unittest.main()
