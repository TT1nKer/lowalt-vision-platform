import unittest

import numpy as np

from parking.parking_representation_e0 import analyze_parking_mask


class ParkingRepresentationE0Tests(unittest.TestCase):
    def test_component_representation_avoids_global_obb_bridge(self):
        mask = np.zeros((100, 100), dtype=np.uint8)
        mask[10:20, 10:30] = 255
        mask[70:80, 70:90] = 255

        result = analyze_parking_mask(mask)

        self.assertEqual(result["valid_components"], 2)
        self.assertGreater(result["global_obb_area_ratio"], 5)
        self.assertLess(result["component_obb_area_ratio"], 1.2)
        self.assertAlmostEqual(result["largest_component_retained_fraction"], 0.5)

    def test_low_intensity_encoding_noise_is_not_a_component(self):
        mask = np.zeros((30, 30), dtype=np.uint8)
        mask[0, 0] = 3
        mask[5:15, 5:15] = 255

        result = analyze_parking_mask(mask)

        self.assertEqual(result["valid_components"], 1)
        self.assertEqual(result["mask_area_px"], 100)


if __name__ == "__main__":
    unittest.main()
