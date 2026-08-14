import unittest

import cv2
import numpy as np

from audits.vehicle_mask_geometry_audit import analyze_mask_geometry


class VehicleMaskGeometryAuditTests(unittest.TestCase):
    def test_ignores_low_intensity_lossy_compression_halo(self):
        mask = np.zeros((20, 20), dtype=np.uint8)
        mask[1, 1] = 3
        mask[5:10, 6:12] = 255

        result = analyze_mask_geometry(mask, [6, 5, 12, 10])

        self.assertEqual(result["mask_bbox"], [6.0, 5.0, 12.0, 10.0])
        self.assertEqual(result["low_intensity_nonzero_px"], 1)

    def test_reports_separate_material_components_without_collapsing_them(self):
        mask = np.zeros((100, 100), dtype=np.uint8)
        mask[10:30, 10:30] = 255
        mask[60:80, 60:80] = 255

        result = analyze_mask_geometry(mask, [10, 10, 80, 80])

        self.assertEqual(result["components"], 2)
        self.assertEqual(result["material_components"], 2)
        self.assertAlmostEqual(result["largest_component_fraction"], 0.5)
        self.assertGreater(result["aabb_background_fraction"], 0.8)

    def test_single_rotated_instance_has_tighter_obb_than_aabb(self):
        mask = np.zeros((100, 100), dtype=np.uint8)
        polygon = cv2.boxPoints(((50, 50), (60, 15), 35)).astype(np.int32)
        cv2.fillPoly(mask, [polygon], 255)

        result = analyze_mask_geometry(mask, [15, 20, 85, 80])

        self.assertEqual(result["material_components"], 1)
        self.assertLess(result["obb_background_fraction"], result["aabb_background_fraction"])
        self.assertGreater(result["mask_bbox_iou"], 0.5)


if __name__ == "__main__":
    unittest.main()
