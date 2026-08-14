import unittest

import numpy as np

from audits.vehicle_teacher_agreement import (
    bbox_from_mask_array,
    compare_target_boxes,
    pair_predictions_with_images,
)


class VehicleTeacherAgreementTests(unittest.TestCase):
    def test_predictions_are_paired_by_input_order_not_synthetic_result_path(self):
        predictions = [{"path": "image0.jpg"}, {"path": "image1.jpg"}]

        paired = list(pair_predictions_with_images(["real-a.jpg", "real-b.jpg"], predictions))

        self.assertEqual(paired, [
            ("real-a.jpg", {"path": "image0.jpg"}),
            ("real-b.jpg", {"path": "image1.jpg"}),
        ])

    def test_mask_bbox_accepts_three_channel_masks(self):
        mask = np.zeros((10, 10, 3), dtype=np.uint8)
        mask[0, 0, :] = 2
        mask[2:5, 3:7, :] = 255

        self.assertEqual(bbox_from_mask_array(mask), [3.0, 2.0, 7.0, 5.0])

    def test_reports_best_iou_and_unmatched_targets(self):
        targets = [[0, 0, 10, 10], [50, 50, 60, 60]]
        detections = [[1, 0, 11, 10]]

        result = compare_target_boxes(targets, detections)

        self.assertAlmostEqual(result["target_best_iou"][0], 90 / 110)
        self.assertEqual(result["target_best_iou"][1], 0.0)
        self.assertEqual(result["targets_matched_at_0_5"], 1)
        self.assertEqual(result["detections_unmatched_at_0_3"], 0)

    def test_reports_duplicate_targets_matching_one_detection(self):
        result = compare_target_boxes(
            [[0, 0, 10, 10], [1, 0, 11, 10]],
            [[0, 0, 10, 10]],
        )

        self.assertEqual(result["detections_with_multiple_targets_at_0_5"], 1)


if __name__ == "__main__":
    unittest.main()
