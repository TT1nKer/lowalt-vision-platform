from __future__ import annotations

import unittest

import numpy as np

from training import parkseg12k_finetune_data


class ShaoxingHardNegativeLabelTest(unittest.TestCase):
    def test_only_predicted_pixels_inside_exclusions_become_background(self) -> None:
        raw_mask = np.array([
            [255, 255, 0],
            [255, 0, 0],
        ], dtype=np.uint8)
        exclusion_mask = np.array([
            [255, 0, 255],
            [0, 0, 0],
        ], dtype=np.uint8)

        label = parkseg12k_finetune_data.build_hard_negative_label(
            raw_mask,
            exclusion_mask,
            confirmed_negative=False,
        )

        self.assertEqual(label.tolist(), [[0, 255, 255], [255, 255, 255]])

    def test_confirmed_negative_marks_all_predicted_pixels_as_background(self) -> None:
        raw_mask = np.array([[255, 0], [255, 255]], dtype=np.uint8)
        exclusion_mask = np.zeros((2, 2), dtype=np.uint8)

        label = parkseg12k_finetune_data.build_hard_negative_label(
            raw_mask,
            exclusion_mask,
            confirmed_negative=True,
        )

        self.assertEqual(label.tolist(), [[0, 255], [0, 0]])

    def test_mismatched_masks_are_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "same-sized"):
            parkseg12k_finetune_data.build_hard_negative_label(
                np.zeros((2, 2), dtype=np.uint8),
                np.zeros((3, 2), dtype=np.uint8),
                confirmed_negative=False,
            )


if __name__ == "__main__":
    unittest.main()
