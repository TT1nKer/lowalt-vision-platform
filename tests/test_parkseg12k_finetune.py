from __future__ import annotations

import unittest

import numpy as np

try:
    import torch
    from torch import nn
    from training import parkseg12k_finetune
    HAS_ML_DEPS = True
except (ImportError, RuntimeError):
    HAS_ML_DEPS = False



class _TinySegmentationModel:
    pass  # placeholder when torch is missing
if HAS_ML_DEPS:
    class _TinySegmentationModel(nn.Module):  # type: ignore[name-defined]
        def __init__(self) -> None:
            super().__init__()
            self.segformer = nn.Linear(2, 2)
            self.decode_head = nn.Linear(2, 2)


@unittest.skipUnless(HAS_ML_DEPS, 'requires torch/transformers')
class ParkSeg12kFineTuneTest(unittest.TestCase):
    def test_positive_mask_becomes_background_and_parking_class_ids(self) -> None:
        mask = np.array([[0, 127], [128, 255]], dtype=np.uint8)

        label = parkseg12k_finetune.prepare_label(mask, positive_example=True)

        self.assertEqual(label.tolist(), [[0, 0], [1, 1]])

    def test_partial_negative_label_preserves_ignore_pixels(self) -> None:
        mask = np.array([[0, 255], [255, 0]], dtype=np.uint8)

        label = parkseg12k_finetune.prepare_label(mask, positive_example=False)

        self.assertEqual(label.tolist(), [[0, 255], [255, 0]])

    def test_freeze_encoder_leaves_only_decode_head_trainable(self) -> None:
        model = _TinySegmentationModel()

        trainable = parkseg12k_finetune.freeze_for_head_finetune(model)

        self.assertEqual(trainable, sum(parameter.numel() for parameter in model.decode_head.parameters()))
        self.assertTrue(all(not parameter.requires_grad for parameter in model.segformer.parameters()))
        self.assertTrue(all(parameter.requires_grad for parameter in model.decode_head.parameters()))


if __name__ == "__main__":
    unittest.main()
