from __future__ import annotations

import unittest
import tempfile
from pathlib import Path

import numpy as np
from PIL import Image
from types import SimpleNamespace

try:
    import torch
    from torch import nn
    from training import parkseg12k_infer
    HAS_ML_DEPS = True
except (ImportError, RuntimeError):
    HAS_ML_DEPS = False



PROJECT_ROOT = Path(__file__).resolve().parents[1]
PROCESSOR_CONFIG = PROJECT_ROOT / "models" / "nvidia_mit_b5_config"


@unittest.skipUnless(HAS_ML_DEPS, 'requires torch/transformers')
class ParkSeg12kInferenceTest(unittest.TestCase):
    def test_masks_only_inference_does_not_write_probability_or_overlay(self) -> None:
        class Processor:
            def __call__(self, *, images, return_tensors):
                return {"pixel_values": torch.zeros((1, 3, 8, 8))}

        class Model(nn.Module):
            def forward(self, pixel_values):
                logits = torch.zeros((1, 2, 2, 2))
                logits[:, 1] = 2
                return SimpleNamespace(logits=logits)

        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            image_path = root / "tile.png"
            Image.new("RGB", (16, 12), "white").save(image_path)

            parkseg12k_infer.run_inference(
                Model(), Processor(), torch.device("cpu"), image_path, root, 0.5,
                save_diagnostics=False,
            )

            self.assertTrue((root / "tile_mask.png").is_file())
            self.assertFalse((root / "tile_pred.png").exists())
            self.assertFalse((root / "tile_parking_prob.npy").exists())

    def test_existing_complete_masks_can_be_skipped(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            first = root / "first.png"
            second = root / "second.png"
            Image.new("L", (2, 2)).save(first)
            Image.new("L", (2, 2)).save(second)
            Image.new("L", (2, 2)).save(root / "first_mask.png")

            pending, skipped = parkseg12k_infer.select_pending_images(
                [first, second], root, skip_existing=True
            )

            self.assertEqual(pending, [second])
            self.assertEqual(skipped, 1)

    def test_preprocess_uses_model_resize_and_normalization(self) -> None:
        processor = parkseg12k_infer.load_processor(PROCESSOR_CONFIG)
        image = Image.fromarray(np.full((32, 48, 3), 255, dtype=np.uint8), mode="RGB")

        pixel_values = parkseg12k_infer.preprocess_image(image, processor)

        self.assertEqual(tuple(pixel_values.shape), (1, 3, 512, 512))
        expected_red = (1.0 - 0.485) / 0.229
        self.assertAlmostEqual(float(pixel_values[0, 0, 0, 0]), expected_red, places=5)

    def test_decode_head_delta_requires_matching_base_revision(self) -> None:
        model = nn.Module()
        model.decode_head = nn.Linear(2, 2, bias=False)
        replacement = torch.full_like(model.decode_head.weight, 3.0)
        with tempfile.TemporaryDirectory() as temporary_directory:
            checkpoint = Path(temporary_directory) / "head.pt"
            torch.save({
                "base_model_revision": parkseg12k_infer.MODEL_REVISION,
                "decode_head": {"weight": replacement},
            }, checkpoint)

            parkseg12k_infer.apply_decode_head_checkpoint(model, checkpoint)

            self.assertTrue(torch.equal(model.decode_head.weight, replacement))

            torch.save({
                "base_model_revision": "wrong",
                "decode_head": {"weight": replacement},
            }, checkpoint)
            with self.assertRaisesRegex(ValueError, "base model revision"):
                parkseg12k_infer.apply_decode_head_checkpoint(model, checkpoint)


if __name__ == "__main__":
    unittest.main()
