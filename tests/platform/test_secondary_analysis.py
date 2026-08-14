from __future__ import annotations

import base64
import io
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from PIL import Image

from lowalt_platform.services.secondary_analysis import (
    SECONDARY_PROMPTS,
    ParkingSecondaryAnalyzer,
)


def _mask_base64() -> str:
    stream = io.BytesIO()
    Image.new("L", (8, 8), 255).save(stream, format="PNG")
    return base64.b64encode(stream.getvalue()).decode("ascii")


def _partial_mask_base64(box: tuple[int, int, int, int]) -> str:
    image = Image.new("L", (8, 8), 0)
    for x in range(box[0], box[2]):
        for y in range(box[1], box[3]):
            image.putpixel((x, y), 255)
    stream = io.BytesIO()
    image.save(stream, format="PNG")
    return base64.b64encode(stream.getvalue()).decode("ascii")


class SecondaryAnalysisTests(unittest.TestCase):
    def test_saves_each_prompt_independently_and_resumes_completed_candidate(self) -> None:
        calls = []

        def fake_client(image_path: Path, prompt: str) -> dict:
            calls.append((image_path.name, prompt))
            return {
                "result": {
                    "targets": [{"bbox": [0, 0, 8, 8], "confidence": 0.9, "class_name": prompt, "single_mask": _mask_base64()}]
                }
            }

        with TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            image = root / "aoi-1.png"
            Image.new("RGB", (8, 8), "white").save(image)
            output_root = root / "output"
            analyzer = ParkingSecondaryAnalyzer(fake_client, output_root, save_instance_masks=True)

            first = analyzer.analyze("aoi-1", image)
            second = analyzer.analyze("aoi-1", image)

            self.assertEqual(len(calls), len(SECONDARY_PROMPTS))
            self.assertEqual(first["status"], "completed")
            self.assertEqual(second["status"], "completed")
            self.assertTrue(second["resumed"])
            manifest = json.loads((output_root / "aoi-1" / "manifest.json").read_text(encoding="utf-8"))
            self.assertEqual(set(manifest["evidence"]), set(SECONDARY_PROMPTS))
            for evidence_type in SECONDARY_PROMPTS:
                self.assertTrue((output_root / "aoi-1" / f"{evidence_type}.png").is_file())

    def test_combines_all_instance_masks_without_losing_individual_masks(self) -> None:
        def two_instance_client(_image_path: Path, prompt: str) -> dict:
            masks = [_partial_mask_base64((0, 0, 2, 2)), _partial_mask_base64((6, 6, 8, 8))]
            return {"result": {"targets": [{"single_mask": mask, "class_name": prompt} for mask in masks]}}

        with TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            image = root / "aoi.png"
            Image.new("RGB", (8, 8), "white").save(image)

            ParkingSecondaryAnalyzer(two_instance_client, root / "output", save_instance_masks=True).analyze("aoi", image)

            evidence_root = root / "output" / "aoi"
            with Image.open(evidence_root / "vehicle.png") as combined:
                self.assertEqual(combined.getpixel((0, 0)), 255)
                self.assertEqual(combined.getpixel((7, 7)), 255)
            self.assertTrue((evidence_root / "vehicle_instances" / "000.png").is_file())
            self.assertTrue((evidence_root / "vehicle_instances" / "001.png").is_file())

    def test_default_output_does_not_store_instance_mask_files(self) -> None:
        def one_instance_client(_image_path: Path, prompt: str) -> dict:
            return {"result": {"targets": [{"single_mask": _mask_base64(), "class_name": prompt}]}}

        with TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            image = root / "aoi.png"
            Image.new("RGB", (8, 8), "white").save(image)

            ParkingSecondaryAnalyzer(one_instance_client, root / "output").analyze("aoi", image)

            self.assertFalse((root / "output" / "aoi" / "vehicle_instances").exists())
            self.assertTrue((root / "output" / "aoi" / "vehicle.png").is_file())

    def test_clips_secondary_evidence_to_primary_candidate_mask(self) -> None:
        def full_image_client(_image_path: Path, prompt: str) -> dict:
            return {"result": {"targets": [{"single_mask": _mask_base64(), "class_name": prompt}]}}

        with TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            image = root / "aoi.png"
            candidate_mask = root / "candidate.png"
            Image.new("RGB", (8, 8), "white").save(image)
            Image.open(io.BytesIO(base64.b64decode(_partial_mask_base64((0, 0, 4, 8))))).save(candidate_mask)

            ParkingSecondaryAnalyzer(full_image_client, root / "output").analyze(
                "aoi",
                image,
                candidate_mask_path=candidate_mask,
            )

            with Image.open(root / "output" / "aoi" / "vehicle.png") as clipped:
                self.assertEqual(clipped.getpixel((1, 1)), 255)
                self.assertEqual(clipped.getpixel((7, 1)), 0)

    def test_failed_prompt_does_not_publish_completed_manifest(self) -> None:
        def failing_client(_image_path: Path, prompt: str) -> dict:
            if prompt == SECONDARY_PROMPTS["internal_aisle"]:
                return {"error": "service unavailable"}
            return {"result": {"targets": []}}

        with TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            image = root / "aoi-2.png"
            Image.new("RGB", (8, 8), "white").save(image)
            analyzer = ParkingSecondaryAnalyzer(failing_client, root / "output")

            with self.assertRaisesRegex(RuntimeError, "internal_aisle"):
                analyzer.analyze("aoi-2", image)

            self.assertFalse((root / "output" / "aoi-2" / "manifest.json").exists())


if __name__ == "__main__":
    unittest.main()
