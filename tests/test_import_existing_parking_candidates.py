from __future__ import annotations

import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

import cv2
import numpy as np

from parking.import_existing_parking_candidates import import_candidates


class ImportExistingParkingCandidatesTests(unittest.TestCase):
    def test_splits_components_and_builds_native_index(self) -> None:
        with TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            (root / "imagery").mkdir()
            mask_root = root / "quality" / "parkseg_imagery_masks"
            mask_root.mkdir(parents=True)
            (root / "imagery" / "sample.png").write_bytes(b"image")
            mask = np.zeros((12, 12), dtype=np.uint8)
            mask[1:4, 1:4] = 255
            mask[7:11, 8:11] = 255
            cv2.imwrite(str(mask_root / "sample_mask.png"), mask)
            candidates = root / "candidates.geojson"
            candidates.write_text(json.dumps({"features": [{"properties": {
                "aoi_id": "sample", "support_level": "vehicle_row_supported", "gate_status": "accepted"
            }}]}), encoding="utf-8")
            config = root / "config.yaml"
            config.write_text("paths:\n  merged_dir: imagery\nsam3:\n  prompt_mode: batch\n  batch_name: test_import\n", encoding="utf-8")

            summary = import_candidates(root, config, candidates)

            self.assertEqual(summary["images"], 1)
            self.assertEqual(summary["targets"], 2)
            self.assertEqual(summary["indexed"], 2)
            run = root / "sam3_runs" / "test_import" / "merged"
            result = json.loads((run / "sam3_results" / "sample.json").read_text(encoding="utf-8"))
            self.assertEqual(len(result["targets"]), 2)
            state = json.loads((run / "review" / "target_state.json").read_text(encoding="utf-8"))
            self.assertEqual({record["label"] for record in state.values()}, {"accept"})
            self.assertTrue(all(record["is_auto"] for record in state.values()))


if __name__ == "__main__":
    unittest.main()
