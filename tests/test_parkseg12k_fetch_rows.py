from __future__ import annotations

import io
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image

from training import parkseg12k_fetch_rows


def _image_bytes(mode: str, values: np.ndarray, image_format: str = "PNG") -> bytes:
    buffer = io.BytesIO()
    Image.fromarray(values, mode=mode).save(buffer, format=image_format)
    return buffer.getvalue()


class ParkSeg12kRowDownloadTest(unittest.TestCase):
    def test_download_writes_rgb_and_lossless_binary_mask(self) -> None:
        revision = parkseg12k_fetch_rows.DATASET_REVISION
        rgb_url = f"https://assets.example/--/{revision}/--/train/7/rgb/image.jpg"
        mask_url = f"https://assets.example/--/{revision}/--/train/7/mask/image.jpg"
        page = {
            "rows": [{
                "row_idx": 7,
                "row": {
                    "rgb": {"src": rgb_url, "height": 2, "width": 3},
                    "mask": {"src": mask_url, "height": 2, "width": 3},
                    "nir": {"src": "https://assets.example/unused"},
                },
            }],
        }
        responses = {
            "page": json.dumps(page).encode("utf-8"),
            rgb_url: _image_bytes("RGB", np.full((2, 3, 3), 90, dtype=np.uint8)),
            mask_url: _image_bytes(
                "L",
                np.array([[0, 127, 128], [255, 20, 240]], dtype=np.uint8),
            ),
        }

        def fetch_bytes(url: str) -> bytes:
            return responses["page"] if "rows?" in url else responses[url]

        with tempfile.TemporaryDirectory() as temporary_directory:
            output = Path(temporary_directory)

            manifest = parkseg12k_fetch_rows.download_rows(
                output,
                offset=7,
                count=1,
                page_size=1,
                fetch_bytes=fetch_bytes,
            )

            with Image.open(output / "rgb" / "000007.jpg") as rgb:
                rgb_size = rgb.size
            with Image.open(output / "masks" / "000007.png") as mask_image:
                mask = np.asarray(mask_image).copy()
            self.assertEqual(rgb_size, (3, 2))
            self.assertEqual(mask.tolist(), [[0, 0, 255], [255, 0, 255]])
            self.assertEqual(manifest["rows"], [7])
            self.assertEqual(manifest["dataset_revision"], revision)

    def test_download_rejects_assets_from_another_revision(self) -> None:
        page = {
            "rows": [{
                "row_idx": 0,
                "row": {
                    "rgb": {"src": "https://assets.example/--/wrong/--/rgb.jpg"},
                    "mask": {"src": "https://assets.example/--/wrong/--/mask.jpg"},
                },
            }],
        }

        with tempfile.TemporaryDirectory() as temporary_directory:
            with self.assertRaisesRegex(ValueError, "dataset revision"):
                parkseg12k_fetch_rows.download_rows(
                    Path(temporary_directory),
                    offset=0,
                    count=1,
                    page_size=1,
                    fetch_bytes=lambda _url: json.dumps(page).encode("utf-8"),
                )


if __name__ == "__main__":
    unittest.main()
