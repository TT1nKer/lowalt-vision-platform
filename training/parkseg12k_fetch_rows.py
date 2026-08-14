from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import io
import json
from pathlib import Path
from typing import Callable
from urllib.parse import urlencode
from urllib.request import urlopen

import numpy as np
from PIL import Image


DATASET_ID = "UTEL-UIUC/parkseg12k"
DATASET_REVISION = "f6f2ab69260d1a58a115be4aaa071659b7c95a2c"
ROWS_ENDPOINT = "https://datasets-server.huggingface.co/rows"


def _fetch_bytes(url: str) -> bytes:
    with urlopen(url, timeout=60) as response:
        return response.read()


def _asset_revision(url: str) -> str:
    marker = "/--/"
    parts = url.split(marker)
    if len(parts) < 3:
        raise ValueError("asset URL does not contain a dataset revision")
    return parts[1]


def _atomic_save_image(image: Image.Image, path: Path, image_format: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_suffix(path.suffix + ".tmp")
    try:
        image.save(temporary_path, format=image_format)
        temporary_path.replace(path)
    finally:
        temporary_path.unlink(missing_ok=True)


def _atomic_write_json(document: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_suffix(path.suffix + ".tmp")
    try:
        temporary_path.write_text(
            json.dumps(document, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        temporary_path.replace(path)
    finally:
        temporary_path.unlink(missing_ok=True)


def _rows_url(split: str, offset: int, length: int) -> str:
    query = urlencode({
        "dataset": DATASET_ID,
        "config": "default",
        "split": split,
        "offset": offset,
        "length": length,
    })
    return f"{ROWS_ENDPOINT}?{query}"


def download_rows(
    output_directory: Path,
    *,
    offset: int,
    count: int,
    page_size: int = 100,
    workers: int = 8,
    split: str = "train",
    fetch_bytes: Callable[[str], bytes] = _fetch_bytes,
) -> dict:
    if offset < 0 or count <= 0 or page_size <= 0 or workers <= 0:
        raise ValueError("offset must be non-negative and counts must be positive")
    if not split:
        raise ValueError("split is required")

    downloaded_rows: list[int] = []
    next_offset = offset
    stop_offset = offset + count
    while next_offset < stop_offset:
        length = min(page_size, stop_offset - next_offset)
        page = json.loads(fetch_bytes(_rows_url(split, next_offset, length)))
        rows = list(page.get("rows") or [])
        if not rows:
            raise RuntimeError(f"dataset server returned no rows at offset {next_offset}")

        row_assets = []
        for item in rows:
            row_index = int(item["row_idx"])
            row = item.get("row") or {}
            rgb_url = str((row.get("rgb") or {}).get("src") or "")
            mask_url = str((row.get("mask") or {}).get("src") or "")
            if not rgb_url or not mask_url:
                raise ValueError(f"row {row_index} is missing rgb or mask")
            if {
                _asset_revision(rgb_url),
                _asset_revision(mask_url),
            } != {DATASET_REVISION}:
                raise ValueError(f"row {row_index} asset dataset revision is not pinned")
            row_assets.append((row_index, rgb_url, mask_url))

        def download_row(asset: tuple[int, str, str]) -> int:
            row_index, rgb_url, mask_url = asset
            rgb_path = output_directory / "rgb" / f"{row_index:06d}.jpg"
            mask_path = output_directory / "masks" / f"{row_index:06d}.png"
            if not rgb_path.exists():
                with Image.open(io.BytesIO(fetch_bytes(rgb_url))) as rgb_image:
                    _atomic_save_image(rgb_image.convert("RGB"), rgb_path, "JPEG")
            if not mask_path.exists():
                with Image.open(io.BytesIO(fetch_bytes(mask_url))) as mask_image:
                    mask_values = np.asarray(mask_image.convert("L"))
                    binary_mask = Image.fromarray(
                        (mask_values >= 128).astype(np.uint8) * 255,
                        mode="L",
                    )
                    _atomic_save_image(binary_mask, mask_path, "PNG")
            return row_index

        with ThreadPoolExecutor(max_workers=min(workers, len(row_assets))) as executor:
            downloaded_rows.extend(executor.map(download_row, row_assets))

        next_offset += len(rows)

    manifest = {
        "dataset_id": DATASET_ID,
        "dataset_revision": DATASET_REVISION,
        "split": split,
        "rows": downloaded_rows,
    }
    _atomic_write_json(manifest, output_directory / "manifest.json")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description="Download pinned ParkSeg12k RGB/mask rows")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument("--count", type=int, default=512)
    parser.add_argument("--page-size", type=int, default=100)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--split", default="train")
    args = parser.parse_args()
    manifest = download_rows(
        args.output_dir,
        offset=args.offset,
        count=args.count,
        page_size=args.page_size,
        workers=args.workers,
        split=args.split,
    )
    print(json.dumps(manifest, ensure_ascii=False))


if __name__ == "__main__":
    main()
