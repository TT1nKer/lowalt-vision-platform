from __future__ import annotations

import json
from pathlib import Path
import re

from PIL import Image


BLOCK_PATTERN = re.compile(r"block_z(?P<z>\d+)_br(?P<br>\d+)_bc(?P<bc>\d+)_r(?P<r>\d+)_c(?P<c>\d+)\.png$", re.IGNORECASE)


def _longitude(zoom: int, tile_col: int) -> float:
    return tile_col / 2 ** (zoom + 1) * 360.0 - 180.0


def _latitude(zoom: int, tile_row: int) -> float:
    return 90.0 - tile_row / 2**zoom * 180.0


def build_overview(image_root: Path, output_image: Path, output_manifest: Path, *, thumbnail_size: int = 16) -> dict:
    if thumbnail_size <= 0:
        raise ValueError("thumbnail size must be positive")
    records = []
    for path in sorted(image_root.glob("block_z*_br*_bc*_r*_c*.png")):
        match = BLOCK_PATTERN.fullmatch(path.name)
        if match:
            records.append((path, {key: int(value) for key, value in match.groupdict().items()}))
    if not records:
        raise ValueError(f"no WMTS block images found in {image_root}")
    zooms = {record[1]["z"] for record in records}
    if len(zooms) != 1:
        raise ValueError("overview requires one WMTS zoom level")
    min_br = min(item["br"] for _, item in records)
    max_br = max(item["br"] for _, item in records)
    min_bc = min(item["bc"] for _, item in records)
    max_bc = max(item["bc"] for _, item in records)
    canvas = Image.new("RGB", ((max_bc - min_bc + 1) * thumbnail_size, (max_br - min_br + 1) * thumbnail_size), "#d8dde3")
    for path, item in records:
        with Image.open(path) as source:
            thumbnail = source.convert("RGB").resize((thumbnail_size, thumbnail_size), Image.Resampling.BILINEAR)
        canvas.paste(thumbnail, ((item["bc"] - min_bc) * thumbnail_size, (item["br"] - min_br) * thumbnail_size))
    output_image.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output_image, format="JPEG", quality=86, optimize=False, progressive=False)
    zoom = zooms.pop()
    west = _longitude(zoom, min(item["c"] for _, item in records))
    north = _latitude(zoom, min(item["r"] for _, item in records))
    east = _longitude(zoom, max(item["c"] for _, item in records) + 4)
    south = _latitude(zoom, max(item["r"] for _, item in records) + 4)
    manifest = {
        "image_count": len(records),
        "zoom": zoom,
        "bounds": [west, south, east, north],
        "grid": {"min_block_row": min_br, "max_block_row": max_br, "min_block_col": min_bc, "max_block_col": max_bc},
        "size": {"width": canvas.width, "height": canvas.height},
        "source_coordinate_frame": "WMTS geographic matrix / OGC:CRS84",
    }
    output_manifest.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    return manifest
