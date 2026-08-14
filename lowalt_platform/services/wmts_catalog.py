from __future__ import annotations

from dataclasses import dataclass
import math
from pathlib import Path
import re


BLOCK_FILENAME = re.compile(
    r"block_z(?P<zoom>\d+)_br(?P<block_row>\d+)_bc(?P<block_col>\d+)_r(?P<tile_row>\d+)_c(?P<tile_col>\d+)\.png$",
    re.IGNORECASE,
)
BLOCK_TILE_SPAN = 4


def _longitude(zoom: int, tile_col: int) -> float:
    return tile_col / 2 ** (zoom + 1) * 360.0 - 180.0


def _latitude(zoom: int, tile_row: int) -> float:
    return 90.0 - tile_row / 2**zoom * 180.0


@dataclass(frozen=True)
class WmtsBlock:
    block_id: str
    path: Path
    zoom: int
    bounds: tuple[float, float, float, float]

    def as_response(self) -> dict:
        west, south, east, north = self.bounds
        return {
            "block_id": self.block_id,
            "native_zoom": self.zoom,
            "bounds": [west, south, east, north],
            "image_url": f"/api/platform/imagery/blocks/{self.block_id}",
        }


class WmtsBlockCatalog:
    def __init__(self, image_root: Path):
        blocks = []
        for path in sorted(image_root.glob("block_z*_br*_bc*_r*_c*.png")):
            match = BLOCK_FILENAME.fullmatch(path.name)
            if not match:
                continue
            fields = {name: int(value) for name, value in match.groupdict().items()}
            zoom = fields["zoom"]
            tile_row = fields["tile_row"]
            tile_col = fields["tile_col"]
            bounds = (
                _longitude(zoom, tile_col),
                _latitude(zoom, tile_row + BLOCK_TILE_SPAN),
                _longitude(zoom, tile_col + BLOCK_TILE_SPAN),
                _latitude(zoom, tile_row),
            )
            blocks.append(WmtsBlock(path.stem, path.resolve(), zoom, bounds))
        if not blocks:
            raise ValueError(f"no WMTS block images found in {image_root}")
        self._blocks = blocks
        self._by_id = {block.block_id: block for block in blocks}

    def query(
        self,
        bounds: tuple[float, float, float, float] | None,
        *,
        limit: int = 256,
    ) -> list[dict]:
        if not 1 <= limit <= 256:
            raise ValueError("block limit must be between 1 and 256")
        if bounds:
            west, south, east, north = bounds
            if not all(math.isfinite(value) for value in bounds):
                raise ValueError("query bounds must be finite")
            if west > east or south > north:
                raise ValueError("query bounds are reversed")
        selected = []
        for block in self._blocks:
            if bounds:
                block_west, block_south, block_east, block_north = block.bounds
                if block_east < west or block_west > east or block_north < south or block_south > north:
                    continue
            selected.append(block.as_response())
            if len(selected) >= limit:
                break
        return selected

    def image_path(self, block_id: str) -> Path:
        try:
            return self._by_id[block_id].path
        except KeyError as exc:
            raise KeyError(f"unknown WMTS block: {block_id}") from exc
