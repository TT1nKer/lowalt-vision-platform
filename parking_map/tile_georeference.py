from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Any

import cv2
import numpy as np
from shapely.geometry import GeometryCollection, MultiPolygon, Polygon, mapping, shape
from shapely.validation import make_valid


CRS84 = "OGC:CRS84"
TILE_SIZE_PX = 256


@dataclass(frozen=True)
class TileBlock:
    zoom: int
    block_row: int
    block_col: int
    tile_row: int
    tile_col: int


@dataclass(frozen=True)
class MaskGeometryResult:
    geometry: dict[str, Any]
    component_count: int
    source_component_count: int
    mask_area_px: int
    crs: str = CRS84


def parse_block_filename(filename: str) -> TileBlock:
    match = re.search(
        r"block_z(?P<zoom>\d+)_br(?P<br>\d+)_bc(?P<bc>\d+)_r(?P<row>\d+)_c(?P<col>\d+)\.(?:png|jpe?g)$",
        filename,
        re.IGNORECASE,
    )
    if not match:
        raise ValueError(f"unsupported block filename: {filename}")
    values = {key: int(value) for key, value in match.groupdict().items()}
    return TileBlock(
        zoom=values["zoom"],
        block_row=values["br"],
        block_col=values["bc"],
        tile_row=values["row"],
        tile_col=values["col"],
    )


def pixel_to_lonlat(
    zoom: int,
    tile_row: int,
    tile_col: int,
    pixel_x: float,
    pixel_y: float,
) -> tuple[float, float]:
    if zoom < 0 or tile_row < 0 or tile_col < 0:
        raise ValueError("zoom and tile coordinates must be non-negative")
    matrix_width = 2 ** (zoom + 1)
    matrix_height = 2**zoom
    absolute_col = tile_col + pixel_x / TILE_SIZE_PX
    absolute_row = tile_row + pixel_y / TILE_SIZE_PX
    longitude = absolute_col / matrix_width * 360.0 - 180.0
    latitude = 90.0 - absolute_row / matrix_height * 180.0
    return longitude, latitude


def lonlat_to_pixel(
    zoom: int,
    tile_row: int,
    tile_col: int,
    longitude: float,
    latitude: float,
) -> tuple[float, float]:
    if zoom < 0 or tile_row < 0 or tile_col < 0:
        raise ValueError("zoom and tile coordinates must be non-negative")
    if not -180 <= longitude <= 180 or not -90 <= latitude <= 90:
        raise ValueError("longitude or latitude is outside CRS84 bounds")
    matrix_width = 2 ** (zoom + 1)
    matrix_height = 2**zoom
    absolute_col = (longitude + 180.0) / 360.0 * matrix_width
    absolute_row = (90.0 - latitude) / 180.0 * matrix_height
    return (
        (absolute_col - tile_col) * TILE_SIZE_PX,
        (absolute_row - tile_row) * TILE_SIZE_PX,
    )


def _signed_ring_area(ring: list[list[float]]) -> float:
    return sum(
        ring[index][0] * ring[index + 1][1]
        - ring[index + 1][0] * ring[index][1]
        for index in range(len(ring) - 1)
    ) / 2.0


def _contour_to_ring(
    contour: np.ndarray,
    block: TileBlock,
    *,
    counter_clockwise: bool,
) -> list[list[float]]:
    points = contour.reshape(-1, 2)
    ring = [
        list(pixel_to_lonlat(block.zoom, block.tile_row, block.tile_col, float(x), float(y)))
        for x, y in points
    ]
    if ring[0] != ring[-1]:
        ring.append(ring[0].copy())
    is_counter_clockwise = _signed_ring_area(ring) > 0
    if is_counter_clockwise != counter_clockwise:
        ring = list(reversed(ring))
    return ring


def mask_to_geographic_geometry(
    mask: np.ndarray,
    block_filename: str,
    *,
    threshold: int = 128,
    minimum_component_area: int = 10,
    simplify_fraction: float = 0.002,
) -> MaskGeometryResult:
    if mask.ndim not in {2, 3}:
        raise ValueError("mask must be a 2D grayscale or 3D color array")
    if not 0 <= threshold <= 255 or minimum_component_area <= 0:
        raise ValueError("invalid threshold or minimum_component_area")

    grayscale = mask.max(axis=2) if mask.ndim == 3 else mask
    binary = (grayscale >= threshold).astype(np.uint8)
    count, labels, stats, _ = cv2.connectedComponentsWithStats(binary, 8)
    valid_labels = [
        label
        for label in range(1, count)
        if int(stats[label, cv2.CC_STAT_AREA]) >= minimum_component_area
    ]
    if not valid_labels:
        raise ValueError("mask has no component above minimum_component_area")

    valid_mask = np.isin(labels, valid_labels).astype(np.uint8)
    contours, hierarchy = cv2.findContours(valid_mask, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
    if hierarchy is None:
        raise ValueError("mask contours could not be extracted")

    block = parse_block_filename(block_filename)
    polygons: list[list[list[list[float]]]] = []
    hierarchy_rows = hierarchy[0]
    for contour_index, contour in enumerate(contours):
        if hierarchy_rows[contour_index][3] != -1:
            continue
        epsilon = max(0.0, simplify_fraction) * cv2.arcLength(contour, True)
        outer = cv2.approxPolyDP(contour, epsilon, True)
        if len(outer) < 3:
            continue
        rings = [_contour_to_ring(outer, block, counter_clockwise=True)]
        child_index = hierarchy_rows[contour_index][2]
        while child_index != -1:
            hole = contours[child_index]
            hole_epsilon = max(0.0, simplify_fraction) * cv2.arcLength(hole, True)
            hole = cv2.approxPolyDP(hole, hole_epsilon, True)
            if len(hole) >= 3:
                rings.append(_contour_to_ring(hole, block, counter_clockwise=False))
            child_index = hierarchy_rows[child_index][0]
        polygons.append(rings)

    if not polygons:
        raise ValueError("mask has no polygonal component")
    geometry: dict[str, Any]
    if len(polygons) == 1:
        geometry = {"type": "Polygon", "coordinates": polygons[0]}
    else:
        geometry = {"type": "MultiPolygon", "coordinates": polygons}
    raster_geometry = shape(geometry)
    valid_geometry = raster_geometry if raster_geometry.is_valid else make_valid(raster_geometry)

    valid_polygons: list[Polygon] = []

    def collect_polygons(candidate: object) -> None:
        if isinstance(candidate, Polygon):
            if not candidate.is_empty and candidate.area > 0:
                valid_polygons.append(candidate)
        elif isinstance(candidate, (MultiPolygon, GeometryCollection)):
            for child in candidate.geoms:
                collect_polygons(child)

    collect_polygons(valid_geometry)
    if not valid_polygons:
        raise ValueError("mask geometry could not be made into valid polygons")
    normalized_geometry = valid_polygons[0] if len(valid_polygons) == 1 else MultiPolygon(valid_polygons)
    return MaskGeometryResult(
        geometry=mapping(normalized_geometry),
        component_count=len(valid_polygons),
        source_component_count=len(valid_labels),
        mask_area_px=int(valid_mask.sum()),
    )
