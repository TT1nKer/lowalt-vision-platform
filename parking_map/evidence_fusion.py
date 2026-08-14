from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable

from shapely.geometry import GeometryCollection, MultiPolygon, Polygon, mapping, shape
from shapely.ops import unary_union
from shapely.validation import make_valid


@dataclass(frozen=True)
class ExclusionResult:
    geometry: dict[str, Any]
    component_count: int
    source_area: float
    remaining_area: float
    removed_fraction: float


def _collect_polygons(geometry: object) -> list[Polygon]:
    if isinstance(geometry, Polygon):
        return [geometry] if not geometry.is_empty and geometry.area > 0 else []
    if isinstance(geometry, (MultiPolygon, GeometryCollection)):
        polygons: list[Polygon] = []
        for child in geometry.geoms:
            polygons.extend(_collect_polygons(child))
        return polygons
    return []


def subtract_exclusions(
    candidate_geometry: dict[str, Any],
    exclusion_geometries: Iterable[dict[str, Any]],
) -> ExclusionResult | None:
    candidate = make_valid(shape(candidate_geometry))
    candidate_polygons = _collect_polygons(candidate)
    if not candidate_polygons:
        raise ValueError("candidate_geometry must contain a non-empty polygon")
    candidate = unary_union(candidate_polygons)
    source_area = float(candidate.area)

    exclusions = []
    for geometry in exclusion_geometries:
        exclusions.extend(_collect_polygons(make_valid(shape(geometry))))
    remaining = candidate if not exclusions else candidate.difference(unary_union(exclusions))
    remaining = make_valid(remaining)
    remaining_polygons = _collect_polygons(remaining)
    if not remaining_polygons:
        return None

    normalized = (
        remaining_polygons[0]
        if len(remaining_polygons) == 1
        else MultiPolygon(remaining_polygons)
    )
    remaining_area = float(normalized.area)
    return ExclusionResult(
        geometry=mapping(normalized),
        component_count=len(remaining_polygons),
        source_area=source_area,
        remaining_area=remaining_area,
        removed_fraction=max(0.0, min(1.0, 1.0 - remaining_area / source_area)),
    )
