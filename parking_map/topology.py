from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

from parking_map.schema import MapFeature, ReviewState


Point = tuple[float, float]


@dataclass(frozen=True)
class TopologyGateResult:
    decision: ReviewState
    issues: tuple[str, ...]


def _orientation(a: Point, b: Point, c: Point) -> float:
    return (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0])


def _on_segment(a: Point, b: Point, point: Point, tolerance: float = 1e-12) -> bool:
    return (
        abs(_orientation(a, b, point)) <= tolerance
        and min(a[0], b[0]) - tolerance <= point[0] <= max(a[0], b[0]) + tolerance
        and min(a[1], b[1]) - tolerance <= point[1] <= max(a[1], b[1]) + tolerance
    )


def _segments_intersect(a: Point, b: Point, c: Point, d: Point) -> bool:
    ab_c = _orientation(a, b, c)
    ab_d = _orientation(a, b, d)
    cd_a = _orientation(c, d, a)
    cd_b = _orientation(c, d, b)
    if (ab_c > 0 > ab_d or ab_c < 0 < ab_d) and (cd_a > 0 > cd_b or cd_a < 0 < cd_b):
        return True
    return any(
        (
            abs(value) <= 1e-12,
            _on_segment(start, end, point),
        ) == (True, True)
        for value, start, end, point in (
            (ab_c, a, b, c),
            (ab_d, a, b, d),
            (cd_a, c, d, a),
            (cd_b, c, d, b),
        )
    )


def _ring_self_intersects(ring: list[list[float]]) -> bool:
    edges = list(zip(ring[:-1], ring[1:]))
    for first_index, (a, b) in enumerate(edges):
        for second_index, (c, d) in enumerate(edges[first_index + 1 :], first_index + 1):
            if second_index in {first_index, first_index + 1}:
                continue
            if first_index == 0 and second_index == len(edges) - 1:
                continue
            if _segments_intersect(tuple(a), tuple(b), tuple(c), tuple(d)):
                return True
    return False


def _polygons(geometry: dict) -> list[list[list[list[float]]]]:
    if geometry.get("type") == "Polygon":
        return [geometry.get("coordinates", [])]
    if geometry.get("type") == "MultiPolygon":
        return geometry.get("coordinates", [])
    return []


def validate_geometry(geometry: dict) -> tuple[str, ...]:
    if geometry.get("type") not in {"Polygon", "MultiPolygon"}:
        return ("unsupported_geometry_type",)
    issues: list[str] = []
    polygons = _polygons(geometry)
    if not polygons:
        return ("empty_geometry",)
    for polygon in polygons:
        if not polygon:
            issues.append("empty_polygon")
            continue
        for ring in polygon:
            if len(ring) < 4 or ring[0] != ring[-1]:
                issues.append("unclosed_or_short_ring")
                continue
            if _ring_self_intersects(ring):
                issues.append("self_intersection")
    return tuple(dict.fromkeys(issues))


def _point_in_ring(point: Point, ring: list[list[float]]) -> bool:
    for start, end in zip(ring[:-1], ring[1:]):
        if _on_segment(tuple(start), tuple(end), point):
            return True
    inside = False
    x, y = point
    for start, end in zip(ring[:-1], ring[1:]):
        x1, y1 = start
        x2, y2 = end
        if (y1 > y) != (y2 > y):
            crossing_x = (x2 - x1) * (y - y1) / (y2 - y1) + x1
            if x < crossing_x:
                inside = not inside
    return inside


def _point_in_polygon(point: Point, polygon: list[list[list[float]]]) -> bool:
    if not polygon or not _point_in_ring(point, polygon[0]):
        return False
    return not any(_point_in_ring(point, hole) for hole in polygon[1:])


def _geometry_within(child: dict, parent: dict) -> bool:
    parent_polygons = _polygons(parent)
    for child_polygon in _polygons(child):
        for ring in child_polygon:
            for point in ring[:-1]:
                if not any(_point_in_polygon(tuple(point), polygon) for polygon in parent_polygons):
                    return False
    return True


def gate_map(features: Iterable[MapFeature]) -> TopologyGateResult:
    feature_list = list(features)
    by_id: dict[str, MapFeature] = {}
    issues: list[str] = []
    for feature in feature_list:
        if feature.object_id in by_id:
            issues.append(f"duplicate_object_id:{feature.object_id}")
        by_id[feature.object_id] = feature
        issues.extend(f"{issue}:{feature.object_id}" for issue in validate_geometry(feature.geometry))

    for feature in feature_list:
        if feature.parent_id is None:
            continue
        parent = by_id.get(feature.parent_id)
        if parent is None:
            issues.append(f"missing_parent:{feature.object_id}")
            continue
        if parent.object_type is not feature.parent_type:
            issues.append(f"parent_type_mismatch:{feature.object_id}")
        if not _geometry_within(feature.geometry, parent.geometry):
            issues.append(f"child_outside_parent:{feature.object_id}")

    unique_issues = tuple(dict.fromkeys(issues))
    decision = ReviewState.AUTO_REJECT if unique_issues else ReviewState.AUTO_ACCEPT
    return TopologyGateResult(decision=decision, issues=unique_issues)
