from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any, Iterable

import cv2
import numpy as np


@dataclass(frozen=True)
class VehicleRow:
    row_id: str
    detection_indices: tuple[int, ...]
    angle_degrees: float
    mask: np.ndarray


@dataclass(frozen=True)
class VehicleRowResult:
    rows: tuple[VehicleRow, ...]
    unassigned_detection_indices: tuple[int, ...]


@dataclass(frozen=True)
class _VehiclePoint:
    original_index: int
    center: np.ndarray
    short_side: float
    long_side: float


def _read_vehicle_points(
    detections: Iterable[dict[str, Any]],
    facility: np.ndarray,
    exclusion: np.ndarray,
) -> tuple[list[_VehiclePoint], set[int]]:
    points: list[_VehiclePoint] = []
    rejected: set[int] = set()
    for index, detection in enumerate(detections):
        box = detection.get("bbox")
        try:
            values = np.asarray(box, dtype=np.float64)
        except (TypeError, ValueError):
            rejected.add(index)
            continue
        if values.shape != (4,) or not np.isfinite(values).all():
            rejected.add(index)
            continue
        x1, y1, x2, y2 = values
        width, height = x2 - x1, y2 - y1
        center_x, center_y = (x1 + x2) / 2, (y1 + y2) / 2
        pixel_x, pixel_y = int(round(center_x)), int(round(center_y))
        if (
            width <= 0
            or height <= 0
            or not (0 <= pixel_x < facility.shape[1] and 0 <= pixel_y < facility.shape[0])
            or facility[pixel_y, pixel_x] == 0
            or exclusion[pixel_y, pixel_x] != 0
        ):
            rejected.add(index)
            continue
        points.append(_VehiclePoint(
            index,
            np.asarray([center_x, center_y], dtype=np.float64),
            float(min(width, height)),
            float(max(width, height)),
        ))
    points.sort(key=lambda point: (float(point.center[0]), float(point.center[1]), point.original_index))
    return points, rejected


def _canonical_axis(vector: np.ndarray) -> np.ndarray:
    axis = vector / np.linalg.norm(vector)
    if axis[0] < 0 or (abs(float(axis[0])) < 1e-12 and axis[1] < 0):
        axis = -axis
    return axis


def _split_by_longitudinal_gap(
    member_indices: np.ndarray,
    projections: np.ndarray,
    maximum_gap: float,
) -> list[tuple[int, ...]]:
    ordered = member_indices[np.argsort(projections[member_indices], kind="stable")]
    groups: list[list[int]] = [[]]
    previous_projection: float | None = None
    for member_index in ordered:
        projection = float(projections[member_index])
        if previous_projection is not None and projection - previous_projection > maximum_gap:
            groups.append([])
        groups[-1].append(int(member_index))
        previous_projection = projection
    return [tuple(group) for group in groups if len(group) >= 3]


def _best_row_candidate(
    points: list[_VehiclePoint],
    available: set[int],
    perpendicular_tolerance: float,
    maximum_gap: float,
) -> tuple[int, ...] | None:
    available_indices = sorted(available)
    centers = np.asarray([point.center for point in points])
    best: tuple[tuple[float, ...], tuple[int, ...]] | None = None
    for left_position, left_index in enumerate(available_indices):
        for right_index in available_indices[left_position + 1:]:
            vector = centers[right_index] - centers[left_index]
            distance = float(np.linalg.norm(vector))
            if distance < 1e-6 or distance > maximum_gap * 2:
                continue
            axis = _canonical_axis(vector)
            normal = np.asarray([-axis[1], axis[0]])
            offsets = centers - centers[left_index]
            perpendicular = np.abs(offsets @ normal)
            projections = centers @ axis
            inliers = np.asarray([
                index for index in available_indices
                if perpendicular[index] <= perpendicular_tolerance
            ], dtype=np.int32)
            for group in _split_by_longitudinal_gap(inliers, projections, maximum_gap):
                group_array = np.asarray(group, dtype=np.int32)
                residual = float(np.mean(perpendicular[group_array]))
                span = float(np.ptp(projections[group_array]))
                key = (float(len(group)), span, -residual, *(-float(v) for v in group))
                if best is None or key > best[0]:
                    best = (key, group)
    return None if best is None else best[1]


def _fit_row_mask(
    members: list[_VehiclePoint],
    image_shape: tuple[int, int],
    facility: np.ndarray,
    exclusion: np.ndarray,
) -> tuple[np.ndarray, float]:
    centers = np.asarray([member.center for member in members])
    centered = centers - centers.mean(axis=0)
    _, _, axes = np.linalg.svd(centered, full_matrices=False)
    axis = _canonical_axis(axes[0])
    normal = np.asarray([-axis[1], axis[0]])
    longitudinal = centers @ axis
    lateral = centers @ normal
    extension = float(np.median([member.short_side for member in members]))
    half_width = max(
        float(np.median([member.long_side for member in members])) * 0.65,
        float(np.ptp(lateral)) / 2 + extension * 0.5,
    )
    start = float(longitudinal.min() - extension)
    end = float(longitudinal.max() + extension)
    lateral_center = float(np.mean(lateral))
    corners = np.asarray([
        axis * start + normal * (lateral_center - half_width),
        axis * end + normal * (lateral_center - half_width),
        axis * end + normal * (lateral_center + half_width),
        axis * start + normal * (lateral_center + half_width),
    ])
    mask = np.zeros(image_shape, dtype=np.uint8)
    cv2.fillPoly(mask, [np.rint(corners).astype(np.int32)], 255)
    mask[(facility == 0) | (exclusion != 0)] = 0
    angle = math.degrees(math.atan2(float(axis[1]), float(axis[0]))) % 180
    return mask, angle


def build_vehicle_rows(
    agreed_vehicle_detections: Iterable[dict[str, Any]],
    facility_mask: np.ndarray,
    exclusion_mask: np.ndarray,
) -> VehicleRowResult:
    """Build conservative parking rows from mutually agreed vehicle centres.

    Pixel thresholds scale with the observed boxes, so this path does not claim
    physical dimensions when WMTS GSD metadata is unavailable.
    """
    if (
        facility_mask.ndim != 2
        or exclusion_mask.shape != facility_mask.shape
        or exclusion_mask.ndim != 2
    ):
        raise ValueError("facility and exclusion masks must be same-sized grayscale arrays")
    detections = list(agreed_vehicle_detections)
    points, rejected = _read_vehicle_points(detections, facility_mask, exclusion_mask)
    if len(points) < 3:
        return VehicleRowResult((), tuple(sorted(rejected | {p.original_index for p in points})))

    median_short_side = float(np.median([point.short_side for point in points]))
    median_diagonal = float(np.median([
        math.hypot(point.short_side, point.long_side) for point in points
    ]))
    perpendicular_tolerance = max(4.0, median_short_side * 0.8)
    maximum_gap = max(12.0, median_diagonal * 1.6)

    available = set(range(len(points)))
    raw_rows: list[tuple[tuple[int, ...], np.ndarray, float]] = []
    while len(available) >= 3:
        member_indices = _best_row_candidate(
            points,
            available,
            perpendicular_tolerance,
            maximum_gap,
        )
        if member_indices is None:
            break
        members = [points[index] for index in member_indices]
        mask, angle = _fit_row_mask(members, facility_mask.shape, facility_mask, exclusion_mask)
        if np.any(mask):
            raw_rows.append((member_indices, mask, angle))
        available.difference_update(member_indices)

    raw_rows.sort(key=lambda row: (
        float(np.mean([points[index].center[1] for index in row[0]])),
        float(np.mean([points[index].center[0] for index in row[0]])),
        row[2],
    ))
    rows = tuple(
        VehicleRow(
            row_id=f"vehicle-row-{row_number:03d}",
            detection_indices=tuple(sorted(points[index].original_index for index in members)),
            angle_degrees=angle,
            mask=mask,
        )
        for row_number, (members, mask, angle) in enumerate(raw_rows, start=1)
    )
    assigned = {index for row in rows for index in row.detection_indices}
    unassigned = tuple(sorted(set(range(len(detections))) - assigned))
    return VehicleRowResult(rows, unassigned)
