from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Iterable

import cv2
import numpy as np

from parking_map.tile_georeference import lonlat_to_pixel, parse_block_filename


@dataclass(frozen=True)
class MarkingEvidenceMeasurement:
    strength: float
    segment_count: int
    orientation_consensus: float


@dataclass(frozen=True)
class VehicleArrangementMeasurement:
    strength: float
    vehicle_count: int
    alignment_ratio: float
    spacing_consistency: float


@dataclass(frozen=True)
class AgreedVehicleDetection:
    bbox: tuple[float, float, float, float]
    agreement_iou: float
    first_confidence: float
    second_confidence: float


def _polygon_coordinates(geometry: dict) -> list[list[list[list[float]]]]:
    if geometry.get("type") == "Polygon":
        return [geometry.get("coordinates") or []]
    if geometry.get("type") == "MultiPolygon":
        return geometry.get("coordinates") or []
    raise ValueError("geometry must be Polygon or MultiPolygon")


def geographic_geometry_to_mask(
    geometry: dict,
    block_filename: str,
    image_shape: tuple[int, int],
) -> np.ndarray:
    height, width = image_shape
    if height <= 0 or width <= 0:
        raise ValueError("image_shape must contain positive height and width")
    block = parse_block_filename(block_filename)
    mask = np.zeros((height, width), dtype=np.uint8)

    def pixel_ring(ring: list[list[float]]) -> np.ndarray:
        points = [
            lonlat_to_pixel(
                block.zoom,
                block.tile_row,
                block.tile_col,
                float(longitude),
                float(latitude),
            )
            for longitude, latitude in ring
        ]
        return np.rint(points).astype(np.int32)

    for polygon in _polygon_coordinates(geometry):
        if not polygon:
            continue
        cv2.fillPoly(mask, [pixel_ring(polygon[0])], 255)
        for hole in polygon[1:]:
            cv2.fillPoly(mask, [pixel_ring(hole)], 0)
    return mask


def measure_marking_evidence(
    image: np.ndarray,
    candidate_mask: np.ndarray,
) -> MarkingEvidenceMeasurement:
    if image.ndim not in {2, 3} or candidate_mask.ndim != 2:
        raise ValueError("image and candidate_mask dimensions are invalid")
    if image.shape[:2] != candidate_mask.shape:
        raise ValueError("image and candidate_mask dimensions must match")
    if not np.any(candidate_mask):
        return MarkingEvidenceMeasurement(0.0, 0, 0.0)

    grayscale = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image
    # Detect on the source image first so the artificial mask edge cannot become evidence.
    edges = cv2.Canny(grayscale, 50, 150)
    interior_mask = cv2.erode(candidate_mask, np.ones((5, 5), np.uint8))
    edges = cv2.bitwise_and(edges, edges, mask=interior_mask)
    lines = cv2.HoughLinesP(
        edges,
        rho=1,
        theta=np.pi / 180,
        threshold=15,
        minLineLength=15,
        maxLineGap=4,
    )
    if lines is None:
        return MarkingEvidenceMeasurement(0.0, 0, 0.0)

    if lines.ndim == 3:
        lines = lines.reshape(-1, 4)
    angles = []
    for x1, y1, x2, y2 in lines:
        length = math.hypot(float(x2 - x1), float(y2 - y1))
        if not 15 <= length <= 120:
            continue
        midpoint_x = max(0, min(candidate_mask.shape[1] - 1, (int(x1) + int(x2)) // 2))
        midpoint_y = max(0, min(candidate_mask.shape[0] - 1, (int(y1) + int(y2)) // 2))
        if candidate_mask[midpoint_y, midpoint_x] == 0:
            continue
        angles.append(math.atan2(float(y2 - y1), float(x2 - x1)) % math.pi)
    if not angles:
        return MarkingEvidenceMeasurement(0.0, 0, 0.0)

    bins = np.histogram(angles, bins=18, range=(0, math.pi))[0]
    consensus = float(bins.max() / len(angles))
    count_score = min(1.0, len(angles) / 6.0)
    strength = count_score * consensus
    return MarkingEvidenceMeasurement(strength, len(angles), consensus)


def _bbox_iou(first: list[float], second: list[float]) -> float:
    left = max(first[0], second[0])
    top = max(first[1], second[1])
    right = min(first[2], second[2])
    bottom = min(first[3], second[3])
    intersection = max(0.0, right - left) * max(0.0, bottom - top)
    first_area = max(0.0, first[2] - first[0]) * max(0.0, first[3] - first[1])
    second_area = max(0.0, second[2] - second[0]) * max(0.0, second[3] - second[1])
    union = first_area + second_area - intersection
    return intersection / union if union else 0.0


def match_vehicle_detections(
    first: Iterable[dict],
    second: Iterable[dict],
    *,
    minimum_iou: float = 0.3,
) -> list[AgreedVehicleDetection]:
    if not 0 < minimum_iou <= 1:
        raise ValueError("minimum_iou must be between zero and one")
    first_list = list(first)
    second_list = list(second)
    possible_matches = []
    for first_index, first_detection in enumerate(first_list):
        for second_index, second_detection in enumerate(second_list):
            overlap = _bbox_iou(first_detection["bbox"], second_detection["bbox"])
            if overlap >= minimum_iou:
                possible_matches.append((overlap, first_index, second_index))

    used_first: set[int] = set()
    used_second: set[int] = set()
    matches = []
    for overlap, first_index, second_index in sorted(possible_matches, reverse=True):
        if first_index in used_first or second_index in used_second:
            continue
        used_first.add(first_index)
        used_second.add(second_index)
        first_detection = first_list[first_index]
        second_detection = second_list[second_index]
        averaged_bbox = tuple(
            (float(first_detection["bbox"][index]) + float(second_detection["bbox"][index])) / 2
            for index in range(4)
        )
        matches.append(AgreedVehicleDetection(
            bbox=averaged_bbox,
            agreement_iou=float(overlap),
            first_confidence=float(first_detection["confidence"]),
            second_confidence=float(second_detection["confidence"]),
        ))
    return matches


def measure_vehicle_arrangement(
    boxes: Iterable[list[float]],
) -> VehicleArrangementMeasurement:
    box_list = [box for box in boxes if len(box) == 4 and box[2] > box[0] and box[3] > box[1]]
    if len(box_list) < 3:
        return VehicleArrangementMeasurement(0.0, len(box_list), 0.0, 0.0)
    centers = np.asarray(
        [[(box[0] + box[2]) / 2, (box[1] + box[3]) / 2] for box in box_list],
        dtype=float,
    )
    covariance = np.cov(centers, rowvar=False)
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    largest = float(max(eigenvalues))
    smallest = float(min(eigenvalues))
    alignment_ratio = largest / max(smallest, 1e-6)

    principal_axis = eigenvectors[:, int(np.argmax(eigenvalues))]
    projected = np.sort(centers @ principal_axis)
    spacings = np.diff(projected)
    mean_spacing = float(np.mean(spacings))
    spacing_consistency = (
        max(0.0, 1.0 - float(np.std(spacings)) / mean_spacing)
        if mean_spacing > 0
        else 0.0
    )
    count_score = min(1.0, len(box_list) / 4.0)
    alignment_score = min(1.0, max(0.0, alignment_ratio - 1.0) / 4.0)
    strength = 0.5 * count_score + 0.3 * alignment_score + 0.2 * spacing_consistency
    return VehicleArrangementMeasurement(
        strength=float(max(0.0, min(1.0, strength))),
        vehicle_count=len(box_list),
        alignment_ratio=alignment_ratio,
        spacing_consistency=spacing_consistency,
    )
