from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from typing import Any, Iterable

import cv2
import numpy as np


@dataclass(frozen=True)
class AnchoredSurfaceSelection:
    union_mask: np.ndarray
    accepted_target_ids: tuple[str, ...]
    component_count: int
    rejected_reason_counts: dict[str, int]


def select_anchor_candidate_ids(
    measurements: Iterable[dict[str, Any]],
    *,
    image_area_px: int,
    minimum_marking_strength: float = 0.6,
    minimum_agreed_vehicles: int = 2,
    maximum_area_fraction: float = 0.05,
) -> set[str]:
    if image_area_px <= 0 or not 0 < maximum_area_fraction <= 1:
        raise ValueError("image area and maximum area fraction are invalid")
    selected = set()
    maximum_area_px = image_area_px * maximum_area_fraction
    for measurement in measurements:
        area_px = int(measurement.get("mask_area_px") or 0)
        has_support = (
            float(measurement.get("marking_strength") or 0) >= minimum_marking_strength
            or int(measurement.get("agreed_vehicle_count") or 0) >= minimum_agreed_vehicles
        )
        if 0 < area_px <= maximum_area_px and has_support:
            selected.add(str(measurement["candidate_id"]))
    return selected


def _binary(mask: np.ndarray, image_shape: tuple[int, int]) -> np.ndarray | None:
    if not isinstance(mask, np.ndarray) or mask.shape[:2] != image_shape:
        return None
    binary = mask >= 128
    if binary.ndim == 3:
        binary = binary.max(axis=2)
    return binary.astype(np.uint8)


def select_anchored_surfaces(
    candidates: Iterable[dict[str, Any]],
    anchor_mask: np.ndarray,
    exclusion_mask: np.ndarray,
    *,
    minimum_confidence: float = 0.4,
    minimum_surface_area_px: int = 500,
    minimum_anchor_overlap_px: int = 100,
) -> AnchoredSurfaceSelection:
    if anchor_mask.ndim != 2 or exclusion_mask.shape != anchor_mask.shape:
        raise ValueError("anchor_mask and exclusion_mask must be same-sized grayscale masks")
    if not 0 <= minimum_confidence <= 1:
        raise ValueError("minimum_confidence must be between zero and one")
    if minimum_surface_area_px <= 0 or minimum_anchor_overlap_px <= 0:
        raise ValueError("surface area and anchor overlap thresholds must be positive")

    image_shape = anchor_mask.shape
    anchors = anchor_mask >= 128
    exclusions = exclusion_mask >= 128
    union = np.zeros(image_shape, dtype=np.uint8)
    accepted_target_ids = []
    rejected_reasons = Counter()
    for candidate in candidates:
        candidate_mask = _binary(candidate.get("mask"), image_shape)
        if candidate_mask is None:
            rejected_reasons["missing_or_mismatched_mask"] += 1
            continue
        if float(candidate.get("confidence") or 0) < minimum_confidence:
            rejected_reasons["confidence_below_threshold"] += 1
            continue
        remaining = candidate_mask.astype(bool) & ~exclusions
        if int(np.count_nonzero(remaining)) < minimum_surface_area_px:
            rejected_reasons["area_below_minimum_after_exclusion"] += 1
            continue
        if int(np.count_nonzero(remaining & anchors)) < minimum_anchor_overlap_px:
            rejected_reasons["anchor_overlap_below_minimum"] += 1
            continue
        union[remaining] = 255
        accepted_target_ids.append(str(candidate.get("target_id") or ""))

    component_count, labels, stats, _ = cv2.connectedComponentsWithStats(
        (union >= 128).astype(np.uint8),
        8,
    )
    retained = np.zeros(image_shape, dtype=np.uint8)
    retained_components = 0
    for label in range(1, component_count):
        component = labels == label
        if int(stats[label, cv2.CC_STAT_AREA]) < minimum_surface_area_px:
            rejected_reasons["component_area_below_minimum"] += 1
            continue
        if int(np.count_nonzero(component & anchors)) < minimum_anchor_overlap_px:
            rejected_reasons["component_anchor_overlap_below_minimum"] += 1
            continue
        retained[component] = 255
        retained_components += 1

    return AnchoredSurfaceSelection(
        union_mask=retained,
        accepted_target_ids=tuple(accepted_target_ids),
        component_count=retained_components,
        rejected_reason_counts=dict(rejected_reasons),
    )
