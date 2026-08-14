from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any, Iterable

import numpy as np

from parking_map.image_evidence import measure_marking_evidence


@dataclass(frozen=True)
class RevalidatedBandMeasurement:
    candidate_id: str
    clipped_area_px: int
    marking_strength: float
    marking_segment_count: int
    agreed_vehicle_count: int
    accepted: bool
    support_kinds: tuple[str, ...]
    rejection_reasons: tuple[str, ...]


@dataclass(frozen=True)
class RevalidatedBandSeed:
    candidate_id: str
    mask: np.ndarray
    support_kinds: tuple[str, ...]


@dataclass(frozen=True)
class RevalidatedBandSelection:
    accepted: tuple[RevalidatedBandSeed, ...]
    measurements: tuple[RevalidatedBandMeasurement, ...]
    aggregate_gate_status: str
    accepted_union_facility_fraction: float


def _center_is_inside(box: list[float], mask: np.ndarray) -> bool:
    if len(box) != 4:
        return False
    center_x = int(round((float(box[0]) + float(box[2])) / 2))
    center_y = int(round((float(box[1]) + float(box[3])) / 2))
    return (
        0 <= center_x < mask.shape[1]
        and 0 <= center_y < mask.shape[0]
        and mask[center_y, center_x] != 0
    )


def select_revalidated_band_seeds(
    image: np.ndarray,
    facility_mask: np.ndarray,
    candidates: Iterable[dict[str, Any]],
    agreed_vehicle_detections: Iterable[dict[str, Any]],
    *,
    minimum_marking_strength: float = 0.6,
    minimum_agreed_vehicles: int = 2,
    minimum_seed_area_px: int = 100,
    maximum_marking_only_union_fraction: float = 0.8,
) -> RevalidatedBandSelection:
    if image.ndim != 3 or facility_mask.ndim != 2 or image.shape[:2] != facility_mask.shape:
        raise ValueError("image and facility mask dimensions are invalid")
    if not 0 <= minimum_marking_strength <= 1:
        raise ValueError("minimum_marking_strength must be between zero and one")
    if minimum_agreed_vehicles <= 0 or minimum_seed_area_px <= 0:
        raise ValueError("vehicle and area thresholds must be positive")
    if not 0 < maximum_marking_only_union_fraction <= 1:
        raise ValueError("maximum_marking_only_union_fraction must be between zero and one")

    facility = facility_mask >= 128
    vehicle_detections = list(agreed_vehicle_detections)
    accepted = []
    measurements = []
    for candidate in candidates:
        candidate_id = str(candidate.get("candidate_id") or "")
        candidate_mask = candidate.get("mask")
        if (
            not candidate_id
            or not isinstance(candidate_mask, np.ndarray)
            or candidate_mask.ndim != 2
            or candidate_mask.shape != facility_mask.shape
        ):
            raise ValueError("candidate_id and same-sized grayscale candidate mask are required")
        clipped_mask = ((candidate_mask >= 128) & facility).astype(np.uint8) * 255
        clipped_area_px = int(np.count_nonzero(clipped_mask))
        marking = measure_marking_evidence(image, clipped_mask)
        vehicle_count = int(sum(
            _center_is_inside(list(detection.get("bbox") or []), clipped_mask)
            for detection in vehicle_detections
        ))
        support_kinds = []
        if marking.strength >= minimum_marking_strength:
            support_kinds.append("parking_marking")
        if vehicle_count >= minimum_agreed_vehicles:
            support_kinds.append("agreed_vehicles")
        is_accepted = clipped_area_px >= minimum_seed_area_px and bool(support_kinds)
        rejection_reasons = []
        if clipped_area_px < minimum_seed_area_px:
            rejection_reasons.append("area_below_minimum_after_clipping")
        if not support_kinds:
            rejection_reasons.append("insufficient_support_after_clipping")
        measurements.append(RevalidatedBandMeasurement(
            candidate_id=candidate_id,
            clipped_area_px=clipped_area_px,
            marking_strength=marking.strength,
            marking_segment_count=marking.segment_count,
            agreed_vehicle_count=vehicle_count,
            accepted=is_accepted,
            support_kinds=tuple(support_kinds),
            rejection_reasons=tuple(rejection_reasons),
        ))
        if is_accepted:
            accepted.append(RevalidatedBandSeed(
                candidate_id=candidate_id,
                mask=clipped_mask,
                support_kinds=tuple(support_kinds),
            ))

    accepted_union = np.zeros(facility_mask.shape, dtype=bool)
    for seed in accepted:
        accepted_union |= seed.mask >= 128
    facility_area_px = int(np.count_nonzero(facility))
    accepted_union_fraction = (
        float(np.count_nonzero(accepted_union) / facility_area_px) if facility_area_px else 0.0
    )
    aggregate_gate_status = "passed" if accepted else "no_locally_qualified_anchor"
    marking_only = accepted and all(
        seed.support_kinds == ("parking_marking",) for seed in accepted
    )
    if marking_only and accepted_union_fraction > maximum_marking_only_union_fraction:
        aggregate_gate_status = "rejected_marking_only_union_too_large"
        rejected_candidate_ids = {seed.candidate_id for seed in accepted}
        measurements = [
            replace(
                measurement,
                accepted=False,
                rejection_reasons=(
                    *measurement.rejection_reasons,
                    "marking_only_union_too_large",
                ),
            )
            if measurement.candidate_id in rejected_candidate_ids
            else measurement
            for measurement in measurements
        ]
        accepted = []

    return RevalidatedBandSelection(
        tuple(accepted),
        tuple(measurements),
        aggregate_gate_status,
        accepted_union_fraction,
    )
