from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from typing import Any, Iterable

import numpy as np


@dataclass(frozen=True)
class SamExclusionSelection:
    union_mask: np.ndarray
    accepted_target_ids: tuple[str, ...]
    rejected_reason_counts: dict[str, int]


_CONFIDENCE_THRESHOLDS = {
    "building": 0.7,
    "public road": 0.65,
}


def _touches_border(binary_mask: np.ndarray) -> bool:
    return bool(
        np.any(binary_mask[0, :])
        or np.any(binary_mask[-1, :])
        or np.any(binary_mask[:, 0])
        or np.any(binary_mask[:, -1])
    )


def select_exclusion_masks(
    prompt: str,
    candidates: Iterable[dict[str, Any]],
    *,
    image_shape: tuple[int, int] | None = None,
    minimum_area_px: int = 100,
) -> SamExclusionSelection:
    if prompt not in _CONFIDENCE_THRESHOLDS:
        raise ValueError(f"unsupported exclusion prompt: {prompt}")
    if minimum_area_px <= 0:
        raise ValueError("minimum_area_px must be positive")

    candidate_list = list(candidates)
    if image_shape is None:
        first_mask = next(
            (candidate.get("mask") for candidate in candidate_list if candidate.get("mask") is not None),
            None,
        )
        if first_mask is None:
            raise ValueError("image_shape is required when candidates contain no mask")
        image_shape = first_mask.shape[:2]
    union_mask = np.zeros(image_shape, dtype=np.uint8)
    accepted_target_ids = []
    rejected_reasons = Counter()

    for candidate in candidate_list:
        candidate_mask = candidate.get("mask")
        if not isinstance(candidate_mask, np.ndarray) or candidate_mask.shape[:2] != image_shape:
            rejected_reasons["missing_or_mismatched_mask"] += 1
            continue
        binary_mask = (candidate_mask >= 128).astype(np.uint8)
        if binary_mask.ndim == 3:
            binary_mask = binary_mask.max(axis=2)
        if int(np.count_nonzero(binary_mask)) < minimum_area_px:
            rejected_reasons["area_below_minimum"] += 1
            continue
        if float(candidate.get("confidence") or 0) < _CONFIDENCE_THRESHOLDS[prompt]:
            rejected_reasons["confidence_below_threshold"] += 1
            continue
        if prompt == "public road" and not _touches_border(binary_mask):
            rejected_reasons["public_road_does_not_touch_border"] += 1
            continue
        union_mask[binary_mask != 0] = 255
        accepted_target_ids.append(str(candidate.get("target_id") or ""))

    return SamExclusionSelection(
        union_mask=union_mask,
        accepted_target_ids=tuple(accepted_target_ids),
        rejected_reason_counts=dict(rejected_reasons),
    )


def measure_exclusion_effects(
    source_component_counts: dict[str, int],
    decisions: Iterable[dict[str, Any]],
) -> dict[str, int]:
    effects = Counter()
    for candidate_id, component_count in source_component_counts.items():
        effects["source_already_multicomponent"] += component_count > 1
    for decision in decisions:
        candidate_id = str(decision["candidate_id"])
        if candidate_id not in source_component_counts:
            raise ValueError(f"missing source component count: {candidate_id}")
        removed_fraction = float(decision["removed_fraction"])
        effects["newly_split_after_exclusion"] += (
            int(decision["component_count"]) > source_component_counts[candidate_id]
        )
        effects["candidates_with_any_removed_area"] += removed_fraction > 1e-9
        effects["candidates_with_at_least_1pct_removed"] += removed_fraction >= 0.01
        effects["candidates_with_at_least_10pct_removed"] += removed_fraction >= 0.1
    return dict(effects)
