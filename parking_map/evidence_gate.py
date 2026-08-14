from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Iterable

from parking_map.schema import ReviewState
from parking_map.topology import validate_geometry


class ParkingEvidenceKind(str, Enum):
    EXCLUSION_COVERAGE = "exclusion_coverage"
    PARKING_MARKING = "parking_marking"
    VEHICLE_ARRANGEMENT = "vehicle_arrangement"
    ENTRANCE_CONNECTIVITY = "entrance_connectivity"
    EXTERNAL_PARKING_MAP = "external_parking_map"
    SURFACE_COMPATIBILITY = "surface_compatibility"
    NON_PARKING_CONFLICT = "non_parking_conflict"


@dataclass(frozen=True)
class ParkingEvidenceSignal:
    kind: ParkingEvidenceKind
    source_id: str
    independence_group: str
    strength: float

    def __post_init__(self) -> None:
        if not self.source_id or not self.independence_group:
            raise ValueError("evidence source_id and independence_group are required")
        if not 0 <= self.strength <= 1:
            raise ValueError("evidence strength must be between 0 and 1")


@dataclass(frozen=True)
class ParkingGateDecision:
    state: ReviewState
    reasons: tuple[str, ...]
    qualified_support: tuple[ParkingEvidenceKind, ...] = ()


_SUPPORT_THRESHOLDS = {
    ParkingEvidenceKind.PARKING_MARKING: 0.6,
    ParkingEvidenceKind.VEHICLE_ARRANGEMENT: 0.6,
    ParkingEvidenceKind.ENTRANCE_CONNECTIVITY: 0.6,
    ParkingEvidenceKind.EXTERNAL_PARKING_MAP: 0.7,
    ParkingEvidenceKind.SURFACE_COMPATIBILITY: 0.7,
}
_DIRECT_OR_RELATIONAL = {
    ParkingEvidenceKind.PARKING_MARKING,
    ParkingEvidenceKind.VEHICLE_ARRANGEMENT,
    ParkingEvidenceKind.ENTRANCE_CONNECTIVITY,
    ParkingEvidenceKind.EXTERNAL_PARKING_MAP,
}


def decide_parking_candidate(
    geometry: dict,
    signals: Iterable[ParkingEvidenceSignal],
) -> ParkingGateDecision:
    geometry_issues = validate_geometry(geometry)
    if geometry_issues:
        return ParkingGateDecision(
            state=ReviewState.AUTO_REJECT,
            reasons=tuple(f"invalid_geometry:{issue}" for issue in geometry_issues),
        )

    signal_list = list(signals)
    if any(
        signal.kind is ParkingEvidenceKind.NON_PARKING_CONFLICT
        and signal.strength >= 0.9
        for signal in signal_list
    ):
        return ParkingGateDecision(
            state=ReviewState.AUTO_REJECT,
            reasons=("strong_non_parking_conflict",),
        )

    exclusion_covered = any(
        signal.kind is ParkingEvidenceKind.EXCLUSION_COVERAGE
        and signal.strength >= 0.8
        for signal in signal_list
    )
    qualified = [
        signal
        for signal in signal_list
        if signal.kind in _SUPPORT_THRESHOLDS
        and signal.strength >= _SUPPORT_THRESHOLDS[signal.kind]
    ]
    qualified_kinds = tuple(dict.fromkeys(signal.kind for signal in qualified))
    support_groups = {signal.independence_group for signal in qualified}

    has_enough_support = len(qualified_kinds) >= 2
    support_is_independent = len(support_groups) >= 2
    has_structural_or_relational_support = any(
        kind in _DIRECT_OR_RELATIONAL for kind in qualified_kinds
    )
    if (
        exclusion_covered
        and has_enough_support
        and support_is_independent
        and has_structural_or_relational_support
    ):
        return ParkingGateDecision(
            state=ReviewState.AUTO_ACCEPT,
            reasons=(),
            qualified_support=qualified_kinds,
        )

    reasons: list[str] = []
    if not exclusion_covered:
        reasons.append("missing_exclusion_coverage")
    if not has_enough_support:
        reasons.append("insufficient_independent_support")
    elif not support_is_independent:
        reasons.append("support_not_independent")
    if not has_structural_or_relational_support:
        reasons.append("missing_structural_or_relational_support")
    return ParkingGateDecision(
        state=ReviewState.ABSTAIN,
        reasons=tuple(reasons),
        qualified_support=qualified_kinds,
    )
