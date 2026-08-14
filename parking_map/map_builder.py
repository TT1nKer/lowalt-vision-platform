from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable

from parking_map.evidence_fusion import subtract_exclusions
from parking_map.evidence_gate import (
    ParkingEvidenceSignal,
    ParkingGateDecision,
    decide_parking_candidate,
)
from parking_map.schema import ReviewState


@dataclass(frozen=True)
class ParkingCandidateBuildResult:
    candidate_id: str
    geometry: dict[str, Any] | None
    component_count: int
    removed_fraction: float
    decision: ParkingGateDecision


def build_parking_candidate(
    *,
    candidate_id: str,
    candidate_geometry: dict[str, Any],
    exclusion_geometries: Iterable[dict[str, Any]],
    signals: Iterable[ParkingEvidenceSignal],
    exclusions_are_authoritative: bool = True,
) -> ParkingCandidateBuildResult:
    if not candidate_id:
        raise ValueError("candidate_id is required")
    exclusion_result = subtract_exclusions(candidate_geometry, exclusion_geometries)
    if exclusion_result is None:
        if not exclusions_are_authoritative:
            component_count = (
                1
                if candidate_geometry.get("type") == "Polygon"
                else len(candidate_geometry.get("coordinates") or [])
            )
            return ParkingCandidateBuildResult(
                candidate_id=candidate_id,
                geometry=candidate_geometry,
                component_count=component_count,
                removed_fraction=1.0,
                decision=ParkingGateDecision(
                    state=ReviewState.ABSTAIN,
                    reasons=("provisionally_fully_excluded",),
                ),
            )
        return ParkingCandidateBuildResult(
            candidate_id=candidate_id,
            geometry=None,
            component_count=0,
            removed_fraction=1.0,
            decision=ParkingGateDecision(
                state=ReviewState.AUTO_REJECT,
                reasons=("fully_excluded",),
            ),
        )

    return ParkingCandidateBuildResult(
        candidate_id=candidate_id,
        geometry=exclusion_result.geometry,
        component_count=exclusion_result.component_count,
        removed_fraction=exclusion_result.removed_fraction,
        decision=decide_parking_candidate(exclusion_result.geometry, signals),
    )
