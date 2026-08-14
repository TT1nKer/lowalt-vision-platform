from __future__ import annotations

from collections import Counter
from typing import Any

from parking_map.evidence_gate import ParkingEvidenceKind, ParkingEvidenceSignal
from parking_map.map_builder import build_parking_candidate
from parking_map.schema import ReviewState


def _parse_signal(record: dict[str, Any]) -> ParkingEvidenceSignal:
    return ParkingEvidenceSignal(
        kind=ParkingEvidenceKind(record["kind"]),
        source_id=str(record["source_id"]),
        independence_group=str(record["independence_group"]),
        strength=float(record["strength"]),
    )


def _geometry(record: dict[str, Any]) -> dict[str, Any]:
    return record.get("geometry", record)


def evaluate_aoi_candidates(
    aoi_id: str,
    candidates: dict[str, Any],
    evidence: dict[str, Any],
) -> dict[str, Any]:
    if not aoi_id or candidates.get("type") != "FeatureCollection":
        raise ValueError("aoi_id and a candidate FeatureCollection are required")
    exclusions = [_geometry(record) for record in evidence.get("exclusions") or []]
    signals_by_candidate = evidence.get("signals") or {}
    summary = Counter({state.value: 0 for state in ReviewState})
    output_features = []
    decisions = []

    for index, feature in enumerate(candidates.get("features") or []):
        properties = feature.get("properties") or {}
        candidate_id = str(properties.get("target_id") or f"{aoi_id}::{index}")
        signals = [
            _parse_signal(record)
            for record in signals_by_candidate.get(candidate_id, [])
        ]
        built = build_parking_candidate(
            candidate_id=candidate_id,
            candidate_geometry=feature.get("geometry") or {},
            exclusion_geometries=exclusions,
            signals=signals,
            exclusions_are_authoritative=bool(
                evidence.get("exclusions_authoritative", True)
            ),
        )
        summary[built.decision.state.value] += 1
        decision_record = {
            "candidate_id": candidate_id,
            "decision": built.decision.state.value,
            "reasons": list(built.decision.reasons),
            "qualified_support": [kind.value for kind in built.decision.qualified_support],
            "component_count": built.component_count,
            "removed_fraction": built.removed_fraction,
        }
        decisions.append(decision_record)
        if built.geometry is None:
            continue
        output_features.append({
            "type": "Feature",
            "geometry": built.geometry,
            "properties": {
                **decision_record,
                "role": "e1_parking_map_candidate",
                "truth_status": "evidence_only_not_ground_truth",
                "crs": "OGC:CRS84",
            },
        })

    return {
        "type": "FeatureCollection",
        "schema_version": 1,
        "aoi_id": aoi_id,
        "truth_status": "evidence_only_not_ground_truth",
        "summary": dict(summary),
        "decisions": decisions,
        "features": output_features,
    }
