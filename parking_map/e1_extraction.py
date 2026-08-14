from __future__ import annotations

from typing import Any, Iterable

import numpy as np

from parking_map.evidence_gate import ParkingEvidenceKind
from parking_map.image_evidence import (
    geographic_geometry_to_mask,
    match_vehicle_detections,
    measure_marking_evidence,
    measure_vehicle_arrangement,
)


def _candidate_id(feature: dict[str, Any], index: int, aoi_id: str) -> str:
    return str((feature.get("properties") or {}).get("target_id") or f"{aoi_id}::{index}")


def _center_is_inside(box: tuple[float, float, float, float], mask: np.ndarray) -> bool:
    center_x = int(round((box[0] + box[2]) / 2))
    center_y = int(round((box[1] + box[3]) / 2))
    return (
        0 <= center_x < mask.shape[1]
        and 0 <= center_y < mask.shape[0]
        and mask[center_y, center_x] != 0
    )


def build_aoi_image_evidence(
    aoi_id: str,
    image_name: str,
    image: np.ndarray,
    candidates: dict[str, Any],
    first_vehicle_detections: Iterable[dict[str, Any]],
    second_vehicle_detections: Iterable[dict[str, Any]],
    *,
    detector_source_id: str = "yolo26n+yolov8n-coco-agreement-v1",
) -> dict[str, Any]:
    if not aoi_id or not image_name or candidates.get("type") != "FeatureCollection":
        raise ValueError("aoi_id, image_name and candidate FeatureCollection are required")
    if image.ndim != 3:
        raise ValueError("image must be a color image")

    agreed_vehicles = match_vehicle_detections(
        first_vehicle_detections,
        second_vehicle_detections,
    )
    signals: dict[str, list[dict[str, Any]]] = {}
    candidate_measurements = []
    for index, feature in enumerate(candidates.get("features") or []):
        candidate_id = _candidate_id(feature, index, aoi_id)
        candidate_mask = geographic_geometry_to_mask(
            feature.get("geometry") or {},
            image_name,
            image.shape[:2],
        )
        marking = measure_marking_evidence(image, candidate_mask)
        candidate_vehicle_boxes = [
            list(detection.bbox)
            for detection in agreed_vehicles
            if _center_is_inside(detection.bbox, candidate_mask)
        ]
        arrangement = measure_vehicle_arrangement(candidate_vehicle_boxes)
        signals[candidate_id] = [
            {
                "kind": ParkingEvidenceKind.PARKING_MARKING.value,
                "source_id": "classical-parallel-line-evidence-v1",
                "independence_group": "classical_image_geometry",
                "strength": marking.strength,
                "details": {
                    "segment_count": marking.segment_count,
                    "orientation_consensus": marking.orientation_consensus,
                },
            },
            {
                "kind": ParkingEvidenceKind.VEHICLE_ARRANGEMENT.value,
                "source_id": detector_source_id,
                "independence_group": "coco_vehicle_detectors",
                "strength": arrangement.strength,
                "details": {
                    "agreed_vehicle_count": arrangement.vehicle_count,
                    "alignment_ratio": arrangement.alignment_ratio,
                    "spacing_consistency": arrangement.spacing_consistency,
                },
            },
        ]
        candidate_measurements.append({
            "candidate_id": candidate_id,
            "mask_area_px": int(np.count_nonzero(candidate_mask)),
            "marking_strength": marking.strength,
            "agreed_vehicle_count": arrangement.vehicle_count,
            "vehicle_arrangement_strength": arrangement.strength,
        })

    return {
        "schema_version": 1,
        "aoi_id": aoi_id,
        "image": image_name,
        "truth_status": "evidence_only_not_ground_truth",
        "exclusion_coverage_status": "unavailable",
        "exclusions": [],
        "signals": signals,
        "agreed_vehicle_detections": [
            {
                "bbox": list(detection.bbox),
                "agreement_iou": detection.agreement_iou,
                "first_confidence": detection.first_confidence,
                "second_confidence": detection.second_confidence,
            }
            for detection in agreed_vehicles
        ],
        "candidate_measurements": candidate_measurements,
        "limitations": [
            "Vehicle detections measure agreement between two COCO-trained teachers, not truth.",
            "Classical line evidence may include roofs, road markings or other repeated edges.",
            "Building, road, vegetation and water exclusion coverage is unavailable.",
        ],
    }
