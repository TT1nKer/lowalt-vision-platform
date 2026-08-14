from __future__ import annotations

from collections import Counter
from dataclasses import asdict
import json
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from parking_map.e2_map_builder import AoiFunctionalMapResult, build_aoi_functional_map


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_overlay(
    image: np.ndarray,
    result: AoiFunctionalMapResult,
    output_path: Path,
) -> None:
    overlay = image.copy()
    layers = (
        (result.facility_mask, (0, 210, 210), 2),
        (result.masks.unknown_region_mask, (120, 120, 120), 1),
        (result.masks.parking_band_mask, (0, 220, 255), 3),
        (result.masks.internal_aisle_mask, (255, 180, 0), 3),
        (result.masks.entrance_exit_mask, (220, 0, 220), 4),
        (result.public_road_mask, (30, 30, 230), 2),
    )
    for mask, color, width in layers:
        contours, _ = cv2.findContours(mask, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(overlay, contours, -1, color, width)
    if not cv2.imwrite(str(output_path), overlay):
        raise OSError(f"could not write overlay: {output_path}")


def run_functional_zoning_batch(
    *,
    images: Path,
    surface_root: Path,
    evidence_root: Path,
    output_dir: Path,
    source_version: str = "parking_map_e2_v1",
    minimum_marking_strength: float = 0.6,
    minimum_agreed_vehicles: int = 2,
    maximum_marking_only_union_fraction: float = 0.8,
    maximum_between_band_distance_px: int = 48,
    entrance_proximity_px: int = 12,
    minimum_component_area_px: int = 100,
    east_west_metres_per_pixel: float | None = None,
    north_south_metres_per_pixel: float | None = None,
) -> dict[str, Any]:
    if (east_west_metres_per_pixel is None) != (north_south_metres_per_pixel is None):
        raise ValueError("both directional metres-per-pixel values are required together")
    if east_west_metres_per_pixel is not None and (
        east_west_metres_per_pixel <= 0 or north_south_metres_per_pixel <= 0
    ):
        raise ValueError("metres-per-pixel values must be positive")
    surface_paths = sorted(surface_root.glob("*/surface_hypothesis.geojson"))
    if not surface_paths:
        raise ValueError("surface_root contains no surface hypotheses")
    output_dir.mkdir(parents=True, exist_ok=True)
    totals = Counter()
    feature_records = []
    for surface_path in surface_paths:
        surface_hypotheses = _load_json(surface_path)
        if not surface_hypotheses.get("features"):
            totals["skipped_no_surface"] += 1
            continue
        aoi_id = surface_path.parent.name
        evidence_directory = evidence_root / aoi_id
        evidence_layers = _load_json(evidence_directory / "evidence_layers.json")
        image_name = str(evidence_layers.get("image") or "")
        if not image_name:
            raise ValueError(f"evidence layer has no image name: {aoi_id}")
        image = cv2.imread(str(images / image_name), cv2.IMREAD_COLOR)
        if image is None:
            raise ValueError(f"unreadable source image: {image_name}")
        result = build_aoi_functional_map(
            aoi_id=aoi_id,
            image_name=image_name,
            image=image,
            surface_hypotheses=surface_hypotheses,
            parking_candidates=_load_json(evidence_directory / "parking_candidates.geojson"),
            evidence_layers=evidence_layers,
            exclusion_layers=_load_json(evidence_directory / "exclusion_layers.geojson"),
            source_version=source_version,
            minimum_marking_strength=minimum_marking_strength,
            minimum_agreed_vehicles=minimum_agreed_vehicles,
            maximum_marking_only_union_fraction=maximum_marking_only_union_fraction,
            maximum_between_band_distance_px=maximum_between_band_distance_px,
            entrance_proximity_px=entrance_proximity_px,
            minimum_component_area_px=minimum_component_area_px,
        )

        aoi_output = output_dir / aoi_id
        aoi_output.mkdir(parents=True, exist_ok=True)
        (aoi_output / "functional_zones.geojson").write_text(
            json.dumps(result.feature_collection, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        (aoi_output / "anchor_revalidation.json").write_text(
            json.dumps(
                {
                    "schema_version": 1,
                    "aoi_id": aoi_id,
                    "truth_status": "evidence_only_not_ground_truth",
                    "aggregate_gate_status": result.aggregate_anchor_gate_status,
                    "accepted_union_facility_fraction": (
                        result.accepted_anchor_union_facility_fraction
                    ),
                    "measurements": [asdict(item) for item in result.revalidation_measurements],
                },
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )
        _write_overlay(image, result, aoi_output / "functional_zones_overlay.jpg")
        feature_records.extend(result.zone_feature_records)
        totals["aois"] += 1
        totals["revalidated_anchor_candidates"] += sum(
            measurement.accepted for measurement in result.revalidation_measurements
        )
        totals["aggregate_marking_only_rejected_aois"] += (
            result.aggregate_anchor_gate_status
            == "rejected_marking_only_union_too_large"
        )
        totals["parking_band_aois"] += bool(np.any(result.masks.parking_band_mask))
        totals["internal_aisle_aois"] += bool(np.any(result.masks.internal_aisle_mask))
        totals["entrance_exit_aois"] += bool(np.any(result.masks.entrance_exit_mask))
        totals["unknown_region_aois"] += bool(np.any(result.masks.unknown_region_mask))

    (output_dir / "zone_features.jsonl").write_text(
        "".join(json.dumps(record, ensure_ascii=False) + "\n" for record in feature_records),
        encoding="utf-8",
    )
    totals["zone_feature_records"] = len(feature_records)
    totals["surface_files"] = len(surface_paths)
    physical_policy: dict[str, Any] = {"gsd_status": "pending"}
    if east_west_metres_per_pixel is not None and north_south_metres_per_pixel is not None:
        physical_policy = {
            "gsd_status": "calculated_proxy_domain_not_provider_confirmed",
            "metres_per_pixel": {
                "east_west": east_west_metres_per_pixel,
                "north_south": north_south_metres_per_pixel,
            },
            "maximum_between_band_distance_metres": {
                "east_west": round(
                    maximum_between_band_distance_px * east_west_metres_per_pixel, 6
                ),
                "north_south": round(
                    maximum_between_band_distance_px * north_south_metres_per_pixel, 6
                ),
            },
            "entrance_proximity_metres": {
                "east_west": round(entrance_proximity_px * east_west_metres_per_pixel, 6),
                "north_south": round(entrance_proximity_px * north_south_metres_per_pixel, 6),
            },
            "minimum_component_area_square_metres": round(
                minimum_component_area_px
                * east_west_metres_per_pixel
                * north_south_metres_per_pixel,
                6,
            ),
        }
    summary = {
        "schema_version": 1,
        "version_id": source_version,
        "truth_status": "evidence_only_not_ground_truth",
        "source_surface_root": str(surface_root),
        "source_evidence_root": str(evidence_root),
        "policy": {
            "minimum_marking_strength_after_clipping": minimum_marking_strength,
            "minimum_agreed_vehicles_after_clipping": minimum_agreed_vehicles,
            "maximum_marking_only_union_fraction": maximum_marking_only_union_fraction,
            "maximum_between_band_distance_px": maximum_between_band_distance_px,
            "entrance_proximity_px": entrance_proximity_px,
            "minimum_component_area_px": minimum_component_area_px,
            **physical_policy,
        },
        "totals": dict(totals),
        "gate_status": "blocked_no_independent_truth_and_target_domain_gsd",
        "limitations": [
            "Functional zones are evidence hypotheses and remain abstain.",
            "Proxy-derived physical thresholds cannot transfer until each target image's GSD is known.",
            "An internal aisle requires two parking-band components; single-sided aisles remain unknown.",
            "An entrance requires both internal-aisle and provisional public-road proximity.",
        ],
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return summary
