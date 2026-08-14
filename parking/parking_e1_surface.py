from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from parking_map.image_evidence import geographic_geometry_to_mask
from parking_map.surface_hypothesis import select_anchor_candidate_ids, select_anchored_surfaces
from parking_map.tile_georeference import mask_to_geographic_geometry


def _union_geometries(
    features: list[dict[str, Any]],
    image_name: str,
    image_shape: tuple[int, int],
) -> np.ndarray:
    union = np.zeros(image_shape, dtype=np.uint8)
    for feature in features:
        mask = geographic_geometry_to_mask(feature["geometry"], image_name, image_shape)
        union[mask != 0] = 255
    return union


def _load_paved_candidates(sam_run: Path, image_name: str) -> list[dict[str, Any]]:
    prompt_directory = sam_run / "raw_prompts" / "paved_area"
    result_path = prompt_directory / "sam3_results" / f"{Path(image_name).stem}.json"
    record = json.loads(result_path.read_text(encoding="utf-8"))
    candidates = []
    for target in record.get("targets") or []:
        mask_path = prompt_directory / "mask" / str(target.get("mask_file") or "")
        candidates.append({
            "target_id": target.get("source_target_id"),
            "confidence": target.get("confidence"),
            "mask": cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE),
        })
    return candidates


def _write_overlay(
    source_image: np.ndarray,
    anchor_mask: np.ndarray,
    exclusion_mask: np.ndarray,
    surface_mask: np.ndarray,
    output_path: Path,
) -> None:
    overlay = source_image.copy()
    for mask, color, width in (
        (exclusion_mask, (40, 40, 230), 2),
        (anchor_mask, (0, 200, 255), 2),
        (surface_mask, (80, 220, 80), 3),
    ):
        contours, _ = cv2.findContours(mask, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(overlay, contours, -1, color, width)
    cv2.imwrite(str(output_path), overlay)


def build_surface_hypotheses(
    *,
    manifest_path: Path,
    images: Path,
    exclusion_evidence_root: Path,
    sam_run: Path,
    output_dir: Path,
) -> dict[str, Any]:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    aoi_by_image = {str(aoi["image"]): aoi for aoi in manifest.get("aois") or []}
    result_directory = sam_run / "raw_prompts" / "paved_area" / "sam3_results"
    result_images = {
        f"{path.stem}.png" for path in result_directory.glob("*.json")
    }
    selected_aois = [aoi_by_image[image] for image in sorted(result_images) if image in aoi_by_image]
    if not selected_aois:
        raise ValueError("SAM paved-area run does not match any manifest AOI")

    output_dir.mkdir(parents=True, exist_ok=True)
    totals = Counter()
    rejected_reasons = Counter()
    surface_area_fractions = []
    for aoi in selected_aois:
        aoi_id = str(aoi["aoi_id"])
        image_name = str(aoi["image"])
        source_image = cv2.imread(str(images / image_name), cv2.IMREAD_COLOR)
        if source_image is None:
            raise ValueError(f"unreadable source image: {image_name}")
        evidence_dir = exclusion_evidence_root / aoi_id
        parking_candidates = json.loads(
            (evidence_dir / "parking_candidates.geojson").read_text(encoding="utf-8")
        )
        evidence_layers = json.loads(
            (evidence_dir / "evidence_layers.json").read_text(encoding="utf-8")
        )
        qualified_anchor_ids = select_anchor_candidate_ids(
            evidence_layers.get("candidate_measurements") or [],
            image_area_px=source_image.shape[0] * source_image.shape[1],
        )
        qualified_anchor_features = [
            feature
            for feature in parking_candidates.get("features") or []
            if str((feature.get("properties") or {}).get("candidate_id")) in qualified_anchor_ids
        ]
        exclusions = json.loads(
            (evidence_dir / "exclusion_layers.geojson").read_text(encoding="utf-8")
        )
        anchor_mask = _union_geometries(
            qualified_anchor_features, image_name, source_image.shape[:2]
        )
        exclusion_mask = _union_geometries(
            exclusions.get("features") or [], image_name, source_image.shape[:2]
        )
        selection = select_anchored_surfaces(
            _load_paved_candidates(sam_run, image_name),
            anchor_mask,
            exclusion_mask,
        )
        rejected_reasons.update(selection.rejected_reason_counts)
        totals["aois"] += 1
        totals["qualified_anchor_candidates"] += len(qualified_anchor_ids)
        totals["aois_with_qualified_anchor"] += bool(qualified_anchor_ids)
        totals["accepted_paved_masks"] += len(selection.accepted_target_ids)
        totals["surface_components"] += selection.component_count
        totals["aois_with_surface_hypothesis"] += selection.component_count > 0
        surface_area_fractions.append(
            float(np.count_nonzero(selection.union_mask) / selection.union_mask.size)
        )

        features = []
        if selection.component_count:
            converted = mask_to_geographic_geometry(
                selection.union_mask,
                image_name,
                minimum_component_area=500,
            )
            features.append({
                "type": "Feature",
                "geometry": converted.geometry,
                "properties": {
                    "object_type": "parking_facility",
                    "review_state": "abstain",
                    "role": "anchored_paved_surface_hypothesis",
                    "source_version": sam_run.name,
                    "truth_status": "teacher_evidence_not_ground_truth",
                    "accepted_paved_target_ids": list(selection.accepted_target_ids),
                    "qualified_anchor_candidate_ids": sorted(qualified_anchor_ids),
                    "component_count": converted.component_count,
                    "crs": "OGC:CRS84",
                },
            })
        result = {
            "type": "FeatureCollection",
            "schema_version": 1,
            "aoi_id": aoi_id,
            "truth_status": "teacher_evidence_not_ground_truth",
            "features": features,
        }
        aoi_output = output_dir / aoi_id
        aoi_output.mkdir(parents=True, exist_ok=True)
        (aoi_output / "surface_hypothesis.geojson").write_text(
            json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        _write_overlay(
            source_image,
            anchor_mask,
            exclusion_mask,
            selection.union_mask,
            aoi_output / "surface_overlay.jpg",
        )

    summary = {
        "schema_version": 1,
        "truth_status": "teacher_evidence_not_ground_truth",
        "source_manifest": str(manifest_path),
        "sam_run": str(sam_run),
        "policy": {
            "paved_area_min_confidence": 0.4,
            "minimum_surface_area_px": 500,
            "minimum_anchor_overlap_px": 100,
            "anchor_minimum_marking_strength": 0.6,
            "anchor_minimum_agreed_vehicles": 2,
            "anchor_maximum_area_fraction": 0.05,
            "no_buffer_hull_or_obb_fill": True,
        },
        "totals": dict(totals),
        "surface_area_fraction": {
            "minimum": min(surface_area_fractions),
            "median": sorted(surface_area_fractions)[len(surface_area_fractions) // 2],
            "maximum": max(surface_area_fractions),
        },
        "rejected_reasons": dict(rejected_reasons),
        "gate_status": "blocked_no_independent_truth_and_incomplete_facility_boundary",
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Build anchored paved-surface parking hypotheses")
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--images", type=Path, required=True)
    parser.add_argument("--exclusion-evidence-root", type=Path, required=True)
    parser.add_argument("--sam-run", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    summary = build_surface_hypotheses(
        manifest_path=args.manifest,
        images=args.images,
        exclusion_evidence_root=args.exclusion_evidence_root,
        sam_run=args.sam_run,
        output_dir=args.output_dir,
    )
    print(json.dumps(summary, ensure_ascii=False))


if __name__ == "__main__":
    main()
