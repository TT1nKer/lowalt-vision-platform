from __future__ import annotations

import argparse
import json
from pathlib import Path

from PIL import Image, ImageChops

from lowalt_platform.services.parking_refinement import binary_mask


REJECTION_OVERLAP = 0.90


def overlap_ratio(candidate_path: Path, building_path: Path, vegetation_path: Path) -> float:
    with Image.open(candidate_path) as candidate, Image.open(building_path) as building, Image.open(vegetation_path) as vegetation:
        candidate_mask = binary_mask(candidate)
        exclusion = ImageChops.lighter(binary_mask(building), binary_mask(vegetation))
        overlap = ImageChops.multiply(candidate_mask, exclusion)
        area = candidate_mask.histogram()[255]
        return overlap.histogram()[255] / area if area else 0.0


def main() -> None:
    parser = argparse.ArgumentParser(description="Apply a minimal object-level gate to parking candidates")
    parser.add_argument("--project-dir", type=Path, default=Path("."))
    args = parser.parse_args()

    root = args.project_dir.resolve()
    source_path = root / "quality" / "parking_direct_first_layers" / "parking_facility_candidates.geojson"
    mask_root = root / "quality" / "parkseg_imagery_masks"
    secondary_root = root / "quality" / "parking_secondary_sam3"
    output_root = root / "quality" / "parking_candidate_gate"
    output_root.mkdir(parents=True, exist_ok=True)

    collection = json.loads(source_path.read_text(encoding="utf-8"))
    counts = {"accepted": 0, "rejected": 0, "unknown": 0}
    processed = 0
    for feature in collection["features"]:
        properties = feature["properties"]
        aoi_id = properties["aoi_id"]
        directory = secondary_root / aoi_id
        evidence_paths = [directory / "building.png", directory / "vegetation.png"]
        ratio = None
        if all(path.is_file() for path in evidence_paths):
            ratio = overlap_ratio(mask_root / f"{aoi_id}_mask.png", *evidence_paths)
            processed += 1

        if properties.get("support_level") == "vehicle_row_supported":
            status, reason = "accepted", "vehicle_row_supported"
        elif ratio is not None and ratio >= REJECTION_OVERLAP:
            status, reason = "rejected", "building_or_vegetation_overlap"
        else:
            status, reason = "unknown", "insufficient_evidence"
        properties["gate_status"] = status
        properties["gate_reason"] = reason
        properties["exclusion_overlap"] = round(ratio, 4) if ratio is not None else None
        counts[status] += 1

    output_path = output_root / "parking_candidates.geojson"
    output_path.write_text(json.dumps(collection, ensure_ascii=False), encoding="utf-8")
    summary = {
        "total": len(collection["features"]),
        "secondary_processed": processed,
        **counts,
        "rule": {"accept": "vehicle_row_supported", "reject_exclusion_overlap": REJECTION_OVERLAP},
        "truth_status": "deterministic_gate_not_ground_truth",
    }
    (output_root / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False))


if __name__ == "__main__":
    main()
