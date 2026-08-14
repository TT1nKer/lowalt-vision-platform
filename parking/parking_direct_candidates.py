from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
from typing import Any


def _manifest_aoi_ids(path: Path) -> set[str]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    return {str(record["aoi_id"]) for record in payload.get("aois") or []}


def _support_level(
    aoi_id: str,
    vehicle_aoi_ids: set[str],
    band_aoi_ids: set[str],
) -> str:
    if aoi_id in band_aoi_ids:
        return "vehicle_row_supported"
    if aoi_id in vehicle_aoi_ids:
        return "vehicle_detected"
    return "segformer_only"


def export_direct_candidates(
    *,
    candidate_root: Path,
    vehicle_manifest: Path,
    band_manifest: Path,
    output_path: Path,
) -> dict[str, Any]:
    candidate_paths = sorted(candidate_root.glob("*/mask_geometries.geojson"))
    if not candidate_paths:
        raise ValueError(f"no SegFormer candidate geometries found in {candidate_root}")

    vehicle_aoi_ids = _manifest_aoi_ids(vehicle_manifest)
    band_aoi_ids = _manifest_aoi_ids(band_manifest)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = output_path.with_name(f".{output_path.name}.tmp")
    support_counts = Counter()
    seen_aoi_ids: set[str] = set()
    first_feature = True

    try:
        with temporary_path.open("w", encoding="utf-8") as stream:
            stream.write(
                '{"type":"FeatureCollection","truth_status":'
                '"model_prediction_not_ground_truth","features":['
            )
            for candidate_path in candidate_paths:
                payload = json.loads(candidate_path.read_text(encoding="utf-8"))
                aoi_id = str(payload.get("aoi_id") or candidate_path.parent.name)
                features = payload.get("features") or []
                if not features:
                    raise ValueError(f"candidate has no geometry: {candidate_path}")
                support_level = _support_level(aoi_id, vehicle_aoi_ids, band_aoi_ids)
                support_counts[support_level] += 1
                seen_aoi_ids.add(aoi_id)
                for feature in features:
                    geometry = feature.get("geometry")
                    if not geometry:
                        raise ValueError(f"candidate geometry is missing: {candidate_path}")
                    output_feature = {
                        "type": "Feature",
                        "geometry": geometry,
                        "properties": {
                            **(feature.get("properties") or {}),
                            "aoi_id": aoi_id,
                            "object_type": "parking_facility_candidate",
                            "review_state": "abstain",
                            "geometry_role": "primary_segformer_prediction",
                            "support_level": support_level,
                            "vehicle_evidence_role": "auxiliary_not_boundary",
                            "truth_status": "model_prediction_not_ground_truth",
                        },
                    }
                    if not first_feature:
                        stream.write(",")
                    json.dump(output_feature, stream, ensure_ascii=False, separators=(",", ":"))
                    first_feature = False
            stream.write("]}")
        temporary_path.replace(output_path)
    except Exception:
        temporary_path.unlink(missing_ok=True)
        raise

    summary = {
        "total_candidates": len(seen_aoi_ids),
        "segformer_only": support_counts["segformer_only"],
        "vehicle_detected": support_counts["vehicle_detected"],
        "vehicle_row_supported": support_counts["vehicle_row_supported"],
        "geometry_source": "UTEL-UIUC/SegFormer-large-parking",
        "vehicle_evidence_role": "auxiliary_not_boundary",
        "truth_status": "model_prediction_not_ground_truth",
        "output": str(output_path),
    }
    output_path.with_name("summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Export SegFormer parking geometry without vehicle-gating its boundary"
    )
    parser.add_argument("--candidate-root", type=Path, required=True)
    parser.add_argument("--vehicle-manifest", type=Path, required=True)
    parser.add_argument("--band-manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    summary = export_direct_candidates(
        candidate_root=args.candidate_root,
        vehicle_manifest=args.vehicle_manifest,
        band_manifest=args.band_manifest,
        output_path=args.output,
    )
    print(json.dumps(summary, ensure_ascii=False))


if __name__ == "__main__":
    main()
