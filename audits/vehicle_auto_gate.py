from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

from audits.vehicle_label_audit import VEHICLE_PROMPTS


def _valid_bbox(bbox: object) -> bool:
    return (
        isinstance(bbox, list)
        and len(bbox) == 4
        and all(isinstance(value, (int, float)) for value in bbox)
        and bbox[2] > bbox[0]
        and bbox[3] > bbox[1]
    )


def classify_vehicle_candidate(
    target: dict,
    geometry: dict,
    first_teacher_iou: float,
    second_teacher_iou: float,
) -> dict:
    reject_reasons = []
    if not _valid_bbox(target.get("bbox")):
        reject_reasons.append("invalid_bbox")
    if geometry.get("empty_mask"):
        reject_reasons.append("empty_mask")
    prompts = set(map(str, target.get("source_prompts") or ()))
    if prompts and not prompts.issubset(VEHICLE_PROMPTS):
        reject_reasons.append("mixed_non_vehicle_prompt")
    if reject_reasons:
        return {"decision": "auto_reject", "reasons": reject_reasons}

    abstain_reasons = []
    if not geometry:
        abstain_reasons.append("missing_mask_geometry")
    if int(geometry.get("material_components") or 0) != 1:
        abstain_reasons.append("fragmented_or_merged_mask")
    if float(geometry.get("largest_component_fraction") or 0) < 0.95:
        abstain_reasons.append("weak_component_dominance")
    if float(geometry.get("mask_bbox_iou") or 0) < 0.3:
        abstain_reasons.append("mask_bbox_export_instability")
    if float(target.get("confidence") or 0) < 0.7:
        abstain_reasons.append("sam_confidence_below_0_7")
    if int(target.get("prompt_hits") or 0) < 2:
        abstain_reasons.append("single_prompt_only")
    if target.get("audit_strata", {}).get("edge") == "edge":
        abstain_reasons.append("image_edge")
    bbox = target.get("bbox")
    if _valid_bbox(bbox) and min(bbox[2] - bbox[0], bbox[3] - bbox[1]) < 8:
        abstain_reasons.append("below_8px_short_side")
    if first_teacher_iou < 0.5 or second_teacher_iou < 0.5:
        abstain_reasons.append("independent_model_disagreement")
    return {
        "decision": "abstain" if abstain_reasons else "auto_accept",
        "reasons": abstain_reasons or ["all_high_purity_rules_passed"],
    }


def build_gate_result(
    manifest_path: Path,
    geometry_path: Path,
    first_agreement_path: Path,
    second_agreement_path: Path,
) -> dict:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    geometry_profile = json.loads(geometry_path.read_text(encoding="utf-8"))
    first = json.loads(first_agreement_path.read_text(encoding="utf-8"))
    second = json.loads(second_agreement_path.read_text(encoding="utf-8"))
    geometry_by_id = {
        str(target["target_id"]): target.get("geometry") or {}
        for target in geometry_profile["targets"]
    }
    first_by_id = {str(target["target_id"]): target for target in first["targets"]}
    second_by_id = {str(target["target_id"]): target for target in second["targets"]}
    decisions = Counter()
    reasons = Counter()
    strata: dict[str, Counter] = defaultdict(Counter)
    results = []
    for target in manifest["targets"]:
        target_id = str(target["target_id"])
        first_iou = float(first_by_id.get(target_id, {}).get("mask_bbox_best_iou") or 0)
        second_iou = float(second_by_id.get(target_id, {}).get("mask_bbox_best_iou") or 0)
        result = classify_vehicle_candidate(
            target, geometry_by_id.get(target_id, {}), first_iou, second_iou
        )
        decisions[result["decision"]] += 1
        reasons.update(result["reasons"])
        for dimension in ("size", "confidence", "review", "prompt_hits"):
            key = f"{dimension}={target['audit_strata'][dimension]}"
            strata[key][result["decision"]] += 1
        results.append({
            "target_id": target_id,
            **result,
            "evidence": {
                "sam_confidence": target.get("confidence"),
                "prompt_hits": target.get("prompt_hits"),
                "first_teacher_iou": first_iou,
                "second_teacher_iou": second_iou,
            },
        })
    return {
        "schema_version": 1,
        "scope": "480-target stratified audit sample; not a production-wide decision set",
        "ground_truth_limitation": "model agreement and geometry only; no independent human truth",
        "policy": {
            "auto_accept": "high SAM confidence, >=2 prompts, stable single mask, interior, and both detector IoU >=0.5",
            "auto_reject": "only malformed bbox, empty mask, or mixed non-vehicle prompt",
            "abstain": "all other evidence gaps; never used as background",
        },
        "decisions": dict(decisions),
        "reasons": dict(reasons),
        "strata": {key: dict(value) for key, value in strata.items()},
        "targets": results,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Conservative three-way vehicle label gate")
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--geometry", type=Path, required=True)
    parser.add_argument("--first-agreement", type=Path, required=True)
    parser.add_argument("--second-agreement", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = build_gate_result(
        args.manifest, args.geometry, args.first_agreement, args.second_agreement
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"output": str(args.output), **result["decisions"]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
