from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Iterable


VEHICLE_PROMPTS = {
    "parked car",
    "parked vehicle",
    "vehicle in parking lot",
    "car in parking spot",
}


def _quantile(values: list[float], fraction: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    index = round((len(ordered) - 1) * fraction)
    return ordered[index]


def _valid_bbox(bbox: object) -> bool:
    return (
        isinstance(bbox, list)
        and len(bbox) == 4
        and all(isinstance(value, (int, float)) for value in bbox)
        and bbox[2] > bbox[0]
        and bbox[3] > bbox[1]
    )


def _profile_raw_prompt(result_dir: Path, image_size: tuple[int, int]) -> tuple[str, dict]:
    width, height = image_size
    confidences: list[float] = []
    short_sides: list[float] = []
    area_ratios: list[float] = []
    classes: Counter[str] = Counter()
    size_buckets: Counter[str] = Counter()
    metrics = Counter()
    prompt = result_dir.parent.name.replace("_", " ")

    for result_path in result_dir.glob("*.json"):
        metrics["json_files"] += 1
        try:
            record = json.loads(result_path.read_text(encoding="utf-8"))
        except (OSError, UnicodeError, json.JSONDecodeError):
            metrics["parse_errors"] += 1
            continue
        prompt = str(record.get("text_prompt") or prompt)
        targets = record.get("targets") or []
        if targets:
            metrics["images_with_targets"] += 1
        for target in targets:
            metrics["targets"] += 1
            if target.get("mask_file"):
                metrics["mask_references"] += 1
            classes[str(target.get("class_name") or "(missing)")] += 1
            confidence = target.get("confidence")
            if isinstance(confidence, (int, float)):
                confidences.append(float(confidence))
            bbox = target.get("bbox")
            if not _valid_bbox(bbox):
                metrics["invalid_bbox"] += 1
                continue
            box_width, box_height = bbox[2] - bbox[0], bbox[3] - bbox[1]
            short_side = min(box_width, box_height)
            short_sides.append(short_side)
            area_ratios.append(box_width * box_height / (width * height))
            if bbox[0] <= 0 or bbox[1] <= 0 or bbox[2] >= width or bbox[3] >= height:
                metrics["edge_touching"] += 1
            bucket = "<16" if short_side < 16 else "16-32" if short_side < 32 else "32-64" if short_side < 64 else ">=64"
            size_buckets[bucket] += 1

    profile = dict(metrics)
    profile.update({
        "class_names": dict(classes),
        "size_buckets_px": dict(size_buckets),
        "confidence": {name: _quantile(confidences, fraction) for name, fraction in (("p01", .01), ("p10", .1), ("p50", .5), ("p90", .9), ("p99", .99))},
        "short_side_px": {name: _quantile(short_sides, fraction) for name, fraction in (("p01", .01), ("p10", .1), ("p50", .5), ("p90", .9), ("p99", .99))},
        "bbox_area_ratio": {name: _quantile(area_ratios, fraction) for name, fraction in (("p01", .01), ("p50", .5), ("p90", .9), ("p99", .99))},
    })
    return prompt, profile


def _iter_jsonl(path: Path) -> Iterable[dict]:
    if not path.exists():
        return
    with path.open("r", encoding="utf-8") as stream:
        for line in stream:
            try:
                yield json.loads(line)
            except json.JSONDecodeError:
                yield {"_parse_error": True}


def build_vehicle_inventory(
    project_dir: Path,
    batch_dir: Path,
    image_size: tuple[int, int],
    source_image_dir: Path | None = None,
    source_domain: str = "unknown",
) -> dict:
    run_meta_path = batch_dir / "run_meta.json"
    run_meta = json.loads(run_meta_path.read_text(encoding="utf-8")) if run_meta_path.exists() else {}
    raw_profiles = {}
    raw_root = batch_dir / "raw_prompts"
    if raw_root.exists():
        for prompt_dir in sorted(path for path in raw_root.iterdir() if path.is_dir()):
            normalized = prompt_dir.name.replace("_", " ")
            if normalized not in VEHICLE_PROMPTS:
                continue
            prompt, profile = _profile_raw_prompt(prompt_dir / "sam3_results", image_size)
            raw_profiles[prompt] = profile

    state_path = batch_dir / "merged" / "review" / "target_state.json"
    state = json.loads(state_path.read_text(encoding="utf-8")) if state_path.exists() else {}
    merged_counts = Counter()
    merged_classes: Counter[str] = Counter()
    prompt_hit_counts: Counter[str] = Counter()
    vehicle_target_ids = set()
    index_path = batch_dir / "merged" / "review_cache" / "target_index.jsonl"
    for target in _iter_jsonl(index_path):
        if target.get("_parse_error"):
            merged_counts["parse_errors"] += 1
            continue
        merged_counts["total"] += 1
        prompts = {str(value) for value in target.get("source_prompts") or []}
        if not prompts.intersection(VEHICLE_PROMPTS):
            continue
        target_id = str(target.get("target_id") or "")
        if target_id:
            vehicle_target_ids.add(target_id)
        merged_counts["vehicle_related"] += 1
        if prompts.issubset(VEHICLE_PROMPTS):
            merged_counts["vehicle_only"] += 1
        else:
            merged_counts["mixed_with_non_vehicle"] += 1
        merged_classes[str(target.get("class_name") or "(missing)")] += 1
        prompt_hit_counts[str(len(prompts))] += 1

    review_labels: Counter[str] = Counter()
    review_sources: Counter[str] = Counter()
    auto_count = 0
    missing_state = 0
    for target_id in vehicle_target_ids:
        review = state.get(target_id)
        if not review:
            missing_state += 1
            continue
        review_labels[str(review.get("label") or "(missing)")] += 1
        review_sources[str(review.get("source") or "(missing)")] += 1
        if review.get("is_auto"):
            auto_count += 1

    return {
        "schema_version": 1,
        "project_dir": str(project_dir),
        "batch_dir": str(batch_dir),
        "source_domain": source_domain,
        "source_image_dir": str(source_image_dir) if source_image_dir else None,
        "image_size": list(image_size),
        "run_meta": run_meta,
        "vehicle_prompts": sorted(VEHICLE_PROMPTS),
        "raw_prompts": raw_profiles,
        "merged": {**dict(merged_counts), "class_names": dict(merged_classes), "prompt_hits": dict(prompt_hit_counts)},
        "review": {
            "vehicle_related_targets": len(vehicle_target_ids),
            "labels": dict(review_labels),
            "sources": dict(review_sources),
            "auto": auto_count,
            "missing_state": missing_state,
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Read-only SAM3 vehicle label inventory")
    parser.add_argument("--project", type=Path, required=True)
    parser.add_argument("--batch", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--width", type=int, required=True)
    parser.add_argument("--height", type=int, required=True)
    parser.add_argument("--source-images", type=Path)
    parser.add_argument("--source-domain", default="unknown")
    args = parser.parse_args()
    result = build_vehicle_inventory(
        args.project, args.batch, (args.width, args.height), args.source_images, args.source_domain
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"output": str(args.output), "raw_prompts": list(result["raw_prompts"]), "merged": result["merged"], "review": result["review"]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
