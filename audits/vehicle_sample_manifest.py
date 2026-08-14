from __future__ import annotations

import argparse
import hashlib
import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Iterable

from audits.vehicle_label_audit import VEHICLE_PROMPTS


DATE_PATTERN = re.compile(r"^(\d{4}-\d{2}-\d{2})")


def _stable_order(target: dict) -> str:
    return hashlib.sha256(str(target.get("target_id", "")).encode()).hexdigest()


def _audit_strata(target: dict, review_state: dict, image_size: tuple[int, int]) -> dict[str, str]:
    width, height = image_size
    bbox = target["bbox"]
    short_side = min(float(bbox[2]) - float(bbox[0]), float(bbox[3]) - float(bbox[1]))
    size = "<16" if short_side < 16 else "16-32" if short_side < 32 else ">=32"
    confidence = float(target.get("confidence") or 0)
    confidence_band = "low" if confidence < 0.45 else "medium" if confidence < 0.7 else "high"
    edge = (
        float(bbox[0]) <= 0
        or float(bbox[1]) <= 0
        or float(bbox[2]) >= width
        or float(bbox[3]) >= height
    )
    image = str(target.get("image") or "")
    date_match = DATE_PATTERN.match(image)
    prompts = sorted(set(map(str, target.get("source_prompts") or ())) & VEHICLE_PROMPTS)
    return {
        "date": date_match.group(1) if date_match else "unknown",
        "prompt_combination": " | ".join(prompts) if prompts else "unknown",
        "size": size,
        "confidence": confidence_band,
        "edge": "edge" if edge else "interior",
        "review": str(review_state.get(str(target.get("target_id")), {}).get("label") or "missing"),
        "prompt_hits": str(target.get("prompt_hits") or len(prompts)),
    }


def select_stratified_targets(
    records: Iterable[dict],
    review_state: dict,
    max_samples: int,
    image_size: tuple[int, int],
) -> list[dict]:
    candidates = []
    for source in records:
        prompts = set(map(str, source.get("source_prompts") or ()))
        if not prompts.intersection(VEHICLE_PROMPTS) or not source.get("mask_file"):
            continue
        target = {
            key: source.get(key)
            for key in (
                "target_id", "image", "bbox", "confidence", "class_name", "mask_file",
                "source_prompts", "prompt_hits", "source_target_ids",
            )
        }
        target["audit_strata"] = _audit_strata(target, review_state, image_size)
        candidates.append(target)

    ordered = sorted(candidates, key=_stable_order)
    groups: dict[str, dict[str, list[dict]]] = {}
    for dimension in ("date", "prompt_combination", "size", "confidence", "edge", "review", "prompt_hits"):
        grouped: dict[str, list[dict]] = defaultdict(list)
        for target in ordered:
            grouped[target["audit_strata"][dimension]].append(target)
        groups[dimension] = grouped

    quotas = {
        "date": 2,
        "prompt_combination": 20,
        "size": 30,
        "confidence": 30,
        "edge": 30,
        "review": 30,
        "prompt_hits": 10,
    }
    selected: list[dict] = []
    selected_ids = set()

    def add(target: dict) -> None:
        target_id = str(target["target_id"])
        if target_id not in selected_ids and len(selected) < max_samples:
            selected_ids.add(target_id)
            selected.append(target)

    for dimension, grouped in groups.items():
        for value in sorted(grouped):
            added = 0
            for target in grouped[value]:
                before = len(selected)
                add(target)
                added += len(selected) - before
                if added >= quotas[dimension] or len(selected) >= max_samples:
                    break
            if len(selected) >= max_samples:
                break
        if len(selected) >= max_samples:
            break
    for target in ordered:
        add(target)
        if len(selected) >= max_samples:
            break
    return selected


def _iter_jsonl(path: Path) -> Iterable[dict]:
    with path.open("r", encoding="utf-8") as stream:
        for line in stream:
            yield json.loads(line)


def build_manifest(index_path: Path, state_path: Path, max_samples: int) -> dict:
    state = json.loads(state_path.read_text(encoding="utf-8"))
    targets = select_stratified_targets(_iter_jsonl(index_path), state, max_samples, (640, 640))
    coverage: dict[str, Counter] = defaultdict(Counter)
    for target in targets:
        for dimension, value in target["audit_strata"].items():
            coverage[dimension][value] += 1
    return {
        "schema_version": 1,
        "sampling_method": "deterministic stratified coverage, then hash-ordered fill",
        "population": "merged targets intersecting the four vehicle prompts",
        "max_samples": max_samples,
        "sample_size": len(targets),
        "coverage": {key: dict(value) for key, value in coverage.items()},
        "targets": targets,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a reproducible vehicle mask audit sample")
    parser.add_argument("--index", type=Path, required=True)
    parser.add_argument("--state", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-samples", type=int, default=480)
    args = parser.parse_args()
    result = build_manifest(args.index, args.state, args.max_samples)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"output": str(args.output), "sample_size": result["sample_size"], "coverage": result["coverage"]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
