from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Iterable

from audits.vehicle_label_audit import VEHICLE_PROMPTS


EXPANSION_RATIO = 0.5


def _expand_bbox(bbox: list[float], ratio: float = EXPANSION_RATIO) -> list[float]:
    x1, y1, x2, y2 = bbox
    width, height = x2 - x1, y2 - y1
    return [x1 - width * ratio, y1 - height * ratio,
            x2 + width * ratio, y2 + height * ratio]


def _overlaps(first: list[float], second: list[float]) -> bool:
    return (
        first[0] < second[2]
        and first[2] > second[0]
        and first[1] < second[3]
        and first[3] > second[1]
    )


def classify_geometry_filter_evidence(
    targets: list[dict], geometry_by_id: dict[str, dict]
) -> dict[str, str]:
    """Reproduce the existing parking-line filter and expose why it accepted."""
    line_boxes = [
        _expand_bbox(target["bbox"])
        for target in targets
        if geometry_by_id.get(str(target.get("target_id")), {}).get("has_lines")
    ]
    evidence = {}
    for target in targets:
        target_id = str(target.get("target_id"))
        geometry = geometry_by_id.get(target_id, {})
        if geometry.get("has_lines"):
            evidence[target_id] = "direct_has_lines"
        elif any(_overlaps(_expand_bbox(target["bbox"]), line_box) for line_box in line_boxes):
            evidence[target_id] = "adjacent_to_lines"
        else:
            evidence[target_id] = "no_line_evidence"
    return evidence


def _iter_jsonl(path: Path) -> Iterable[dict]:
    with path.open("r", encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            try:
                yield json.loads(line)
            except json.JSONDecodeError:
                yield {"_parse_error": line_number}


def _load_geometry(path: Path) -> tuple[dict[str, dict], Counter]:
    geometry_by_id: dict[str, dict] = {}
    metrics = Counter()
    for record in _iter_jsonl(path):
        if record.get("_parse_error"):
            metrics["parse_errors"] += 1
            continue
        target_id = str(record.get("target_id") or "")
        if not target_id:
            metrics["missing_target_id"] += 1
            continue
        geometry_by_id[target_id] = record
        metrics["records"] += 1
        if record.get("has_lines"):
            metrics["has_lines"] += 1
    return geometry_by_id, metrics


def _is_vehicle_related(target: dict) -> bool:
    return bool(set(map(str, target.get("source_prompts") or ())) & VEHICLE_PROMPTS)


def review_label_for_evidence(evidence: str) -> str:
    return "reject" if evidence == "no_line_evidence" else "accept"


def build_geometry_filter_profile(
    index_path: Path, geometry_path: Path, state_path: Path
) -> dict:
    geometry_by_id, geometry_metrics = _load_geometry(geometry_path)
    state = json.loads(state_path.read_text(encoding="utf-8"))
    evidence_counts = Counter()
    state_counts = Counter()
    reproduction = Counter()
    index_metrics = Counter()

    current_image: str | None = None
    image_targets: list[dict] = []

    def profile_image(targets: list[dict]) -> None:
        if not targets:
            return
        evidence_by_id = classify_geometry_filter_evidence(targets, geometry_by_id)
        for target in targets:
            if not _is_vehicle_related(target):
                continue
            target_id = str(target.get("target_id"))
            evidence = evidence_by_id[target_id]
            evidence_counts[evidence] += 1
            if target_id not in geometry_by_id:
                evidence_counts["missing_geometry"] += 1
            label = str(state.get(target_id, {}).get("label") or "missing")
            state_counts[label] += 1
            expected = review_label_for_evidence(evidence)
            reproduction["matches" if label == expected else "mismatches"] += 1

    for target in _iter_jsonl(index_path):
        if target.get("_parse_error"):
            index_metrics["parse_errors"] += 1
            continue
        index_metrics["records"] += 1
        image = str(target.get("image") or "")
        if current_image is not None and image != current_image:
            profile_image(image_targets)
            image_targets = []
            index_metrics["images"] += 1
        current_image = image
        image_targets.append(target)
    if image_targets:
        profile_image(image_targets)
        index_metrics["images"] += 1

    return {
        "schema_version": 1,
        "filter_semantics": {
            "direct_has_lines": "geometry API found parking-line evidence inside this target bbox",
            "adjacent_to_lines": "expanded bbox overlaps another target with parking-line evidence",
            "no_line_evidence": "neither condition was met",
            "independent_vehicle_validation": False,
        },
        "geometry": dict(geometry_metrics),
        "index": dict(index_metrics),
        "vehicle_evidence": dict(evidence_counts),
        "vehicle_state_labels": dict(state_counts),
        "state_reproduction": dict(reproduction),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit provenance of the existing geometry filter")
    parser.add_argument("--index", type=Path, required=True)
    parser.add_argument("--geometry", type=Path, required=True)
    parser.add_argument("--state", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = build_geometry_filter_profile(args.index, args.geometry, args.state)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"output": str(args.output), **result["vehicle_evidence"]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
