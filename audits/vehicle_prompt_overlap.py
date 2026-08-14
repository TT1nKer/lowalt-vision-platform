from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path


def bbox_iou(first: list[float], second: list[float]) -> float:
    left = max(first[0], second[0])
    top = max(first[1], second[1])
    right = min(first[2], second[2])
    bottom = min(first[3], second[3])
    intersection = max(0.0, right - left) * max(0.0, bottom - top)
    first_area = max(0.0, first[2] - first[0]) * max(0.0, first[3] - first[1])
    second_area = max(0.0, second[2] - second[0]) * max(0.0, second[3] - second[1])
    union = first_area + second_area - intersection
    return intersection / union if union else 0.0


def profile_prompt_overlap(
    candidates: list[dict], iou_thresholds: tuple[float, ...] = (0.5, 0.65, 0.8)
) -> dict:
    """Profile transitive candidate groups without changing pipeline labels."""
    canonical = sorted(
        candidates,
        key=lambda item: (str(item.get("prompt", "")), tuple(item.get("bbox") or ())),
    )
    result = {}
    for threshold in iou_thresholds:
        parents = list(range(len(canonical)))

        def find(index: int) -> int:
            while parents[index] != index:
                parents[index] = parents[parents[index]]
                index = parents[index]
            return index

        def union(left: int, right: int) -> None:
            left_root, right_root = find(left), find(right)
            if left_root != right_root:
                parents[max(left_root, right_root)] = min(left_root, right_root)

        pair_matches = 0
        cross_prompt_pairs = 0
        for left in range(len(canonical)):
            left_box = canonical[left].get("bbox")
            if not left_box or len(left_box) != 4:
                continue
            for right in range(left + 1, len(canonical)):
                right_box = canonical[right].get("bbox")
                if not right_box or len(right_box) != 4:
                    continue
                if bbox_iou(left_box, right_box) >= threshold:
                    union(left, right)
                    pair_matches += 1
                    if canonical[left].get("prompt") != canonical[right].get("prompt"):
                        cross_prompt_pairs += 1

        groups: dict[int, list[dict]] = {}
        for index, candidate in enumerate(canonical):
            groups.setdefault(find(index), []).append(candidate)
        result[f"{threshold:g}"] = {
            "groups": len(groups),
            "multi_candidate_groups": sum(1 for group in groups.values() if len(group) > 1),
            "cross_prompt_groups": sum(
                1 for group in groups.values()
                if len({item.get("prompt") for item in group}) > 1
            ),
            "pair_matches": pair_matches,
            "cross_prompt_pairs": cross_prompt_pairs,
        }
    return result


def count_pipeline_greedy_groups(candidates: list[dict], threshold: float = 0.65) -> int:
    """Reproduce the current representative-based merge count for diagnostics."""
    representatives: list[dict] = []
    for candidate in candidates:
        for index, representative in enumerate(representatives):
            if bbox_iou(candidate["bbox"], representative["bbox"]) < threshold:
                continue
            if (candidate.get("confidence") or 0) > (representative.get("confidence") or 0):
                representatives[index] = candidate
            break
        else:
            representatives.append(candidate)
    return len(representatives)


def build_dataset_overlap_profile(
    raw_root: Path,
    prompt_slugs: tuple[str, ...],
    thresholds: tuple[float, ...] = (0.5, 0.65, 0.8),
) -> dict:
    result_dirs = [raw_root / slug / "sam3_results" for slug in prompt_slugs]
    filenames = sorted({path.name for directory in result_dirs for path in directory.glob("*.json")})
    aggregate = {f"{threshold:g}": Counter() for threshold in thresholds}
    greedy = Counter()
    parse_errors = 0
    candidate_count = 0
    images_with_candidates = 0

    for image_index, filename in enumerate(filenames, start=1):
        candidates = []
        for directory in result_dirs:
            path = directory / filename
            if not path.exists():
                continue
            try:
                record = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, UnicodeError, json.JSONDecodeError):
                parse_errors += 1
                continue
            prompt = str(record.get("text_prompt") or directory.parent.name.replace("_", " "))
            for target in record.get("targets") or []:
                bbox = target.get("bbox")
                if not isinstance(bbox, list) or len(bbox) != 4:
                    continue
                candidates.append({
                    "bbox": bbox,
                    "prompt": prompt,
                    "confidence": target.get("confidence"),
                })
        candidate_count += len(candidates)
        if candidates:
            images_with_candidates += 1
        profile = profile_prompt_overlap(candidates, thresholds)
        for threshold_key, metrics in profile.items():
            aggregate[threshold_key].update(metrics)
        forward = count_pipeline_greedy_groups(candidates)
        reverse = count_pipeline_greedy_groups(list(reversed(candidates)))
        greedy["forward_groups"] += forward
        greedy["reverse_groups"] += reverse
        if forward != reverse:
            greedy["order_sensitive_images"] += 1
            greedy["absolute_group_delta"] += abs(forward - reverse)
        if image_index % 500 == 0:
            print(f"progress={image_index}/{len(filenames)}", flush=True)

    thresholds_out = {}
    for threshold_key, metrics in aggregate.items():
        groups = metrics["groups"]
        thresholds_out[threshold_key] = {
            **dict(metrics),
            "collapsed_candidates": candidate_count - groups,
            "collapsed_fraction": (candidate_count - groups) / candidate_count if candidate_count else None,
        }
    return {
        "schema_version": 1,
        "raw_root": str(raw_root),
        "prompt_slugs": list(prompt_slugs),
        "images": len(filenames),
        "images_with_candidates": images_with_candidates,
        "candidates": candidate_count,
        "parse_errors": parse_errors,
        "thresholds": thresholds_out,
        "pipeline_greedy_0.65": dict(greedy),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Read-only cross-prompt overlap profile")
    parser.add_argument("--raw-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--prompt-slug", action="append", required=True)
    args = parser.parse_args()
    result = build_dataset_overlap_profile(args.raw_root, tuple(args.prompt_slug))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
