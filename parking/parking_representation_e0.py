from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path

import cv2
import numpy as np


POSITIVE_LABELS = {"accept", "hard_positive"}
MIN_COMPONENT_AREA = 10


def _obb_area(points: np.ndarray, minimum_area: int) -> float:
    (_, _), (width, height), _ = cv2.minAreaRect(points.astype(np.float32))
    return max(float(width * height), float(minimum_area))


def analyze_parking_mask(mask: np.ndarray) -> dict:
    grayscale = mask.max(axis=2) if mask.ndim == 3 else mask
    binary = (grayscale >= 128).astype(np.uint8)
    count, labels, stats, _ = cv2.connectedComponentsWithStats(binary, 8)
    valid_labels = [
        label for label in range(1, count)
        if int(stats[label, cv2.CC_STAT_AREA]) >= MIN_COMPONENT_AREA
    ]
    if not valid_labels:
        return {"empty_valid_mask": True, "valid_components": 0, "mask_area_px": 0}

    valid_mask = np.isin(labels, valid_labels).astype(np.uint8)
    ys, xs = np.nonzero(valid_mask)
    mask_area = int(valid_mask.sum())
    all_points = np.column_stack((xs, ys))
    global_obb_area = _obb_area(all_points, mask_area)
    component_obb_area = 0.0
    polygon_area = 0.0
    polygon_vertices = 0
    component_areas = []
    for label in valid_labels:
        component = (labels == label).astype(np.uint8)
        component_area = int(component.sum())
        component_areas.append(component_area)
        component_y, component_x = np.nonzero(component)
        component_points = np.column_stack((component_x, component_y))
        component_obb_area += _obb_area(component_points, component_area)
        contours, _ = cv2.findContours(component, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for contour in contours:
            epsilon = 0.01 * cv2.arcLength(contour, True)
            polygon = cv2.approxPolyDP(contour, epsilon, True)
            polygon_area += cv2.contourArea(polygon)
            polygon_vertices += len(polygon)
    return {
        "empty_valid_mask": False,
        "valid_components": len(valid_labels),
        "mask_area_px": mask_area,
        "global_obb_area_ratio": global_obb_area / mask_area,
        "component_obb_area_ratio": component_obb_area / mask_area,
        "global_obb_irrelevant_fraction": 1 - mask_area / global_obb_area,
        "component_obb_irrelevant_fraction": 1 - mask_area / component_obb_area,
        "largest_component_retained_fraction": max(component_areas) / mask_area,
        "simplified_polygon_area_ratio": polygon_area / mask_area,
        "simplified_polygon_vertices": polygon_vertices,
    }


def _quantiles(values: list[float]) -> dict[str, float | None]:
    if not values:
        return {"p10": None, "p50": None, "p90": None, "p99": None}
    ordered = sorted(values)
    return {
        name: ordered[round((len(ordered) - 1) * fraction)]
        for name, fraction in (("p10", .1), ("p50", .5), ("p90", .9), ("p99", .99))
    }


def _stable_order(record: dict) -> str:
    return hashlib.sha256(str(record["target_id"]).encode()).hexdigest()


def _select_e0_sample(records: list[dict]) -> tuple[list[dict], dict]:
    selected = []
    selected_ids = set()

    def add_group(predicate, limit: int) -> int:
        added = 0
        for record in sorted(filter(predicate, records), key=_stable_order):
            if record["target_id"] in selected_ids:
                continue
            selected.append(record)
            selected_ids.add(record["target_id"])
            added += 1
            if added >= limit:
                break
        return added

    coverage = {
        "multi_component": add_group(lambda row: row["metrics"]["valid_components"] >= 2, 100),
        "global_obb_inflation_ge_2": add_group(
            lambda row: row["metrics"].get("global_obb_area_ratio", 0) >= 2, 50
        ),
        "single_component": add_group(lambda row: row["metrics"]["valid_components"] == 1, 50),
    }
    return selected, coverage


def run_e0(index_path: Path, state_path: Path, mask_dir: Path) -> dict:
    state = json.loads(state_path.read_text(encoding="utf-8"))
    records = []
    counts = Counter()
    encodings = Counter()
    numeric: dict[str, list[float]] = {}
    with index_path.open("r", encoding="utf-8") as stream:
        for line in stream:
            target = json.loads(line)
            target_id = str(target.get("target_id") or "")
            if state.get(target_id, {}).get("label") not in POSITIVE_LABELS:
                continue
            counts["positive_targets"] += 1
            mask_path = mask_dir / str(target.get("mask_file") or "")
            signature = mask_path.read_bytes()[:8] if mask_path.exists() else b""
            encoding = "jpeg" if signature.startswith(b"\xff\xd8") else "png" if signature.startswith(b"\x89PNG") else "other"
            encodings[encoding] += 1
            mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
            if mask is None:
                counts["unreadable_masks"] += 1
                continue
            metrics = analyze_parking_mask(mask)
            counts["analyzed"] += 1
            counts["empty_valid_masks"] += metrics["empty_valid_mask"]
            counts["multi_component"] += metrics["valid_components"] >= 2
            counts["global_obb_inflation_ge_2"] += metrics.get("global_obb_area_ratio", 0) >= 2
            records.append({
                "target_id": target_id,
                "image": target.get("image"),
                "mask_file": target.get("mask_file"),
                "metrics": metrics,
            })
            for key, value in metrics.items():
                if isinstance(value, (int, float)) and not isinstance(value, bool):
                    numeric.setdefault(key, []).append(float(value))
    sample, coverage = _select_e0_sample(records)
    return {
        "schema_version": 1,
        "task": "parking representation E0; no model training",
        "semantic_limit": "geometry can measure topology loss, not decide parking-area business meaning",
        "positive_labels": sorted(POSITIVE_LABELS),
        "minimum_component_area_px": MIN_COMPONENT_AREA,
        "counts": dict(counts),
        "content_encodings": dict(encodings),
        "quantiles": {key: _quantiles(values) for key, values in numeric.items()},
        "sample": {
            "method": "deterministic: 100 multi-component, 50 additional >=2x global OBB, 50 single-component",
            "coverage": coverage,
            "size": len(sample),
            "targets": sample,
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Parking representation E0 audit")
    parser.add_argument("--index", type=Path, required=True)
    parser.add_argument("--state", type=Path, required=True)
    parser.add_argument("--masks", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = run_e0(args.index, args.state, args.masks)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"output": str(args.output), **result["counts"], "sample": result["sample"]["coverage"]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
