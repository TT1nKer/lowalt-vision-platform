from __future__ import annotations

import argparse
import json
import math
from collections import Counter
from pathlib import Path

import cv2
import numpy as np


def _bbox_iou(first: list[float], second: list[float]) -> float:
    left, top = max(first[0], second[0]), max(first[1], second[1])
    right, bottom = min(first[2], second[2]), min(first[3], second[3])
    intersection = max(0.0, right - left) * max(0.0, bottom - top)
    first_area = max(0.0, first[2] - first[0]) * max(0.0, first[3] - first[1])
    second_area = max(0.0, second[2] - second[0]) * max(0.0, second[3] - second[1])
    union = first_area + second_area - intersection
    return intersection / union if union else 0.0


def analyze_mask_geometry(mask: np.ndarray, source_bbox: list[float]) -> dict:
    grayscale = mask.max(axis=2) if mask.ndim == 3 else mask
    binary = (grayscale >= 128).astype(np.uint8)
    foreground_y, foreground_x = np.nonzero(binary)
    if not len(foreground_x):
        return {"empty_mask": True}

    mask_area = int(binary.sum())
    component_count, _, component_stats, _ = cv2.connectedComponentsWithStats(binary, 8)
    component_areas = sorted(map(int, component_stats[1:, cv2.CC_STAT_AREA]), reverse=True)
    material_threshold = max(4, math.ceil(mask_area * 0.02))
    material_components = sum(area >= material_threshold for area in component_areas)

    x1, x2 = int(foreground_x.min()), int(foreground_x.max()) + 1
    y1, y2 = int(foreground_y.min()), int(foreground_y.max()) + 1
    mask_bbox = [float(x1), float(y1), float(x2), float(y2)]
    aabb_area = float((x2 - x1) * (y2 - y1))
    points = np.column_stack((foreground_x, foreground_y)).astype(np.float32)
    (_, _), (obb_width, obb_height), obb_angle = cv2.minAreaRect(points)
    obb_area = max(float(obb_width * obb_height), float(mask_area))

    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    perimeter = sum(cv2.arcLength(contour, True) for contour in contours)
    compactness = 4 * math.pi * mask_area / (perimeter * perimeter) if perimeter else 0.0

    return {
        "empty_mask": False,
        "mask_area_px": mask_area,
        "low_intensity_nonzero_px": int(((grayscale > 0) & (grayscale < 128)).sum()),
        "components": component_count - 1,
        "material_components": material_components,
        "largest_component_fraction": component_areas[0] / mask_area,
        "material_component_area_threshold_px": material_threshold,
        "mask_bbox": mask_bbox,
        "mask_bbox_iou": _bbox_iou(mask_bbox, list(map(float, source_bbox))),
        "aabb_area_px": aabb_area,
        "aabb_background_fraction": 1 - mask_area / aabb_area,
        "obb_area_px": obb_area,
        "obb_background_fraction": 1 - mask_area / obb_area,
        "obb_angle_degrees": float(obb_angle),
        "compactness": compactness,
    }


def _quantile(values: list[float], fraction: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    return ordered[round((len(ordered) - 1) * fraction)]


def audit_manifest(manifest_path: Path, mask_dir: Path) -> dict:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    records = manifest["targets"] if isinstance(manifest, dict) else manifest
    results = []
    counts = Counter()
    numeric: dict[str, list[float]] = {}
    encodings = Counter()
    for record in records:
        mask_path = mask_dir / str(record["mask_file"])
        signature = mask_path.read_bytes()[:8] if mask_path.exists() else b""
        encoding = "jpeg" if signature.startswith(b"\xff\xd8") else "png" if signature.startswith(b"\x89PNG") else "other"
        encodings[encoding] += 1
        mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        if mask is None:
            counts["unreadable_masks"] += 1
            continue
        geometry = analyze_mask_geometry(mask, record["bbox"])
        result = {**record, "geometry": geometry}
        results.append(result)
        counts["analyzed"] += 1
        if geometry.get("empty_mask"):
            counts["empty_masks"] += 1
            continue
        if geometry["material_components"] > 1:
            counts["multiple_material_components"] += 1
        if geometry["largest_component_fraction"] < 0.9:
            counts["largest_component_below_90_percent"] += 1
        if geometry["mask_bbox_iou"] < 0.8:
            counts["mask_bbox_iou_below_80_percent"] += 1
        for key, value in geometry.items():
            if isinstance(value, (int, float)) and not isinstance(value, bool):
                numeric.setdefault(key, []).append(float(value))
    quantiles = {
        key: {name: _quantile(values, fraction) for name, fraction in
              (("p10", .1), ("p50", .5), ("p90", .9), ("p99", .99))}
        for key, values in numeric.items()
    }
    return {
        "schema_version": 1,
        "manifest": str(manifest_path),
        "mask_dir": str(mask_dir),
        "counts": dict(counts),
        "content_encodings": dict(encodings),
        "quantiles": quantiles,
        "targets": results,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare vehicle mask, AABB, and OBB geometry")
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--mask-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = audit_manifest(args.manifest, args.mask_dir)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"output": str(args.output), **result["counts"]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
