from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path

import cv2
import numpy as np


VEHICLE_CLASSES = {"car", "bus", "truck"}


def _bbox_iou(first: list[float], second: list[float]) -> float:
    left, top = max(first[0], second[0]), max(first[1], second[1])
    right, bottom = min(first[2], second[2]), min(first[3], second[3])
    intersection = max(0.0, right - left) * max(0.0, bottom - top)
    first_area = max(0.0, first[2] - first[0]) * max(0.0, first[3] - first[1])
    second_area = max(0.0, second[2] - second[0]) * max(0.0, second[3] - second[1])
    union = first_area + second_area - intersection
    return intersection / union if union else 0.0


def compare_target_boxes(targets: list[list[float]], detections: list[list[float]]) -> dict:
    overlaps = [[_bbox_iou(target, detection) for detection in detections] for target in targets]
    target_best = [max(row, default=0.0) for row in overlaps]
    detection_matches = [
        sum(overlaps[target_index][detection_index] >= 0.5 for target_index in range(len(targets)))
        for detection_index in range(len(detections))
    ]
    detection_best = [
        max((overlaps[target_index][detection_index] for target_index in range(len(targets))), default=0.0)
        for detection_index in range(len(detections))
    ]
    return {
        "target_best_iou": target_best,
        "targets_matched_at_0_3": sum(value >= 0.3 for value in target_best),
        "targets_matched_at_0_5": sum(value >= 0.5 for value in target_best),
        "detections_unmatched_at_0_3": sum(value < 0.3 for value in detection_best),
        "detections_with_multiple_targets_at_0_5": sum(value > 1 for value in detection_matches),
    }


def bbox_from_mask_array(mask: np.ndarray) -> list[float] | None:
    grayscale = mask.max(axis=2) if mask.ndim == 3 else mask
    foreground = grayscale >= 128
    ys, xs = foreground.nonzero()
    if not len(xs):
        return None
    return [float(xs.min()), float(ys.min()), float(xs.max() + 1), float(ys.max() + 1)]


def _mask_bbox(mask_path: Path) -> list[float] | None:
    mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
    return None if mask is None else bbox_from_mask_array(mask)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _quantiles(values: list[float]) -> dict[str, float | None]:
    if not values:
        return {"p10": None, "p50": None, "p90": None}
    ordered = sorted(values)
    return {
        name: ordered[round((len(ordered) - 1) * fraction)]
        for name, fraction in (("p10", .1), ("p50", .5), ("p90", .9))
    }


def pair_predictions_with_images(image_names, predictions):
    """Ultralytics assigns synthetic paths to list inputs, so preserve input order."""
    return zip(image_names, predictions, strict=True)


def run_teacher_agreement(
    manifest_path: Path,
    image_dir: Path,
    mask_dir: Path,
    weights_path: Path,
    image_size: int = 640,
) -> dict:
    from ultralytics import YOLO

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    targets_by_image: dict[str, list[dict]] = defaultdict(list)
    unreadable_masks = 0
    for target in manifest["targets"]:
        mask_bbox = _mask_bbox(mask_dir / str(target["mask_file"]))
        if mask_bbox is None:
            unreadable_masks += 1
            continue
        targets_by_image[str(target["image"])].append({**target, "mask_bbox": mask_bbox})

    model = YOLO(str(weights_path))
    image_names = sorted(targets_by_image)
    image_paths = [str(image_dir / image) for image in image_names]
    predictions = model.predict(
        source=image_paths,
        imgsz=image_size,
        conf=0.05,
        iou=0.7,
        batch=16,
        device=0,
        half=True,
        verbose=False,
        stream=True,
    )
    totals = Counter()
    class_counts = Counter()
    strata_matches: dict[str, Counter] = defaultdict(Counter)
    target_results = []
    recorded_ious: list[float] = []
    mask_ious: list[float] = []
    for image, prediction in pair_predictions_with_images(image_names, predictions):
        targets = targets_by_image[image]
        detections = []
        for box in prediction.boxes:
            class_name = str(prediction.names[int(box.cls.item())])
            if class_name not in VEHICLE_CLASSES:
                continue
            detections.append(list(map(float, box.xyxy[0].tolist())))
            class_counts[class_name] += 1
        recorded = compare_target_boxes([target["bbox"] for target in targets], detections)
        masks = compare_target_boxes([target["mask_bbox"] for target in targets], detections)
        totals["images"] += 1
        totals["detections"] += len(detections)
        totals["images_without_vehicle_detection"] += not detections
        for index, target in enumerate(targets):
            recorded_iou = recorded["target_best_iou"][index]
            mask_iou = masks["target_best_iou"][index]
            recorded_ious.append(recorded_iou)
            mask_ious.append(mask_iou)
            totals["targets"] += 1
            totals["targets_mask_matched_at_0_3"] += mask_iou >= 0.3
            totals["targets_mask_matched_at_0_5"] += mask_iou >= 0.5
            for dimension in ("size", "confidence", "review", "prompt_hits"):
                key = f"{dimension}={target['audit_strata'][dimension]}"
                strata_matches[key]["targets"] += 1
                strata_matches[key]["matched_at_0_3"] += mask_iou >= 0.3
            target_results.append({
                "target_id": target["target_id"],
                "recorded_bbox_best_iou": recorded_iou,
                "mask_bbox_best_iou": mask_iou,
            })
    return {
        "schema_version": 1,
        "agreement_only_not_ground_truth": True,
        "scope_limit": "Only sampled SAM targets were matched; detector recall and unmatched detections are not measured.",
        "vehicle_classes": sorted(VEHICLE_CLASSES),
        "weights": {"path": str(weights_path), "sha256": _sha256(weights_path)},
        "inference": {"imgsz": image_size, "conf": 0.05, "iou": 0.7, "fp16": True},
        "unreadable_or_empty_masks": unreadable_masks,
        "totals": dict(totals),
        "detection_classes": dict(class_counts),
        "recorded_bbox_best_iou": _quantiles(recorded_ious),
        "mask_bbox_best_iou": _quantiles(mask_ious),
        "strata": {key: dict(value) for key, value in strata_matches.items()},
        "targets": target_results,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare SAM3 candidates with an independent detector")
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--images", type=Path, required=True)
    parser.add_argument("--masks", type=Path, required=True)
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--imgsz", type=int, default=640)
    args = parser.parse_args()
    result = run_teacher_agreement(args.manifest, args.images, args.masks, args.weights, args.imgsz)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"output": str(args.output), **result["totals"]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
