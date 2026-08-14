#!/usr/bin/env python3
"""Calibrate a candidate against stable parking-space topology from nearby labeled frames."""
from __future__ import annotations

import argparse
import json
import re
from datetime import date, datetime, time
from pathlib import Path

import numpy as np
from ultralytics import YOLO

from console.core import safe_write_json


STAMP = re.compile(r"(\d{4}-\d{2}-\d{2})_(\d{2})_(\d{2})_(\d{2})")


def _stamp(name: str):
    match = STAMP.match(name)
    if not match: return None
    return match.group(1), datetime.combine(date.min, time(*map(int, match.groups()[1:])))


def _references(label_dir: Path) -> dict[str, list[tuple[datetime, Path]]]:
    result = {}
    for path in label_dir.glob("*.txt"):
        stamp = _stamp(path.name)
        if stamp: result.setdefault(stamp[0], []).append((stamp[1], path))
    return result


def _nearest(name: str, references):
    stamp = _stamp(name)
    if not stamp or stamp[0] not in references: return None, None
    seconds, path = min((abs((stamp[1] - moment).total_seconds()), path) for moment, path in references[stamp[0]])
    return path, seconds / 60


def _boxes(path: Path, width: int, height: int) -> tuple[np.ndarray, np.ndarray]:
    boxes = []
    classes = []
    for line in path.read_text(encoding="utf-8").splitlines():
        values = line.split()
        if len(values) != 5 or int(values[0]) not in {1, 2}: continue
        cx, cy, w, h = map(float, values[1:])
        boxes.append(((cx-w/2)*width, (cy-h/2)*height, (cx+w/2)*width, (cy+h/2)*height))
        classes.append(0 if int(values[0]) == 2 else 1)
    return np.asarray(boxes, dtype=np.float32), np.asarray(classes, dtype=np.int32)


def _iou(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    if not len(a) or not len(b): return np.zeros((len(a), len(b)), dtype=np.float32)
    tl = np.maximum(a[:, None, :2], b[None, :, :2]); br = np.minimum(a[:, None, 2:], b[None, :, 2:])
    inter = np.maximum(br - tl, 0).prod(2)
    aa = np.maximum(a[:, 2:] - a[:, :2], 0).prod(1)[:, None]
    bb = np.maximum(b[:, 2:] - b[:, :2], 0).prod(1)[None, :]
    return inter / np.maximum(aa + bb - inter, 1e-9)


def _match(pred: np.ndarray, truth: np.ndarray, threshold: float = .30):
    matrix = _iou(pred, truth); matched_p, matched_t = set(), set()
    while matrix.size:
        p, t = np.unravel_index(np.argmax(matrix), matrix.shape)
        if matrix[p, t] < threshold: break
        matched_p.add(int(p)); matched_t.add(int(t)); matrix[p, :] = -1; matrix[:, t] = -1
    return len(matched_p), len(pred)-len(matched_p), len(truth)-len(matched_t)


def _class_match(pred: np.ndarray, classes: np.ndarray, truth: np.ndarray, truth_classes: np.ndarray, threshold: float = .30):
    metrics = {}
    for class_id, name in ((0, "occupied"), (1, "empty")):
        metrics[name] = dict(zip(("tp", "fp", "fn"), _match(pred[classes == class_id], truth[truth_classes == class_id], threshold)))
    return metrics


def calibrate(project_dir: str, model_path: str, source_dir: str, label_dir: str) -> dict:
    root = Path(project_dir).resolve(); shadow = root / "quality/business_shadow_v1"
    model_path = Path(model_path)
    if not model_path.is_absolute(): model_path = root / model_path
    source_dir = Path(source_dir)
    if not source_dir.is_absolute(): source_dir = root / source_dir
    label_dir = Path(label_dir)
    if not label_dir.is_absolute(): label_dir = root / label_dir
    records = json.loads((shadow / "shadow_report.json").read_text(encoding="utf-8"))["records"][:30]
    refs, model = _references(label_dir), YOLO(str(model_path))
    thresholds = [.25, .35, .45, .55, .65, .75]
    totals = {str(c): {"tp": 0, "fp": 0, "fn": 0, "classes": {name: {"tp": 0, "fp": 0, "fn": 0} for name in ("occupied", "empty")}} for c in thresholds}; samples = []
    for index, record in enumerate(records, 1):
        ref, delta = _nearest(record["image"], refs)
        if ref is None: continue
        image = source_dir / record["image"]
        result = model.predict(str(image), imgsz=640, conf=min(thresholds), device=0, verbose=False)[0]
        truth, truth_classes = _boxes(ref, result.orig_shape[1], result.orig_shape[0])
        pred_boxes = result.obb.xyxy.cpu().numpy() if result.obb is not None else np.empty((0, 4))
        confidence = result.obb.conf.cpu().numpy() if result.obb is not None else np.empty((0,))
        pred_classes = result.obb.cls.cpu().numpy().astype(np.int32) if result.obb is not None else np.empty((0,), dtype=np.int32)
        row = {"image": record["image"], "reference": ref.name, "reference_minutes": round(delta, 1), "spaces": len(truth), "thresholds": {}}
        for conf in thresholds:
            tp, fp, fn = _match(pred_boxes[confidence >= conf], truth)
            row["thresholds"][str(conf)] = {"tp": tp, "fp": fp, "fn": fn}
            for key, value in zip(("tp", "fp", "fn"), (tp, fp, fn)): totals[str(conf)][key] += value
            for name, values in _class_match(pred_boxes[confidence >= conf], pred_classes[confidence >= conf], truth, truth_classes).items():
                for key, value in values.items(): totals[str(conf)]["classes"][name][key] += value
        samples.append(row); print(f"@@PROGRESS {index} {len(records)} {record['image']}", flush=True)
    for conf, values in totals.items():
        tp, fp, fn = values["tp"], values["fp"], values["fn"]
        values.update({"precision": tp/(tp+fp) if tp+fp else 0, "recall": tp/(tp+fn) if tp+fn else 0,
                       "f1": 2*tp/(2*tp+fp+fn) if 2*tp+fp+fn else 0})
        for metrics in values["classes"].values():
            tp, fp, fn = metrics["tp"], metrics["fp"], metrics["fn"]
            metrics.update({"precision": tp/(tp+fp) if tp+fp else 0, "recall": tp/(tp+fn) if tp+fn else 0,
                            "f1": 2*tp/(2*tp+fp+fn) if 2*tp+fp+fn else 0})
    best = max(totals, key=lambda key: totals[key]["f1"])
    report = {"schema_version": 1, "status": "diagnostic_topology_calibration",
              "model": str(model_path.resolve()), "evaluated_images": len(samples),
              "iou_threshold": .30, "thresholds": totals, "recommended_confidence": float(best),
              "recommended_metrics": totals[best], "samples": samples,
              "limitations": ["Validates stable parking-space geometry, not occupied/empty class correctness.",
                              "Nearby-frame topology is excluded when no same-day official label exists."]}
    safe_write_json(str(shadow / "topology_calibration.json"), report); return report


if __name__ == "__main__":
    p=argparse.ArgumentParser();p.add_argument("--project-dir",default=".");p.add_argument("--model",default="quality/official_finetune/finetune_official_v1/weights/best.pt");p.add_argument("--source",default=r"H:\pklotdataset\train\images");p.add_argument("--labels",default=r"H:\pklotdataset\test\labels")
    a=p.parse_args();print(json.dumps(calibrate(a.project_dir,a.model,a.source,a.labels),ensure_ascii=False,indent=2))
