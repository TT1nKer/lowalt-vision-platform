#!/usr/bin/env python3
"""Run a candidate model in shadow mode on unlabeled business images."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO

from console.core import safe_write_json


def _rank(name: str) -> str:
    return hashlib.sha256(("business-shadow-v1:" + name).encode("utf-8")).hexdigest()


def _sample(source: Path, count: int) -> list[Path]:
    groups = defaultdict(list)
    for path in source.glob("*.jpg"):
        groups[path.name[:10]].append(path)
    for paths in groups.values():
        paths.sort(key=lambda path: _rank(path.name))
    selected = []
    while len(selected) < count:
        added = False
        for day in sorted(groups, key=_rank):
            if groups[day]:
                selected.append(groups[day].pop(0))
                added = True
                if len(selected) >= count:
                    break
        if not added:
            break
    return selected


def _stats(result) -> dict:
    if result.obb is None or len(result.obb) == 0:
        return {"detections": 0, "classes": {}, "mean_confidence": 0.0, "large_boxes": 0}
    classes = result.obb.cls.cpu().numpy().astype(int)
    confidence = result.obb.conf.cpu().numpy()
    points = result.obb.xyxyxyxy.cpu().numpy()
    height, width = result.orig_shape
    image_area = max(height * width, 1)
    large = 0
    for polygon in points:
        area = abs(cv2.contourArea(polygon.astype(np.float32))) / image_area
        large += area > 0.03
    counts = Counter(result.names.get(int(class_id), str(class_id)) for class_id in classes)
    return {
        "detections": int(len(classes)),
        "classes": dict(counts),
        "mean_confidence": round(float(confidence.mean()), 4),
        "large_boxes": int(large),
    }


def _panel(result, title: str, explicit_classes: bool = False) -> np.ndarray:
    if explicit_classes:
        image = result.orig_img.copy()
        if result.obb is not None:
            polygons = result.obb.xyxyxyxy.cpu().numpy()
            classes = result.obb.cls.cpu().numpy().astype(int)
            confidence = result.obb.conf.cpu().numpy()
            for polygon, class_id, score in zip(polygons, classes, confidence):
                name = result.names.get(int(class_id), str(class_id))
                color = (45, 70, 230) if name == "occupied" else (45, 190, 80)
                points = polygon.astype(np.int32).reshape((-1, 1, 2))
                cv2.polylines(image, [points], True, color, 2, cv2.LINE_AA)
                x, y = polygon.astype(np.int32).min(axis=0)
                cv2.putText(image, f"{name[0].upper()} {score:.2f}", (max(0, x), max(52, y - 3)),
                            cv2.FONT_HERSHEY_SIMPLEX, .34, color, 1, cv2.LINE_AA)
    else:
        image = result.plot(labels=False, conf=False, line_width=2)
    cv2.rectangle(image, (0, 0), (image.shape[1], 40), (20, 25, 34), -1)
    cv2.putText(image, title, (12, 27), cv2.FONT_HERSHEY_SIMPLEX, 0.58, (255, 255, 255), 2, cv2.LINE_AA)
    if explicit_classes:
        cv2.putText(image, "RED=occupied", (230, 26), cv2.FONT_HERSHEY_SIMPLEX, .40, (45, 70, 230), 1, cv2.LINE_AA)
        cv2.putText(image, "GREEN=empty", (420, 26), cv2.FONT_HERSHEY_SIMPLEX, .40, (45, 190, 80), 1, cv2.LINE_AA)
    return image


def evaluate(source_dir: str, baseline_path: str, candidate_path: str, output_dir: str, count: int = 60,
             baseline_conf: float = .25, candidate_conf: float = .25) -> dict:
    source, output = Path(source_dir).resolve(), Path(output_dir).resolve()
    output.mkdir(parents=True, exist_ok=True)
    images = _sample(source, count)
    baseline, candidate = YOLO(baseline_path), YOLO(candidate_path)
    records = []
    aggregate = {"baseline": Counter(), "candidate": Counter()}
    for index, image_path in enumerate(images, 1):
        old = baseline.predict(str(image_path), imgsz=640, conf=baseline_conf, device=0, verbose=False)[0]
        new = candidate.predict(str(image_path), imgsz=640, conf=candidate_conf, device=0, verbose=False)[0]
        old_stats, new_stats = _stats(old), _stats(new)
        for key, value in old_stats["classes"].items():
            aggregate["baseline"][key] += value
        for key, value in new_stats["classes"].items():
            aggregate["candidate"][key] += value
        aggregate["baseline"]["detections"] += old_stats["detections"]
        aggregate["candidate"]["detections"] += new_stats["detections"]
        aggregate["baseline"]["large_boxes"] += old_stats["large_boxes"]
        aggregate["candidate"]["large_boxes"] += new_stats["large_boxes"]
        name = f"shadow_{index:03d}.jpg"
        separator = np.full((old.orig_shape[0], 4, 3), 235, dtype=np.uint8)
        combined = np.hstack((_panel(old, "BEFORE"), separator,
                              _panel(new, f"CANDIDATE {candidate_conf:.2f}", explicit_classes=True)))
        cv2.imwrite(str(output / name), combined, [cv2.IMWRITE_JPEG_QUALITY, 90])
        records.append({"image": image_path.name, "comparison": name, "baseline": old_stats, "candidate": new_stats})
        print(f"@@PROGRESS {index} {len(images)} {image_path.name}", flush=True)

    summary = {
        "schema_version": 1, "generated_at": datetime.now().isoformat(),
        "status": "awaiting_human_acceptance", "source_dir": str(source),
        "sample_method": "capture-day stratified deterministic", "sample_count": len(images),
        "baseline_model": str(Path(baseline_path).resolve()),
        "candidate_model": str(Path(candidate_path).resolve()),
        "inference": {"baseline_confidence": baseline_conf, "candidate_confidence": candidate_conf},
        "aggregate": {key: dict(value) for key, value in aggregate.items()},
        "records": records,
        "acceptance_policy": {
            "human_sample_required": 30,
            "min_per_class_precision": 0.90,
            "min_per_class_recall": 0.85,
            "max_critical_false_positive_rate": 0.02,
            "rollback_required": True,
        },
    }
    safe_write_json(str(output / "shadow_report.json"), summary)
    cards = "\n".join(
        f'<article><h2>{item["image"]}</h2><img src="{item["comparison"]}"><p>旧：{item["baseline"]["detections"]} 框 / 大框 {item["baseline"]["large_boxes"]}；候选：{item["candidate"]["detections"]} 框 / 大框 {item["candidate"]["large_boxes"]}</p></article>'
        for item in records
    )
    (output / "index.html").write_text(f"""<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>业务影子验证</title><style>body{{margin:0;background:#f3f5f8;color:#263249;font:14px Arial}}header{{padding:24px 5vw;background:white;border-bottom:1px solid #dde4ed}}header h1{{margin:0}}header p{{color:#647188}}main{{padding:20px 5vw}}article{{padding:14px;margin-bottom:16px;background:white;border:1px solid #dde4ed}}h2{{font-size:13px;margin:0 0 10px}}img{{display:block;width:100%;background:#111}}article p{{color:#68758a}}</style><header><h1>业务影子验证</h1><p>真实业务源图的旧模型/候选模型同图输出。候选尚未发布，需完成 30 张人工验收。</p></header><main>{cards}</main></html>""", encoding="utf-8")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", default=r"H:\pklotdataset\train\images")
    parser.add_argument("--baseline", default="sam3_runs/pklot_v1/merged/yolo_obb/train/weights/best.pt")
    parser.add_argument("--candidate", default="quality/official_finetune/finetune_official_v1/weights/best.pt")
    parser.add_argument("--output", default="quality/business_shadow_v1")
    parser.add_argument("--count", type=int, default=60)
    parser.add_argument("--baseline-conf", type=float, default=.25)
    parser.add_argument("--candidate-conf", type=float, default=.25)
    args = parser.parse_args()
    result = evaluate(args.source, args.baseline, args.candidate, args.output, args.count,
                      args.baseline_conf, args.candidate_conf)
    printable = {key: value for key, value in result.items() if key != "records"}
    print(json.dumps(printable, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
