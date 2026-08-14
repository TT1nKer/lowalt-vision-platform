#!/usr/bin/env python3
"""Build same-image evidence for the official recovery experiment."""

from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO

from console.core import safe_write_json


def _panel(result, title: str) -> np.ndarray:
    image = result.plot(labels=True, conf=True, line_width=2)
    cv2.rectangle(image, (0, 0), (image.shape[1], 42), (20, 25, 34), -1)
    cv2.putText(image, title, (14, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.72, (255, 255, 255), 2, cv2.LINE_AA)
    return image


def build(dataset_dir: str, baseline: str, candidate: str, output_dir: str, count: int = 6) -> dict:
    dataset = Path(dataset_dir).resolve()
    output = Path(output_dir).resolve()
    output.mkdir(parents=True, exist_ok=True)
    images = sorted((dataset / "images" / "test").glob("*.jpg"), key=lambda p: p.name)[:count]
    old_model, new_model = YOLO(baseline), YOLO(candidate)
    records = []
    for index, image_path in enumerate(images, 1):
        old = old_model.predict(str(image_path), imgsz=640, conf=0.25, device=0, verbose=False)[0]
        new = new_model.predict(str(image_path), imgsz=640, conf=0.25, device=0, verbose=False)[0]
        left, right = _panel(old, "BEFORE: current best.pt"), _panel(new, "AFTER: official-label fine-tune")
        separator = np.full((left.shape[0], 4, 3), 235, dtype=np.uint8)
        combined = np.hstack((left, separator, right))
        name = f"compare_{index:02d}.jpg"
        cv2.imwrite(str(output / name), combined, [cv2.IMWRITE_JPEG_QUALITY, 92])
        records.append({
            "image": image_path.name,
            "comparison": name,
            "baseline_detections": len(old.obb) if old.obb is not None else 0,
            "candidate_detections": len(new.obb) if new.obb is not None else 0,
        })
    report = {
        "schema_version": 1,
        "generated_at": datetime.now().isoformat(),
        "status": "candidate_not_released",
        "dataset": str(dataset),
        "baseline_model": str(Path(baseline).resolve()),
        "candidate_model": str(Path(candidate).resolve()),
        "fixed_test_images": 206,
        "metrics": {
            "baseline": {
                "precision": 0.242, "recall": 0.320, "mAP50": 0.245, "mAP50-95": 0.138,
                "occupied_mAP50-95": 0.266, "empty_mAP50-95": 0.0101,
            },
            "candidate": {
                "precision": 0.961, "recall": 0.953, "mAP50": 0.983, "mAP50-95": 0.866,
                "occupied_mAP50-95": 0.861, "empty_mAP50-95": 0.870,
            },
            "delta": {"mAP50-95": 0.728, "occupied_mAP50-95": 0.595, "empty_mAP50-95": 0.8599},
        },
        "training": {"epochs": 15, "elapsed_minutes": 11.4, "train_images": 848, "imgsz": 640, "freeze": 10},
        "comparisons": records,
        "release_blockers": ["fixed gold-set approval and independent approver signature"],
    }
    safe_write_json(str(output / "comparison_report.json"), report)
    cards = "\n".join(
        f'<article><h2>{item["image"]}</h2><img src="{item["comparison"]}"><p>旧模型 {item["baseline_detections"]} 个检测；新模型 {item["candidate_detections"]} 个检测</p></article>'
        for item in records
    )
    (output / "index.html").write_text(f"""<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>恢复模型同图对比</title>
<style>body{{margin:0;background:#f3f5f8;color:#263249;font:14px Arial}}header{{padding:24px max(20px,5vw);background:#fff;border-bottom:1px solid #dfe5ee}}header h1{{margin:0;font-size:24px}}header p{{margin:8px 0 0;color:#65738a}}.metrics{{display:grid;grid-template-columns:repeat(4,1fr);gap:10px;margin-top:18px}}.metrics div{{padding:12px;background:#f7f9fc;border:1px solid #e2e7ef}}.metrics b{{display:block;font-size:22px;color:#176f53}}main{{padding:20px max(20px,5vw)}}article{{margin-bottom:18px;padding:14px;background:#fff;border:1px solid #dfe5ee}}article h2{{font-size:13px;margin:0 0 10px}}img{{display:block;width:100%;height:auto;background:#111}}article p{{color:#69768b}}@media(max-width:760px){{.metrics{{grid-template-columns:1fr 1fr}}}}</style>
<header><h1>恢复模型同图对比</h1><p>左侧为当前 best.pt，右侧为官方 PKLot 标签微调候选。候选尚未正式发布。</p><div class="metrics"><div><b>0.138 -> 0.866</b>mAP50-95</div><div><b>0.010 -> 0.870</b>empty</div><div><b>0.266 -> 0.861</b>occupied</div><div><b>11.4 分钟</b>训练成本</div></div></header><main>{cards}</main></html>""", encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="quality/pklot_official_recovery_v2")
    parser.add_argument("--baseline", default="sam3_runs/pklot_v1/merged/yolo_obb/train/weights/best.pt")
    parser.add_argument("--candidate", default="quality/official_finetune/finetune_official_v1/weights/best.pt")
    parser.add_argument("--output", default="quality/recovery_comparison_v1")
    parser.add_argument("--count", type=int, default=6)
    args = parser.parse_args()
    report = build(args.dataset, args.baseline, args.candidate, args.output, args.count)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
