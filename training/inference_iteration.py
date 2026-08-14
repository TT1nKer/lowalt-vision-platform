#!/usr/bin/env python3
"""Render controlled inference variants for visual/Gemma iteration."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO

from console.business_shadow_eval import _panel, _stats


def render(model_path: str, image_path: str, output_dir: str) -> dict:
    model, image = YOLO(model_path), Path(image_path).resolve()
    output = Path(output_dir).resolve(); output.mkdir(parents=True, exist_ok=True)
    variants = [(640, .65), (1024, .65), (1024, .45), (1280, .55), (1280, .40)]
    records = []
    original = cv2.imread(str(image))
    for imgsz, confidence in variants:
        result = model.predict(str(image), imgsz=imgsz, conf=confidence, device=0, verbose=False)[0]
        panel = _panel(result, f"imgsz={imgsz} conf={confidence:.2f}", explicit_classes=True)
        name = f"variant_{imgsz}_{int(confidence*100):02d}.jpg"
        cv2.imwrite(str(output / name), panel, [cv2.IMWRITE_JPEG_QUALITY, 92])
        records.append({"imgsz": imgsz, "confidence": confidence, "file": name, **_stats(result)})
    canvas = np.hstack([cv2.resize(cv2.imread(str(output / item["file"])), (640, 640)) for item in records])
    cv2.imwrite(str(output / "variants_all.jpg"), canvas, [cv2.IMWRITE_JPEG_QUALITY, 92])
    report = {"image": image.name, "model": str(Path(model_path).resolve()), "variants": records}
    (output / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return report


if __name__ == "__main__":
    p=argparse.ArgumentParser();p.add_argument("--model",required=True);p.add_argument("--image",required=True);p.add_argument("--output",required=True)
    a=p.parse_args();print(json.dumps(render(a.model,a.image,a.output),ensure_ascii=False,indent=2))
