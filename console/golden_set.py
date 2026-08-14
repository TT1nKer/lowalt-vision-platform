#!/usr/bin/env python3
"""Build a reviewable, unsigned golden-set candidate from a sanitized dataset."""

from __future__ import annotations

import csv
import hashlib
import html
import json
import os
import shutil
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
import yaml

from console.core import safe_load_json, safe_write_json


def _label_rows(path: Path) -> list[tuple[int, list[float]]]:
    rows = []
    for raw in path.read_text(encoding="utf-8").splitlines():
        values = raw.split()
        if len(values) != 9:
            continue
        try:
            rows.append((int(values[0]), [float(value) for value in values[1:]]))
        except ValueError:
            continue
    return rows


def _image_for_stem(image_dir: Path, stem: str) -> Path | None:
    for ext in (".jpg", ".jpeg", ".png", ".webp", ".bmp"):
        path = image_dir / (stem + ext)
        if path.is_file():
            return path
    return None


def _signature(rows: list[tuple[int, list[float]]], class_names: dict[int, str]) -> str:
    counts = Counter(class_names.get(class_id, str(class_id)) for class_id, _ in rows)
    return "+".join(sorted(counts)) or "background"


def _density_bucket(count: int) -> str:
    if count <= 20:
        return "sparse"
    if count <= 60:
        return "medium"
    if count <= 100:
        return "dense"
    return "very_dense"


def _stable_rank(name: str) -> str:
    return hashlib.sha256(("golden-v1:" + name).encode("utf-8")).hexdigest()


def _select(records: list[dict], count: int, risk_stems: set[str]) -> list[dict]:
    for record in records:
        record["risk"] = record["stem"] in risk_stems
        record["stratum"] = f"{record['density']}|{record['signature']}"
    selected = []
    selected_names = set()
    # Include all known risky test images first, then one deterministic sample per stratum.
    for record in sorted((r for r in records if r["risk"]), key=lambda r: _stable_rank(r["name"])):
        if len(selected) >= count:
            break
        selected.append(record)
        selected_names.add(record["name"])
    groups = defaultdict(list)
    for record in records:
        if record["name"] not in selected_names:
            groups[record["stratum"]].append(record)
    for group in groups.values():
        group.sort(key=lambda r: _stable_rank(r["name"]))
    while len(selected) < count:
        added = False
        for key in sorted(groups):
            group = groups[key]
            if group:
                record = group.pop(0)
                selected.append(record)
                selected_names.add(record["name"])
                added = True
                if len(selected) >= count:
                    break
        if not added:
            break
    return selected


def _render(image_path: Path, rows: list[tuple[int, list[float]]], names: dict[int, str], output: Path) -> None:
    image = cv2.imread(str(image_path))
    if image is None:
        raise RuntimeError(f"无法读取图片: {image_path}")
    height, width = image.shape[:2]
    colors = {0: (255, 90, 40), 1: (40, 190, 255), 2: (70, 210, 110)}
    for class_id, coords in rows:
        points = np.array([
            [round(coords[index] * width), round(coords[index + 1] * height)]
            for index in range(0, 8, 2)
        ], dtype=np.int32)
        color = colors.get(class_id, (220, 220, 220))
        cv2.polylines(image, [points], True, color, 2, cv2.LINE_AA)
        anchor = tuple(points[np.argmin(points[:, 1])])
        cv2.putText(image, names.get(class_id, str(class_id)), anchor,
                    cv2.FONT_HERSHEY_SIMPLEX, 0.42, color, 1, cv2.LINE_AA)
    cv2.imwrite(str(output), image, [cv2.IMWRITE_JPEG_QUALITY, 90])


def build_golden_candidate(dataset_dir: str, output_dir: str, count: int = 300) -> dict:
    dataset = Path(dataset_dir).resolve()
    output = Path(output_dir).resolve()
    if output.exists():
        raise FileExistsError(f"金标候选目录已存在，为保护审核记录不覆盖: {output}")
    data = yaml.safe_load((dataset / "data.yaml").read_text(encoding="utf-8")) or {}
    raw_names = data.get("names") or {}
    names = ({int(key): str(value) for key, value in raw_names.items()}
             if isinstance(raw_names, dict)
             else {index: str(value) for index, value in enumerate(raw_names)})
    image_dir = dataset / "images" / "test"
    label_dir = dataset / "labels" / "test"
    records = []
    for label in sorted(label_dir.glob("*.txt")):
        image = _image_for_stem(image_dir, label.stem)
        if image is None:
            continue
        rows = _label_rows(label)
        records.append({
            "name": image.name, "stem": label.stem, "image": image, "label": label,
            "rows": rows, "objects": len(rows), "signature": _signature(rows, names),
            "density": _density_bucket(len(rows)),
        })
    queue = safe_load_json(str(dataset / "quality_review_queue.json"), {})
    risk_stems = {item.get("image_stem") for item in queue.get("items", []) if item.get("image_stem")}
    selected = _select(records, min(count, len(records)), risk_stems)
    staging = output.with_name(output.name + f".staging.{os.getpid()}")
    preview_dir = staging / "previews"
    image_out = staging / "images"
    label_out = staging / "labels"
    preview_dir.mkdir(parents=True)
    image_out.mkdir()
    label_out.mkdir()
    review_rows = []
    for index, record in enumerate(selected, 1):
        shutil.copy2(record["image"], image_out / record["name"])
        shutil.copy2(record["label"], label_out / record["label"].name)
        preview_name = record["stem"] + ".jpg"
        _render(record["image"], record["rows"], names, preview_dir / preview_name)
        review_rows.append({
            "index": index, "image": record["name"], "objects": record["objects"],
            "stratum": record["stratum"], "risk_sample": "yes" if record["risk"] else "no",
            "review_status": "pending", "reviewer": "", "reviewed_at": "", "notes": "",
        })
    with open(staging / "review.csv", "w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=review_rows[0].keys())
        writer.writeheader()
        writer.writerows(review_rows)
    (staging / "fixed_test_list.txt").write_text(
        "\n".join(record["name"] for record in selected) + "\n", encoding="utf-8"
    )
    cards = "\n".join(
        f'<article><img src="previews/{html.escape(record["stem"])}.jpg" loading="lazy">'
        f'<div><b>{index}. {html.escape(record["name"])}</b>'
        f'<span>{record["objects"]} objects | {html.escape(record["stratum"])}'
        f'{" | RISK" if record["risk"] else ""}</span></div></article>'
        for index, record in enumerate(selected, 1)
    )
    (staging / "index.html").write_text(f"""<!doctype html><meta charset="utf-8"><title>金标测试集复核包</title>
<style>body{{font:13px Arial;margin:24px;background:#f4f6f9;color:#25324a}}header{{margin-bottom:18px}}main{{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:14px}}article{{background:#fff;border:1px solid #dfe5ee;padding:10px}}img{{display:block;width:100%;height:420px;object-fit:contain;background:#111}}article div{{padding:9px 2px 2px}}b,span{{display:block}}span{{margin-top:4px;color:#68758b;font-size:11px}}@media(max-width:800px){{main{{grid-template-columns:1fr}}}}</style>
<header><h1>金标测试集复核包</h1><p>逐图检查漏框、错框、类别和边界。结果填写 review.csv；全部通过后由有权限人员签署 manifest。</p></header><main>{cards}</main>""", encoding="utf-8")
    manifest = {
        "schema_version": 1, "approved": False, "reviewer": "", "approver": "",
        "approved_at": "", "annotation_standard": "parking_obb_v1",
        "classes": [names[key] for key in sorted(names)],
        "dataset_source": str(dataset), "selection_method": "stratified_density_class_risk_v1",
        "created_at": datetime.now().isoformat(), "images": [record["name"] for record in selected],
        "review_csv": "review.csv", "notes": "不得在未逐图复核前设置 approved=true",
    }
    safe_write_json(str(staging / "golden_test_manifest.json"), manifest)
    summary = {
        "created_at": datetime.now().isoformat(), "dataset": str(dataset), "count": len(selected),
        "risk_samples": sum(record["risk"] for record in selected),
        "density": dict(Counter(record["density"] for record in selected)),
        "signatures": dict(Counter(record["signature"] for record in selected)),
    }
    safe_write_json(str(staging / "selection_summary.json"), summary)
    os.replace(staging, output)
    return {"output_dir": str(output), **summary}


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Build an unsigned golden test review package.")
    parser.add_argument("dataset_dir")
    parser.add_argument("output_dir")
    parser.add_argument("--count", type=int, default=300)
    args = parser.parse_args()
    if args.count <= 0:
        parser.error("--count must be greater than zero")
    print(json.dumps(
        build_golden_candidate(args.dataset_dir, args.output_dir, count=args.count),
        ensure_ascii=False,
        indent=2,
    ))
