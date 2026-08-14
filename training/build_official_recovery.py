#!/usr/bin/env python3
"""Build a low-cost recovery dataset from the preserved official PKLot labels."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
from collections import Counter
from datetime import datetime
from pathlib import Path

from console.core import safe_write_json


SOURCE_CLASS_MAP = {1: 1, 2: 0}  # space-empty -> empty, space-occupied -> occupied


def _rank(name: str) -> str:
    return hashlib.sha256(("official-pklot-v1:" + name).encode("utf-8")).hexdigest()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _obb_line(raw: str) -> tuple[str, int]:
    values = raw.split()
    if len(values) != 5:
        raise ValueError(f"invalid YOLO box: {raw}")
    source_class = int(values[0])
    if source_class not in SOURCE_CLASS_MAP:
        raise ValueError(f"unsupported source class: {source_class}")
    cx, cy, width, height = (float(value) for value in values[1:])
    x1, y1 = cx - width / 2, cy - height / 2
    x2, y2 = cx + width / 2, cy + height / 2
    coords = (x1, y1, x2, y1, x2, y2, x1, y2)
    if any(value < -1e-6 or value > 1 + 1e-6 for value in coords) or width <= 0 or height <= 0:
        raise ValueError(f"invalid geometry: {raw}")
    target_class = SOURCE_CLASS_MAP[source_class]
    return f"{target_class} " + " ".join(f"{max(0, min(value, 1)):.6f}" for value in coords), target_class


def build(source_dir: str, output_dir: str, *, test_count: int = 200, val_count: int = 150) -> dict:
    source = Path(source_dir).resolve()
    output = Path(output_dir).resolve()
    image_dir = source / "images"
    label_dir = source / "labels"
    if output.exists():
        raise FileExistsError(f"recovery candidate already exists: {output}")
    records = []
    for label in sorted(label_dir.glob("*.txt")):
        image = image_dir / f"{label.stem}.jpg"
        if image.is_file():
            records.append((image, label))
    if len(records) < test_count + val_count + 1:
        raise RuntimeError(f"not enough source samples: {len(records)}")
    records.sort(key=lambda pair: _rank(pair[0].name))
    groups = {}
    for record in records:
        day = record[0].name[:10]
        groups.setdefault(day, []).append(record)
    ranked_days = sorted(groups, key=_rank)
    test_days, val_days = set(), set()
    running = 0
    for day in ranked_days:
        if running < test_count:
            test_days.add(day)
            running += len(groups[day])
    running = 0
    for day in ranked_days:
        if day in test_days:
            continue
        if running < val_count:
            val_days.add(day)
            running += len(groups[day])
    assignment = {}
    for image, _ in records:
        day = image.name[:10]
        assignment[image.name] = "test" if day in test_days else "val" if day in val_days else "train"

    staging = output.with_name(output.name + f".staging.{os.getpid()}")
    class_counts = Counter()
    split_counts = Counter()
    source_hashes = {}
    try:
        for split in ("train", "val", "test"):
            (staging / "images" / split).mkdir(parents=True)
            (staging / "labels" / split).mkdir(parents=True)
        for image, label in records:
            split = assignment[image.name]
            target_image = staging / "images" / split / image.name
            try:
                os.link(image, target_image)
            except OSError:
                shutil.copy2(image, target_image)
            converted = []
            for raw in label.read_text(encoding="utf-8").splitlines():
                if not raw.strip():
                    continue
                line, class_id = _obb_line(raw)
                converted.append(line)
                class_counts[class_id] += 1
            (staging / "labels" / split / label.name).write_text("\n".join(converted) + "\n", encoding="utf-8")
            split_counts[split] += 1
            source_hashes[label.name] = _sha256(label)
        (staging / "data.yaml").write_text(
            f"path: {output}\ntrain: images/train\nval: images/val\ntest: images/test\n"
            "names:\n  0: occupied\n  1: empty\n",
            encoding="utf-8",
        )
        fixed = sorted(name for name, split in assignment.items() if split == "test")
        (staging / "fixed_test_list.txt").write_text("\n".join(fixed) + "\n", encoding="utf-8")
        provenance = {
            "schema_version": 1,
            "created_at": datetime.now().isoformat(),
            "source": str(source),
            "source_dataset": "PKLot Roboflow export",
            "source_license": "CC BY 4.0",
            "conversion": "YOLO xywh AABB to four-corner OBB",
            "class_map": {"1 space-empty": "1 empty", "2 space-occupied": "0 occupied"},
            "split_method": "capture-day grouped; day groups ranked by sha256 official-pklot-v1",
            "split_groups": {
                "test_days": sorted(test_days),
                "val_days": sorted(val_days),
                "train_days": sorted(set(groups) - test_days - val_days),
            },
            "split_counts": dict(split_counts),
            "class_counts": {"occupied": class_counts[0], "empty": class_counts[1]},
            "source_label_sha256": source_hashes,
        }
        safe_write_json(str(staging / "source_provenance.json"), provenance)
        safe_write_json(str(staging / "export_meta.json"), {
            "schema_version": 1,
            "split_mode": "fixed_test_list",
            "fixed_test_list": str(output / "fixed_test_list.txt"),
            "review_provenance": {
                "reviewed": sum(class_counts.values()),
                "human": sum(class_counts.values()),
                "automated": 0,
                "source": "published PKLot ground-truth annotations",
            },
            "source_provenance": str(output / "source_provenance.json"),
        })
        os.replace(staging, output)
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    return {"output_dir": str(output), **provenance}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", default=r"H:\pklotdataset\test")
    parser.add_argument("--output", default="quality/pklot_official_recovery_v1")
    parser.add_argument("--test-count", type=int, default=200)
    parser.add_argument("--val-count", type=int, default=150)
    args = parser.parse_args()
    print(json.dumps(build(args.source, args.output, test_count=args.test_count, val_count=args.val_count), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
