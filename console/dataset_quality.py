#!/usr/bin/env python3
"""Deterministic quality gates for exported YOLO OBB datasets.

The audit is intentionally independent from training code so it can run in CI,
after export, and immediately before training.  A failed audit is a release
blocker, not a warning hidden in a training log.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import shutil
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Iterable

from console.core import cfg_get, safe_load_json, safe_write_json


SCHEMA_VERSION = 1
SPLITS = ("train", "val", "test")


@dataclass(frozen=True)
class QualityPolicy:
    enforce: bool = True
    require_fixed_test: bool = True
    require_golden_manifest: bool = True
    require_review_provenance: bool = True
    min_human_review_fraction: float = 0.05
    golden_manifest: str = ""
    min_test_images: int = 100
    max_object_area_ratio: float = 0.08
    max_oversized_fraction: float = 0.005
    max_aspect_ratio: float = 8.0
    max_extreme_aspect_fraction: float = 0.01
    max_duplicate_fraction: float = 0.005
    max_class_conflict_fraction: float = 0.002
    max_objects_per_image: int = 120
    geometry_exempt_classes: tuple[str, ...] = ("parking_area",)

    @classmethod
    def from_config(cls, cfg: dict) -> "QualityPolicy":
        values = cfg_get(cfg, "yolo_obb.quality", {}) or {}
        allowed = cls.__dataclass_fields__
        selected = {key: value for key, value in values.items() if key in allowed}
        if "geometry_exempt_classes" in selected:
            selected["geometry_exempt_classes"] = tuple(selected["geometry_exempt_classes"] or ())
        return cls(**selected)


def _polygon_area(points: list[tuple[float, float]]) -> float:
    return abs(sum(
        points[i][0] * points[(i + 1) % len(points)][1]
        - points[(i + 1) % len(points)][0] * points[i][1]
        for i in range(len(points))
    )) / 2.0


def _aspect_ratio(points: list[tuple[float, float]]) -> float:
    if len(points) == 4:
        edges = [math.dist(points[i], points[(i + 1) % 4]) for i in range(4)]
        shortest = min(edges)
        return math.inf if shortest <= 1e-9 else max(edges) / shortest
    min_x, min_y, max_x, max_y = _bounds(points)
    width, height = max_x - min_x, max_y - min_y
    shortest = min(width, height)
    return math.inf if shortest <= 1e-9 else max(width, height) / shortest


def _bounds(points: list[tuple[float, float]]) -> tuple[float, float, float, float]:
    xs, ys = zip(*points)
    return min(xs), min(ys), max(xs), max(ys)


def _bbox_iou(a: tuple[float, ...], b: tuple[float, ...]) -> float:
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    if inter <= 0:
        return 0.0
    area_a = max(0.0, a[2] - a[0]) * max(0.0, a[3] - a[1])
    area_b = max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])
    return inter / max(area_a + area_b - inter, 1e-12)


def _fingerprint(paths: Iterable[Path]) -> str:
    digest = hashlib.sha256()
    for path in sorted(paths, key=lambda item: item.as_posix()):
        digest.update(path.as_posix().encode("utf-8"))
        digest.update(path.read_bytes())
    return digest.hexdigest()


def _issue(code: str, message: str, actual=None, limit=None) -> dict:
    return {"code": code, "message": message, "actual": actual, "limit": limit}


def _golden_manifest_status(project_dir: str, policy: QualityPolicy) -> tuple[dict | None, str]:
    path = policy.golden_manifest.strip()
    if path and not os.path.isabs(path):
        path = os.path.join(project_dir, path)
    if not path or not os.path.isfile(path):
        return None, path
    try:
        data = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None, path
    required = (
        data.get("approved") is True
        and data.get("reviewer")
        and data.get("approver")
        and data.get("reviewer") != data.get("approver")
        and data.get("approved_at")
    )
    return data if required else None, path


def audit_yolo_dataset(dataset_dir: str, cfg: dict, *, project_dir: str | None = None,
                       write_report: bool = True) -> dict:
    """Audit one exported dataset and return a stable, JSON-serializable report."""
    root = Path(dataset_dir).resolve()
    policy = QualityPolicy.from_config(cfg)
    meta_path = root / "export_meta.json"
    meta = json.loads(meta_path.read_text(encoding="utf-8")) if meta_path.is_file() else {}
    label_format = str(meta.get("format") or "obb").lower()
    data_path = root / "data.yaml"
    data = {}
    if data_path.is_file():
        try:
            import yaml
            data = yaml.safe_load(data_path.read_text(encoding="utf-8")) or {}
        except (OSError, yaml.YAMLError):
            data = {}
    raw_names = data.get("names") or {}
    if isinstance(raw_names, dict):
        class_names = {int(key): str(value) for key, value in raw_names.items()}
    else:
        class_names = {index: str(value) for index, value in enumerate(raw_names)}
    geometry_exempt = set(policy.geometry_exempt_classes)
    counters = {
        "images": 0, "label_files": 0, "objects": 0, "malformed": 0,
        "out_of_bounds": 0, "degenerate": 0, "oversized": 0,
        "extreme_aspect": 0, "duplicates": 0, "class_conflicts": 0,
        "border_touching": 0, "overcrowded_images": 0,
    }
    split_stats = {}
    samples = {key: [] for key in (
        "malformed", "oversized", "extreme_aspect", "duplicates",
        "class_conflicts", "overcrowded_images",
    )}
    flagged_files: dict[str, set[str]] = {}

    def flag(file_key: str, code: str) -> None:
        flagged_files.setdefault(file_key, set()).add(code)
    label_paths: list[Path] = []
    seen_images: dict[str, str] = {}
    leakage = []

    for split in SPLITS:
        image_dir = root / "images" / split
        label_dir = root / "labels" / split
        images = list(image_dir.glob("*")) if image_dir.is_dir() else []
        labels = sorted(label_dir.glob("*.txt")) if label_dir.is_dir() else []
        split_objects = 0
        counters["images"] += sum(path.is_file() for path in images)
        counters["label_files"] += len(labels)
        for image in images:
            if not image.is_file():
                continue
            key = image.name.lower()
            if key in seen_images and len(leakage) < 20:
                leakage.append({"image": image.name, "splits": [seen_images[key], split]})
            seen_images[key] = split

        for label_path in labels:
            label_paths.append(label_path)
            records = []
            for line_number, raw in enumerate(label_path.read_text(encoding="utf-8").splitlines(), 1):
                values = raw.split()
                coordinate_count = len(values) - 1
                valid_obb = label_format == "obb" and coordinate_count == 8
                valid_seg = label_format == "seg" and coordinate_count >= 6 and coordinate_count % 2 == 0
                if not (valid_obb or valid_seg):
                    counters["malformed"] += 1
                    flag(f"{split}/{label_path.name}", "MALFORMED_LABEL")
                    if len(samples["malformed"]) < 20:
                        samples["malformed"].append(f"{split}/{label_path.name}:{line_number}")
                    continue
                try:
                    class_id = int(values[0])
                    coords = [float(value) for value in values[1:]]
                except ValueError:
                    counters["malformed"] += 1
                    continue
                points = list(zip(coords[::2], coords[1::2]))
                class_name = class_names.get(class_id, f"class_{class_id}")
                check_object_geometry = class_name not in geometry_exempt
                counters["objects"] += 1
                split_objects += 1
                if any(value < 0.0 or value > 1.0 for value in coords):
                    counters["out_of_bounds"] += 1
                    flag(f"{split}/{label_path.name}", "OUT_OF_BOUNDS")
                area = _polygon_area(points)
                if area <= 1e-8:
                    counters["degenerate"] += 1
                    flag(f"{split}/{label_path.name}", "DEGENERATE")
                if check_object_geometry and area >= policy.max_object_area_ratio:
                    counters["oversized"] += 1
                    flag(f"{split}/{label_path.name}", "OVERSIZED_OBJECT")
                    if len(samples["oversized"]) < 20:
                        samples["oversized"].append({"file": f"{split}/{label_path.name}", "area": round(area, 4)})
                aspect = _aspect_ratio(points)
                if check_object_geometry and aspect >= policy.max_aspect_ratio:
                    counters["extreme_aspect"] += 1
                    flag(f"{split}/{label_path.name}", "EXTREME_ASPECT")
                    if len(samples["extreme_aspect"]) < 20:
                        samples["extreme_aspect"].append({"file": f"{split}/{label_path.name}", "aspect": round(aspect, 2)})
                if any(value <= 1e-6 or value >= 1 - 1e-6 for value in coords):
                    counters["border_touching"] += 1
                records.append((class_id, tuple(round(value, 6) for value in coords), _bounds(points)))

            if len(records) > policy.max_objects_per_image:
                counters["overcrowded_images"] += 1
                flag(f"{split}/{label_path.name}", "OVERCROWDED_IMAGE")
                if len(samples["overcrowded_images"]) < 20:
                    samples["overcrowded_images"].append({"file": f"{split}/{label_path.name}", "objects": len(records)})
            exact = set()
            for index, (class_id, coords, bounds) in enumerate(records):
                key = (class_id, coords)
                if key in exact:
                    counters["duplicates"] += 1
                    flag(f"{split}/{label_path.name}", "DUPLICATE_LABEL")
                    if len(samples["duplicates"]) < 20:
                        samples["duplicates"].append(f"{split}/{label_path.name}")
                exact.add(key)
                for other_class, _, other_bounds in records[:index]:
                    if other_class != class_id and _bbox_iou(bounds, other_bounds) >= 0.90:
                        counters["class_conflicts"] += 1
                        flag(f"{split}/{label_path.name}", "CLASS_CONFLICT")
                        if len(samples["class_conflicts"]) < 20:
                            samples["class_conflicts"].append(f"{split}/{label_path.name}")
        split_stats[split] = {"images": sum(path.is_file() for path in images), "labels": len(labels), "objects": split_objects}

    errors = []
    warnings = []
    objects = max(counters["objects"], 1)
    ratios = {
        "oversized": counters["oversized"] / objects,
        "extreme_aspect": counters["extreme_aspect"] / objects,
        "duplicates": counters["duplicates"] / objects,
        "class_conflicts": counters["class_conflicts"] / objects,
        "border_touching": counters["border_touching"] / objects,
    }
    if counters["malformed"]:
        errors.append(_issue("MALFORMED_LABEL", "存在无法解析的标签行", counters["malformed"], 0))
    if counters["out_of_bounds"] or counters["degenerate"]:
        errors.append(_issue("INVALID_GEOMETRY", "存在越界或零面积标注", counters["out_of_bounds"] + counters["degenerate"], 0))
    for key, limit, code, message in (
        ("oversized", policy.max_oversized_fraction, "OVERSIZED_OBJECTS", "大面积目标占比过高，疑似整排/整区错误框"),
        ("extreme_aspect", policy.max_extreme_aspect_fraction, "EXTREME_ASPECT", "极端长宽比目标占比过高"),
        ("duplicates", policy.max_duplicate_fraction, "DUPLICATE_LABELS", "重复标注占比过高"),
        ("class_conflicts", policy.max_class_conflict_fraction, "CLASS_CONFLICTS", "高度重叠目标存在类别冲突"),
    ):
        if ratios[key] > limit:
            errors.append(_issue(code, message, round(ratios[key], 6), limit))
    if leakage:
        errors.append(_issue("SPLIT_LEAKAGE", "同名图片跨数据分区出现", len(leakage), 0))
    if split_stats.get("test", {}).get("images", 0) < policy.min_test_images:
        errors.append(_issue("TEST_SET_TOO_SMALL", "测试集规模不足", split_stats.get("test", {}).get("images", 0), policy.min_test_images))

    source_provenance = {}
    provenance_path = meta.get("source_provenance") if isinstance(meta, dict) else ""
    if provenance_path:
        provenance_file = Path(provenance_path)
        if not provenance_file.is_absolute():
            provenance_file = root / provenance_file
        source_provenance = safe_load_json(str(provenance_file), {}) or {}
    conversion = str(source_provenance.get("conversion", "")).lower()
    if "aabb to four-corner obb" in conversion:
        errors.append(_issue(
            "AABB_TO_OBB_INVALID",
            "Source rotation was discarded before export; expanding AABB corners cannot create valid parking-space OBB labels.",
            source_provenance.get("conversion"),
            "native rotated polygons or fixed camera topology",
        ))
    source_dataset = str(source_provenance.get("source_dataset", "")).lower()
    if "roboflow" in source_dataset and "aabb" in conversion:
        errors.append(_issue(
            "SOURCE_DATA_NOT_GOLDEN",
            "The Roboflow detection export is suitable for crop classification, not OBB golden-set acceptance.",
            source_provenance.get("source_dataset"),
            "native PKLot rotated annotations",
        ))
    if policy.require_fixed_test and meta.get("split_mode") != "fixed_test_list":
        errors.append(_issue("TEST_SET_NOT_FIXED", "企业验收要求固定、隔离的测试集清单", meta.get("split_mode"), "fixed_test_list"))
    manifest, manifest_path = _golden_manifest_status(project_dir or os.getcwd(), policy)
    if policy.require_golden_manifest and manifest is None:
        errors.append(_issue("GOLDEN_SET_NOT_APPROVED", "缺少经人工签署的金标测试集清单", manifest_path or "未配置", "approved manifest"))
    provenance = meta.get("review_provenance") if isinstance(meta, dict) else None
    if policy.require_review_provenance and not isinstance(provenance, dict):
        errors.append(_issue("REVIEW_PROVENANCE_MISSING", "导出数据缺少人工/自动审核来源记录", "missing", "review_provenance"))
    elif isinstance(provenance, dict):
        reviewed = max(int(provenance.get("reviewed", 0)), 1)
        human_fraction = int(provenance.get("human", 0)) / reviewed
        if human_fraction < policy.min_human_review_fraction:
            errors.append(_issue("HUMAN_REVIEW_TOO_LOW", "人工复核比例不足", round(human_fraction, 6), policy.min_human_review_fraction))
    if ratios["border_touching"] > 0.05:
        warnings.append(_issue("MANY_BORDER_OBJECTS", "大量目标贴边，建议人工抽检截断框", round(ratios["border_touching"], 6), 0.05))
    if counters["overcrowded_images"]:
        warnings.append(_issue("OVERCROWDED_IMAGES", "部分图片目标数量异常高", counters["overcrowded_images"], 0))

    report = {
        "schema_version": SCHEMA_VERSION,
        "audited_at": datetime.now().isoformat(),
        "dataset_dir": str(root),
        "label_format": label_format,
        "dataset_fingerprint": _fingerprint(label_paths),
        "status": "failed" if errors else "passed",
        "enforced": policy.enforce,
        "policy": asdict(policy),
        "summary": counters,
        "ratios": {key: round(value, 6) for key, value in ratios.items()},
        "splits": split_stats,
        "errors": errors,
        "warnings": warnings,
        "samples": samples,
        "split_leakage_samples": leakage,
        "golden_manifest": manifest_path,
        "review_provenance": provenance,
        "source_provenance": source_provenance,
        "review_queue_images": len(flagged_files),
        "class_names": class_names,
        "geometry_exempt_classes": sorted(geometry_exempt),
    }
    if write_report:
        queue_path = root / "quality_review_queue.json"
        safe_write_json(str(queue_path), {
            "schema_version": SCHEMA_VERSION,
            "dataset_fingerprint": report["dataset_fingerprint"],
            "items": [
                {"label": path, "image_stem": Path(path).stem, "issues": sorted(codes)}
                for path, codes in sorted(flagged_files.items())
            ],
        })
        report["review_queue_path"] = str(queue_path)
        safe_write_json(str(root / "quality_report.json"), report)
    return report


def require_quality_pass(dataset_dir: str, cfg: dict, *, project_dir: str | None = None) -> dict:
    report = audit_yolo_dataset(dataset_dir, cfg, project_dir=project_dir, write_report=True)
    if report["status"] != "passed" and report["enforced"]:
        details = "\n".join(f" - {item['code']}: {item['message']}" for item in report["errors"])
        raise RuntimeError(f"数据质量门禁未通过，禁止训练/验收：\n{details}\n报告: {Path(dataset_dir) / 'quality_report.json'}")
    return report


def build_recovery_plan(report: dict) -> dict:
    """Build a conservative, low-cost remediation plan from an audit report."""
    summary = report.get("summary", {})
    total_objects = max(int(summary.get("objects", 0)), 1)
    deterministic_removals = int(summary.get("degenerate", 0)) + int(summary.get("duplicates", 0))
    quarantine_signals = (
        int(summary.get("oversized", 0))
        + int(summary.get("extreme_aspect", 0))
        + int(summary.get("class_conflicts", 0))
    )
    conservative_removed = min(total_objects, deterministic_removals + quarantine_signals)
    retained = total_objects - conservative_removed
    return {
        "strategy": "sanitize_then_finetune",
        "source_model": "train/weights/best.pt",
        "full_retrain_required": False,
        "total_objects": total_objects,
        "automatic_removals": deterministic_removals,
        "quarantine_signals": quarantine_signals,
        "estimated_retained_objects": retained,
        "estimated_retained_fraction": round(retained / total_objects, 4),
        "manual_golden_images": 300,
        "recommended_finetune_epochs": {"min": 15, "max": 30},
        "steps": [
            {"id": "auto_clean", "title": "自动清洗确定性坏标注", "scope": "零面积、完全重复、极端几何和类别冲突对象", "cost": "低"},
            {"id": "golden_set", "title": "建立小规模金标测试集", "scope": "分层抽取 300 张，由人工逐框复核", "cost": "中"},
            {"id": "finetune", "title": "基于现有 best.pt 微调", "scope": "15-30 epochs，低学习率，不从零训练", "cost": "低"},
            {"id": "acceptance", "title": "同类别历史版本验收", "scope": "固定金标集，检查分场景误检和漏检", "cost": "低"},
        ],
    }


def _valid_record(raw: str, class_names: dict[int, str], policy: QualityPolicy) -> tuple[int, list[float], tuple[float, float, float, float]] | None:
    values = raw.split()
    if len(values) != 9:
        return None
    try:
        class_id = int(values[0])
        coords = [float(value) for value in values[1:]]
    except ValueError:
        return None
    if any(value < 0.0 or value > 1.0 for value in coords):
        return None
    points = list(zip(coords[::2], coords[1::2]))
    if _polygon_area(points) <= 1e-8:
        return None
    class_name = class_names.get(class_id, f"class_{class_id}")
    if class_name not in set(policy.geometry_exempt_classes):
        if _polygon_area(points) >= policy.max_object_area_ratio:
            return None
        if _aspect_ratio(points) >= policy.max_aspect_ratio:
            return None
    return class_id, coords, _bounds(points)


def build_sanitized_candidate(dataset_dir: str, cfg: dict, output_dir: str) -> dict:
    """Create a non-destructive training candidate with only deterministic bad objects removed.

    Images are hard-linked when possible so the candidate consumes minimal disk.
    Ambiguous geometry is quarantined from *training labels*, never rewritten into
    the source dataset.  It still requires a human-approved gold set before release.
    """
    source = Path(dataset_dir).resolve()
    output = Path(output_dir).resolve()
    if output.exists():
        raise FileExistsError(f"清洗候选集目录已存在，为保护已有产物不覆盖: {output}")
    policy = QualityPolicy.from_config(cfg)
    data_path = source / "data.yaml"
    if not data_path.is_file():
        raise FileNotFoundError(f"找不到 data.yaml: {data_path}")
    import yaml
    data = yaml.safe_load(data_path.read_text(encoding="utf-8")) or {}
    raw_names = data.get("names") or {}
    class_names = ({int(key): str(value) for key, value in raw_names.items()}
                   if isinstance(raw_names, dict)
                   else {index: str(value) for index, value in enumerate(raw_names)})
    staging = output.with_name(output.name + f".staging.{os.getpid()}")
    stats = {"images": 0, "input_objects": 0, "kept_objects": 0,
             "removed_invalid": 0, "removed_duplicate": 0, "removed_conflict": 0}
    try:
        for split in SPLITS:
            src_images = source / "images" / split
            src_labels = source / "labels" / split
            dst_images = staging / "images" / split
            dst_labels = staging / "labels" / split
            dst_images.mkdir(parents=True, exist_ok=True)
            dst_labels.mkdir(parents=True, exist_ok=True)
            for image in src_images.iterdir() if src_images.is_dir() else []:
                if not image.is_file():
                    continue
                target = dst_images / image.name
                try:
                    os.link(image, target)
                except OSError:
                    shutil.copy2(image, target)
                stats["images"] += 1
            for label in src_labels.glob("*.txt") if src_labels.is_dir() else []:
                parsed = []
                for raw in label.read_text(encoding="utf-8").splitlines():
                    stats["input_objects"] += 1
                    item = _valid_record(raw, class_names, policy)
                    if item is None:
                        stats["removed_invalid"] += 1
                    else:
                        parsed.append((raw, *item))
                kept = []
                exact = set()
                for raw, class_id, coords, bounds in parsed:
                    key = (class_id, tuple(round(value, 6) for value in coords))
                    if key in exact:
                        stats["removed_duplicate"] += 1
                        continue
                    exact.add(key)
                    kept.append([raw, class_id, bounds, True])
                for index, item in enumerate(kept):
                    if not item[3]:
                        continue
                    for other in kept[:index]:
                        if other[3] and item[1] != other[1] and _bbox_iou(item[2], other[2]) >= 0.90:
                            item[3] = other[3] = False
                            stats["removed_conflict"] += 2
                safe_lines = [item[0] for item in kept if item[3]]
                stats["kept_objects"] += len(safe_lines)
                (dst_labels / label.name).write_text("\n".join(safe_lines) + ("\n" if safe_lines else ""), encoding="utf-8")
        data["path"] = str(output)
        (staging / "data.yaml").write_text(yaml.safe_dump(data, allow_unicode=True, sort_keys=False), encoding="utf-8")
        source_meta = safe_load_json(str(source / "export_meta.json"), {})
        source_meta["sanitized_candidate"] = {
            "created_at": datetime.now().isoformat(),
            "source_dataset": str(source),
            "source_fingerprint": _fingerprint(list(source.glob("labels/*/*.txt"))),
            "policy": asdict(policy),
            "stats": stats,
        }
        source_meta["review_provenance"] = source_meta.get("review_provenance") or {
            "reviewed": 0, "auto": 0, "human": 0, "status": "legacy_unknown",
        }
        safe_write_json(str(staging / "export_meta.json"), source_meta)
        os.replace(staging, output)
    except Exception:
        if staging.exists():
            shutil.rmtree(staging)
        raise
    report = audit_yolo_dataset(str(output), cfg, project_dir=os.getcwd(), write_report=True)
    report["sanitization"] = stats
    safe_write_json(str(output / "quality_report.json"), report)
    stats["removed_total"] = stats["input_objects"] - stats["kept_objects"]
    return {"output_dir": str(output), "stats": stats, "quality_status": report["status"]}
