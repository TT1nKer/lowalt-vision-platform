#!/usr/bin/env python3
"""Validate and install a human-reviewed golden test package.

Validation is deliberately read-only. Installation is atomic and is allowed
only after every review and approval invariant has passed.
"""

from __future__ import annotations

import csv
import hashlib
import json
import os
import shutil
import tempfile
import threading
from datetime import datetime
from pathlib import Path

from console.core import safe_load_json, safe_write_json


FINAL_REVIEW_STATUSES = frozenset({"approved", "corrected"})
REQUIRED_REVIEW_COLUMNS = frozenset({
    "image", "review_status", "reviewer", "reviewed_at", "notes",
})
EDITABLE_REVIEW_STATUSES = frozenset({
    "pending", "approved", "needs_correction", "corrected",
})
_REVIEW_LOCK = threading.Lock()


def _issue(code: str, message: str, actual=None) -> dict:
    return {"code": code, "message": message, "actual": actual}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _parse_datetime(value: str) -> bool:
    try:
        datetime.fromisoformat(value.strip().replace("Z", "+00:00"))
        return True
    except (AttributeError, TypeError, ValueError):
        return False


def read_review_rows(package_dir: str) -> list[dict]:
    path = Path(package_dir).resolve() / "review.csv"
    if not path.is_file():
        raise FileNotFoundError(f"review.csv not found: {path}")
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def save_review_row(package_dir: str, index: int, values: dict) -> dict:
    """Atomically update one review decision while preserving row identity."""
    path = Path(package_dir).resolve() / "review.csv"
    status = str(values.get("review_status", "")).strip().lower()
    reviewer = str(values.get("reviewer", "local-review")).strip()[:100] or "local-review"
    notes = str(values.get("notes", "")).strip()[:2000]
    if status not in EDITABLE_REVIEW_STATUSES:
        raise ValueError(f"unsupported review_status: {status}")
    if status in {"needs_correction", "corrected"} and not notes:
        raise ValueError(f"notes are required when status is {status}")

    with _REVIEW_LOCK:
        rows = read_review_rows(package_dir)
        if index < 0 or index >= len(rows):
            raise IndexError("review row index out of range")
        rows[index]["review_status"] = status
        rows[index]["reviewer"] = reviewer if status != "pending" else ""
        rows[index]["reviewed_at"] = datetime.now().astimezone().isoformat() if status != "pending" else ""
        rows[index]["notes"] = notes if status != "pending" else ""
        fieldnames = list(rows[0].keys())
        fd, temp_name = tempfile.mkstemp(prefix=path.name + ".", suffix=".tmp", dir=str(path.parent))
        try:
            with os.fdopen(fd, "w", encoding="utf-8-sig", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(rows)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temp_name, path)
        except Exception:
            try:
                os.unlink(temp_name)
            except OSError:
                pass
            raise
    return rows[index]


def validate_golden_package(package_dir: str) -> dict:
    """Return an evidence-rich report without modifying the package."""
    root = Path(package_dir).resolve()
    review_path = root / "review.csv"
    manifest_path = root / "golden_test_manifest.json"
    fixed_path = root / "fixed_test_list.txt"
    errors: list[dict] = []
    rows: list[dict] = []
    manifest = safe_load_json(str(manifest_path), {})

    if not review_path.is_file():
        errors.append(_issue("REVIEW_CSV_MISSING", "缺少 review.csv"))
    else:
        try:
            with review_path.open("r", encoding="utf-8-sig", newline="") as handle:
                reader = csv.DictReader(handle)
                columns = set(reader.fieldnames or [])
                missing = sorted(REQUIRED_REVIEW_COLUMNS - columns)
                if missing:
                    errors.append(_issue("REVIEW_COLUMNS_MISSING", "审核表缺少必要列", missing))
                rows = list(reader)
        except (OSError, csv.Error) as exc:
            errors.append(_issue("REVIEW_CSV_INVALID", "review.csv 无法解析", str(exc)))

    if not isinstance(manifest, dict) or not manifest:
        errors.append(_issue("MANIFEST_MISSING", "缺少或无法解析 golden_test_manifest.json"))
        manifest = {}

    fixed_images: list[str] = []
    if not fixed_path.is_file():
        errors.append(_issue("FIXED_LIST_MISSING", "缺少 fixed_test_list.txt"))
    else:
        fixed_images = [line.strip() for line in fixed_path.read_text(encoding="utf-8").splitlines() if line.strip()]

    row_images = [str(row.get("image", "")).strip() for row in rows]
    manifest_images = manifest.get("images") if isinstance(manifest.get("images"), list) else []
    if not rows:
        errors.append(_issue("REVIEW_EMPTY", "审核表没有样本"))
    if len(row_images) != len(set(row_images)):
        errors.append(_issue("REVIEW_DUPLICATES", "审核表包含重复图片"))
    if row_images and (row_images != manifest_images or row_images != fixed_images):
        errors.append(_issue("GOLDEN_LIST_MISMATCH", "审核表、manifest 与固定测试清单不一致"))

    pending = 0
    invalid_rows = []
    reviewers = set()
    for line_number, row in enumerate(rows, 2):
        status = str(row.get("review_status", "")).strip().lower()
        reviewer = str(row.get("reviewer", "")).strip()
        reviewed_at = str(row.get("reviewed_at", "")).strip()
        notes = str(row.get("notes", "")).strip()
        if status not in FINAL_REVIEW_STATUSES:
            pending += 1
            invalid_rows.append(line_number)
            continue
        if not reviewer or not _parse_datetime(reviewed_at) or (status == "corrected" and not notes):
            invalid_rows.append(line_number)
        if reviewer:
            reviewers.add(reviewer)
    if invalid_rows:
        errors.append(_issue(
            "REVIEW_INCOMPLETE", "每行须为 approved/corrected，填写审核人和 ISO 时间；corrected 还须写说明",
            {"rows": invalid_rows[:20], "count": len(invalid_rows)},
        ))

    missing_assets = []
    for image_name in row_images:
        stem = Path(image_name).stem
        for relative in (Path("images") / image_name, Path("labels") / f"{stem}.txt"):
            if not (root / relative).is_file():
                missing_assets.append(relative.as_posix())
    if missing_assets:
        errors.append(_issue("GOLDEN_ASSETS_MISSING", "金标图片或标签缺失", missing_assets[:20]))

    manifest_reviewer = str(manifest.get("reviewer", "")).strip()
    approver = str(manifest.get("approver", "")).strip()
    approved_at = str(manifest.get("approved_at", "")).strip()
    if manifest.get("approved") is not True:
        errors.append(_issue("MANIFEST_NOT_APPROVED", "manifest 尚未设置 approved=true"))
    if not manifest_reviewer or not approver or not _parse_datetime(approved_at):
        errors.append(_issue("MANIFEST_SIGNATURE_MISSING", "manifest 须填写 reviewer、approver 和 ISO approved_at"))
    elif manifest_reviewer == approver:
        errors.append(_issue("FOUR_EYES_VIOLATION", "审核人与批准人必须是不同人员"))
    if reviewers and manifest_reviewer and manifest_reviewer not in reviewers:
        errors.append(_issue("REVIEWER_MISMATCH", "manifest reviewer 未出现在逐图审核记录中", manifest_reviewer))

    hashes = {}
    for path in (review_path, manifest_path, fixed_path):
        if path.is_file():
            hashes[path.name] = _sha256(path)
    return {
        "schema_version": 1,
        "validated_at": datetime.now().isoformat(),
        "package_dir": str(root),
        "status": "passed" if not errors else "failed",
        "sample_count": len(rows),
        "completed_count": len(rows) - pending,
        "pending_count": pending,
        "reviewers": sorted(reviewers),
        "approved": manifest.get("approved") is True,
        "approver": approver,
        "errors": errors,
        "artifact_sha256": hashes,
    }


def install_golden_package(package_dir: str, project_dir: str, dataset_dir: str) -> dict:
    """Install approved artifacts and attest that the current split is isolated."""
    report = validate_golden_package(package_dir)
    if report["status"] != "passed":
        codes = ", ".join(item["code"] for item in report["errors"])
        raise RuntimeError(f"金标审核包未通过校验，禁止安装: {codes}")

    root = Path(package_dir).resolve()
    dataset = Path(dataset_dir).resolve()
    fixed_images = [line.strip() for line in (root / "fixed_test_list.txt").read_text(encoding="utf-8").splitlines() if line.strip()]
    leakage = []
    for image_name in fixed_images:
        for split in ("train", "val"):
            if (dataset / "images" / split / image_name).exists():
                leakage.append(f"{split}/{image_name}")
        if not (dataset / "images" / "test" / image_name).is_file():
            leakage.append(f"missing-test/{image_name}")
    if leakage:
        raise RuntimeError(f"固定金标集未与训练/验证集隔离，禁止安装: {leakage[:10]}")

    quality_dir = Path(project_dir).resolve() / "quality"
    quality_dir.mkdir(parents=True, exist_ok=True)
    targets = {
        root / "golden_test_manifest.json": quality_dir / "golden_test_manifest.json",
        root / "fixed_test_list.txt": quality_dir / "fixed_test_list.txt",
        root / "review.csv": quality_dir / "golden_review.csv",
    }
    staging = Path(tempfile.mkdtemp(prefix="golden-install-", dir=str(quality_dir)))
    try:
        for source, target in targets.items():
            staged = staging / target.name
            shutil.copy2(source, staged)
            os.replace(staged, target)
    finally:
        shutil.rmtree(staging, ignore_errors=True)

    meta_path = dataset / "export_meta.json"
    meta = safe_load_json(str(meta_path), {})
    meta["split_mode"] = "fixed_test_list"
    meta["fixed_test_list"] = str(quality_dir / "fixed_test_list.txt")
    meta["golden_test"] = {
        "manifest": str(quality_dir / "golden_test_manifest.json"),
        "sample_count": report["sample_count"],
        "installed_at": datetime.now().isoformat(),
        "artifact_sha256": report["artifact_sha256"],
    }
    safe_write_json(str(meta_path), meta)
    installed = {**report, "installed": True, "dataset_dir": str(dataset)}
    safe_write_json(str(quality_dir / "golden_install_report.json"), installed)
    return installed
