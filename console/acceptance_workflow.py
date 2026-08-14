#!/usr/bin/env python3
"""Create and validate the human business acceptance package."""
from __future__ import annotations

import csv
import hashlib
import json
import os
import tempfile
from datetime import datetime, timezone
from pathlib import Path


FIELDS = ["image", "comparison", "occupied_tp", "occupied_fp", "occupied_fn",
          "empty_tp", "empty_fp", "empty_fn", "critical_fp", "reviewer",
          "reviewed_at", "notes"]
NUMERIC_FIELDS = FIELDS[2:9]


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def create_package(shadow_dir: str, count: int = 30, overwrite: bool = False) -> Path:
    root = Path(shadow_dir).resolve()
    report = json.loads((root / "shadow_report.json").read_text(encoding="utf-8"))
    rows = report.get("records", [])[:count]
    out = root / "human_acceptance.csv"
    if out.exists() and not overwrite:
        raise FileExistsError(f"refusing to overwrite existing human review: {out}")
    with out.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS)
        writer.writeheader()
        for item in rows:
            writer.writerow({"image": item["image"], "comparison": item["comparison"],
                             **{field: "" for field in FIELDS[2:]}})
    meta = {"schema_version": 1, "status": "awaiting_human_review",
            "required_rows": len(rows), "created_at": datetime.now(timezone.utc).isoformat(),
            "policy": report.get("acceptance_policy", {})}
    (root / "human_acceptance.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    return out


def validate_acceptance(csv_path: str) -> dict:
    path = Path(csv_path).resolve()
    if not path.is_file():
        return {"status": "missing", "path": str(path)}
    with path.open(newline="", encoding="utf-8-sig") as f:
        rows = list(csv.DictReader(f))
    errors = []
    reviewed = 0
    for i, row in enumerate(rows, 2):
        row_complete = True
        for field in NUMERIC_FIELDS:
            try:
                value = int(row.get(field, ""))
                if value < 0: raise ValueError
            except (TypeError, ValueError):
                row_complete = False
                errors.append(f"row {i}: {field} must be a non-negative integer")
        if not row.get("reviewer", "").strip() or not row.get("reviewed_at", "").strip():
            row_complete = False
            errors.append(f"row {i}: reviewer and reviewed_at are required")
        if row_complete:
            reviewed += 1
    def score(cls):
        tp = sum(int(r[cls+"_tp"]) for r in rows if r.get(cls+"_tp", "").isdigit())
        fp = sum(int(r[cls+"_fp"]) for r in rows if r.get(cls+"_fp", "").isdigit())
        fn = sum(int(r[cls+"_fn"]) for r in rows if r.get(cls+"_fn", "").isdigit())
        return {"tp": tp, "fp": fp, "fn": fn,
                "precision": tp / (tp + fp) if tp + fp else 0.0,
                "recall": tp / (tp + fn) if tp + fn else 0.0}
    metrics = {"occupied": score("occupied"), "empty": score("empty")}
    critical = sum(int(r["critical_fp"]) for r in rows if r.get("critical_fp", "").isdigit())
    total = len(rows)
    if total != 30: errors.append(f"exactly 30 reviewed rows required (got {total})")
    if any(m["precision"] < .90 or m["recall"] < .85 for m in metrics.values()):
        errors.append("per-class precision/recall threshold not met")
    if total and critical / total > .02: errors.append("critical false-positive rate exceeds 0.02")
    return {"status": "passed" if not errors else "failed", "rows": total,
            "reviewed": reviewed, "remaining": max(total - reviewed, 0),
            "metrics": metrics, "critical_fp": critical,
            "critical_fp_rate": critical / total if total else 0.0, "errors": errors,
            "sha256": _sha256(path)}


def read_acceptance(csv_path: str) -> list[dict]:
    path = Path(csv_path).resolve()
    with path.open(newline="", encoding="utf-8-sig") as f:
        return list(csv.DictReader(f))


def save_acceptance_row(csv_path: str, index: int, values: dict) -> dict:
    """Atomically save one review row without allowing image identity changes."""
    path = Path(csv_path).resolve()
    rows = read_acceptance(str(path))
    if index < 0 or index >= len(rows):
        raise IndexError("acceptance row index out of range")
    clean = {}
    for field in NUMERIC_FIELDS:
        raw = values.get(field, "")
        if raw in (None, ""):
            clean[field] = ""
            continue
        try:
            value = int(raw)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"{field} must be a non-negative integer") from exc
        if value < 0:
            raise ValueError(f"{field} must be a non-negative integer")
        clean[field] = str(value)
    clean["reviewer"] = str(values.get("reviewer", "")).strip()[:100]
    clean["notes"] = str(values.get("notes", "")).strip()[:2000]
    complete = all(clean[field] != "" for field in NUMERIC_FIELDS) and bool(clean["reviewer"])
    clean["reviewed_at"] = datetime.now(timezone.utc).isoformat() if complete else ""
    rows[index].update(clean)
    fd, temp_name = tempfile.mkstemp(prefix=path.name + ".", suffix=".tmp", dir=str(path.parent))
    try:
        with os.fdopen(fd, "w", newline="", encoding="utf-8-sig") as f:
            writer = csv.DictWriter(f, fieldnames=FIELDS)
            writer.writeheader(); writer.writerows(rows)
            f.flush(); os.fsync(f.fileno())
        os.replace(temp_name, path)
    except Exception:
        try: os.unlink(temp_name)
        except OSError: pass
        raise
    return rows[index]


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser(); p.add_argument("shadow_dir", nargs="?", default="quality/business_shadow_v1"); p.add_argument("--validate", action="store_true"); p.add_argument("--force", action="store_true")
    a = p.parse_args(); path = Path(a.shadow_dir).resolve() / "human_acceptance.csv"
    result = validate_acceptance(str(path)) if a.validate else {"created": str(create_package(a.shadow_dir, overwrite=a.force))}
    print(json.dumps(result, ensure_ascii=False, indent=2))
