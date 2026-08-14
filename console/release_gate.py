#!/usr/bin/env python3
"""Build an auditable candidate manifest and evaluate the production release gate."""
from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

from console.acceptance_workflow import validate_acceptance


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _valid_time(value) -> bool:
    try:
        datetime.fromisoformat(str(value).replace("Z", "+00:00"))
        return True
    except (TypeError, ValueError):
        return False


def validate_release_approval(approval: dict, *, candidate_sha256: str,
                              dataset_fingerprint: str, acceptance_sha256: str,
                              reviewers: set[str]) -> dict:
    approver = str(approval.get("approver", "")).strip()
    errors = []
    if approval.get("decision") != "approved": errors.append("decision is not approved")
    if not approver: errors.append("approver is missing")
    if approver in reviewers: errors.append("approver must be independent from business reviewers")
    if not _valid_time(approval.get("approved_at")): errors.append("approved_at is not a valid ISO-8601 timestamp")
    expected = {"candidate_sha256": candidate_sha256, "dataset_fingerprint": dataset_fingerprint,
                "acceptance_sha256": acceptance_sha256}
    for field, value in expected.items():
        if approval.get(field) != value: errors.append(f"{field} does not match current evidence")
    return {"valid": not errors, "approver": approver or None, "errors": errors,
            "evidence": expected}


def validate_golden_attestation(root: Path, dataset_fingerprint: str) -> dict:
    manifest_path = root / "quality/golden_test_manifest.json"
    fixed_path = root / "quality/pklot_official_recovery_v2/fixed_test_list.txt"
    if not manifest_path.is_file():
        return {"valid": False, "errors": ["signed golden_test_manifest.json is missing"]}
    try: manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {"valid": False, "errors": ["golden manifest is invalid JSON"]}
    errors = []
    reviewer, approver = str(manifest.get("reviewer", "")).strip(), str(manifest.get("approver", "")).strip()
    if manifest.get("approved") is not True: errors.append("golden manifest is not approved")
    if not reviewer or not approver or reviewer == approver: errors.append("golden review requires two distinct people")
    if not _valid_time(manifest.get("approved_at")): errors.append("golden approved_at is invalid")
    if manifest.get("dataset_fingerprint") != dataset_fingerprint: errors.append("golden dataset fingerprint mismatch")
    fixed_hash = sha256(fixed_path)
    if manifest.get("fixed_test_list_sha256") != fixed_hash: errors.append("fixed test list hash mismatch")
    fixed_images = [line.strip() for line in fixed_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if manifest.get("images") != fixed_images: errors.append("golden image list does not match fixed test list")
    return {"valid": not errors, "errors": errors, "reviewer": reviewer or None,
            "approver": approver or None, "fixed_test_list_sha256": fixed_hash,
            "image_count": len(fixed_images)}


def build(project_dir: str = ".") -> dict:
    root = Path(project_dir).resolve()
    candidate = root / "quality/official_finetune/finetune_official_v1/weights/best.pt"
    baseline = root / "sam3_runs/pklot_v1/merged/yolo_obb/train/weights/best.pt"
    dataset_report = json.loads((root / "quality/pklot_official_recovery_v2/quality_report.json").read_text(encoding="utf-8"))
    comparison = json.loads((root / "quality/recovery_comparison_v1/comparison_report.json").read_text(encoding="utf-8"))
    acceptance_path = root / "quality/business_shadow_v1/human_acceptance.csv"
    acceptance = validate_acceptance(str(acceptance_path))
    approval_path = root / "quality/business_shadow_v1/release_approval.json"
    approval = json.loads(approval_path.read_text(encoding="utf-8")) if approval_path.is_file() else {}
    reviewers = set()
    if acceptance_path.is_file():
        import csv
        with acceptance_path.open(newline="", encoding="utf-8-sig") as f:
            reviewers = {r.get("reviewer", "").strip() for r in csv.DictReader(f) if r.get("reviewer", "").strip()}
    candidate_hash, baseline_hash = sha256(candidate), sha256(baseline)
    acceptance_hash = acceptance.get("sha256", "")
    approval_report = validate_release_approval(
        approval, candidate_sha256=candidate_hash,
        dataset_fingerprint=dataset_report.get("dataset_fingerprint", ""),
        acceptance_sha256=acceptance_hash, reviewers=reviewers,
    )
    golden_report = validate_golden_attestation(root, dataset_report.get("dataset_fingerprint", ""))
    blockers = []
    if acceptance.get("status") != "passed": blockers.append("30-image human business acceptance has not passed")
    if not approval_report["valid"]: blockers.append("independent release approval is missing, stale, or invalid")
    if not golden_report["valid"]: blockers.append("formal signed gold-set attestation is missing or invalid")
    manifest = {
        "schema_version": 1, "generated_at": datetime.now(timezone.utc).isoformat(),
        "release_status": "approved_for_promotion" if not blockers else "blocked",
        "candidate": {"path": str(candidate), "sha256": candidate_hash},
        "rollback": {"path": str(baseline), "sha256": baseline_hash},
        "dataset": {"path": str(root / "quality/pklot_official_recovery_v2"),
                    "fingerprint": dataset_report.get("dataset_fingerprint")},
        "fixed_test": comparison.get("metrics", {}), "acceptance": acceptance,
        "independent_approval": approval_report, "golden_attestation": golden_report,
        "blockers": blockers,
        "promotion_policy": "Versioned copy plus atomic default pointer update; preserve rollback model.",
    }
    out = root / "quality/model_release_manifest.json"
    out.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    return manifest


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser(); p.add_argument("project_dir", nargs="?", default=".")
    print(json.dumps(build(p.parse_args().project_dir), ensure_ascii=False, indent=2))
