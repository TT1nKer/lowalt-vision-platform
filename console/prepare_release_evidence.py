#!/usr/bin/env python3
"""Prepare unsigned, evidence-bound templates for recovery release approval."""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

from console.acceptance_workflow import validate_acceptance
from console.release_gate import sha256


def prepare(project_dir: str = ".") -> dict:
    root = Path(project_dir).resolve()
    dataset = root / "quality/pklot_official_recovery_v2"
    shadow = root / "quality/business_shadow_v1"
    candidate = root / "quality/official_finetune/finetune_official_v1/weights/best.pt"
    quality = json.loads((dataset / "quality_report.json").read_text(encoding="utf-8"))
    fixed = dataset / "fixed_test_list.txt"
    images = [line.strip() for line in fixed.read_text(encoding="utf-8").splitlines() if line.strip()]
    acceptance = validate_acceptance(str(shadow / "human_acceptance.csv"))
    approval = {
        "schema_version": 1, "decision": "pending", "approver": "",
        "approved_at": "", "notes": "",
        "candidate_sha256": sha256(candidate),
        "dataset_fingerprint": quality["dataset_fingerprint"],
        "acceptance_sha256": acceptance.get("sha256", ""),
        "instructions": "Only an independent approver may set decision=approved after human acceptance passes.",
    }
    golden = {
        "schema_version": 1, "approved": False, "reviewer": "", "approver": "",
        "approved_at": "", "annotation_standard": "parking_obb_v1",
        "classes": ["occupied", "empty"],
        "dataset_fingerprint": quality["dataset_fingerprint"],
        "fixed_test_list_sha256": sha256(fixed), "images": images, "notes": "",
        "instructions": "Set approved=true only after human review; reviewer and approver must differ.",
    }
    approval_path = shadow / "release_approval.pending.json"
    golden_path = root / "quality/golden_test_manifest.pending.json"
    approval_path.write_text(json.dumps(approval, ensure_ascii=False, indent=2), encoding="utf-8")
    golden_path.write_text(json.dumps(golden, ensure_ascii=False, indent=2), encoding="utf-8")
    return {"generated_at": datetime.now(timezone.utc).isoformat(),
            "release_approval_template": str(approval_path),
            "golden_manifest_template": str(golden_path), "fixed_test_images": len(images)}


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser(); p.add_argument("project_dir", nargs="?", default=".")
    print(json.dumps(prepare(p.parse_args().project_dir), ensure_ascii=False, indent=2))
