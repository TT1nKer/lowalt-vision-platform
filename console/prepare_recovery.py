#!/usr/bin/env python3
"""Prepare auditable recovery metadata and a low-cost fine-tune command."""

import argparse
import json
import os
from datetime import datetime
from pathlib import Path

from console.core import safe_load_json, safe_write_json


def prepare(project_dir: str) -> dict:
    project = Path(project_dir).resolve()
    candidate = project / "sam3_runs" / "pklot_v1" / "merged" / "yolo_obb_clean_candidate"
    golden = project / "quality" / "golden_review_v1"
    provenance_path = project / "quality" / "review_provenance.json"
    for required in (candidate / "export_meta.json", golden / "golden_test_manifest.json", provenance_path):
        if not required.is_file():
            raise FileNotFoundError(required)
    meta = safe_load_json(str(candidate / "export_meta.json"), {})
    provenance = safe_load_json(str(provenance_path), {})
    meta["review_provenance"] = provenance
    meta["golden_candidate"] = {
        "manifest": str(golden / "golden_test_manifest.json"),
        "fixed_test_list": str(golden / "fixed_test_list.txt"),
        "review_csv": str(golden / "review.csv"),
        "status": "pending_human_review",
    }
    safe_write_json(str(candidate / "export_meta.json"), meta)
    finetune = {
        "schema_version": 1,
        "created_at": datetime.now().isoformat(),
        "status": "blocked_pending_golden_approval",
        "source_model": str(project / "sam3_runs" / "pklot_v1" / "merged" / "yolo_obb" / "train" / "weights" / "best.pt"),
        "dataset": str(candidate / "data.yaml"),
        "epochs": 20,
        "imgsz": 1024,
        "device": 0,
        "workers": 0,
        "lr0": 0.001,
        "freeze": 10,
        "output_name": "finetune_clean_v1",
        "required_before_start": [
            "review.csv all rows have review_status=approved",
            "golden_test_manifest.json approved=true with reviewer, approver, approved_at",
            "fixed test list is configured and isolated from training",
        ],
    }
    safe_write_json(str(project / "quality" / "finetune_clean_v1.json"), finetune)
    return {"candidate": str(candidate), "golden": str(golden), "finetune": finetune}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-dir", default=".")
    print(json.dumps(prepare(os.path.abspath(parser.parse_args().project_dir)), ensure_ascii=False, indent=2))
