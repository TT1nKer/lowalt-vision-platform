#!/usr/bin/env python3
"""Create a non-production inference profile from topology calibration evidence."""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

from console.core import safe_write_json
from console.release_gate import sha256


def create(project_dir: str = ".") -> dict:
    root = Path(project_dir).resolve(); shadow = root / "quality/business_shadow_v1"
    calibration = json.loads((shadow / "topology_calibration.json").read_text(encoding="utf-8"))
    metric = calibration["recommended_metrics"]
    classes = metric.get("classes", {})
    required = {"occupied": (0.90, 0.85), "empty": (0.90, 0.85)}
    class_pass = all(classes.get(name, {}).get("precision", 0) >= p and classes.get(name, {}).get("recall", 0) >= r
                     for name, (p, r) in required.items())
    candidate = root / "quality/official_finetune/finetune_official_v1/weights/best.pt"
    profile = {
        "schema_version": 1, "created_at": datetime.now(timezone.utc).isoformat(),
        "status": "candidate_shadow_only" if class_pass else "rejected",
        "model": {"path": str(candidate), "sha256": sha256(candidate)},
        "inference": {"imgsz": 640, "confidence": calibration["recommended_confidence"], "iou": 0.7},
        "validation": {"source": str(shadow / "topology_calibration.json"),
                       "evaluated_images": calibration["evaluated_images"],
                       "geometry": {k: metric[k] for k in ("precision", "recall", "f1")},
                       "classes": classes, "thresholds": required},
        "limitations": ["Validation uses same-day nearby-frame parking topology.",
                        "This profile is not production until 30-image human acceptance and independent approval pass."],
        "rollback": "Keep existing production model and configuration unchanged.",
    }
    safe_write_json(str(root / "quality/calibrated_candidate_profile_v1.json"), profile)
    return profile


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser(); p.add_argument("project_dir", nargs="?", default=".")
    print(json.dumps(create(p.parse_args().project_dir), ensure_ascii=False, indent=2))
