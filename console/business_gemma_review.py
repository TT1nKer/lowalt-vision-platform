#!/usr/bin/env python3
"""Resumable Gemma pre-review for the 30-image business acceptance set."""
from __future__ import annotations

import csv
import json
import os
import threading
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path

from console.core import load_config, safe_write_json
from console.golden_gemma_review import _call_gemma


PROMPT = """You are performing machine pre-review of a parking-space detector.

Image 1 is the clean business image. Image 2 is the candidate output; it may be a side-by-side comparison or a single annotated candidate image. Audit only the candidate output. RED boxes labeled O are occupied, GREEN boxes labeled E are empty, and GRAY boxes labeled ? are unknown. Unknown is a safe abstention: do not count it as an occupied/empty FP or FN, but mention excessive unknown coverage. When YELLOW monitored-topology boxes are shown, ONLY those yellow spaces are in the business scope. Never count visible spaces or vehicles outside monitored topology as false negatives.

Classes:
- occupied: one parking space containing a vehicle
- empty: one visible empty parking space

Count obvious candidate true positives, false positives, and false negatives for each class. A wrong occupied/empty class is an FP for the predicted class and an FN for the true class. Mark critical_fp when a false detection could directly cause an unsafe or materially wrong parking-availability action. If exact counting is ambiguous, use needs_review and conservative estimates.

File: {image}
Candidate detections: {candidate_detections}

Output ONLY JSON:
{{
  "verdict": "pass" | "needs_review" | "fail",
  "confidence": 0.0,
  "issues": ["missing_space" | "wrong_class" | "false_positive" | "bad_boundary" | "critical_false_positive" | "other"],
  "evidence": "brief visible evidence and the areas a human should check",
  "estimated_problem_count": 0,
  "occupied_tp": 0, "occupied_fp": 0, "occupied_fn": 0,
  "empty_tp": 0, "empty_fp": 0, "empty_fn": 0,
  "critical_fp": 0
}}
"""

COUNT_FIELDS = ("occupied_tp", "occupied_fp", "occupied_fn", "empty_tp", "empty_fp", "empty_fn", "critical_fp")


def _completed(path: Path) -> dict[str, dict]:
    result = {}
    if not path.is_file(): return result
    for line in path.read_text(encoding="utf-8").splitlines():
        try: item = json.loads(line)
        except json.JSONDecodeError: continue
        if item.get("image") and item.get("status") == "completed": result[item["image"]] = item
    return result


def run(project_dir: str, config_path: str, workers: int = 2,
        shadow_dir: str = "quality/business_shadow_v1", reset: bool = False) -> dict:
    root = Path(project_dir).resolve()
    shadow = Path(shadow_dir)
    if not shadow.is_absolute(): shadow = root / shadow
    report = json.loads((shadow / "shadow_report.json").read_text(encoding="utf-8"))
    rows = report.get("records", [])[:30]
    source = Path(report["source_dir"])
    journal = shadow / "gemma_business_review.jsonl"
    if reset and journal.is_file(): journal.unlink()
    done = _completed(journal)
    pending = [r for r in rows if r["image"] not in done]
    cfg = load_config(config_path)
    lock = threading.Lock()

    def audit(row: dict) -> dict:
        started = datetime.now().isoformat()
        try:
            parsed, raw = _call_gemma(cfg, PROMPT.format(
                image=row["image"], candidate_detections=row["candidate"]["detections"]),
                source / row["image"], shadow / row["comparison"])
            for field in COUNT_FIELDS:
                try: parsed[field] = max(0, int(parsed.get(field, 0)))
                except (TypeError, ValueError): parsed[field] = 0
            return {"schema_version": 1, "status": "completed", "source": "gemma",
                    "image": row["image"], "comparison": row["comparison"],
                    "started_at": started, "reviewed_at": datetime.now().isoformat(), **parsed, "raw": raw}
        except Exception as exc:
            return {"schema_version": 1, "status": "failed", "source": "gemma",
                    "image": row["image"], "comparison": row["comparison"],
                    "started_at": started, "reviewed_at": datetime.now().isoformat(), "error": str(exc)[:1000]}

    with ThreadPoolExecutor(max_workers=max(1, workers)) as pool:
        futures = [pool.submit(audit, row) for row in pending]
        for index, future in enumerate(as_completed(futures), 1):
            item = future.result()
            with lock, journal.open("a", encoding="utf-8") as f:
                f.write(json.dumps(item, ensure_ascii=False) + "\n"); f.flush(); os.fsync(f.fileno())
            print(f"@@PROGRESS {index} {len(pending)} {item['image']} {item['status']}", flush=True)

    latest = _completed(journal)
    ordered = [latest.get(row["image"], {"image": row["image"], "status": "missing"}) for row in rows]
    fields = ["image", "status", "verdict", "confidence", "issues", "evidence", *COUNT_FIELDS, "reviewed_at", "error"]
    with (shadow / "gemma_business_review.csv").open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=fields); writer.writeheader()
        for item in ordered:
            writer.writerow({k: json.dumps(item.get(k), ensure_ascii=False) if k == "issues" else item.get(k, "") for k in fields})
    summary = {"schema_version": 1, "generated_at": datetime.now().isoformat(), "source": "gemma",
               "model": cfg.get("gemma", {}).get("model"), "total_images": len(rows),
               "completed_images": sum(x.get("status") == "completed" for x in ordered),
               "failed_images": sum(x.get("status") == "failed" for x in ordered),
               "verdicts": dict(Counter(x.get("verdict") for x in ordered if x.get("status") == "completed")),
               "human_approval_created": False, "journal": str(journal)}
    safe_write_json(str(shadow / "gemma_business_review_summary.json"), summary)
    return summary


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser(); p.add_argument("--project-dir", default="."); p.add_argument("--config", default="config.yaml"); p.add_argument("--workers", type=int, default=2); p.add_argument("--shadow-dir", default="quality/business_shadow_v1"); p.add_argument("--reset", action="store_true")
    a = p.parse_args(); config = Path(a.config); config = config if config.is_absolute() else Path(a.project_dir).resolve() / config
    print(json.dumps(run(a.project_dir, str(config), a.workers, a.shadow_dir, a.reset), ensure_ascii=False, indent=2))
