#!/usr/bin/env python3
"""Resumable Gemma pre-review for a golden OBB package.

The machine review is stored separately from human review.csv. It is useful
for prioritization and evidence, but it never creates human approval records.
"""

from __future__ import annotations

import argparse
import base64
import csv
import json
import os
import re
import threading
import urllib.error
import urllib.request
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path

from console.core import cfg_get, load_config, safe_write_json


PROMPT = """You are auditing an oriented-bounding-box annotation for a top-down parking-lot image.

You receive two images in this order:
1. ORIGINAL: clean source image.
2. ANNOTATED PREVIEW: the same image with colored oriented boxes and class labels.

Business classes:
- occupied: one parking space occupied by a vehicle
- empty: one visible empty parking space
- parking_area: the polygon covering the overall parking area; this is intentionally much larger than a single space

Audit the annotation, not the photographic quality. Look for clear missing parking spaces, duplicate boxes, wrong occupied/empty classes, boxes on non-parking objects, and severely misplaced boundaries. Dense scenes may contain many valid boxes. Do not reject merely because exact counting is difficult. Use needs_review when evidence is ambiguous.

File: {image}
Recorded objects: {objects}
Stratum: {stratum}
Known risk sample: {risk_sample}

Output ONLY valid JSON with this schema:
{{
  "verdict": "pass" | "needs_review" | "fail",
  "confidence": 0.0,
  "issues": ["missing_box" | "duplicate_box" | "wrong_class" | "false_positive" | "bad_boundary" | "other"],
  "evidence": "brief visible evidence",
  "estimated_problem_count": 0
}}
"""


def _data_url(path: Path) -> str:
    ext = path.suffix.lower()
    mime = "image/png" if ext == ".png" else "image/jpeg"
    return f"data:{mime};base64,{base64.b64encode(path.read_bytes()).decode('ascii')}"


def _parse_json(text: str) -> dict | None:
    if not text:
        return None
    match = re.search(r"```(?:json)?\s*(.*?)\s*```", text, re.DOTALL | re.IGNORECASE)
    candidate = match.group(1) if match else text[text.find("{"):text.rfind("}") + 1]
    try:
        value = json.loads(candidate)
        return value if isinstance(value, dict) else None
    except (TypeError, ValueError, json.JSONDecodeError):
        return None


def _call_gemma(cfg: dict, prompt: str, original: Path, preview: Path) -> tuple[dict, str]:
    url = str(cfg_get(cfg, "gemma.url", ""))
    model = str(cfg_get(cfg, "gemma.model", ""))
    api_key = str(cfg_get(cfg, "gemma.api_key", "") or "")
    timeout = int(cfg_get(cfg, "gemma.timeout", 120))
    schema = str(cfg_get(cfg, "gemma.schema", "openai")).lower()
    if schema != "openai":
        raise RuntimeError("golden Gemma audit currently requires gemma.schema=openai")
    content = [
        {"type": "text", "text": prompt},
        {"type": "image_url", "image_url": {"url": _data_url(original)}},
        {"type": "image_url", "image_url": {"url": _data_url(preview)}},
    ]
    body = {
        "model": model,
        "messages": [{"role": "user", "content": content}],
        "temperature": 0,
        "max_tokens": 300,
    }
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    request = urllib.request.Request(
        url, data=json.dumps(body).encode("utf-8"), headers=headers, method="POST"
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        payload = json.loads(response.read().decode("utf-8", errors="replace"))
    raw = payload["choices"][0]["message"]["content"]
    parsed = _parse_json(raw)
    if parsed is None:
        raise RuntimeError(f"Gemma returned invalid JSON: {raw[:300]}")
    verdict = str(parsed.get("verdict", "")).lower()
    if verdict not in {"pass", "needs_review", "fail"}:
        raise RuntimeError(f"Gemma returned invalid verdict: {verdict}")
    try:
        confidence = float(parsed.get("confidence", 0))
    except (TypeError, ValueError):
        confidence = 0.0
    parsed["verdict"] = verdict
    parsed["confidence"] = max(0.0, min(confidence, 1.0))
    parsed["issues"] = [str(issue) for issue in (parsed.get("issues") or [])]
    parsed["evidence"] = str(parsed.get("evidence", ""))[:1000]
    try:
        parsed["estimated_problem_count"] = max(0, int(parsed.get("estimated_problem_count", 0)))
    except (TypeError, ValueError):
        parsed["estimated_problem_count"] = 0
    return parsed, raw


def _load_rows(package: Path) -> list[dict]:
    with (package / "review.csv").open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def _load_completed(path: Path) -> dict[str, dict]:
    completed = {}
    if not path.is_file():
        return completed
    for line in path.read_text(encoding="utf-8").splitlines():
        try:
            item = json.loads(line)
        except json.JSONDecodeError:
            continue
        if item.get("image") and item.get("status") == "completed":
            completed[item["image"]] = item
    return completed


def run_gemma_review(package_dir: str, cfg: dict, *, workers: int = 1, limit: int | None = None) -> dict:
    package = Path(package_dir).resolve()
    journal = package / "gemma_review.jsonl"
    rows = _load_rows(package)
    completed = _load_completed(journal)
    pending = [row for row in rows if row["image"] not in completed]
    if limit is not None:
        pending = pending[:max(0, limit)]
    lock = threading.Lock()

    def audit(row: dict) -> dict:
        image = row["image"]
        original = package / "images" / image
        preview = package / "previews" / f"{Path(image).stem}.jpg"
        started = datetime.now().isoformat()
        try:
            parsed, raw = _call_gemma(cfg, PROMPT.format(**row), original, preview)
            return {
                "schema_version": 1, "status": "completed", "source": "gemma",
                "image": image, "started_at": started, "reviewed_at": datetime.now().isoformat(),
                **parsed, "raw": raw,
            }
        except Exception as exc:
            return {
                "schema_version": 1, "status": "failed", "source": "gemma",
                "image": image, "started_at": started, "reviewed_at": datetime.now().isoformat(),
                "error": str(exc)[:1000],
            }

    if pending:
        with ThreadPoolExecutor(max_workers=max(1, workers)) as executor:
            futures = {executor.submit(audit, row): row for row in pending}
            done_count = 0
            for future in as_completed(futures):
                result = future.result()
                with lock, journal.open("a", encoding="utf-8") as handle:
                    handle.write(json.dumps(result, ensure_ascii=False) + "\n")
                    handle.flush()
                    os.fsync(handle.fileno())
                done_count += 1
                print(f"@@PROGRESS {done_count} {len(pending)} {result['image']} {result['status']}", flush=True)

    latest = {}
    if journal.is_file():
        for line in journal.read_text(encoding="utf-8").splitlines():
            try:
                item = json.loads(line)
            except json.JSONDecodeError:
                continue
            if item.get("image"):
                latest[item["image"]] = item
    ordered = sorted(latest.values(), key=lambda item: (
        {"fail": 0, "needs_review": 1, "pass": 2}.get(item.get("verdict"), 3),
        -float(item.get("confidence", 0)), item["image"],
    ))
    csv_path = package / "gemma_review.csv"
    fields = ["image", "status", "verdict", "confidence", "issues", "estimated_problem_count", "evidence", "reviewed_at", "error"]
    with csv_path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for item in ordered:
            writer.writerow({key: json.dumps(item.get(key), ensure_ascii=False) if key == "issues" else item.get(key, "") for key in fields})
    verdicts = Counter(item.get("verdict") for item in ordered if item.get("status") == "completed")
    summary = {
        "schema_version": 1, "generated_at": datetime.now().isoformat(), "source": "gemma",
        "package_dir": str(package), "total_images": len(rows), "attempted_images": len(latest),
        "completed_images": sum(item.get("status") == "completed" for item in ordered),
        "failed_images": sum(item.get("status") == "failed" for item in ordered),
        "verdicts": dict(verdicts), "human_approval_created": False,
        "issue_counts": dict(Counter(
            issue for item in ordered if item.get("status") == "completed"
            for issue in (item.get("issues") or [])
        )),
        "priority_review_count": sum(item.get("verdict") in {"fail", "needs_review"} for item in ordered),
        "priority_review_images": [item["image"] for item in ordered if item.get("verdict") in {"fail", "needs_review"}],
        "journal": str(journal), "csv": str(csv_path),
    }
    safe_write_json(str(package / "gemma_review_summary.json"), summary)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-dir", default=".")
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--package", default="quality/golden_review_v1")
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--limit", type=int)
    args = parser.parse_args()
    project = Path(args.project_dir).resolve()
    config_path = Path(args.config)
    if not config_path.is_absolute():
        config_path = project / config_path
    package = Path(args.package)
    if not package.is_absolute():
        package = project / package
    cfg = load_config(str(config_path))
    summary = run_gemma_review(str(package), cfg, workers=args.workers, limit=args.limit)
    printable = {key: value for key, value in summary.items() if key != "priority_review_images"}
    print(json.dumps(printable, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
