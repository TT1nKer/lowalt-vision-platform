from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
from pathlib import Path
import threading

from console.core import cfg_get, load_config
from console.pipeline1_sam3 import call_sam3_with_prompt
from lowalt_platform.services.secondary_analysis import ParkingSecondaryAnalyzer
from lowalt_platform.settings import PlatformSettings


def _candidate_ids(candidate_geojson: Path) -> list[str]:
    payload = json.loads(candidate_geojson.read_text(encoding="utf-8"))
    return sorted({str(feature["properties"]["aoi_id"]) for feature in payload.get("features") or []})


def main() -> None:
    parser = argparse.ArgumentParser(description="Run SAM3 evidence extraction inside parking candidates")
    parser.add_argument("--project-dir", type=Path, default=Path("."))
    parser.add_argument("--config", type=Path, default=Path("config.yaml"))
    parser.add_argument("--aoi-id")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--workers", type=int)
    args = parser.parse_args()

    project_root = args.project_dir.resolve()
    platform_settings = PlatformSettings.from_project(project_root)
    config_path = args.config if args.config.is_absolute() else project_root / args.config
    config = load_config(str(config_path))
    candidate_geojson = project_root / "quality" / "parking_direct_first_layers" / "parking_facility_candidates.geojson"
    image_root = platform_settings.image_root
    candidate_mask_root = platform_settings.mask_root
    output_root = project_root / "quality" / "parking_secondary_sam3"
    candidate_ids = [args.aoi_id] if args.aoi_id else _candidate_ids(candidate_geojson)
    if args.limit is not None:
        if args.limit <= 0:
            raise SystemExit("--limit must be positive")
        candidate_ids = candidate_ids[: args.limit]

    thread_local = threading.local()

    def analyzer() -> ParkingSecondaryAnalyzer:
        if not hasattr(thread_local, "analyzer"):
            client = lambda image_path, prompt: call_sam3_with_prompt(config, str(image_path), prompt)
            thread_local.analyzer = ParkingSecondaryAnalyzer(client, output_root)
        return thread_local.analyzer

    def analyze_aoi(aoi_id: str) -> tuple[str, str]:
        image_path = image_root / f"{aoi_id}.png"
        candidate_mask_path = candidate_mask_root / f"{aoi_id}_mask.png"
        if not image_path.is_file():
            raise FileNotFoundError(f"candidate image is missing: {image_path}")
        if not candidate_mask_path.is_file():
            raise FileNotFoundError(f"candidate mask is missing: {candidate_mask_path}")
        result = analyzer().analyze(aoi_id, image_path, candidate_mask_path=candidate_mask_path)
        return aoi_id, "resumed" if result.get("resumed") else "completed"

    workers = args.workers or min(4, int(cfg_get(config, "infer.batch_concurrency", 4)))
    completed = resumed = failed = 0
    with ThreadPoolExecutor(max_workers=max(1, workers)) as executor:
        futures = {executor.submit(analyze_aoi, aoi_id): aoi_id for aoi_id in candidate_ids}
        for index, future in enumerate(as_completed(futures), start=1):
            aoi_id = futures[future]
            try:
                _, status = future.result()
                completed += status == "completed"
                resumed += status == "resumed"
                print(f"[{index}/{len(candidate_ids)}] {aoi_id}: {status}", flush=True)
            except Exception as exc:
                failed += 1
                print(f"[{index}/{len(candidate_ids)}] {aoi_id}: failed: {exc}", flush=True)
    print(json.dumps({"total": len(candidate_ids), "completed": completed, "resumed": resumed, "failed": failed}, ensure_ascii=False))
    if failed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
