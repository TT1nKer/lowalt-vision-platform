from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import cv2
import numpy as np

from console.core import ensure_dir, load_config, run_dirs, safe_load_json, safe_write_json
from console.pipeline2_review import build_index


def _write_component_masks(source_mask: Path, destination: Path, image_name: str) -> list[dict]:
    mask = cv2.imread(str(source_mask), cv2.IMREAD_GRAYSCALE)
    if mask is None:
        raise ValueError(f"cannot read mask: {source_mask}")
    binary = (mask > 0).astype(np.uint8)
    count, labels, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
    components = []
    for component_id in range(1, count):
        x, y, width, height, area = (int(value) for value in stats[component_id])
        if area <= 0:
            continue
        mask_name = f"{Path(image_name).stem}_t{len(components)}.png"
        mask_path = destination / mask_name
        if count == 2:
            if not mask_path.exists():
                os.link(source_mask, mask_path)
        else:
            component = np.where(labels == component_id, 255, 0).astype(np.uint8)
            temporary_path = mask_path.with_name(f".{mask_path.name}.tmp.png")
            if not cv2.imwrite(str(temporary_path), component):
                raise OSError(f"cannot write mask: {temporary_path}")
            os.replace(temporary_path, mask_path)
        components.append({
            "bbox": [float(x), float(y), float(x + width), float(y + height)],
            "mask_file": mask_name,
            "area": area,
        })
    return components


def import_candidates(project_root: Path, config_path: Path, candidate_path: Path) -> dict:
    config = load_config(str(config_path))
    directories = run_dirs(str(project_root), config)
    image_root = project_root / "imagery"
    source_mask_root = project_root / "quality" / "parkseg_imagery_masks"
    result_root = Path(directories["merged"])
    mask_root = Path(directories["mask"])
    review_root = Path(directories["review"])
    for path in (result_root, mask_root, review_root):
        ensure_dir(str(path))

    collection = json.loads(candidate_path.read_text(encoding="utf-8"))
    state_path = Path(directories["state"])
    existing_state = safe_load_json(str(state_path), {}) or {}
    initial_state = dict(existing_state)
    image_count = target_count = accepted = rejected = missing = 0

    for feature in collection.get("features") or []:
        properties = feature.get("properties") or {}
        aoi_id = str(properties.get("aoi_id") or "")
        image_name = f"{aoi_id}.png"
        source_mask = source_mask_root / f"{aoi_id}_mask.png"
        if not aoi_id or not (image_root / image_name).is_file() or not source_mask.is_file():
            missing += 1
            continue

        components = _write_component_masks(source_mask, mask_root, image_name)
        targets = []
        for target_index, component in enumerate(components):
            target_id = f"{image_name}::{target_index}"
            targets.append({
                "source_target_id": target_id,
                "class_name": "parking_area",
                "confidence": None,
                "bbox": component["bbox"],
                "mask_file": component["mask_file"],
                "source_prompt": "UTEL-UIUC/SegFormer-large-parking",
            })
            if target_id in existing_state:
                record = initial_state[target_id]
                if record.get("source") in {"segformer_vehicle_gate", "segformer_exclusion_gate"}:
                    record["is_auto"] = True
                    accepted += record.get("label") == "accept"
                    rejected += record.get("label") == "reject"
                continue
            gate_status = properties.get("gate_status")
            if gate_status == "accepted" or properties.get("support_level") == "vehicle_row_supported":
                initial_state[target_id] = {
                    "label": "accept",
                    "note": "Imported teacher label with vehicle-row support",
                    "source": "segformer_vehicle_gate",
                    "is_auto": True,
                }
                accepted += 1
            elif gate_status == "rejected":
                initial_state[target_id] = {
                    "label": "reject",
                    "note": "Rejected by building/vegetation overlap gate",
                    "source": "segformer_exclusion_gate",
                    "is_auto": True,
                }
                rejected += 1
        safe_write_json(str(result_root / f"{aoi_id}.json"), {
            "source_file": image_name,
            "text_prompt": "parking area",
            "prompt_mode": "imported_teacher",
            "targets": targets,
        })
        image_count += 1
        target_count += len(targets)

    safe_write_json(str(state_path), initial_state)
    index_count = build_index(str(project_root), config)
    summary = {
        "source": str(candidate_path),
        "images": image_count,
        "targets": target_count,
        "indexed": index_count,
        "initial_accepted": accepted,
        "initial_rejected": rejected,
        "unreviewed": target_count - len(initial_state),
        "missing_inputs": missing,
        "mask_storage": "hardlink_for_single_component",
        "truth_status": "teacher_labels_not_independent_ground_truth",
    }
    safe_write_json(str(Path(directories["base"]) / "import_summary.json"), summary)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Import existing parking masks into the legacy review pipeline")
    parser.add_argument("--project-dir", type=Path, default=Path("."))
    parser.add_argument("--config", type=Path, default=Path("config.yaml"))
    parser.add_argument("--candidates", type=Path)
    args = parser.parse_args()
    project_root = args.project_dir.resolve()
    config_path = args.config if args.config.is_absolute() else project_root / args.config
    candidate_path = args.candidates or project_root / "quality" / "parking_candidate_gate" / "parking_candidates.geojson"
    print(json.dumps(import_candidates(project_root, config_path, candidate_path), ensure_ascii=False))


if __name__ == "__main__":
    main()
