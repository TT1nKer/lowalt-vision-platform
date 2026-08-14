from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Iterable

import cv2
import numpy as np


VEHICLE_CLASSES = {"car", "bus", "truck"}


def detections_inside_mask(
    detections: Iterable[dict[str, Any]],
    candidate_mask: np.ndarray,
) -> list[dict[str, Any]]:
    if candidate_mask.ndim == 3 and candidate_mask.shape[2] == 1:
        candidate_mask = candidate_mask[:, :, 0]
    if candidate_mask.ndim != 2:
        raise ValueError(f"candidate mask must be grayscale, got shape {candidate_mask.shape}")
    selected = []
    for detection in detections:
        try:
            box = np.asarray(detection.get("bbox"), dtype=np.float64)
        except (TypeError, ValueError):
            continue
        if box.shape != (4,) or not np.isfinite(box).all():
            continue
        center_x = int(round(float((box[0] + box[2]) / 2)))
        center_y = int(round(float((box[1] + box[3]) / 2)))
        if (
            0 <= center_x < candidate_mask.shape[1]
            and 0 <= center_y < candidate_mask.shape[0]
            and candidate_mask[center_y, center_x] >= 128
        ):
            selected.append(detection)
    return selected


def _vehicle_detections(prediction: Any) -> list[dict[str, Any]]:
    detections = []
    for box in prediction.boxes:
        class_name = str(prediction.names[int(box.cls.item())])
        if class_name not in VEHICLE_CLASSES:
            continue
        detections.append({
            "bbox": list(map(float, box.xyxy[0].tolist())),
            "confidence": float(box.conf.item()),
            "class_name": class_name,
        })
    return detections


def screen_vehicle_candidates(
    *,
    manifest_path: Path,
    image_dir: Path,
    mask_dir: Path,
    weights: Path,
    output_manifest: Path,
    device: str,
    image_size: int = 1024,
    batch_size: int = 16,
) -> dict[str, int]:
    from ultralytics import YOLO

    if image_size <= 0 or batch_size <= 0:
        raise ValueError("image and batch sizes must be positive")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    aois = list(manifest.get("aois") or [])
    if not aois:
        raise ValueError("manifest contains no AOIs")
    if not weights.is_file():
        raise FileNotFoundError(f"detector weights not found: {weights}")

    model = YOLO(str(weights))
    selected_aois = []
    selected_detection_count = 0
    for chunk_start in range(0, len(aois), batch_size):
        chunk = aois[chunk_start:chunk_start + batch_size]
        image_paths = [image_dir / str(aoi["image"]) for aoi in chunk]
        missing = [path for path in image_paths if not path.is_file()]
        if missing:
            raise FileNotFoundError(f"source image missing: {missing[0]}")
        predictions = list(model.predict(
            source=[str(path) for path in image_paths],
            imgsz=image_size,
            conf=0.05,
            iou=0.7,
            batch=len(chunk),
            device=device,
            half=device.lower() != "cpu",
            verbose=False,
        ))
        if len(predictions) != len(chunk):
            raise RuntimeError("detector prediction count does not match manifest chunk")
        for aoi, prediction in zip(chunk, predictions, strict=True):
            aoi_id = str(aoi["aoi_id"])
            mask_path = mask_dir / f"{aoi_id}_mask.png"
            candidate_mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
            if candidate_mask is None:
                raise FileNotFoundError(f"candidate mask missing or unreadable: {mask_path}")
            inside = detections_inside_mask(_vehicle_detections(prediction), candidate_mask)
            if not inside:
                continue
            selected_aois.append({
                **aoi,
                "vehicle_screen": {
                    "status": "selected_for_tiled_confirmation",
                    "detector": weights.name,
                    "detection_count": len(inside),
                    "detections": inside,
                    "truth_status": "single_model_prefilter_not_ground_truth",
                },
            })
            selected_detection_count += len(inside)
        processed = min(chunk_start + len(chunk), len(aois))
        if processed % 512 == 0:
            print(json.dumps({
                "screened_aois": processed,
                "selected_aois_so_far": len(selected_aois),
            }), flush=True)

    output = {
        **{key: value for key, value in manifest.items() if key not in {"aois", "count"}},
        "count": len(selected_aois),
        "selection_method": "single full-image detector prefilter inside SegFormer candidate",
        "truth_status": "model_prefilter_not_ground_truth",
        "aois": selected_aois,
    }
    output_manifest.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = output_manifest.with_suffix(output_manifest.suffix + ".tmp")
    temporary_path.write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary_path.replace(output_manifest)
    summary = {
        "screened_aois": len(aois),
        "selected_aois": len(selected_aois),
        "selected_detections": selected_detection_count,
    }
    summary_path = output_manifest.with_name(f"{output_manifest.stem}_summary.json")
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Cheap full-image vehicle prefilter before tiled dual-model confirmation"
    )
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--images", type=Path, required=True)
    parser.add_argument("--masks", type=Path, required=True)
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--output-manifest", type=Path, required=True)
    parser.add_argument("--device", default="0")
    parser.add_argument("--imgsz", type=int, default=1024)
    parser.add_argument("--batch-size", type=int, default=16)
    args = parser.parse_args()
    summary = screen_vehicle_candidates(
        manifest_path=args.manifest,
        image_dir=args.images,
        mask_dir=args.masks,
        weights=args.weights,
        output_manifest=args.output_manifest,
        device=args.device,
        image_size=args.imgsz,
        batch_size=args.batch_size,
    )
    print(json.dumps(summary, ensure_ascii=False))


if __name__ == "__main__":
    main()
