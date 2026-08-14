from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import shutil
from typing import Any

import cv2

from parking_map.e1_extraction import build_aoi_image_evidence
from parking_map.e1_runner import evaluate_aoi_candidates
from parking_map.image_evidence import geographic_geometry_to_mask
from parking_map.tiled_detection import merge_overlapping_detections, tile_windows


_VEHICLE_CLASSES = {"car", "bus", "truck"}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _vehicle_detections(prediction: Any) -> list[dict[str, Any]]:
    detections = []
    for box in prediction.boxes:
        class_name = str(prediction.names[int(box.cls.item())])
        if class_name not in _VEHICLE_CLASSES:
            continue
        detections.append({
            "bbox": list(map(float, box.xyxy[0].tolist())),
            "confidence": float(box.conf.item()),
            "class_name": class_name,
        })
    return detections


def _tiled_vehicle_detections(
    model: Any,
    image,
    *,
    image_size: int,
    tile_size: int,
    tile_overlap: int,
    device: str,
) -> list[dict[str, Any]]:
    windows = tile_windows(
        image.shape[1],
        image.shape[0],
        tile_size=tile_size,
        overlap=tile_overlap,
    )
    crops = [image[top:bottom, left:right] for left, top, right, bottom in windows]
    predictions = list(model.predict(
        source=crops,
        imgsz=image_size,
        conf=0.05,
        iou=0.7,
        batch=len(crops),
        device=device,
        half=device.lower() != "cpu",
        verbose=False,
    ))
    if len(predictions) != len(windows):
        raise RuntimeError("tiled detector prediction count does not match windows")
    detections = []
    for (left, top, _right, _bottom), prediction in zip(windows, predictions, strict=True):
        for detection in _vehicle_detections(prediction):
            x1, y1, x2, y2 = detection["bbox"]
            detections.append({
                **detection,
                "bbox": [x1 + left, y1 + top, x2 + left, y2 + top],
            })
    return merge_overlapping_detections(detections, maximum_iou=0.5)


def _write_overlay(
    image,
    image_name: str,
    evaluated: dict[str, Any],
    agreed_vehicles: list[dict[str, Any]],
    output_path: Path,
) -> None:
    overlay = image.copy()
    colors = {"auto_accept": (80, 200, 80), "auto_reject": (60, 60, 230), "abstain": (0, 190, 255)}
    for feature in evaluated["features"]:
        mask = geographic_geometry_to_mask(feature["geometry"], image_name, image.shape[:2])
        contours, _ = cv2.findContours(mask, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        color = colors[feature["properties"]["decision"]]
        cv2.drawContours(overlay, contours, -1, color, 2)
    for detection in agreed_vehicles:
        left, top, right, bottom = map(round, detection["bbox"])
        cv2.rectangle(overlay, (left, top), (right, bottom), (255, 120, 0), 2)
    cv2.imwrite(str(output_path), overlay)


def run_e1_evidence(
    *,
    manifest_path: Path,
    evidence_root: Path,
    image_dir: Path,
    first_weights: Path,
    second_weights: Path,
    output_dir: Path,
    device: str,
    image_size: int,
    tile_size: int = 0,
    tile_overlap: int = 0,
    exclusion_root: Path | None = None,
    write_overlays: bool = True,
) -> dict[str, Any]:
    from ultralytics import YOLO

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    aois = manifest.get("aois") or []
    if not aois:
        raise ValueError("manifest contains no AOIs")
    image_paths = [image_dir / str(aoi["image"]) for aoi in aois]
    missing = [str(path) for path in image_paths if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"missing AOI image: {missing[0]}")
    for path in (first_weights, second_weights):
        if not path.is_file():
            raise FileNotFoundError(f"missing detector weights: {path}")

    first_model = YOLO(str(first_weights))
    second_model = YOLO(str(second_weights))
    if tile_size:
        if tile_overlap < 0 or tile_overlap >= tile_size:
            raise ValueError("tile overlap must be non-negative and smaller than tile size")
        first_detections = []
        second_detections = []
        for image_path in image_paths:
            image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
            if image is None:
                raise ValueError(f"unreadable AOI image: {image_path}")
            first_detections.append(_tiled_vehicle_detections(
                first_model,
                image,
                image_size=image_size,
                tile_size=tile_size,
                tile_overlap=tile_overlap,
                device=device,
            ))
            second_detections.append(_tiled_vehicle_detections(
                second_model,
                image,
                image_size=image_size,
                tile_size=tile_size,
                tile_overlap=tile_overlap,
                device=device,
            ))
    else:
        predict_options = {
            "source": [str(path) for path in image_paths],
            "imgsz": image_size,
            "conf": 0.05,
            "iou": 0.7,
            "batch": 8,
            "device": device,
            "half": device.lower() != "cpu",
            "verbose": False,
        }
        first_predictions = list(first_model.predict(**predict_options))
        second_predictions = list(second_model.predict(**predict_options))
        if len(first_predictions) != len(aois) or len(second_predictions) != len(aois):
            raise RuntimeError("detector prediction count does not match AOI count")
        first_detections = [_vehicle_detections(prediction) for prediction in first_predictions]
        second_detections = [_vehicle_detections(prediction) for prediction in second_predictions]

    output_dir.mkdir(parents=True, exist_ok=True)
    decision_counts = Counter()
    totals = Counter()
    for aoi, image_path, first_image_detections, second_image_detections in zip(
        aois, image_paths, first_detections, second_detections, strict=True
    ):
        aoi_id = str(aoi["aoi_id"])
        package_dir = evidence_root / aoi_id
        candidates_path = package_dir / "mask_geometries.geojson"
        candidates = json.loads(candidates_path.read_text(encoding="utf-8"))
        image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if image is None:
            raise ValueError(f"unreadable AOI image: {image_path}")
        evidence = build_aoi_image_evidence(
            aoi_id,
            str(aoi["image"]),
            image,
            candidates,
            first_image_detections,
            second_image_detections,
            detector_source_id=(
                f"{first_weights.stem}+{second_weights.stem}-coco-agreement-"
                f"{'tiled' + str(tile_size) if tile_size else 'full'}-v1"
            ),
        )
        evaluated = evaluate_aoi_candidates(aoi_id, candidates, evidence)
        aoi_output = output_dir / aoi_id
        aoi_output.mkdir(parents=True, exist_ok=True)
        (aoi_output / "evidence_layers.json").write_text(
            json.dumps(evidence, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        (aoi_output / "parking_candidates.geojson").write_text(
            json.dumps(evaluated, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        exclusions = (
            exclusion_root / aoi_id / "exclusion_layers.geojson"
            if exclusion_root is not None
            else package_dir / "exclusion_layers.geojson"
        )
        if exclusions.is_file():
            shutil.copy2(exclusions, aoi_output / exclusions.name)
        if write_overlays:
            _write_overlay(
                image,
                str(aoi["image"]),
                evaluated,
                evidence["agreed_vehicle_detections"],
                aoi_output / "evidence_overlay.jpg",
            )
        decision_counts.update(evaluated["summary"])
        totals["aois"] += 1
        totals["candidates"] += len(evaluated["decisions"])
        totals["agreed_vehicle_detections"] += len(evidence["agreed_vehicle_detections"])
        for measurement in evidence["candidate_measurements"]:
            totals["marking_strength_at_least_0_6"] += measurement["marking_strength"] >= 0.6
            totals["vehicle_arrangement_at_least_0_6"] += measurement["vehicle_arrangement_strength"] >= 0.6
            totals["both_supports_at_least_0_6"] += (
                measurement["marking_strength"] >= 0.6
                and measurement["vehicle_arrangement_strength"] >= 0.6
            )

    summary = {
        "schema_version": 1,
        "truth_status": "evidence_only_not_ground_truth",
        "source_manifest": str(manifest_path),
        "models": [
            {"path": str(first_weights), "sha256": _sha256(first_weights)},
            {"path": str(second_weights), "sha256": _sha256(second_weights)},
        ],
        "inference": {
            "device": device,
            "image_size": image_size,
            "tile_size": tile_size or None,
            "tile_overlap": tile_overlap if tile_size else None,
            "confidence": 0.05,
            "iou": 0.7,
            "overlays_written": write_overlays,
        },
        "totals": dict(totals),
        "decisions": dict(decision_counts),
        "gate_status": "blocked_missing_exclusion_coverage",
        "limitations": [
            "No independent parking truth is present.",
            "Two COCO detectors measure model agreement and share a training-domain family.",
            "No complete building, road, vegetation or water exclusion layer is available.",
        ],
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Build bounded E1 parking evidence for selected WMTS AOIs")
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--evidence-root", type=Path, required=True)
    parser.add_argument("--images", type=Path, required=True)
    parser.add_argument("--first-weights", type=Path, required=True)
    parser.add_argument("--second-weights", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="0")
    parser.add_argument("--imgsz", type=int, default=1024)
    parser.add_argument("--tile-size", type=int, default=0)
    parser.add_argument("--tile-overlap", type=int, default=0)
    parser.add_argument("--exclusion-root", type=Path)
    parser.add_argument("--no-overlays", action="store_true")
    args = parser.parse_args()
    summary = run_e1_evidence(
        manifest_path=args.manifest,
        evidence_root=args.evidence_root,
        image_dir=args.images,
        first_weights=args.first_weights,
        second_weights=args.second_weights,
        output_dir=args.output_dir,
        device=args.device,
        image_size=args.imgsz,
        tile_size=args.tile_size,
        tile_overlap=args.tile_overlap,
        exclusion_root=args.exclusion_root,
        write_overlays=not args.no_overlays,
    )
    print(json.dumps(summary, ensure_ascii=False))


if __name__ == "__main__":
    main()
