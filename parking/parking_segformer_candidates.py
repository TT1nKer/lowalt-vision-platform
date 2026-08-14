from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from parking_map.tile_georeference import mask_to_geographic_geometry, parse_block_filename


def _retained_mask(mask: np.ndarray, minimum_component_area_px: int) -> np.ndarray:
    count, labels, stats, _ = cv2.connectedComponentsWithStats((mask >= 128).astype(np.uint8), 8)
    retained = np.zeros(mask.shape, dtype=np.uint8)
    for component_id in range(1, count):
        if int(stats[component_id, cv2.CC_STAT_AREA]) >= minimum_component_area_px:
            retained[labels == component_id] = 255
    return retained


def _write_json(path: Path, value: dict[str, Any]) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


def export_positive_masks(
    *,
    images: Path,
    masks: Path,
    output_dir: Path,
    minimum_component_area_px: int = 100,
) -> dict[str, Any]:
    if minimum_component_area_px <= 0:
        raise ValueError("minimum component area must be positive")
    mask_paths = sorted(masks.glob("*_mask.png"))
    if not mask_paths:
        raise ValueError(f"no SegFormer masks found in {masks}")

    output_dir.mkdir(parents=True, exist_ok=True)
    aois = []
    totals = Counter()
    for mask_path in mask_paths:
        aoi_id = mask_path.stem.removesuffix("_mask")
        image_name = f"{aoi_id}.png"
        if not (images / image_name).is_file():
            raise FileNotFoundError(f"source image missing for mask: {image_name}")
        mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        if mask is None:
            raise ValueError(f"unreadable SegFormer mask: {mask_path}")
        retained = _retained_mask(mask, minimum_component_area_px)
        totals["masks"] += 1
        if not np.any(retained):
            totals["empty_or_below_minimum"] += 1
            continue

        converted = mask_to_geographic_geometry(
            retained,
            image_name,
            minimum_component_area=minimum_component_area_px,
            simplify_fraction=0.002,
        )
        block = parse_block_filename(image_name)
        target_id = f"{aoi_id}::segformer-parking"
        feature_collection = {
            "type": "FeatureCollection",
            "schema_version": 1,
            "aoi_id": aoi_id,
            "truth_status": "model_prediction_not_ground_truth",
            "features": [{
                "type": "Feature",
                "geometry": converted.geometry,
                "properties": {
                    "target_id": target_id,
                    "candidate_id": target_id,
                    "source": "UTEL-UIUC/SegFormer-large-parking",
                    "component_count": converted.component_count,
                    "mask_area_px": converted.mask_area_px,
                    "truth_status": "model_prediction_not_ground_truth",
                    "crs": "OGC:CRS84",
                },
            }],
        }
        aoi_output = output_dir / aoi_id
        aoi_output.mkdir(parents=True, exist_ok=True)
        _write_json(aoi_output / "mask_geometries.geojson", feature_collection)
        _write_json(aoi_output / "surface_hypothesis.geojson", {
            "type": "FeatureCollection",
            "schema_version": 1,
            "aoi_id": aoi_id,
            "truth_status": "model_prediction_not_ground_truth",
            "features": [{
                "type": "Feature",
                "geometry": converted.geometry,
                "properties": {
                    "source": "UTEL-UIUC/SegFormer-large-parking",
                    "qualified_anchor_candidate_ids": [target_id],
                    "truth_status": "model_prediction_not_ground_truth",
                    "crs": "OGC:CRS84",
                },
            }],
        })
        _write_json(aoi_output / "exclusion_layers.geojson", {
            "type": "FeatureCollection",
            "schema_version": 1,
            "aoi_id": aoi_id,
            "coverage_status": "not_available",
            "features": [],
        })
        aois.append({
            "aoi_id": aoi_id,
            "image": image_name,
            "tile_origin": {
                "zoom": block.zoom,
                "row": block.tile_row,
                "col": block.tile_col,
            },
            "block_grid": {"row": block.block_row, "col": block.block_col},
            "geo_cell": f"{block.block_row // 32}:{block.block_col // 32}",
            "strata": ["segformer_positive"],
            "evidence": {
                "target_id": target_id,
                "component_count": converted.component_count,
                "mask_area_px": converted.mask_area_px,
            },
        })
        totals["positive_aois"] += 1
        totals["components"] += converted.component_count
        totals["positive_pixels"] += converted.mask_area_px

    manifest = {
        "schema_version": 1,
        "dataset_version": "imagery_segformer_base",
        "truth_status": "model_prediction_not_ground_truth",
        "gsd_status": "pending_confirmation",
        "count": len(aois),
        "aois": aois,
    }
    _write_json(output_dir / "manifest.json", manifest)
    summary = dict(totals)
    _write_json(output_dir / "summary.json", summary)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Export positive SegFormer WMTS masks as existing AOI candidate packages"
    )
    parser.add_argument("--images", type=Path, required=True)
    parser.add_argument("--masks", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--minimum-component-area-px", type=int, default=100)
    args = parser.parse_args()
    summary = export_positive_masks(
        images=args.images,
        masks=args.masks,
        output_dir=args.output_dir,
        minimum_component_area_px=args.minimum_component_area_px,
    )
    print(json.dumps(summary, ensure_ascii=False))


if __name__ == "__main__":
    main()
