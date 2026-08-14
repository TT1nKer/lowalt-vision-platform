from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from parking_map.e2_map_builder import AoiFunctionalMapResult, build_aoi_functional_map
from parking_map.e2_runner import RevalidatedBandSeed
from parking_map.image_evidence import geographic_geometry_to_mask
from parking_map.tile_georeference import mask_to_geographic_geometry
from parking_map.vehicle_rows import build_vehicle_rows


@dataclass(frozen=True)
class CleanedParkingMask:
    cleaned_mask: np.ndarray
    exclusion_mask: np.ndarray
    removed_pixels: int
    component_count: int


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def merge_exclusion_layers(
    original: dict[str, Any],
    additional: dict[str, Any] | None,
) -> dict[str, Any]:
    return {
        "type": "FeatureCollection",
        "features": [
            *(original.get("features") or []),
            *((additional or {}).get("features") or []),
        ],
    }


def resolve_image_paths(images: Path, manifest_path: Path | None) -> list[Path]:
    if manifest_path is None:
        return sorted(images.glob("*.png")) + sorted(images.glob("*.jpg"))
    image_root = images.resolve()
    manifest = _load_json(manifest_path)
    result = []
    for aoi in manifest.get("aois") or []:
        image_name = str(aoi.get("image") or "")
        image_path = images / image_name
        resolved_path = image_path.resolve()
        if not image_name or resolved_path.parent != image_root:
            raise ValueError(f"unsafe manifest image path: {image_name}")
        if not resolved_path.is_file():
            raise FileNotFoundError(f"manifest image does not exist: {resolved_path}")
        result.append(image_path)
    if not result:
        raise ValueError("manifest contains no AOI images")
    return result


def clean_parking_mask(
    raw_mask: np.ndarray,
    exclusion_layers: dict[str, Any],
    image_name: str,
    *,
    minimum_component_area_px: int = 100,
) -> CleanedParkingMask:
    if raw_mask.ndim != 2 or minimum_component_area_px <= 0:
        raise ValueError("raw_mask must be grayscale and minimum area must be positive")

    raw = raw_mask >= 128
    exclusion = np.zeros(raw.shape, dtype=np.uint8)
    for feature in exclusion_layers.get("features") or []:
        kind = str((feature.get("properties") or {}).get("exclusion_kind") or "")
        if kind not in {"building", "public_road"}:
            continue
        feature_mask = geographic_geometry_to_mask(
            feature.get("geometry") or {},
            image_name,
            raw.shape,
        )
        exclusion[feature_mask != 0] = 255

    candidate = (raw & (exclusion == 0)).astype(np.uint8)
    count, labels, stats, _ = cv2.connectedComponentsWithStats(candidate, 8)
    cleaned = np.zeros(raw.shape, dtype=np.uint8)
    component_count = 0
    for component_id in range(1, count):
        if int(stats[component_id, cv2.CC_STAT_AREA]) < minimum_component_area_px:
            continue
        cleaned[labels == component_id] = 255
        component_count += 1

    return CleanedParkingMask(
        cleaned_mask=cleaned,
        exclusion_mask=exclusion,
        removed_pixels=int(np.count_nonzero(raw) - np.count_nonzero(cleaned)),
        component_count=component_count,
    )


def _surface_from_cleaned_mask(
    cleaned_mask: np.ndarray,
    image_name: str,
    previous_surface: dict[str, Any],
    minimum_component_area_px: int,
) -> dict[str, Any]:
    qualified_ids = sorted({
        str(candidate_id)
        for feature in previous_surface.get("features") or []
        for candidate_id in (feature.get("properties") or {}).get(
            "qualified_anchor_candidate_ids", []
        )
    })
    converted = mask_to_geographic_geometry(
        cleaned_mask,
        image_name,
        minimum_component_area=minimum_component_area_px,
        simplify_fraction=0,
    )
    return {
        "type": "FeatureCollection",
        "features": [{
            "type": "Feature",
            "geometry": converted.geometry,
            "properties": {
                "source": "segformer_large_parking_after_exclusions",
                "qualified_anchor_candidate_ids": qualified_ids,
            },
        }],
    }


def _tint(image: np.ndarray, mask: np.ndarray, color: tuple[int, int, int]) -> np.ndarray:
    result = image.copy()
    color_layer = np.zeros_like(result)
    color_layer[mask != 0] = color
    selected = mask != 0
    result[selected] = (result[selected] * 0.65 + color_layer[selected] * 0.35).astype(np.uint8)
    return result


def _label(image: np.ndarray, text: str) -> np.ndarray:
    result = image.copy()
    cv2.rectangle(result, (0, 0), (430, 42), (0, 0, 0), -1)
    cv2.putText(result, text, (12, 29), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    return result


def _map_overlay(image: np.ndarray, result: AoiFunctionalMapResult) -> np.ndarray:
    overlay = image.copy()
    layers = (
        (result.facility_mask, (0, 210, 210), 2),
        (result.masks.unknown_region_mask, (130, 130, 130), 1),
        (result.masks.parking_band_mask, (0, 220, 255), 4),
        (result.masks.internal_aisle_mask, (255, 160, 0), 4),
    )
    for mask, color, width in layers:
        contours, _ = cv2.findContours(mask, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(overlay, contours, -1, color, width)
    return overlay


def _write_preview(
    image: np.ndarray,
    raw_mask: np.ndarray,
    cleaned: CleanedParkingMask,
    map_result: AoiFunctionalMapResult,
    output_path: Path,
) -> None:
    panels = (
        _label(_tint(image, raw_mask, (0, 220, 0)), "SegFormer raw"),
        _label(_tint(image, cleaned.exclusion_mask, (0, 0, 230)), "building / road exclusions"),
        _label(_tint(image, cleaned.cleaned_mask, (0, 220, 0)), "cleaned facility"),
        _label(_map_overlay(image, map_result), "bands / aisles / unknown"),
    )
    preview = np.vstack((np.hstack(panels[:2]), np.hstack(panels[2:])))
    if not cv2.imwrite(str(output_path), preview):
        raise OSError(f"could not write preview: {output_path}")


def build_cleaned_parking_previews(
    *,
    images: Path,
    predictions: Path,
    surface_root: Path,
    evidence_root: Path,
    output_dir: Path,
    minimum_component_area_px: int = 100,
    vehicle_row_bands: bool = False,
    manifest_path: Path | None = None,
    write_diagnostics: bool = True,
    additional_exclusion_root: Path | None = None,
) -> dict[str, Any]:
    image_paths = resolve_image_paths(images, manifest_path)
    if not image_paths:
        raise ValueError(f"no images found in {images}")
    output_dir.mkdir(parents=True, exist_ok=True)
    totals = Counter()
    records = []

    for image_path in image_paths:
        aoi_id = image_path.stem
        raw_mask = cv2.imread(
            str(predictions / f"{aoi_id}_mask.png"),
            cv2.IMREAD_GRAYSCALE,
        )
        image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if image is None or raw_mask is None or raw_mask.shape != image.shape[:2]:
            raise ValueError(f"missing or mismatched image/mask for {aoi_id}")

        evidence_dir = evidence_root / aoi_id
        additional_exclusions = None
        if additional_exclusion_root is not None:
            additional_path = additional_exclusion_root / aoi_id / "exclusion_layers.geojson"
            if not additional_path.is_file():
                raise FileNotFoundError(f"additional exclusions do not exist: {additional_path}")
            additional_exclusions = _load_json(additional_path)
        exclusion_layers = merge_exclusion_layers(
            _load_json(evidence_dir / "exclusion_layers.geojson"),
            additional_exclusions,
        )
        cleaned = clean_parking_mask(
            raw_mask,
            exclusion_layers,
            image_path.name,
            minimum_component_area_px=minimum_component_area_px,
        )
        if not np.any(cleaned.cleaned_mask):
            totals["empty_after_cleanup"] += 1
            continue

        surface = _surface_from_cleaned_mask(
            cleaned.cleaned_mask,
            image_path.name,
            _load_json(surface_root / aoi_id / "surface_hypothesis.geojson"),
            minimum_component_area_px,
        )
        evidence_layers = _load_json(evidence_dir / "evidence_layers.json")
        vehicle_rows = None
        direct_band_seeds = None
        if vehicle_row_bands:
            vehicle_rows = build_vehicle_rows(
                evidence_layers.get("agreed_vehicle_detections") or [],
                cleaned.cleaned_mask,
                cleaned.exclusion_mask,
            )
            direct_band_seeds = tuple(
                RevalidatedBandSeed(
                    candidate_id=row.row_id,
                    mask=row.mask,
                    support_kinds=("agreed_vehicle_alignment",),
                )
                for row in vehicle_rows.rows
            )

        map_result = build_aoi_functional_map(
            aoi_id=aoi_id,
            image_name=image_path.name,
            image=image,
            surface_hypotheses=surface,
            parking_candidates=_load_json(evidence_dir / "parking_candidates.geojson"),
            evidence_layers=evidence_layers,
            exclusion_layers=exclusion_layers,
            source_version="segformer_cleanup",
            minimum_component_area_px=minimum_component_area_px,
            direct_band_seeds=direct_band_seeds,
        )

        aoi_output = output_dir / aoi_id
        aoi_output.mkdir(parents=True, exist_ok=True)
        (aoi_output / "parking_map.geojson").write_text(
            json.dumps(map_result.feature_collection, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        if write_diagnostics:
            cv2.imwrite(str(aoi_output / "raw_mask.png"), raw_mask)
            cv2.imwrite(str(aoi_output / "cleaned_mask.png"), cleaned.cleaned_mask)
            _write_preview(image, raw_mask, cleaned, map_result, aoi_output / "preview.png")

        object_types = Counter(
            (feature.get("properties") or {}).get("object_type")
            for feature in map_result.feature_collection.get("features") or []
        )
        raw_pixels = int(np.count_nonzero(raw_mask))
        cleaned_pixels = int(np.count_nonzero(cleaned.cleaned_mask))
        records.append({
            "aoi_id": aoi_id,
            "raw_pixels": raw_pixels,
            "cleaned_pixels": cleaned_pixels,
            "removed_pixels": cleaned.removed_pixels,
            "removed_fraction": cleaned.removed_pixels / raw_pixels if raw_pixels else 0.0,
            "components": cleaned.component_count,
            "map_objects": dict(object_types),
            "vehicle_row_candidates": len(vehicle_rows.rows) if vehicle_rows else None,
            "vehicles_not_assigned_to_rows": (
                len(vehicle_rows.unassigned_detection_indices) if vehicle_rows else None
            ),
        })
        totals["processed"] += 1

    summary = {
        "totals": dict(totals),
        "manifest": str(manifest_path) if manifest_path else None,
        "diagnostics_written": write_diagnostics,
        "images": records,
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Clean SegFormer parking masks and build map previews")
    parser.add_argument("--images", type=Path, required=True)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--surface-root", type=Path, required=True)
    parser.add_argument("--evidence-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--additional-exclusion-root", type=Path)
    parser.add_argument("--maps-only", action="store_true")
    parser.add_argument(
        "--vehicle-row-bands",
        action="store_true",
        help="derive conservative parking bands from aligned agreed vehicle detections",
    )
    args = parser.parse_args()
    summary = build_cleaned_parking_previews(
        images=args.images,
        predictions=args.predictions,
        surface_root=args.surface_root,
        evidence_root=args.evidence_root,
        output_dir=args.output_dir,
        vehicle_row_bands=args.vehicle_row_bands,
        manifest_path=args.manifest,
        write_diagnostics=not args.maps_only,
        additional_exclusion_root=args.additional_exclusion_root,
    )
    print(json.dumps(summary, ensure_ascii=False))


if __name__ == "__main__":
    main()
