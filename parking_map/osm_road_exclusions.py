from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import dataclass
import json
import math
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from parking_map.tile_georeference import (
    lonlat_to_pixel,
    mask_to_geographic_geometry,
    parse_block_filename,
)


# These are visual screening widths for the fixed z19 WMTS proxy, not surveyed road widths.
ROAD_WIDTH_PIXELS = {
    "motorway": 36,
    "motorway_link": 24,
    "trunk": 32,
    "trunk_link": 22,
    "primary": 28,
    "primary_link": 20,
    "secondary": 24,
    "secondary_link": 18,
    "tertiary": 20,
    "tertiary_link": 16,
    "residential": 16,
    "unclassified": 16,
    "living_street": 14,
    "service": 10,
}
NON_PUBLIC_SERVICE_TYPES = {"parking_aisle", "driveway", "drive-through"}
NON_PUBLIC_ACCESS = {"private", "no"}


@dataclass(frozen=True)
class AoiRoadExclusion:
    mask: np.ndarray
    feature_collection: dict[str, Any]
    used_way_ids: tuple[int, ...]
    component_count: int
    coverage_status: str
    skipped_reason_counts: dict[str, int]


def _way_rejection_reason(element: dict[str, Any]) -> str | None:
    if element.get("type") != "way":
        return "not_way"
    tags = element.get("tags") or {}
    highway = str(tags.get("highway") or "")
    if highway not in ROAD_WIDTH_PIXELS:
        return "unsupported_highway"
    if str(tags.get("access") or "") in NON_PUBLIC_ACCESS:
        return "non_public_access"
    if highway == "service" and str(tags.get("service") or "") in NON_PUBLIC_SERVICE_TYPES:
        return "internal_service_road"
    if str(tags.get("tunnel") or "") in {"yes", "building_passage"}:
        return "tunnel"
    if str(tags.get("bridge") or "") == "yes":
        return "bridge"
    geometry = element.get("geometry") or []
    if len(geometry) < 2:
        return "insufficient_geometry"
    for point in geometry:
        longitude = point.get("lon")
        latitude = point.get("lat")
        if not isinstance(longitude, (int, float)) or not isinstance(latitude, (int, float)):
            return "invalid_coordinate"
        if not math.isfinite(longitude) or not math.isfinite(latitude):
            return "invalid_coordinate"
        if not -180 <= longitude <= 180 or not -90 <= latitude <= 90:
            return "invalid_coordinate"
    return None


def _draw_way(
    mask: np.ndarray,
    element: dict[str, Any],
    block_filename: str,
) -> bool:
    height, width = mask.shape
    block = parse_block_filename(block_filename)
    points = [
        lonlat_to_pixel(
            block.zoom,
            block.tile_row,
            block.tile_col,
            float(point["lon"]),
            float(point["lat"]),
        )
        for point in element["geometry"]
    ]
    highway = str(element["tags"]["highway"])
    line_width = ROAD_WIDTH_PIXELS[highway]
    drawn = False
    for first, second in zip(points, points[1:]):
        first_pixel = tuple(int(round(value)) for value in first)
        second_pixel = tuple(int(round(value)) for value in second)
        visible, clipped_first, clipped_second = cv2.clipLine(
            (0, 0, width, height),
            first_pixel,
            second_pixel,
        )
        if not visible:
            continue
        cv2.line(mask, clipped_first, clipped_second, 255, line_width, cv2.LINE_8)
        drawn = True
    return drawn


def build_aoi_road_exclusion(
    osm_document: dict[str, Any],
    block_filename: str,
    image_shape: tuple[int, int],
) -> AoiRoadExclusion:
    height, width = image_shape
    if height <= 0 or width <= 0:
        raise ValueError("image_shape must contain positive dimensions")
    mask = np.zeros((height, width), dtype=np.uint8)
    used_way_ids = []
    skipped_reasons = Counter()

    elements = sorted(
        osm_document.get("elements") or [],
        key=lambda element: (str(element.get("type") or ""), int(element.get("id") or 0)),
    )
    for element in elements:
        rejection_reason = _way_rejection_reason(element)
        if rejection_reason:
            skipped_reasons[rejection_reason] += 1
            continue
        if _draw_way(mask, element, block_filename):
            used_way_ids.append(int(element["id"]))

    if not used_way_ids:
        feature_collection = {
            "type": "FeatureCollection",
            "features": [],
            "metadata": {
                "coverage_status": "no_matching_public_roads",
                "source": "OpenStreetMap",
                "attribution": "© OpenStreetMap contributors",
            },
        }
        return AoiRoadExclusion(
            mask=mask,
            feature_collection=feature_collection,
            used_way_ids=(),
            component_count=0,
            coverage_status="no_matching_public_roads",
            skipped_reason_counts=dict(skipped_reasons),
        )

    converted = mask_to_geographic_geometry(
        mask,
        block_filename,
        minimum_component_area=1,
        simplify_fraction=0,
    )
    feature_collection = {
        "type": "FeatureCollection",
        "features": [{
            "type": "Feature",
            "geometry": converted.geometry,
            "properties": {
                "exclusion_kind": "public_road",
                "source": "OpenStreetMap",
                "attribution": "© OpenStreetMap contributors",
                "license": "ODbL 1.0",
                "source_timestamp": (osm_document.get("osm3s") or {}).get("timestamp_osm_base"),
                "osm_way_ids": sorted(used_way_ids),
                "width_basis": "configured visual screening pixels at WMTS z19; not surveyed width",
                "coverage_status": "provisional_osm_public_roads",
            },
        }],
        "metadata": {
            "coverage_status": "provisional_osm_public_roads",
            "source": "OpenStreetMap",
            "attribution": "© OpenStreetMap contributors",
        },
    }
    return AoiRoadExclusion(
        mask=mask,
        feature_collection=feature_collection,
        used_way_ids=tuple(sorted(used_way_ids)),
        component_count=converted.component_count,
        coverage_status="provisional_osm_public_roads",
        skipped_reason_counts=dict(skipped_reasons),
    )


def write_manifest_road_exclusions(
    *,
    osm_json: Path,
    images: Path,
    manifest_path: Path,
    output_root: Path,
) -> dict[str, Any]:
    osm_document = json.loads(osm_json.read_text(encoding="utf-8"))
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    image_root = images.resolve()
    records = []
    totals = Counter()
    output_root.mkdir(parents=True, exist_ok=True)

    for aoi in manifest.get("aois") or []:
        image_name = str(aoi.get("image") or "")
        image_path = (images / image_name).resolve()
        if not image_name or image_path.parent != image_root:
            raise ValueError(f"unsafe manifest image path: {image_name}")
        image = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
        if image is None:
            raise FileNotFoundError(f"could not read manifest image: {image_path}")
        result = build_aoi_road_exclusion(osm_document, image_name, image.shape)
        aoi_id = Path(image_name).stem
        aoi_output = output_root / aoi_id
        aoi_output.mkdir(parents=True, exist_ok=True)
        (aoi_output / "exclusion_layers.geojson").write_text(
            json.dumps(result.feature_collection, ensure_ascii=False, indent=2, allow_nan=False),
            encoding="utf-8",
        )
        records.append({
            "aoi_id": aoi_id,
            "road_count": len(result.used_way_ids),
            "road_pixels": int(np.count_nonzero(result.mask)),
            "component_count": result.component_count,
            "coverage_status": result.coverage_status,
        })
        totals["aois"] += 1
        totals["aois_with_roads"] += bool(result.used_way_ids)
        totals["road_way_intersections"] += len(result.used_way_ids)
        totals["road_pixels"] += int(np.count_nonzero(result.mask))

    summary = {
        "source": "OpenStreetMap",
        "attribution": "© OpenStreetMap contributors",
        "license": "ODbL 1.0",
        "source_timestamp": (osm_document.get("osm3s") or {}).get("timestamp_osm_base"),
        "input": str(osm_json),
        "manifest": str(manifest_path),
        "road_width_pixels": ROAD_WIDTH_PIXELS,
        "totals": dict(totals),
        "aois": records,
    }
    (output_root / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=False),
        encoding="utf-8",
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Build provisional OSM public-road exclusions")
    parser.add_argument("--osm-json", type=Path, required=True)
    parser.add_argument("--images", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    summary = write_manifest_road_exclusions(
        osm_json=args.osm_json,
        images=args.images,
        manifest_path=args.manifest,
        output_root=args.output_root,
    )
    print(json.dumps(summary["totals"], ensure_ascii=False))


if __name__ == "__main__":
    main()
