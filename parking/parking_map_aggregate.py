from __future__ import annotations

import argparse
from contextlib import ExitStack
from collections import Counter
import json
from pathlib import Path
from typing import Any

from shapely.geometry import mapping, shape


LAYER_FILES = {
    "supported_facility": "parking_facilities.geojson",
    "candidate_facility": "facility_candidates_abstain.geojson",
    "parking_zone": "parking_zones.geojson",
    "parking_band": "parking_bands.geojson",
    "internal_aisle": "internal_aisles.geojson",
    "entrance_exit": "entrances.geojson",
    "unknown_region": "unknown_regions.geojson",
}


def _layer_for_feature(object_type: str, *, facility_supported: bool) -> str | None:
    if object_type == "parking_facility":
        return "supported_facility" if facility_supported else "candidate_facility"
    return object_type if object_type in LAYER_FILES else None


def aggregate_map_layers(
    input_root: Path,
    output_dir: Path,
    *,
    simplify_tolerance: float = 1e-6,
) -> dict[str, Any]:
    if simplify_tolerance < 0:
        raise ValueError("simplify tolerance cannot be negative")
    map_paths = sorted(input_root.glob("*/parking_map.geojson"))
    if not map_paths:
        raise ValueError(f"no AOI parking maps found in {input_root}")
    output_dir.mkdir(parents=True, exist_ok=True)
    temporary_paths = {
        layer: output_dir / f".{filename}.tmp"
        for layer, filename in LAYER_FILES.items()
    }
    counts = Counter()
    first_feature = {layer: True for layer in LAYER_FILES}

    try:
        with ExitStack() as stack:
            streams = {
                layer: stack.enter_context(path.open("w", encoding="utf-8"))
                for layer, path in temporary_paths.items()
            }
            for stream in streams.values():
                stream.write(
                    '{"type":"FeatureCollection","truth_status":'
                    '"model_and_geometry_inference_not_ground_truth","features":['
                )

            for map_path in map_paths:
                record = json.loads(map_path.read_text(encoding="utf-8"))
                aoi_id = str(record.get("aoi_id") or map_path.parent.name)
                features = list(record.get("features") or [])
                facility_supported = any(
                    (feature.get("properties") or {}).get("object_type") == "parking_band"
                    for feature in features
                )
                for feature in features:
                    properties = dict(feature.get("properties") or {})
                    object_type = str(properties.get("object_type") or "")
                    layer = _layer_for_feature(
                        object_type,
                        facility_supported=facility_supported,
                    )
                    if layer is None:
                        continue
                    geometry = shape(feature.get("geometry") or {})
                    if geometry.is_empty or not geometry.is_valid:
                        raise ValueError(f"invalid geometry in {map_path}")
                    if simplify_tolerance:
                        geometry = geometry.simplify(
                            simplify_tolerance,
                            preserve_topology=True,
                        )
                    output_feature = {
                        "type": "Feature",
                        "geometry": mapping(geometry),
                        "properties": {**properties, "aoi_id": aoi_id},
                    }
                    stream = streams[layer]
                    if not first_feature[layer]:
                        stream.write(",")
                    json.dump(output_feature, stream, ensure_ascii=False, separators=(",", ":"))
                    first_feature[layer] = False
                    counts[layer] += 1

            for stream in streams.values():
                stream.write("]}")

        for layer, filename in LAYER_FILES.items():
            temporary_paths[layer].replace(output_dir / filename)
    except Exception:
        for path in temporary_paths.values():
            path.unlink(missing_ok=True)
        raise

    summary = {
        "source_map_files": len(map_paths),
        "simplify_tolerance_degrees": simplify_tolerance,
        "supported_parking_facilities": counts["supported_facility"],
        "candidate_only_facilities": counts["candidate_facility"],
        "parking_zones": counts["parking_zone"],
        "parking_bands": counts["parking_band"],
        "internal_aisles": counts["internal_aisle"],
        "entrances": counts["entrance_exit"],
        "unknown_regions": counts["unknown_region"],
        "truth_status": "model_and_geometry_inference_not_ground_truth",
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Aggregate per-AOI parking maps into QGIS layers")
    parser.add_argument("--input-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--simplify-tolerance", type=float, default=1e-6)
    args = parser.parse_args()
    summary = aggregate_map_layers(
        args.input_root,
        args.output_dir,
        simplify_tolerance=args.simplify_tolerance,
    )
    print(json.dumps(summary, ensure_ascii=False))


if __name__ == "__main__":
    main()
