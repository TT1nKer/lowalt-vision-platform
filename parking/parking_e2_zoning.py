from __future__ import annotations

import argparse
import json
from pathlib import Path

from parking_map.e2_batch import run_functional_zoning_batch


def main() -> None:
    parser = argparse.ArgumentParser(description="Build abstaining parking functional-zone evidence")
    parser.add_argument("--images", type=Path, required=True)
    parser.add_argument("--surface-root", type=Path, required=True)
    parser.add_argument("--evidence-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--maximum-between-band-distance-px", type=int, default=48)
    parser.add_argument("--maximum-marking-only-union-fraction", type=float, default=0.8)
    parser.add_argument("--entrance-proximity-px", type=int, default=12)
    parser.add_argument("--minimum-component-area-px", type=int, default=100)
    parser.add_argument("--east-west-metres-per-pixel", type=float)
    parser.add_argument("--north-south-metres-per-pixel", type=float)
    args = parser.parse_args()
    summary = run_functional_zoning_batch(
        images=args.images,
        surface_root=args.surface_root,
        evidence_root=args.evidence_root,
        output_dir=args.output_dir,
        maximum_marking_only_union_fraction=args.maximum_marking_only_union_fraction,
        maximum_between_band_distance_px=args.maximum_between_band_distance_px,
        entrance_proximity_px=args.entrance_proximity_px,
        minimum_component_area_px=args.minimum_component_area_px,
        east_west_metres_per_pixel=args.east_west_metres_per_pixel,
        north_south_metres_per_pixel=args.north_south_metres_per_pixel,
    )
    print(json.dumps(summary, ensure_ascii=False))


if __name__ == "__main__":
    main()
