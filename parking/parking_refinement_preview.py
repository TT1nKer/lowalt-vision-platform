from __future__ import annotations

import argparse
import json
from pathlib import Path

from PIL import Image, ImageChops, ImageDraw

from lowalt_platform.services.parking_refinement import binary_mask, refine_parking_candidate


THUMBNAIL_SIZE = 256
SAMPLE_COUNT = 12


def mask_ratio(mask: Image.Image, candidate: Image.Image) -> float:
    overlap = ImageChops.multiply(binary_mask(mask), binary_mask(candidate))
    candidate_pixels = binary_mask(candidate).histogram()[255]
    return overlap.histogram()[255] / candidate_pixels if candidate_pixels else 0.0


def overlay(image: Image.Image, mask: Image.Image, color: tuple[int, int, int]) -> Image.Image:
    base = image.convert("RGB")
    layer = Image.new("RGB", base.size, color)
    return Image.composite(layer, base, binary_mask(mask).point(lambda value: value * 2 // 5))


def select_evenly(paths: list[Path], count: int) -> list[Path]:
    if len(paths) <= count:
        return paths
    return [paths[index * (len(paths) - 1) // (count - 1)] for index in range(count)]


def main() -> None:
    parser = argparse.ArgumentParser(description="Preview direct parking-candidate cleanup")
    parser.add_argument("--project-dir", type=Path, default=Path("."))
    parser.add_argument("--scan-count", type=int, default=120)
    args = parser.parse_args()

    root = args.project_dir.resolve()
    secondary_root = root / "quality" / "parking_secondary_sam3"
    candidate_root = root / "quality" / "parkseg_imagery_masks"
    image_root = root / "imagery"
    output_root = root / "quality" / "parking_refinement_preview"
    output_root.mkdir(parents=True, exist_ok=True)

    completed = sorted(path.parent for path in secondary_root.glob("*/manifest.json"))
    scanned = select_evenly(completed, min(args.scan_count, len(completed)))
    measurements = []
    for directory in scanned:
        aoi_id = directory.name
        candidate_path = candidate_root / f"{aoi_id}_mask.png"
        required = [candidate_path, directory / "building.png", directory / "vegetation.png", directory / "internal_aisle.png"]
        if not all(path.is_file() for path in required):
            continue
        with Image.open(candidate_path) as candidate, Image.open(required[1]) as building, Image.open(required[2]) as vegetation, Image.open(required[3]) as aisle:
            measurements.append({
                "aoi_id": aoi_id,
                "directory": directory,
                "exclusion_ratio": mask_ratio(ImageChops.lighter(binary_mask(building), binary_mask(vegetation)), candidate),
                "aisle_ratio": mask_ratio(aisle, candidate),
            })

    ranked = sorted(measurements, key=lambda item: item["exclusion_ratio"] + item["aisle_ratio"])
    indices = [0, 1, 2, len(ranked) // 3, len(ranked) // 2, 2 * len(ranked) // 3, -6, -5, -4, -3, -2, -1]
    selected = [ranked[index] for index in indices] if len(ranked) >= SAMPLE_COUNT else ranked

    canvas = Image.new("RGB", (THUMBNAIL_SIZE * 4, THUMBNAIL_SIZE * len(selected)), "white")
    report = []
    for row, item in enumerate(selected):
        aoi_id = item["aoi_id"]
        directory = item["directory"]
        with (
            Image.open(image_root / f"{aoi_id}.png") as source,
            Image.open(candidate_root / f"{aoi_id}_mask.png") as candidate,
            Image.open(directory / "building.png") as building,
            Image.open(directory / "vegetation.png") as vegetation,
            Image.open(directory / "internal_aisle.png") as aisle,
        ):
            exclusions = ImageChops.lighter(binary_mask(building), binary_mask(vegetation))
            exclusions = ImageChops.lighter(exclusions, binary_mask(aisle))
            refined = refine_parking_candidate(candidate, building, vegetation, aisle)
            panels = [source.convert("RGB"), overlay(source, candidate, (255, 176, 40)), overlay(source, exclusions, (220, 55, 65)), overlay(source, refined, (20, 190, 120))]
            for column, panel in enumerate(panels):
                panel.thumbnail((THUMBNAIL_SIZE, THUMBNAIL_SIZE), Image.Resampling.LANCZOS)
                canvas.paste(panel, (column * THUMBNAIL_SIZE, row * THUMBNAIL_SIZE))
            report.append({key: value for key, value in item.items() if key != "directory"})

    draw = ImageDraw.Draw(canvas)
    for column, title in enumerate(("Original", "SegFormer", "Removed", "Refined")):
        draw.rectangle((column * THUMBNAIL_SIZE, 0, column * THUMBNAIL_SIZE + 90, 20), fill="black")
        draw.text((column * THUMBNAIL_SIZE + 4, 4), title, fill="white")
    canvas.save(output_root / "comparison.jpg", quality=90)
    (output_root / "summary.json").write_text(json.dumps({"scanned": len(measurements), "samples": report}, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"scanned": len(measurements), "selected": len(selected), "preview": str(output_root / "comparison.jpg")}, ensure_ascii=False))


if __name__ == "__main__":
    main()
