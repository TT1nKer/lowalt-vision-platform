from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np


TILE_WIDTH = 300
TILE_HEIGHT = 240


def _render_target(target: dict, image_dir: Path, mask_dir: Path) -> np.ndarray | None:
    image = cv2.imread(str(image_dir / str(target["image"])))
    mask = cv2.imread(str(mask_dir / str(target["mask_file"])), cv2.IMREAD_GRAYSCALE)
    if image is None or mask is None:
        return None
    binary = mask > 0
    ys, xs = np.nonzero(binary)
    if not len(xs):
        return None
    bbox = list(map(float, target["bbox"]))
    left = max(0, int(min(xs.min(), bbox[0]) - 24))
    top = max(0, int(min(ys.min(), bbox[1]) - 24))
    right = min(image.shape[1], int(max(xs.max() + 1, bbox[2]) + 24))
    bottom = min(image.shape[0], int(max(ys.max() + 1, bbox[3]) + 24))
    if right <= left or bottom <= top:
        return None

    overlay = image.copy()
    overlay[binary] = (0, 255, 0)
    image = cv2.addWeighted(image, 0.65, overlay, 0.35, 0)
    cv2.rectangle(
        image,
        (round(bbox[0]), round(bbox[1])),
        (round(bbox[2]), round(bbox[3])),
        (0, 0, 255),
        2,
    )
    crop = image[top:bottom, left:right]
    scale = min(TILE_WIDTH / crop.shape[1], (TILE_HEIGHT - 38) / crop.shape[0])
    resized = cv2.resize(crop, None, fx=scale, fy=scale, interpolation=cv2.INTER_NEAREST)
    tile = np.full((TILE_HEIGHT, TILE_WIDTH, 3), 245, dtype=np.uint8)
    x_offset = (TILE_WIDTH - resized.shape[1]) // 2
    y_offset = 32 + (TILE_HEIGHT - 32 - resized.shape[0]) // 2
    tile[y_offset:y_offset + resized.shape[0], x_offset:x_offset + resized.shape[1]] = resized
    strata = target["audit_strata"]
    title = f"{strata['size']} {strata['confidence']} {strata['review']} hits={strata['prompt_hits']}"
    cv2.putText(tile, title, (6, 21), cv2.FONT_HERSHEY_SIMPLEX, .47, (20, 20, 20), 1, cv2.LINE_AA)
    return tile


def _write_sheet(
    name: str, targets: list[dict], image_dir: Path, mask_dir: Path, output_dir: Path
) -> int:
    tiles = []
    for target in targets:
        tile = _render_target(target, image_dir, mask_dir)
        if tile is not None:
            tiles.append(tile)
    if not tiles:
        return 0
    blank = np.full_like(tiles[0], 245)
    tiles.extend([blank] * (16 - len(tiles)))
    rows = [np.hstack(tiles[index:index + 4]) for index in range(0, 16, 4)]
    cv2.imwrite(str(output_dir / f"{name}.png"), np.vstack(rows))
    return min(len(targets), 16)


def build_preview_sheets(
    manifest_path: Path,
    image_dir: Path,
    mask_dir: Path,
    output_dir: Path,
    gate_path: Path | None = None,
) -> dict[str, int]:
    targets = json.loads(manifest_path.read_text(encoding="utf-8"))["targets"]
    if gate_path:
        accepted_ids = {
            str(target["target_id"])
            for target in json.loads(gate_path.read_text(encoding="utf-8"))["targets"]
            if target["decision"] == "auto_accept"
        }
        targets = [target for target in targets if str(target["target_id"]) in accepted_ids]
    categories = {
        "small_under_16": lambda target: target["audit_strata"]["size"] == "<16",
        "medium_16_32": lambda target: target["audit_strata"]["size"] == "16-32",
        "large_32_plus": lambda target: target["audit_strata"]["size"] == ">=32",
        "edge_or_reject": lambda target: (
            target["audit_strata"]["edge"] == "edge"
            or target["audit_strata"]["review"] == "reject"
        ),
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    counts = {}
    for name, predicate in categories.items():
        selected = [target for target in targets if predicate(target)][:16]
        counts[name] = _write_sheet(name, selected, image_dir, mask_dir, output_dir)
    return counts


def main() -> None:
    parser = argparse.ArgumentParser(description="Build bounded SAM3 vehicle mask evidence sheets")
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--images", type=Path, required=True)
    parser.add_argument("--masks", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--gate", type=Path)
    args = parser.parse_args()
    counts = build_preview_sheets(args.manifest, args.images, args.masks, args.output_dir, args.gate)
    print(json.dumps({"output_dir": str(args.output_dir), "counts": counts}))


if __name__ == "__main__":
    main()
