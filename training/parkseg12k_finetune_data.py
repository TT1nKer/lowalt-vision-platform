from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np

from parking.parking_segformer_cleanup import clean_parking_mask


IGNORE_LABEL = 255
BACKGROUND_LABEL = 0


def build_hard_negative_label(
    raw_mask: np.ndarray,
    exclusion_mask: np.ndarray,
    *,
    confirmed_negative: bool,
) -> np.ndarray:
    if (
        raw_mask.ndim != 2
        or exclusion_mask.ndim != 2
        or raw_mask.shape != exclusion_mask.shape
    ):
        raise ValueError("raw and exclusion masks must be same-sized grayscale arrays")
    predicted = raw_mask >= 128
    supervised_background = predicted if confirmed_negative else predicted & (exclusion_mask >= 128)
    label = np.full(raw_mask.shape, IGNORE_LABEL, dtype=np.uint8)
    label[supervised_background] = BACKGROUND_LABEL
    return label


def _atomic_write_bytes(path: Path, content: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_suffix(path.suffix + ".tmp")
    try:
        temporary_path.write_bytes(content)
        temporary_path.replace(path)
    finally:
        temporary_path.unlink(missing_ok=True)


def _write_png(path: Path, image: np.ndarray) -> None:
    encoded, content = cv2.imencode(".png", image)
    if not encoded:
        raise OSError(f"could not encode {path}")
    _atomic_write_bytes(path, content.tobytes())


def build_hard_negatives(
    *,
    images: Path,
    predictions: Path,
    evidence_root: Path,
    output_directory: Path,
    confirmed_negative_aois: set[str],
) -> dict:
    image_paths = sorted(images.glob("*.png")) + sorted(images.glob("*.jpg"))
    if not image_paths:
        raise ValueError(f"no images found in {images}")

    known_aois = {path.stem for path in image_paths}
    unknown_confirmed = confirmed_negative_aois - known_aois
    if unknown_confirmed:
        raise ValueError(f"unknown confirmed-negative AOIs: {sorted(unknown_confirmed)}")

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

        exclusion_path = evidence_root / aoi_id / "exclusion_layers.geojson"
        exclusion_layers = json.loads(exclusion_path.read_text(encoding="utf-8"))
        cleaned = clean_parking_mask(raw_mask, exclusion_layers, image_path.name)
        confirmed_negative = aoi_id in confirmed_negative_aois
        label = build_hard_negative_label(
            raw_mask,
            cleaned.exclusion_mask,
            confirmed_negative=confirmed_negative,
        )
        supervised_pixels = int(np.count_nonzero(label == BACKGROUND_LABEL))
        if supervised_pixels == 0:
            continue

        label_path = output_directory / "labels" / f"{aoi_id}.png"
        _write_png(label_path, label)
        records.append({
            "aoi_id": aoi_id,
            "image": str(image_path),
            "label": str(label_path),
            "supervised_background_pixels": supervised_pixels,
            "source": "confirmed_false_positive" if confirmed_negative else "building_or_public_road_overlap",
        })

    manifest = {
        "label_contract": {"background": BACKGROUND_LABEL, "ignore": IGNORE_LABEL},
        "confirmed_negative_aois": sorted(confirmed_negative_aois),
        "examples": records,
    }
    _atomic_write_bytes(
        output_directory / "manifest.json",
        json.dumps(manifest, ensure_ascii=False, indent=2).encode("utf-8"),
    )
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description="Build conservative Shaoxing hard-negative labels")
    parser.add_argument("--images", type=Path, required=True)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--evidence-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--confirmed-negative-aoi", action="append", default=[])
    args = parser.parse_args()
    manifest = build_hard_negatives(
        images=args.images,
        predictions=args.predictions,
        evidence_root=args.evidence_root,
        output_directory=args.output_dir,
        confirmed_negative_aois=set(args.confirmed_negative_aoi),
    )
    print(json.dumps({"examples": len(manifest["examples"])}, ensure_ascii=False))


if __name__ == "__main__":
    main()
