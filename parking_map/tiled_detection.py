from __future__ import annotations

from typing import Any


def _axis_starts(length: int, tile_size: int, overlap: int) -> list[int]:
    if length <= tile_size:
        return [0]
    step = tile_size - overlap
    starts = list(range(0, length - tile_size + 1, step))
    final_start = length - tile_size
    if starts[-1] != final_start:
        starts.append(final_start)
    return starts


def tile_windows(
    image_width: int,
    image_height: int,
    *,
    tile_size: int,
    overlap: int,
) -> list[tuple[int, int, int, int]]:
    if image_width <= 0 or image_height <= 0 or tile_size <= 0:
        raise ValueError("image dimensions and tile size must be positive")
    if overlap < 0 or overlap >= tile_size:
        raise ValueError("overlap must be non-negative and smaller than tile size")
    window_width = min(tile_size, image_width)
    window_height = min(tile_size, image_height)
    return [
        (x, y, x + window_width, y + window_height)
        for y in _axis_starts(image_height, window_height, min(overlap, window_height - 1))
        for x in _axis_starts(image_width, window_width, min(overlap, window_width - 1))
    ]


def _box_iou(first: list[float], second: list[float]) -> float:
    left = max(first[0], second[0])
    top = max(first[1], second[1])
    right = min(first[2], second[2])
    bottom = min(first[3], second[3])
    intersection = max(0.0, right - left) * max(0.0, bottom - top)
    first_area = max(0.0, first[2] - first[0]) * max(0.0, first[3] - first[1])
    second_area = max(0.0, second[2] - second[0]) * max(0.0, second[3] - second[1])
    union = first_area + second_area - intersection
    return intersection / union if union else 0.0


def merge_overlapping_detections(
    detections: list[dict[str, Any]],
    *,
    maximum_iou: float,
) -> list[dict[str, Any]]:
    if not 0 < maximum_iou <= 1:
        raise ValueError("maximum_iou must be between zero and one")
    ordered = sorted(
        detections,
        key=lambda detection: (
            -float(detection.get("confidence", 0.0)),
            tuple(float(value) for value in detection.get("bbox") or []),
        ),
    )
    retained: list[dict[str, Any]] = []
    for detection in ordered:
        box = list(detection.get("bbox") or [])
        if len(box) != 4 or box[2] <= box[0] or box[3] <= box[1]:
            raise ValueError("detections require valid xyxy bounding boxes")
        if any(_box_iou(box, list(existing["bbox"])) > maximum_iou for existing in retained):
            continue
        retained.append({**detection, "bbox": [float(value) for value in box]})
    return retained
