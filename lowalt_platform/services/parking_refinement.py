from __future__ import annotations

from PIL import Image, ImageChops


def binary_mask(mask: Image.Image) -> Image.Image:
    return mask.convert("L").point(lambda value: 255 if value > 0 else 0)


def refine_parking_candidate(
    candidate: Image.Image,
    building: Image.Image,
    vegetation: Image.Image,
    aisle: Image.Image,
) -> Image.Image:
    masks = [binary_mask(mask) for mask in (candidate, building, vegetation, aisle)]
    if len({mask.size for mask in masks}) != 1:
        raise ValueError("candidate and evidence masks must have the same size")
    candidate_mask, building_mask, vegetation_mask, aisle_mask = masks
    exclusions = ImageChops.lighter(building_mask, vegetation_mask)
    exclusions = ImageChops.lighter(exclusions, aisle_mask)
    return ImageChops.subtract(candidate_mask, exclusions)
