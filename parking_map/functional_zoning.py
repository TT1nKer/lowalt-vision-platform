from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import cv2
import numpy as np


@dataclass(frozen=True)
class FunctionalZoneMasks:
    parking_band_mask: np.ndarray
    internal_aisle_mask: np.ndarray
    entrance_exit_mask: np.ndarray
    unknown_region_mask: np.ndarray
    parking_band_components: int
    internal_aisle_components: int
    entrance_exit_components: int


@dataclass(frozen=True)
class ZoneFeatureMeasurement:
    area_px: int
    facility_area_fraction: float
    component_count: int
    bounding_box_extent: float
    compactness: float
    adjacent_parking_band_components: int
    public_road_proximity_fraction: float
    touches_image_boundary: bool


def _binary(mask: np.ndarray, image_shape: tuple[int, int]) -> np.ndarray:
    if not isinstance(mask, np.ndarray) or mask.ndim != 2 or mask.shape != image_shape:
        raise ValueError("all inputs must be same-sized grayscale masks")
    return mask >= 128


def _retain_components(mask: np.ndarray, minimum_area_px: int) -> tuple[np.ndarray, int]:
    count, labels, stats, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8), 8)
    retained = np.zeros(mask.shape, dtype=bool)
    retained_count = 0
    for label in range(1, count):
        if int(stats[label, cv2.CC_STAT_AREA]) < minimum_area_px:
            continue
        retained[labels == label] = True
        retained_count += 1
    return retained, retained_count


def measure_zone_features(
    zone_mask: np.ndarray,
    facility_mask: np.ndarray,
    public_road_mask: np.ndarray,
    parking_band_mask: np.ndarray,
    *,
    adjacency_distance_px: int = 12,
) -> ZoneFeatureMeasurement:
    if not isinstance(zone_mask, np.ndarray) or zone_mask.ndim != 2:
        raise ValueError("all inputs must be same-sized grayscale masks")
    if adjacency_distance_px <= 0:
        raise ValueError("adjacency_distance_px must be positive")
    image_shape = zone_mask.shape
    zone = _binary(zone_mask, image_shape)
    facility = _binary(facility_mask, image_shape)
    public_road = _binary(public_road_mask, image_shape)
    parking_bands = _binary(parking_band_mask, image_shape)
    area_px = int(np.count_nonzero(zone))
    facility_area_px = int(np.count_nonzero(facility))

    count, labels, stats, _ = cv2.connectedComponentsWithStats(zone.astype(np.uint8), 8)
    component_count = count - 1
    bounding_area = sum(
        int(stats[label, cv2.CC_STAT_WIDTH]) * int(stats[label, cv2.CC_STAT_HEIGHT])
        for label in range(1, count)
    )
    contours, _ = cv2.findContours(zone.astype(np.uint8), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    perimeter = sum(cv2.arcLength(contour, True) for contour in contours)

    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE,
        (2 * adjacency_distance_px + 1, 2 * adjacency_distance_px + 1),
    )
    road_neighborhood = cv2.dilate(public_road.astype(np.uint8), kernel) != 0
    band_count, band_labels = cv2.connectedComponents(parking_bands.astype(np.uint8), 8)
    adjacent_band_components = sum(
        bool(np.any((cv2.dilate((band_labels == label).astype(np.uint8), kernel) != 0) & zone))
        for label in range(1, band_count)
    )

    return ZoneFeatureMeasurement(
        area_px=area_px,
        facility_area_fraction=area_px / facility_area_px if facility_area_px else 0.0,
        component_count=component_count,
        bounding_box_extent=area_px / bounding_area if bounding_area else 0.0,
        compactness=(4.0 * np.pi * area_px / (perimeter * perimeter)) if perimeter else 0.0,
        adjacent_parking_band_components=adjacent_band_components,
        public_road_proximity_fraction=(
            float(np.count_nonzero(zone & road_neighborhood) / area_px) if area_px else 0.0
        ),
        touches_image_boundary=bool(
            np.any(zone[0, :])
            or np.any(zone[-1, :])
            or np.any(zone[:, 0])
            or np.any(zone[:, -1])
        ),
    )


def build_functional_zones(
    facility_mask: np.ndarray,
    parking_band_seed_masks: Iterable[np.ndarray],
    public_road_mask: np.ndarray,
    *,
    maximum_between_band_distance_px: int = 48,
    entrance_proximity_px: int = 12,
    minimum_component_area_px: int = 100,
) -> FunctionalZoneMasks:
    if not isinstance(facility_mask, np.ndarray) or facility_mask.ndim != 2:
        raise ValueError("all inputs must be same-sized grayscale masks")
    if (
        maximum_between_band_distance_px <= 0
        or entrance_proximity_px <= 0
        or minimum_component_area_px <= 0
    ):
        raise ValueError("distance and area thresholds must be positive")

    image_shape = facility_mask.shape
    facility = _binary(facility_mask, image_shape)
    public_road = _binary(public_road_mask, image_shape)
    parking_band_union = np.zeros(image_shape, dtype=bool)
    for seed_mask in parking_band_seed_masks:
        parking_band_union |= _binary(seed_mask, image_shape) & facility
    parking_bands, parking_band_components = _retain_components(
        parking_band_union,
        minimum_component_area_px,
    )

    component_count, component_labels = cv2.connectedComponents(
        parking_bands.astype(np.uint8),
        8,
    )
    nearby_band_count = np.zeros(image_shape, dtype=np.uint16)
    for label in range(1, component_count):
        component = component_labels == label
        distances = cv2.distanceTransform((~component).astype(np.uint8), cv2.DIST_L2, 5)
        nearby_band_count += distances <= maximum_between_band_distance_px

    aisle_candidates = facility & ~parking_bands & (nearby_band_count >= 2)
    internal_aisles, _ = _retain_components(aisle_candidates, minimum_component_area_px)

    proximity_kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE,
        (2 * entrance_proximity_px + 1, 2 * entrance_proximity_px + 1),
    )
    road_neighborhood = cv2.dilate(public_road.astype(np.uint8), proximity_kernel) != 0
    entrances, entrance_exit_components = _retain_components(
        internal_aisles & road_neighborhood,
        minimum_component_area_px,
    )
    internal_aisles &= ~entrances
    internal_aisles, internal_aisle_components = _retain_components(
        internal_aisles,
        minimum_component_area_px,
    )

    unknown = facility & ~parking_bands & ~internal_aisles & ~entrances

    def as_mask(region: np.ndarray) -> np.ndarray:
        return region.astype(np.uint8) * 255

    return FunctionalZoneMasks(
        parking_band_mask=as_mask(parking_bands),
        internal_aisle_mask=as_mask(internal_aisles),
        entrance_exit_mask=as_mask(entrances),
        unknown_region_mask=as_mask(unknown),
        parking_band_components=parking_band_components,
        internal_aisle_components=internal_aisle_components,
        entrance_exit_components=entrance_exit_components,
    )
