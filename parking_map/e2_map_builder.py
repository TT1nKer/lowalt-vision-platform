from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import cv2
import numpy as np

from parking_map.e2_runner import (
    RevalidatedBandMeasurement,
    RevalidatedBandSeed,
    RevalidatedBandSelection,
    select_revalidated_band_seeds,
)
from parking_map.functional_zoning import (
    FunctionalZoneMasks,
    build_functional_zones,
    measure_zone_features,
)
from parking_map.image_evidence import geographic_geometry_to_mask
from parking_map.schema import EvidenceRef, MapFeature, MapObjectType, ReviewState
from parking_map.tile_georeference import mask_to_geographic_geometry


@dataclass(frozen=True)
class AoiFunctionalMapResult:
    feature_collection: dict[str, Any]
    zone_feature_records: tuple[dict[str, Any], ...]
    revalidation_measurements: tuple[RevalidatedBandMeasurement, ...]
    masks: FunctionalZoneMasks
    facility_mask: np.ndarray
    public_road_mask: np.ndarray
    aggregate_anchor_gate_status: str
    accepted_anchor_union_facility_fraction: float


def _union_feature_masks(
    features: list[dict[str, Any]],
    image_name: str,
    image_shape: tuple[int, int],
) -> np.ndarray:
    union = np.zeros(image_shape, dtype=np.uint8)
    for feature in features:
        mask = geographic_geometry_to_mask(feature.get("geometry") or {}, image_name, image_shape)
        union[mask != 0] = 255
    return union


def _feature_from_mask(
    *,
    mask: np.ndarray,
    image_name: str,
    object_id: str,
    object_type: MapObjectType,
    source_version: str,
    parent_id: str | None = None,
    parent_type: MapObjectType | None = None,
    evidence: tuple[EvidenceRef, ...] = (),
    minimum_component_area_px: int,
) -> MapFeature:
    converted = mask_to_geographic_geometry(
        mask,
        image_name,
        minimum_component_area=minimum_component_area_px,
        # Parent and child masks share exact raster boundaries. Independent
        # contour simplification can move those boundaries in opposite directions.
        simplify_fraction=0,
    )
    return MapFeature(
        object_id=object_id,
        object_type=object_type,
        geometry=converted.geometry,
        review_state=ReviewState.ABSTAIN,
        source_version=source_version,
        parent_id=parent_id,
        parent_type=parent_type,
        evidence=evidence,
        attributes={
            "truth_status": "evidence_only_not_ground_truth",
            "component_count": converted.component_count,
            "mask_area_px": converted.mask_area_px,
        },
    )


def _has_component_at_least(mask: np.ndarray, minimum_area_px: int) -> bool:
    count, _, stats, _ = cv2.connectedComponentsWithStats((mask >= 128).astype(np.uint8), 8)
    return any(
        int(stats[label, cv2.CC_STAT_AREA]) >= minimum_area_px
        for label in range(1, count)
    )


def build_aoi_functional_map(
    *,
    aoi_id: str,
    image_name: str,
    image: np.ndarray,
    surface_hypotheses: dict[str, Any],
    parking_candidates: dict[str, Any],
    evidence_layers: dict[str, Any],
    exclusion_layers: dict[str, Any],
    source_version: str = "parking_map_e2_v1",
    minimum_marking_strength: float = 0.6,
    minimum_agreed_vehicles: int = 2,
    maximum_marking_only_union_fraction: float = 0.8,
    maximum_between_band_distance_px: int = 48,
    entrance_proximity_px: int = 12,
    minimum_component_area_px: int = 100,
    direct_band_seeds: tuple[RevalidatedBandSeed, ...] | None = None,
) -> AoiFunctionalMapResult:
    if not aoi_id or not image_name or image.ndim != 3:
        raise ValueError("aoi_id, image_name and a color image are required")
    surface_features = list(surface_hypotheses.get("features") or [])
    if not surface_features:
        raise ValueError("at least one surface hypothesis is required")
    image_shape = image.shape[:2]
    facility_mask = _union_feature_masks(surface_features, image_name, image_shape)

    qualified_candidate_ids = {
        str(candidate_id)
        for feature in surface_features
        for candidate_id in (feature.get("properties") or {}).get(
            "qualified_anchor_candidate_ids", []
        )
    }
    candidate_inputs = []
    for feature in parking_candidates.get("features") or []:
        candidate_id = str((feature.get("properties") or {}).get("candidate_id") or "")
        if candidate_id not in qualified_candidate_ids:
            continue
        candidate_inputs.append({
            "candidate_id": candidate_id,
            "mask": geographic_geometry_to_mask(
                feature.get("geometry") or {}, image_name, image_shape
            ),
        })

    if direct_band_seeds is None:
        revalidated = select_revalidated_band_seeds(
            image,
            facility_mask,
            candidate_inputs,
            evidence_layers.get("agreed_vehicle_detections") or [],
            minimum_marking_strength=minimum_marking_strength,
            minimum_agreed_vehicles=minimum_agreed_vehicles,
            minimum_seed_area_px=minimum_component_area_px,
            maximum_marking_only_union_fraction=maximum_marking_only_union_fraction,
        )
    else:
        if any(seed.mask.shape != image_shape for seed in direct_band_seeds):
            raise ValueError("direct band seeds must match the image dimensions")
        clipped_seeds = tuple(
            RevalidatedBandSeed(
                candidate_id=seed.candidate_id,
                mask=np.where((seed.mask >= 128) & (facility_mask >= 128), 255, 0).astype(np.uint8),
                support_kinds=seed.support_kinds,
            )
            for seed in direct_band_seeds
        )
        accepted_union = np.zeros(image_shape, dtype=bool)
        for seed in clipped_seeds:
            accepted_union |= seed.mask >= 128
        facility_area = int(np.count_nonzero(facility_mask))
        revalidated = RevalidatedBandSelection(
            accepted=tuple(seed for seed in clipped_seeds if np.any(seed.mask)),
            measurements=(),
            aggregate_gate_status="passed" if np.any(accepted_union) else "no_vehicle_rows",
            accepted_union_facility_fraction=(
                float(np.count_nonzero(accepted_union) / facility_area) if facility_area else 0.0
            ),
        )
    public_road_features = [
        feature
        for feature in exclusion_layers.get("features") or []
        if (feature.get("properties") or {}).get("exclusion_kind") == "public_road"
    ]
    public_road_mask = _union_feature_masks(public_road_features, image_name, image_shape)
    masks = build_functional_zones(
        facility_mask,
        [seed.mask for seed in revalidated.accepted],
        public_road_mask,
        maximum_between_band_distance_px=maximum_between_band_distance_px,
        entrance_proximity_px=entrance_proximity_px,
        minimum_component_area_px=minimum_component_area_px,
    )

    facility_id = f"{aoi_id}::parking-facility"
    parking_zone_id = f"{aoi_id}::parking-zone"
    features = [_feature_from_mask(
        mask=facility_mask,
        image_name=image_name,
        object_id=facility_id,
        object_type=MapObjectType.PARKING_FACILITY,
        source_version=source_version,
        evidence=(EvidenceRef("paved_surface", "parking_e1_paved_area_v1"),),
        minimum_component_area_px=minimum_component_area_px,
    )]

    band_evidence = tuple(
        EvidenceRef(
            "revalidated_parking_anchor",
            seed.candidate_id,
            details={"support_kinds": list(seed.support_kinds)},
        )
        for seed in revalidated.accepted
    )
    if np.any(masks.parking_band_mask):
        parking_zone_mask = np.maximum(masks.parking_band_mask, masks.internal_aisle_mask)
        features.extend([
            _feature_from_mask(
                mask=parking_zone_mask,
                image_name=image_name,
                object_id=parking_zone_id,
                object_type=MapObjectType.PARKING_ZONE,
                source_version=source_version,
                parent_id=facility_id,
                parent_type=MapObjectType.PARKING_FACILITY,
                evidence=band_evidence,
                minimum_component_area_px=minimum_component_area_px,
            ),
            _feature_from_mask(
                mask=masks.parking_band_mask,
                image_name=image_name,
                object_id=f"{aoi_id}::parking-band",
                object_type=MapObjectType.PARKING_BAND,
                source_version=source_version,
                parent_id=parking_zone_id,
                parent_type=MapObjectType.PARKING_ZONE,
                evidence=band_evidence,
                minimum_component_area_px=minimum_component_area_px,
            ),
        ])

    optional_zones = (
        (
            masks.internal_aisle_mask,
            MapObjectType.INTERNAL_AISLE,
            "internal-aisle",
            EvidenceRef(
                "between_parking_bands",
                "functional_zoning_geometry_v1",
                details={"maximum_distance_px": maximum_between_band_distance_px},
            ),
        ),
        (
            masks.entrance_exit_mask,
            MapObjectType.ENTRANCE_EXIT,
            "entrance-exit",
            EvidenceRef(
                "aisle_public_road_adjacency",
                "functional_zoning_geometry_v1",
                details={"proximity_px": entrance_proximity_px},
            ),
        ),
        (
            masks.unknown_region_mask,
            MapObjectType.UNKNOWN_REGION,
            "unknown-region",
            EvidenceRef("residual_partition", "functional_zoning_geometry_v1"),
        ),
    )
    for mask, object_type, suffix, evidence in optional_zones:
        if not _has_component_at_least(mask, minimum_component_area_px):
            continue
        features.append(_feature_from_mask(
            mask=mask,
            image_name=image_name,
            object_id=f"{aoi_id}::{suffix}",
            object_type=object_type,
            source_version=source_version,
            parent_id=facility_id,
            parent_type=MapObjectType.PARKING_FACILITY,
            evidence=(evidence,),
            minimum_component_area_px=minimum_component_area_px,
        ))

    zone_feature_records = []
    for object_type, mask in (
        (MapObjectType.PARKING_BAND, masks.parking_band_mask),
        (MapObjectType.INTERNAL_AISLE, masks.internal_aisle_mask),
        (MapObjectType.ENTRANCE_EXIT, masks.entrance_exit_mask),
        (MapObjectType.UNKNOWN_REGION, masks.unknown_region_mask),
    ):
        if not np.any(mask):
            continue
        measurement = measure_zone_features(
            mask,
            facility_mask,
            public_road_mask,
            masks.parking_band_mask,
            adjacency_distance_px=entrance_proximity_px,
        )
        zone_feature_records.append({
            "aoi_id": aoi_id,
            "zone_id": f"{aoi_id}::{object_type.value}",
            "object_type": object_type.value,
            "truth_status": "evidence_only_not_ground_truth",
            "geojson_export_status": (
                "exported"
                if _has_component_at_least(mask, minimum_component_area_px)
                else "below_minimum_component_area"
            ),
            **asdict(measurement),
        })

    return AoiFunctionalMapResult(
        feature_collection={
            "type": "FeatureCollection",
            "schema_version": 1,
            "aoi_id": aoi_id,
            "truth_status": "evidence_only_not_ground_truth",
            "features": [feature.to_geojson() for feature in features],
        },
        zone_feature_records=tuple(zone_feature_records),
        revalidation_measurements=revalidated.measurements,
        masks=masks,
        facility_mask=facility_mask,
        public_road_mask=public_road_mask,
        aggregate_anchor_gate_status=revalidated.aggregate_gate_status,
        accepted_anchor_union_facility_fraction=(
            revalidated.accepted_union_facility_fraction
        ),
    )
