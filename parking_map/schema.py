from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class MapObjectType(str, Enum):
    PARKING_FACILITY = "parking_facility"
    PARKING_ZONE = "parking_zone"
    PARKING_BAND = "parking_band"
    PARKING_SPACE = "parking_space"
    INTERNAL_AISLE = "internal_aisle"
    ENTRANCE_EXIT = "entrance_exit"
    EXCLUSION = "exclusion"
    UNKNOWN_REGION = "unknown_region"


class ReviewState(str, Enum):
    AUTO_ACCEPT = "auto_accept"
    AUTO_REJECT = "auto_reject"
    ABSTAIN = "abstain"


_POLYGON_OBJECTS = {
    MapObjectType.PARKING_FACILITY,
    MapObjectType.PARKING_ZONE,
    MapObjectType.PARKING_BAND,
    MapObjectType.PARKING_SPACE,
    MapObjectType.INTERNAL_AISLE,
    MapObjectType.EXCLUSION,
    MapObjectType.UNKNOWN_REGION,
}

_PARENT_TYPES = {
    MapObjectType.PARKING_ZONE: {MapObjectType.PARKING_FACILITY},
    MapObjectType.PARKING_BAND: {MapObjectType.PARKING_ZONE},
    MapObjectType.PARKING_SPACE: {MapObjectType.PARKING_BAND},
    MapObjectType.INTERNAL_AISLE: {
        MapObjectType.PARKING_FACILITY,
        MapObjectType.PARKING_ZONE,
    },
    MapObjectType.ENTRANCE_EXIT: {MapObjectType.PARKING_FACILITY},
    MapObjectType.EXCLUSION: {
        MapObjectType.PARKING_FACILITY,
        MapObjectType.PARKING_ZONE,
    },
    MapObjectType.UNKNOWN_REGION: {
        MapObjectType.PARKING_FACILITY,
        MapObjectType.PARKING_ZONE,
    },
}


@dataclass(frozen=True)
class EvidenceRef:
    kind: str
    source_id: str
    confidence: float | None = None
    details: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.kind or not self.source_id:
            raise ValueError("evidence kind and source_id are required")
        if self.confidence is not None and not 0 <= self.confidence <= 1:
            raise ValueError("evidence confidence must be between 0 and 1")

    def to_dict(self) -> dict[str, Any]:
        result: dict[str, Any] = {"kind": self.kind, "source_id": self.source_id}
        if self.confidence is not None:
            result["confidence"] = self.confidence
        if self.details:
            result["details"] = self.details
        return result


@dataclass(frozen=True)
class MapFeature:
    object_id: str
    object_type: MapObjectType
    geometry: dict[str, Any]
    review_state: ReviewState
    source_version: str
    parent_id: str | None = None
    parent_type: MapObjectType | None = None
    evidence: tuple[EvidenceRef, ...] = ()
    crs: str = "OGC:CRS84"
    attributes: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.object_id or not self.source_version or not self.crs:
            raise ValueError("object_id, source_version and crs are required")
        geometry_type = self.geometry.get("type")
        if self.object_type in _POLYGON_OBJECTS and geometry_type not in {
            "Polygon",
            "MultiPolygon",
        }:
            raise ValueError(f"{self.object_type.value} requires Polygon or MultiPolygon geometry")
        if self.object_type is MapObjectType.ENTRANCE_EXIT and geometry_type not in {
            "Point",
            "LineString",
            "Polygon",
        }:
            raise ValueError("entrance_exit requires Point, LineString or Polygon geometry")

        allowed_parents = _PARENT_TYPES.get(self.object_type)
        if allowed_parents:
            if not self.parent_id or self.parent_type not in allowed_parents:
                expected = ", ".join(sorted(item.value for item in allowed_parents))
                raise ValueError(
                    f"{self.object_type.value} requires parent_id and parent_type in: {expected}"
                )
        elif self.parent_id is not None or self.parent_type is not None:
            raise ValueError(f"{self.object_type.value} cannot have a parent")

    def to_geojson(self) -> dict[str, Any]:
        properties: dict[str, Any] = {
            "object_id": self.object_id,
            "object_type": self.object_type.value,
            "parent_id": self.parent_id,
            "parent_type": self.parent_type.value if self.parent_type else None,
            "review_state": self.review_state.value,
            "source_version": self.source_version,
            "crs": self.crs,
            "evidence": [item.to_dict() for item in self.evidence],
        }
        properties.update(self.attributes)
        return {"type": "Feature", "geometry": self.geometry, "properties": properties}
