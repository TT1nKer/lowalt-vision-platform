from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum


class SupportLevel(StrEnum):
    SEGFORMER_ONLY = "segformer_only"
    VEHICLE_DETECTED = "vehicle_detected"
    VEHICLE_ROW_SUPPORTED = "vehicle_row_supported"


class CapabilityStatus(StrEnum):
    AVAILABLE = "available"
    CANDIDATE = "candidate"
    NOT_CONNECTED = "not_connected"


class AssetKind(StrEnum):
    WMTS = "wmts"
    IMAGE = "image"
    VIDEO = "video"


@dataclass(frozen=True)
class AnalysisAsset:
    asset_id: str
    kind: AssetKind
    source_path: str
    longitude: float | None = None
    latitude: float | None = None

    def __post_init__(self) -> None:
        if not self.asset_id or not self.source_path:
            raise ValueError("asset id and source path are required")
        if (self.longitude is None) != (self.latitude is None):
            raise ValueError("longitude and latitude must be provided together")

    @property
    def spatial_state(self) -> str:
        return "georeferenced" if self.longitude is not None else "independent_scene"


@dataclass(frozen=True)
class CandidateSummary:
    total: int
    segformer_only: int
    vehicle_detected: int
    vehicle_row_supported: int

    def __post_init__(self) -> None:
        counts = (self.total, self.segformer_only, self.vehicle_detected, self.vehicle_row_supported)
        if any(count < 0 for count in counts):
            raise ValueError("candidate counts cannot be negative")
        if self.total != sum(counts[1:]):
            raise ValueError("candidate support counts must equal total")
