from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Iterable

from lowalt_platform.domain import CandidateSummary, SupportLevel


def _coordinate_pairs(value: object) -> Iterable[tuple[float, float]]:
    if isinstance(value, (list, tuple)):
        if len(value) >= 2 and all(isinstance(item, (int, float)) for item in value[:2]):
            longitude, latitude = float(value[0]), float(value[1])
            if not math.isfinite(longitude) or not math.isfinite(latitude):
                raise ValueError("candidate coordinates must be finite")
            yield longitude, latitude
            return
        for child in value:
            yield from _coordinate_pairs(child)


def _geometry_bounds(geometry: dict) -> tuple[float, float, float, float]:
    points = list(_coordinate_pairs(geometry.get("coordinates")))
    if not points:
        raise ValueError("candidate geometry has no coordinates")
    xs, ys = zip(*points)
    return min(xs), min(ys), max(xs), max(ys)


class ParkingCatalog:
    def __init__(self, features: list[dict], summary: CandidateSummary, image_root: Path, mask_root: Path):
        self._features = sorted(features, key=lambda item: str(item["properties"]["aoi_id"]))
        self._summary = summary
        self._image_root = image_root.resolve()
        self._mask_root = mask_root.resolve()
        self._by_aoi = {str(feature["properties"]["aoi_id"]): feature for feature in self._features}
        if len(self._by_aoi) != len(self._features):
            raise ValueError("candidate AOI ids must be unique")
        self._bounds = {aoi_id: _geometry_bounds(feature["geometry"]) for aoi_id, feature in self._by_aoi.items()}

    @classmethod
    def from_paths(cls, candidate_geojson: Path, summary_json: Path, image_root: Path, mask_root: Path) -> "ParkingCatalog":
        payload = json.loads(candidate_geojson.read_text(encoding="utf-8"))
        summary_payload = json.loads(summary_json.read_text(encoding="utf-8"))
        summary = CandidateSummary(
            total=int(summary_payload["total_candidates"]),
            segformer_only=int(summary_payload["segformer_only"]),
            vehicle_detected=int(summary_payload["vehicle_detected"]),
            vehicle_row_supported=int(summary_payload["vehicle_row_supported"]),
        )
        features = list(payload.get("features") or [])
        if len(features) != summary.total:
            raise ValueError("candidate GeoJSON count does not match summary")
        for feature in features:
            properties = feature.get("properties") or {}
            SupportLevel(str(properties.get("support_level")))
            if not properties.get("aoi_id"):
                raise ValueError("candidate AOI id is required")
        return cls(features, summary, image_root, mask_root)

    def summary(self) -> CandidateSummary:
        return self._summary

    def query(self, *, bounds: tuple[float, float, float, float] | None = None, support_levels: set[str] | None = None, limit: int = 2000) -> list[dict]:
        if not 1 <= limit <= 2000:
            raise ValueError("limit must be between 1 and 2000")
        accepted_support = {SupportLevel(value).value for value in support_levels} if support_levels else None
        if bounds:
            west, south, east, north = bounds
            if not all(math.isfinite(value) for value in bounds):
                raise ValueError("query bounds must be finite")
            if west > east:
                raise ValueError("west must not exceed east")
            if south > north:
                raise ValueError("south must not exceed north")
        selected = []
        for feature in self._features:
            properties = feature["properties"]
            if accepted_support and properties["support_level"] not in accepted_support:
                continue
            if bounds:
                item_west, item_south, item_east, item_north = self._bounds[properties["aoi_id"]]
                if item_east < west or item_west > east or item_north < south or item_south > north:
                    continue
            selected.append(feature)
            if len(selected) >= limit:
                break
        return selected

    def detail(self, aoi_id: str) -> dict:
        try:
            feature = self._by_aoi[aoi_id]
        except KeyError as exc:
            raise KeyError(f"unknown candidate: {aoi_id}") from exc
        image_name = f"{aoi_id}.png"
        mask_name = f"{aoi_id}_mask.png"
        return {
            "aoi_id": aoi_id,
            "feature": feature,
            "image_name": image_name,
            "mask_name": mask_name,
            "image_available": (self._image_root / image_name).is_file(),
            "mask_available": (self._mask_root / mask_name).is_file(),
        }

    def image_path(self, aoi_id: str) -> Path:
        self.detail(aoi_id)
        return self._image_root / f"{aoi_id}.png"

    def mask_path(self, aoi_id: str) -> Path:
        self.detail(aoi_id)
        return self._mask_root / f"{aoi_id}_mask.png"
