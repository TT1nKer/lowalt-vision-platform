from __future__ import annotations

import hashlib
import json
from pathlib import Path
import re
from typing import Any, Iterable

from PIL import Image, ImageDraw, ImageOps

from parking_map.tile_georeference import parse_block_filename


_NEGATIVE_LABELS = {"hard_negative", "empty_ok", "reject", "bad_mask"}
_BLOCK_SIZE_PX = 1024


def _stable_key(candidate: dict[str, Any]) -> str:
    return hashlib.sha256(str(candidate["aoi_id"]).encode("utf-8")).hexdigest()


def build_aoi_manifest(
    candidates: Iterable[dict[str, Any]],
    quotas: dict[str, int],
    *,
    dataset_version: str,
) -> dict[str, Any]:
    if not dataset_version or not quotas or any(limit <= 0 for limit in quotas.values()):
        raise ValueError("dataset_version and positive stratum quotas are required")
    ordered = sorted(candidates, key=_stable_key)
    selected: list[dict[str, Any]] = []
    selected_ids: set[str] = set()
    selected_geo_cells: set[str] = set()
    coverage: dict[str, int] = {}
    for stratum, limit in quotas.items():
        matches = [item for item in ordered if stratum in item.get("strata", [])]
        added = 0
        unseen_cells = [
            item
            for item in matches
            if item.get("geo_cell") is not None
            and str(item["geo_cell"]) not in selected_geo_cells
        ]
        for pool in (unseen_cells, matches):
            for item in pool:
                aoi_id = str(item.get("aoi_id") or "")
                if not aoi_id or aoi_id in selected_ids:
                    continue
                geo_cell = item.get("geo_cell")
                if pool is unseen_cells and geo_cell is not None and str(geo_cell) in selected_geo_cells:
                    continue
                selected.append(item)
                selected_ids.add(aoi_id)
                if geo_cell is not None:
                    selected_geo_cells.add(str(geo_cell))
                added += 1
                if added == limit:
                    break
            if added == limit:
                break
        coverage[stratum] = added

    return {
        "schema_version": 1,
        "dataset_version": dataset_version,
        "truth_status": "unverified_proxy_evidence",
        "gsd_status": "pending_confirmation",
        "selection_method": "deterministic stratified selection by SHA-256 of aoi_id",
        "coverage": coverage,
        "count": len(selected),
        "aois": selected,
    }


def derive_aoi_candidates(
    index_records: Iterable[dict[str, Any]],
    states: dict[str, dict[str, Any]],
    e0_records: Iterable[dict[str, Any]],
) -> list[dict[str, Any]]:
    e0_by_target = {str(item["target_id"]): item for item in e0_records}
    by_image: dict[str, dict[str, Any]] = {}
    for target in index_records:
        target_id = str(target.get("target_id") or "")
        image = str(target.get("image") or "")
        if not target_id or not image:
            continue
        try:
            block = parse_block_filename(image)
        except ValueError:
            continue

        state = states.get(target_id) or {}
        label = str(state.get("label") or "unreviewed")
        metrics = (e0_by_target.get(target_id) or {}).get("metrics") or {}
        bbox = target.get("bbox") or []
        strata = {"geo_spread"}
        if metrics.get("valid_components", 0) >= 2:
            strata.add("multi_component")
        elif metrics.get("valid_components") == 1:
            strata.add("single_component")
        if metrics.get("global_obb_area_ratio", 0) >= 2:
            strata.add("inflated_obb")
        if label in _NEGATIVE_LABELS:
            strata.add("negative")
        if len(bbox) == 4 and (
            bbox[0] <= 32
            or bbox[1] <= 32
            or bbox[2] >= _BLOCK_SIZE_PX - 32
            or bbox[3] >= _BLOCK_SIZE_PX - 32
        ):
            strata.add("edge")

        aoi_id = Path(image).stem
        candidate = by_image.setdefault(
            image,
            {
                "aoi_id": aoi_id,
                "image": image,
                "tile_origin": {"zoom": block.zoom, "row": block.tile_row, "col": block.tile_col},
                "block_grid": {"row": block.block_row, "col": block.block_col},
                "geo_cell": f"{block.block_row // 32}:{block.block_col // 32}",
                "strata": set(),
                "evidence": {"targets": []},
            },
        )
        candidate["strata"].update(strata)
        candidate["evidence"]["targets"].append(
            {
                "target_id": target_id,
                "label": label,
                "class_name": target.get("class_name"),
                "confidence": target.get("confidence"),
                "bbox": bbox,
                "mask_file": target.get("mask_file"),
                "geometry_metrics": metrics,
            }
        )

    result = []
    for candidate in by_image.values():
        candidate["strata"] = sorted(candidate["strata"])
        candidate["evidence"]["targets"].sort(key=lambda item: item["target_id"])
        result.append(candidate)
    return sorted(result, key=lambda item: item["aoi_id"])


def _safe_aoi_directory_name(aoi_id: str) -> str:
    if not re.fullmatch(r"[A-Za-z0-9_.-]+", aoi_id) or aoi_id in {".", ".."}:
        raise ValueError(f"unsafe aoi_id: {aoi_id}")
    return aoi_id


def write_evidence_packages(
    manifest: dict[str, Any],
    image_dir: Path,
    output_dir: Path,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    for aoi in manifest.get("aois") or []:
        aoi_id = _safe_aoi_directory_name(str(aoi["aoi_id"]))
        package_dir = output_dir / aoi_id
        package_dir.mkdir(parents=True, exist_ok=True)
        (package_dir / "evidence.json").write_text(
            json.dumps(aoi, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

        source_path = image_dir / str(aoi["image"])
        if not source_path.is_file():
            raise FileNotFoundError(f"AOI source image does not exist: {source_path}")
        with Image.open(source_path) as source:
            source = ImageOps.exif_transpose(source).convert("RGB")
            source_width, source_height = source.size
            preview = ImageOps.fit(source, (512, 512), method=Image.Resampling.LANCZOS)
        draw = ImageDraw.Draw(preview)
        scale_x = 512 / source_width
        scale_y = 512 / source_height
        for target in aoi.get("evidence", {}).get("targets", []):
            bbox = target.get("bbox") or []
            if len(bbox) != 4:
                continue
            scaled = [
                bbox[0] * scale_x,
                bbox[1] * scale_y,
                bbox[2] * scale_x,
                bbox[3] * scale_y,
            ]
            color = "#ff4d4d" if target.get("label") in _NEGATIVE_LABELS else "#ffd24d"
            draw.rectangle(scaled, outline=color, width=2)
        preview.save(package_dir / "source_preview.jpg", quality=90)
