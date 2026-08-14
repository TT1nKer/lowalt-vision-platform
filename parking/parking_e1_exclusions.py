from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from parking_map.e1_runner import evaluate_aoi_candidates
from parking_map.image_evidence import geographic_geometry_to_mask
from parking_map.sam_exclusions import measure_exclusion_effects, select_exclusion_masks
from parking_map.tile_georeference import mask_to_geographic_geometry


_PROMPT_DIRECTORIES = {"building": "building", "public road": "public_road"}


def _resolve_prompts(requested: list[str] | tuple[str, ...] | None) -> tuple[str, ...]:
    if not requested:
        return tuple(_PROMPT_DIRECTORIES)
    unsupported = sorted(set(requested) - set(_PROMPT_DIRECTORIES))
    if unsupported:
        raise ValueError(f"unsupported exclusion prompt: {unsupported[0]}")
    return tuple(dict.fromkeys(requested))


def _load_prompt_candidates(
    sam_run: Path,
    prompt: str,
    image_name: str,
) -> list[dict[str, Any]]:
    prompt_directory = sam_run / "raw_prompts" / _PROMPT_DIRECTORIES[prompt]
    result_path = prompt_directory / "sam3_results" / f"{Path(image_name).stem}.json"
    record = json.loads(result_path.read_text(encoding="utf-8"))
    candidates = []
    for target in record.get("targets") or []:
        mask_path = prompt_directory / "mask" / str(target.get("mask_file") or "")
        candidates.append({
            "target_id": target.get("source_target_id"),
            "confidence": target.get("confidence"),
            "mask": cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE),
        })
    return candidates


def _exclusion_feature(prompt: str, geometry: dict[str, Any], accepted_ids: tuple[str, ...]) -> dict:
    return {
        "type": "Feature",
        "geometry": geometry,
        "properties": {
            "object_type": "exclusion",
            "exclusion_kind": prompt.replace(" ", "_"),
            "source": "sam3_teacher",
            "source_prompt": prompt,
            "accepted_target_ids": list(accepted_ids),
            "authority": "provisional",
            "truth_status": "teacher_evidence_not_ground_truth",
            "crs": "OGC:CRS84",
        },
    }


def _draw_geometry(image, image_name: str, geometry: dict, color: tuple[int, int, int], width: int) -> None:
    mask = geographic_geometry_to_mask(geometry, image_name, image.shape[:2])
    contours, _ = cv2.findContours(mask, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(image, contours, -1, color, width)


def _write_overlay(
    source_image,
    image_name: str,
    exclusions: list[dict[str, Any]],
    evaluated: dict[str, Any],
    output_path: Path,
) -> None:
    overlay = source_image.copy()
    exclusion_colors = {"building": (40, 40, 230), "public road": (220, 80, 180)}
    for feature in exclusions:
        prompt = feature["properties"]["source_prompt"]
        _draw_geometry(overlay, image_name, feature["geometry"], exclusion_colors[prompt], 2)
    for feature in evaluated["features"]:
        _draw_geometry(overlay, image_name, feature["geometry"], (0, 200, 255), 2)
    cv2.imwrite(str(output_path), overlay)


def _quantiles(values: list[float]) -> dict[str, float | None]:
    if not values:
        return {"p10": None, "p50": None, "p90": None}
    ordered = sorted(values)
    return {
        label: ordered[round((len(ordered) - 1) * fraction)]
        for label, fraction in (("p10", 0.1), ("p50", 0.5), ("p90", 0.9))
    }


def apply_sam_exclusions(
    *,
    manifest_path: Path,
    p1_evidence_root: Path,
    image_dir: Path,
    base_evidence_root: Path,
    sam_run: Path,
    output_dir: Path,
    prompts: list[str] | tuple[str, ...] | None = None,
    write_overlays: bool = True,
) -> dict[str, Any]:
    selected_prompts = _resolve_prompts(prompts)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    output_dir.mkdir(parents=True, exist_ok=True)
    totals = Counter()
    decision_counts = Counter()
    rejected_reasons: dict[str, Counter] = {prompt: Counter() for prompt in selected_prompts}
    removed_fractions = []

    for aoi in manifest.get("aois") or []:
        aoi_id = str(aoi["aoi_id"])
        image_name = str(aoi["image"])
        source_image = cv2.imread(str(image_dir / image_name), cv2.IMREAD_COLOR)
        if source_image is None:
            raise ValueError(f"unreadable source image: {image_name}")
        candidates = json.loads(
            (p1_evidence_root / aoi_id / "mask_geometries.geojson").read_text(encoding="utf-8")
        )
        evidence = json.loads(
            (base_evidence_root / aoi_id / "evidence_layers.json").read_text(encoding="utf-8")
        )
        exclusion_features = []
        for prompt in selected_prompts:
            selection = select_exclusion_masks(
                prompt,
                _load_prompt_candidates(sam_run, prompt, image_name),
                image_shape=source_image.shape[:2],
            )
            totals[f"accepted_{prompt.replace(' ', '_')}_masks"] += len(selection.accepted_target_ids)
            rejected_reasons[prompt].update(selection.rejected_reason_counts)
            if not np.any(selection.union_mask):
                continue
            converted = mask_to_geographic_geometry(
                selection.union_mask,
                image_name,
                minimum_component_area=100,
            )
            exclusion_features.append(
                _exclusion_feature(prompt, converted.geometry, selection.accepted_target_ids)
            )
            totals[f"{prompt.replace(' ', '_')}_polygon_components"] += converted.component_count

        evidence.update({
            "schema_version": 2,
            "exclusions": exclusion_features,
            "exclusions_authoritative": False,
            "exclusion_coverage_status": "provisional_incomplete",
            "exclusion_source_version": sam_run.name,
        })
        evaluated = evaluate_aoi_candidates(aoi_id, candidates, evidence)
        source_component_counts = {
            str((feature.get("properties") or {}).get("target_id")): int(
                (feature.get("properties") or {}).get("component_count")
                or (1 if feature.get("geometry", {}).get("type") == "Polygon" else len(feature.get("geometry", {}).get("coordinates") or []))
            )
            for feature in candidates.get("features") or []
        }
        totals.update(measure_exclusion_effects(source_component_counts, evaluated["decisions"]))
        aoi_output = output_dir / aoi_id
        aoi_output.mkdir(parents=True, exist_ok=True)
        (aoi_output / "evidence_layers.json").write_text(
            json.dumps(evidence, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        (aoi_output / "exclusion_layers.geojson").write_text(
            json.dumps({"type": "FeatureCollection", "features": exclusion_features}, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        (aoi_output / "parking_candidates.geojson").write_text(
            json.dumps(evaluated, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        if write_overlays:
            _write_overlay(
                source_image,
                image_name,
                exclusion_features,
                evaluated,
                aoi_output / "exclusion_overlay.jpg",
            )
        totals["aois"] += 1
        totals["candidates"] += len(evaluated["decisions"])
        totals["aois_with_exclusion"] += bool(exclusion_features)
        decision_counts.update(evaluated["summary"])
        for decision in evaluated["decisions"]:
            removed_fractions.append(float(decision["removed_fraction"]))
            totals["provisionally_fully_excluded"] += (
                "provisionally_fully_excluded" in decision["reasons"]
            )

    summary = {
        "schema_version": 2,
        "truth_status": "evidence_only_not_ground_truth",
        "source_manifest": str(manifest_path),
        "base_evidence": str(base_evidence_root),
        "sam_run": str(sam_run),
        "policy": {
            "prompts": list(selected_prompts),
            "overlays_written": write_overlays,
            "building_min_confidence": 0.7,
            "public_road_min_confidence": 0.65,
            "public_road_requires_border_contact": True,
            "minimum_mask_area_px": 100,
            "exclusions_authoritative": False,
        },
        "totals": dict(totals),
        "decisions": dict(decision_counts),
        "removed_fraction": _quantiles(removed_fractions),
        "rejected_exclusion_reasons": {
            prompt: dict(counts) for prompt, counts in rejected_reasons.items()
        },
        "gate_status": "blocked_missing_complete_exclusion_coverage_and_parking_truth",
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Apply provisional SAM3 building/public-road exclusions")
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--p1-evidence-root", type=Path, required=True)
    parser.add_argument("--images", type=Path, required=True)
    parser.add_argument("--base-evidence-root", type=Path, required=True)
    parser.add_argument("--sam-run", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--prompt",
        action="append",
        choices=tuple(_PROMPT_DIRECTORIES),
        help="limit exclusions to a prompt; repeat to select more than one",
    )
    parser.add_argument("--no-overlays", action="store_true")
    args = parser.parse_args()
    summary = apply_sam_exclusions(
        manifest_path=args.manifest,
        p1_evidence_root=args.p1_evidence_root,
        image_dir=args.images,
        base_evidence_root=args.base_evidence_root,
        sam_run=args.sam_run,
        output_dir=args.output_dir,
        prompts=args.prompt,
        write_overlays=not args.no_overlays,
    )
    print(json.dumps(summary, ensure_ascii=False))


if __name__ == "__main__":
    main()
