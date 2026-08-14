from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Any

import yaml

from uav_data.dlp_adapter import DLP_LOCAL_CRS, load_dlp_scene
from uav_data.ingest import download_registered_files
from uav_data.schema import (
    GsdStatus,
    SequenceManifest,
    SequenceRecord,
    SourceKind,
    SpatialReference,
)
from uav_data.source_files import RegisteredFile
from uav_data.trajectory_features import extract_trajectory_features


DLP_VEHICLE_TYPES = frozenset({"Car", "Medium Vehicle", "Bus"})


def _load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as source:
        value = json.load(source)
    if not isinstance(value, dict):
        raise ValueError(f"JSON root must be an object: {path}")
    return value


def _write_json_atomic(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_suffix(path.suffix + ".part")
    try:
        with partial.open("w", encoding="utf-8", newline="\n") as output:
            json.dump(value, output, ensure_ascii=False, indent=2, sort_keys=True)
            output.write("\n")
        os.replace(partial, path)
    except Exception:
        partial.unlink(missing_ok=True)
        raise


def _write_jsonl_atomic(path: Path, records: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_suffix(path.suffix + ".part")
    try:
        with partial.open("w", encoding="utf-8", newline="\n") as output:
            for record in records:
                output.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")
        os.replace(partial, path)
    except Exception:
        partial.unlink(missing_ok=True)
        raise


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_parking_map(path: Path) -> dict[str, int | float]:
    with path.open("r", encoding="utf-8") as source:
        value = yaml.safe_load(source)
    if not isinstance(value, dict):
        raise ValueError("DLP parking map must contain a mapping")
    map_size = value.get("MAP_SIZE")
    parking_areas = value.get("PARKING_AREAS")
    waypoints = value.get("WAYPOINTS")
    if not isinstance(map_size, dict) or not isinstance(parking_areas, dict) or not isinstance(waypoints, dict):
        raise ValueError("DLP parking map is missing MAP_SIZE, PARKING_AREAS, or WAYPOINTS")
    width = float(map_size.get("x", 0))
    height = float(map_size.get("y", 0))
    if width <= 0 or height <= 0:
        raise ValueError("DLP parking map dimensions must be positive")
    return {
        "width_metres": width,
        "height_metres": height,
        "parking_area_groups": len(parking_areas),
        "waypoint_groups": len(waypoints),
    }


def _registered_file(value: dict[str, Any]) -> RegisteredFile:
    return RegisteredFile(
        name=value["name"],
        url=value["url"],
        size_bytes=int(value["size_bytes"]),
        checksum=value["checksum"],
    )


def run_ingest(
    registry_path: Path,
    download_destination: Path,
    evidence_destination: Path,
    *,
    maximum_bytes: int,
    include_trajectories: bool = False,
) -> dict[str, Any]:
    registry = _load_json(registry_path)
    first_stage_files = tuple(
        _registered_file(item) for item in registry.get("download_files", [])
    )
    if not first_stage_files:
        raise ValueError("registry contains no download_files")
    trajectory_files = tuple(
        _registered_file(item) for item in registry.get("next_gate_files", [])
    ) if include_trajectories else ()
    if include_trajectories and not trajectory_files:
        raise ValueError("registry contains no next_gate_files")
    registered_files = first_stage_files + trajectory_files
    budget_key = (
        "trajectory_stage_maximum_bytes" if include_trajectories
        else "first_stage_maximum_bytes"
    )
    registry_budget = int(registry[budget_key])
    effective_budget = min(maximum_bytes, registry_budget)
    downloaded = download_registered_files(
        registered_files,
        download_destination,
        maximum_bytes=effective_budget,
    )
    downloaded_by_name = {path.name: path for path in downloaded}
    required = {
        "parking_map.yml",
        "DJI_0001_scene.json",
        "DJI_0001_agents.json",
        "DJI_0001_obstacles.json",
    }
    missing = sorted(required - downloaded_by_name.keys())
    if missing:
        raise ValueError(f"DLP first-stage files are missing: {', '.join(missing)}")

    load_options: dict[str, Path] = {}
    if include_trajectories:
        load_options = {
            "frames_path": downloaded_by_name["DJI_0001_frames.json"],
            "instances_path": downloaded_by_name["DJI_0001_instances.json"],
        }
    scene = load_dlp_scene(
        downloaded_by_name["DJI_0001_scene.json"],
        downloaded_by_name["DJI_0001_agents.json"],
        downloaded_by_name["DJI_0001_obstacles.json"],
        **load_options,
    )
    parking_map = _read_parking_map(downloaded_by_name["parking_map.yml"])
    handling = registry.get("handling", {})
    manifest = SequenceManifest(
        sequence=SequenceRecord(
            source_id=registry["source_id"],
            site_id="dragon_lake",
            sequence_id=scene.filename,
            source_kind=SourceKind.UAV_VIDEO_SEQUENCE,
            license_id=registry.get("license", {}).get("resolution", "research-only"),
            spatial_reference=SpatialReference(
                crs=DLP_LOCAL_CRS,
                gsd_status=GsdStatus.UNKNOWN,
                coordinate_note="DLP coordinates are local parking-lot metres transformed from UTM",
            ),
            media_root=str(download_destination),
            attributes={
                "scene_token": scene.scene_token,
                "source_timestamp": scene.source_timestamp,
                "annotation_frame_count": len(scene.frames),
                "truth_scope": handling.get("truth_scope", "algorithm_validation_only"),
            },
        ),
        frames=(),
    )
    expected_split_group = handling.get("split_group", manifest.split_group)
    manifest_value = manifest.to_dict()
    manifest_value["split_group"] = expected_split_group

    file_records = [
        {
            "name": path.name,
            "size_bytes": path.stat().st_size,
            "registry_checksum": registered.checksum,
            "sha256": _sha256(path),
        }
        for registered, path in zip(registered_files, downloaded)
    ]
    trajectory_records: list[dict[str, Any]] = []
    if include_trajectories:
        for agent_token in sorted(scene.agents):
            agent = scene.agents[agent_token]
            features = extract_trajectory_features(scene.agent_trajectory(agent_token))
            trajectory_records.append(
                {
                    "source_id": registry["source_id"],
                    "split_group": expected_split_group,
                    "scene_token": scene.scene_token,
                    "agent_token": agent_token,
                    "agent_type": agent.agent_type,
                    "size_metres": list(agent.size_metres),
                    "features": features.to_dict(),
                }
            )
    agent_type_counts = {
        agent_type: sum(record["agent_type"] == agent_type for record in trajectory_records)
        for agent_type in sorted({record["agent_type"] for record in trajectory_records})
    }
    vehicle_trajectory_count = sum(
        record["agent_type"] in DLP_VEHICLE_TYPES for record in trajectory_records
    )
    summary: dict[str, Any] = {
        "schema_version": 1,
        "source_id": registry["source_id"],
        "source_record_id": registry.get("source", {}).get("zenodo_record_id"),
        "split_group": expected_split_group,
        "truth_scope": handling.get("truth_scope", "algorithm_validation_only"),
        "coordinate_reference": DLP_LOCAL_CRS,
        "files": file_records,
        "downloaded_bytes": sum(item["size_bytes"] for item in file_records),
        "counts": {
            "agents": len(scene.agents),
            "static_obstacles": len(scene.obstacles),
            "frames": len(scene.frames),
            "instances": len(scene.instances),
            "trajectory_feature_records": len(trajectory_records),
            "vehicle_trajectory_feature_records": vehicle_trajectory_count,
        },
        "agent_type_counts": agent_type_counts,
        "parking_map": parking_map,
        "gate_status": (
            "single_scene_trajectory_features_validated" if include_trajectories
            else "metadata_validated_instances_not_downloaded"
        ),
        "limitations": ([
            "No frame or instance file was downloaded in the first-stage budget."
        ] if not include_trajectories else [
            "Trajectory features cover one DLP scene and cannot support a scene-isolated classifier evaluation."
        ]) + [
            "DLP is algorithm-validation evidence and not Shaoxing deployment truth.",
            "DLP local coordinates are not CRS84 and have no image GSD in this ingest.",
        ],
    }
    _write_json_atomic(evidence_destination / "sequence_manifest.json", manifest_value)
    if include_trajectories:
        _write_jsonl_atomic(
            evidence_destination / "trajectory_features.jsonl", trajectory_records
        )
    _write_json_atomic(evidence_destination / "ingest_summary.json", summary)
    return summary


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Ingest the registered DLP E3 metadata subset")
    parser.add_argument("--registry", type=Path, required=True)
    parser.add_argument("--download-destination", type=Path, required=True)
    parser.add_argument("--evidence-destination", type=Path, required=True)
    parser.add_argument("--maximum-mib", type=float, default=1.0)
    parser.add_argument("--include-trajectories", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    if args.maximum_mib <= 0:
        raise ValueError("maximum-mib must be positive")
    summary = run_ingest(
        args.registry,
        args.download_destination,
        args.evidence_destination,
        maximum_bytes=int(args.maximum_mib * 1024 * 1024),
        include_trajectories=args.include_trajectories,
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
