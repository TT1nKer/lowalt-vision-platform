import hashlib
import json
from pathlib import Path
import tempfile
import unittest

from parking.parking_e3_ingest import run_ingest


def write_json(path: Path, value: object) -> Path:
    path.write_text(json.dumps(value), encoding="utf-8")
    return path


def registered_file(path: Path) -> dict[str, object]:
    content = path.read_bytes()
    return {
        "name": path.name,
        "url": path.as_uri(),
        "size_bytes": len(content),
        "checksum": "sha256:" + hashlib.sha256(content).hexdigest(),
    }


class ParkingE3TrajectoryIngestTests(unittest.TestCase):
    def test_second_stage_uses_frame_timestamps_and_emits_agent_features(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            upstream = root / "upstream"
            upstream.mkdir()
            readme = upstream / "README.md"
            readme.write_text("DLP fixture", encoding="utf-8")
            parking_map = upstream / "parking_map.yml"
            parking_map.write_text(
                "MAP_SIZE: {x: 140, y: 80}\nPARKING_AREAS: {A: {bounds: [[0, 0], [2, 2]]}}\n"
                "WAYPOINTS: {R1: {bounds: [[0, 0], [2, 0]], nums: 2}}\n",
                encoding="utf-8",
            )
            scene = write_json(
                upstream / "DJI_0001_scene.json",
                {
                    "scene_token": "scene-token", "filename": "DJI_0001",
                    "timestamp": "2020-06-20 15:03:24", "first_frame": "frame-1",
                    "last_frame": "frame-2", "agents": ["agent-token"], "obstacles": [],
                },
            )
            agents = write_json(
                upstream / "DJI_0001_agents.json",
                {
                    "agent-token": {
                        "agent_token": "agent-token", "scene_token": "scene-token",
                        "type": "Car", "size": [4.5, 2.0],
                        "first_instance": "instance-1", "last_instance": "instance-2",
                    }
                },
            )
            obstacles = write_json(upstream / "DJI_0001_obstacles.json", {})
            frames = write_json(
                upstream / "DJI_0001_frames.json",
                {
                    "frame-1": {
                        "frame_token": "frame-1", "scene_token": "scene-token",
                        "timestamp": 0.0, "prev": "", "next": "frame-2",
                        "instances": ["instance-1"],
                    },
                    "frame-2": {
                        "frame_token": "frame-2", "scene_token": "scene-token",
                        "timestamp": 2.0, "prev": "frame-1", "next": "",
                        "instances": ["instance-2"],
                    },
                },
            )
            instances = write_json(
                upstream / "DJI_0001_instances.json",
                {
                    "instance-1": {
                        "instance_token": "instance-1", "agent_token": "agent-token",
                        "frame_token": "frame-1", "coords": [0.0, 0.0], "heading": 0.0,
                        "speed": 0.0, "acceleration": [0.0, 0.0], "prev": "",
                        "next": "instance-2",
                    },
                    "instance-2": {
                        "instance_token": "instance-2", "agent_token": "agent-token",
                        "frame_token": "frame-2", "coords": [2.0, 0.0], "heading": 0.0,
                        "speed": 0.0, "acceleration": [0.0, 0.0], "prev": "instance-1",
                        "next": "",
                    },
                },
            )
            first_stage = [readme, parking_map, scene, agents, obstacles]
            trajectory_stage = [frames, instances]
            registry = write_json(
                root / "registry.json",
                {
                    "schema_version": 1, "source_id": "dlp-test-record",
                    "source": {"zenodo_record_id": "10084683"},
                    "first_stage_maximum_bytes": 1048576,
                    "trajectory_stage_maximum_bytes": 1048576,
                    "download_files": [registered_file(path) for path in first_stage],
                    "next_gate_files": [
                        {**registered_file(path), "purpose": "test trajectory evidence"}
                        for path in trajectory_stage
                    ],
                    "handling": {
                        "split_group": "dragon_lake/DJI_0001",
                        "truth_scope": "algorithm_validation_only",
                    },
                },
            )

            summary = run_ingest(
                registry, root / "downloads", root / "evidence",
                maximum_bytes=1048576, include_trajectories=True,
            )

            self.assertEqual(summary["counts"]["frames"], 2)
            self.assertEqual(summary["counts"]["instances"], 2)
            self.assertEqual(summary["counts"]["trajectory_feature_records"], 1)
            self.assertEqual(summary["counts"]["vehicle_trajectory_feature_records"], 1)
            self.assertEqual(summary["agent_type_counts"], {"Car": 1})
            self.assertEqual(summary["gate_status"], "single_scene_trajectory_features_validated")
            records = [
                json.loads(line)
                for line in (root / "evidence" / "trajectory_features.jsonl")
                .read_text(encoding="utf-8").splitlines()
            ]
            self.assertEqual(records[0]["agent_token"], "agent-token")
            self.assertEqual(records[0]["features"]["duration_s"], 2.0)
            self.assertEqual(records[0]["features"]["displacement_metres"], 2.0)


if __name__ == "__main__":
    unittest.main()
