import hashlib
import json
from pathlib import Path
import tempfile
import unittest

from parking.parking_e3_ingest import run_ingest


SCENE_TOKEN = "scene-token"


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


class ParkingE3IngestTests(unittest.TestCase):
    def test_real_files_generate_provenance_and_sequence_outputs(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            upstream = root / "upstream"
            upstream.mkdir()
            readme = (upstream / "README.md")
            readme.write_text("DLP fixture", encoding="utf-8")
            parking_map = upstream / "parking_map.yml"
            parking_map.write_text(
                "MAP_SIZE: {x: 140, y: 80}\n"
                "PARKING_AREAS: {A: {bounds: [[0, 0], [1, 1]]}}\n"
                "WAYPOINTS: {R1: {bounds: [[0, 0], [1, 0]], nums: 2}}\n",
                encoding="utf-8",
            )
            scene = write_json(
                upstream / "DJI_0001_scene.json",
                {
                    "scene_token": SCENE_TOKEN,
                    "filename": "DJI_0001",
                    "timestamp": 100.0,
                    "first_frame": "frame-1",
                    "last_frame": "frame-2",
                    "agents": ["agent-token"],
                    "obstacles": ["obstacle-token"],
                },
            )
            agents = write_json(
                upstream / "DJI_0001_agents.json",
                {
                    "agent-token": {
                        "agent_token": "agent-token", "scene_token": SCENE_TOKEN,
                        "type": "Car", "size": [4.5, 2.0],
                        "first_instance": "instance-1", "last_instance": "instance-2",
                    }
                },
            )
            obstacles = write_json(
                upstream / "DJI_0001_obstacles.json",
                {
                    "obstacle-token": {
                        "obstacle_token": "obstacle-token", "scene_token": SCENE_TOKEN,
                        "type": "Car", "size": [4.4, 2.1],
                        "coords": [12.0, 30.0], "heading": 1.5,
                    }
                },
            )
            files = [readme, parking_map, scene, agents, obstacles]
            registry = write_json(
                root / "registry.json",
                {
                    "schema_version": 1,
                    "source_id": "dlp-test-record",
                    "source": {"zenodo_record_id": "10084683"},
                    "first_stage_maximum_bytes": 1048576,
                    "download_files": [registered_file(path) for path in files],
                    "handling": {
                        "split_group": "dragon_lake/DJI_0001",
                        "truth_scope": "algorithm_validation_only",
                    },
                },
            )
            downloads = root / "downloads"
            evidence = root / "evidence"

            summary = run_ingest(
                registry,
                downloads,
                evidence,
                maximum_bytes=1048576,
            )

            self.assertEqual(summary["source_id"], "dlp-test-record")
            self.assertEqual(summary["split_group"], "dragon_lake/DJI_0001")
            self.assertEqual(summary["truth_scope"], "algorithm_validation_only")
            self.assertEqual(summary["coordinate_reference"], "LOCAL:DLP_DRAGON_LAKE_METRES")
            self.assertEqual(summary["counts"]["agents"], 1)
            self.assertEqual(summary["counts"]["static_obstacles"], 1)
            self.assertEqual(summary["parking_map"]["parking_area_groups"], 1)
            self.assertEqual(summary["parking_map"]["waypoint_groups"], 1)
            self.assertEqual(len(summary["files"]), 5)
            self.assertTrue(all(len(item["sha256"]) == 64 for item in summary["files"]))
            self.assertEqual(summary["gate_status"], "metadata_validated_instances_not_downloaded")

            manifest = json.loads(
                (evidence / "sequence_manifest.json").read_text(encoding="utf-8")
            )
            self.assertEqual(manifest["split_group"], "dragon_lake/DJI_0001")
            self.assertEqual(
                manifest["sequence"]["spatial_reference"]["crs"],
                "LOCAL:DLP_DRAGON_LAKE_METRES",
            )
            self.assertEqual(manifest["sequence"]["source_kind"], "uav_video_sequence")
            self.assertNotIn("capture_start_s", manifest["sequence"])
            self.assertEqual(manifest["sequence"]["attributes"]["source_timestamp"], 100.0)
            self.assertEqual(manifest["frames"], [])
            self.assertTrue((evidence / "ingest_summary.json").is_file())


if __name__ == "__main__":
    unittest.main()
