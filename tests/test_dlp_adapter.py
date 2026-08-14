import json
from pathlib import Path
import tempfile
import unittest

from uav_data.dlp_adapter import DLP_LOCAL_CRS, load_dlp_scene


SCENE_TOKEN = "scene-token"
AGENT_TOKEN = "agent-token"


def write_json(path: Path, value: object) -> Path:
    path.write_text(json.dumps(value), encoding="utf-8")
    return path


def valid_source_files(root: Path) -> tuple[Path, Path, Path]:
    scene = write_json(
        root / "DJI_0001_scene.json",
        {
            "scene_token": SCENE_TOKEN,
            "filename": "DJI_0001",
            "timestamp": 100.0,
            "first_frame": "frame-1",
            "last_frame": "frame-2",
            "agents": [AGENT_TOKEN],
            "obstacles": ["obstacle-token"],
        },
    )
    agents = write_json(
        root / "DJI_0001_agents.json",
        {
            AGENT_TOKEN: {
                "agent_token": AGENT_TOKEN,
                "scene_token": SCENE_TOKEN,
                "type": "Car",
                "size": [4.5, 2.0],
                "first_instance": "instance-1",
                "last_instance": "instance-2",
            }
        },
    )
    obstacles = write_json(
        root / "DJI_0001_obstacles.json",
        {
            "obstacle-token": {
                "obstacle_token": "obstacle-token",
                "scene_token": SCENE_TOKEN,
                "type": "Car",
                "size": [4.4, 2.1],
                "coords": [12.0, 30.0],
                "heading": 1.5,
            }
        },
    )
    return scene, agents, obstacles


class DlpAdapterTests(unittest.TestCase):
    def test_real_calendar_scene_timestamp_is_preserved_without_seconds_conversion(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            scene_path, agents_path, obstacles_path = valid_source_files(root)
            raw_scene = json.loads(scene_path.read_text(encoding="utf-8"))
            raw_scene["timestamp"] = "2020-06-20 15:03:24"
            write_json(scene_path, raw_scene)

            scene = load_dlp_scene(scene_path, agents_path, obstacles_path)

            self.assertEqual(scene.source_timestamp, "2020-06-20 15:03:24")

    def test_preserves_tokens_local_crs_and_static_obstacle_boundary(self):
        with tempfile.TemporaryDirectory() as directory:
            scene_path, agents_path, obstacles_path = valid_source_files(Path(directory))

            scene = load_dlp_scene(scene_path, agents_path, obstacles_path)

            self.assertEqual(scene.scene_token, SCENE_TOKEN)
            self.assertEqual(scene.coordinate_reference, DLP_LOCAL_CRS)
            self.assertEqual(scene.agents[AGENT_TOKEN].agent_token, AGENT_TOKEN)
            self.assertEqual(scene.obstacles["obstacle-token"].coords_metres, (12.0, 30.0))
            self.assertEqual(len(scene.instances), 0)

    def test_foreign_scene_agent_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            scene_path, agents_path, obstacles_path = valid_source_files(root)
            agents = json.loads(agents_path.read_text(encoding="utf-8"))
            agents[AGENT_TOKEN]["scene_token"] = "another-scene"
            write_json(agents_path, agents)

            with self.assertRaisesRegex(ValueError, "foreign scene"):
                load_dlp_scene(scene_path, agents_path, obstacles_path)

    def test_broken_instance_chain_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            scene_path, agents_path, obstacles_path = valid_source_files(root)
            frames_path = write_json(
                root / "DJI_0001_frames.json",
                {
                    "frame-1": {
                        "frame_token": "frame-1",
                        "scene_token": SCENE_TOKEN,
                        "timestamp": 0.0,
                        "prev": "",
                        "next": "frame-2",
                        "instances": ["instance-1"],
                    },
                    "frame-2": {
                        "frame_token": "frame-2",
                        "scene_token": SCENE_TOKEN,
                        "timestamp": 1.0,
                        "prev": "frame-1",
                        "next": "",
                        "instances": ["instance-2"],
                    },
                },
            )
            instances_path = write_json(
                root / "DJI_0001_instances.json",
                {
                    "instance-1": {
                        "instance_token": "instance-1",
                        "agent_token": AGENT_TOKEN,
                        "frame_token": "frame-1",
                        "coords": [0.0, 0.0],
                        "heading": 0.0,
                        "speed": 0.0,
                        "acceleration": [0.0, 0.0],
                        "prev": "",
                        "next": "instance-2",
                    },
                    "instance-2": {
                        "instance_token": "instance-2",
                        "agent_token": AGENT_TOKEN,
                        "frame_token": "frame-2",
                        "coords": [1.0, 0.0],
                        "heading": 0.0,
                        "speed": 1.0,
                        "acceleration": [0.0, 0.0],
                        "prev": "wrong-token",
                        "next": "",
                    },
                },
            )

            with self.assertRaisesRegex(ValueError, "broken instance chain"):
                load_dlp_scene(
                    scene_path,
                    agents_path,
                    obstacles_path,
                    frames_path=frames_path,
                    instances_path=instances_path,
                )

    def test_loaded_trajectory_uses_frame_timestamps(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            scene_path, agents_path, obstacles_path = valid_source_files(root)
            frames_path = write_json(
                root / "DJI_0001_frames.json",
                {
                    "frame-1": {
                        "frame_token": "frame-1", "scene_token": SCENE_TOKEN,
                        "timestamp": 0.0, "prev": "", "next": "frame-2",
                        "instances": ["instance-1"],
                    },
                    "frame-2": {
                        "frame_token": "frame-2", "scene_token": SCENE_TOKEN,
                        "timestamp": 1.0, "prev": "frame-1", "next": "",
                        "instances": ["instance-2"],
                    },
                },
            )
            instances_path = write_json(
                root / "DJI_0001_instances.json",
                {
                    "instance-1": {
                        "instance_token": "instance-1", "agent_token": AGENT_TOKEN,
                        "frame_token": "frame-1", "coords": [0.0, 0.0],
                        "heading": 0.0, "speed": 0.0, "acceleration": [0.0, 0.0],
                        "prev": "", "next": "instance-2",
                    },
                    "instance-2": {
                        "instance_token": "instance-2", "agent_token": AGENT_TOKEN,
                        "frame_token": "frame-2", "coords": [1.0, 0.0],
                        "heading": 0.0, "speed": 1.0, "acceleration": [0.0, 0.0],
                        "prev": "instance-1", "next": "",
                    },
                },
            )

            scene = load_dlp_scene(
                scene_path, agents_path, obstacles_path,
                frames_path=frames_path, instances_path=instances_path,
            )
            trajectory = scene.agent_trajectory(AGENT_TOKEN)

            self.assertEqual([point.timestamp_s for point in trajectory], [0.0, 1.0])
            self.assertEqual([point.x_metres for point in trajectory], [0.0, 1.0])


if __name__ == "__main__":
    unittest.main()
