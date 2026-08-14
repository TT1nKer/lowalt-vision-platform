from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from PIL import Image

from lowalt_platform.services.dji_metadata import extract_image_metadata, parse_srt, srt_nearest_position
from lowalt_platform.services.import_runner import ImportService

HAS_CV2 = importlib.util.find_spec("cv2") is not None

DJI_SRT = """1
00:00:00,000 --> 00:00:00,500
F/2.8, SS 1/50, ISO 100, EV 0, GPS (30.123456, 120.654321, 15.0), D 3.2m, H 2.0m, H.S 0.0m/s, V.S 0.0m/s

2
00:00:01,000 --> 00:00:01,500
F/2.8, SS 1/50, ISO 100, EV 0, GPS (30.124000, 120.655000, 16.0), D 3.2m, H 2.0m, H.S 0.0m/s, V.S 0.0m/s

3
00:00:02,000 --> 00:00:02,500
F/2.8, SS 1/50, ISO 100, EV 0, GPS (30.124500, 120.655500, 17.0), D 3.2m, H 2.0m, H.S 0.0m/s, V.S 0.0m/s
"""


def write_gps_jpeg(path: Path) -> None:
    image = Image.new("RGB", (16, 16), (120, 40, 20))
    exif = Image.Exif()
    exif[0x010F] = "DJI"
    exif[0x0110] = "FC3582"
    exif[0x9003] = "2026:08:14 09:30:00"
    exif[0x8825] = {
        1: "N",
        2: (30.0, 7.0, 12.0),
        3: "E",
        4: (120.0, 30.0, 6.0),
        5: 0,
        6: 150.0,
    }
    image.save(path, format="JPEG", exif=exif)


class SrtParsingTests(unittest.TestCase):
    def test_parses_dji_gps_records_with_timestamps(self) -> None:
        records = parse_srt(DJI_SRT)
        self.assertEqual(len(records), 3)
        self.assertAlmostEqual(records[0]["time_seconds"], 0.0)
        self.assertAlmostEqual(records[0]["latitude"], 30.123456)
        self.assertAlmostEqual(records[0]["longitude"], 120.654321)
        self.assertAlmostEqual(records[0]["altitude_m"], 15.0)
        self.assertAlmostEqual(records[2]["time_seconds"], 2.0)

    def test_parses_labeled_lat_lon_variant(self) -> None:
        text = "1\n00:00:05,000 --> 00:00:05,500\nlat: 31.25, lon: 121.50, ALT 8.0m\n"
        records = parse_srt(text)
        self.assertEqual(len(records), 1)
        self.assertAlmostEqual(records[0]["latitude"], 31.25)
        self.assertAlmostEqual(records[0]["longitude"], 121.50)
        self.assertAlmostEqual(records[0]["altitude_m"], 8.0)

    def test_nearest_position_labels_exact_interpolated_and_none(self) -> None:
        records = parse_srt(DJI_SRT)
        _, exact = srt_nearest_position(records, 1.05)
        self.assertEqual(exact, "srt_frame")
        record, interpolated = srt_nearest_position(records, 6.0)
        self.assertIsNotNone(record)
        self.assertEqual(interpolated, "srt_interpolated")
        record, none_source = srt_nearest_position(records, 60.0)
        self.assertIsNone(record)
        self.assertEqual(none_source, "none")


class ExifExtractionTests(unittest.TestCase):
    def test_extracts_gps_time_and_camera(self) -> None:
        with TemporaryDirectory() as temporary_directory:
            path = Path(temporary_directory) / "dji.jpg"
            write_gps_jpeg(path)
            metadata = extract_image_metadata(path)
            self.assertAlmostEqual(metadata["latitude"], 30.12, places=5)
            self.assertAlmostEqual(metadata["longitude"], 120.501667, places=5)
            self.assertAlmostEqual(metadata["altitude_m"], 150.0)
            self.assertEqual(metadata["camera_make"], "DJI")
            self.assertEqual(metadata["captured_at"], "2026-08-14T09:30:00")
            self.assertEqual(metadata["width"], 16)


class ImportServiceTests(unittest.TestCase):
    def setUp(self) -> None:
        self._temporary = TemporaryDirectory()
        root = Path(self._temporary.name)
        self.source = root / "source"
        self.source.mkdir()
        self.import_root = root / "dji_imports"
        self.service = ImportService(self.import_root, (), root)

    def tearDown(self) -> None:
        self._temporary.cleanup()

    def test_rejects_sources_outside_allowed_roots(self) -> None:
        outside = Path(self._temporary.name).parent
        self.assertFalse(self.service.is_allowed_source(outside))
        with self.assertRaises(ValueError):
            self.service.scan_source(outside)

    def test_scan_lists_images_videos_and_srt(self) -> None:
        (self.source / "a.jpg").write_bytes(b"jpeg-bytes")
        (self.source / "b.MP4").write_bytes(b"video-bytes")
        (self.source / "b.SRT").write_text(DJI_SRT, encoding="utf-8")
        scan = self.service.scan_source(self.source)
        self.assertEqual([item["name"] for item in scan["images"]], ["a.jpg"])
        self.assertEqual([item["name"] for item in scan["videos"]], ["b.MP4"])
        self.assertEqual([item["name"] for item in scan["srt"]], ["b.SRT"])

    def test_import_copies_image_and_resolves_gps_from_exif_and_srt(self) -> None:
        write_gps_jpeg(self.source / "photo.jpg")
        (self.source / "flight.SRT").write_text(DJI_SRT, encoding="utf-8")
        run_id = self.service.create_run(self.source, "flight1")
        manifest = self.service.execute_import(run_id)
        self.assertEqual(manifest["status"], "done")
        self.assertEqual(manifest["counts"]["images"], 1)
        self.assertEqual(manifest["counts"]["srt"], 1)
        self.assertEqual(len(manifest["assets"]), 1)
        asset = manifest["assets"][0]
        self.assertEqual(asset["kind"], "image")
        self.assertEqual(asset["gps_source"], "exif")
        self.assertAlmostEqual(asset["latitude"], 30.12, places=5)
        self.assertTrue((self.import_root / run_id / "assets" / asset["file"]).is_file())
        runs = self.service.list_runs()
        self.assertEqual(len(runs), 1)
        self.assertEqual(runs[0].georeferenced_assets, 1)

    def test_image_without_exif_gps_stays_unpositioned(self) -> None:
        Image.new("RGB", (8, 8), (10, 20, 30)).save(self.source / "plain.jpg", format="JPEG")
        (self.source / "flight.SRT").write_text(DJI_SRT, encoding="utf-8")
        run_id = self.service.create_run(self.source)
        manifest = self.service.execute_import(run_id)
        asset = manifest["assets"][0]
        self.assertEqual(asset["gps_source"], "none")
        self.assertIsNone(asset["latitude"])

    def test_geojson_contains_only_georeferenced_assets(self) -> None:
        write_gps_jpeg(self.source / "photo.jpg")
        Image.new("RGB", (8, 8), (10, 20, 30)).save(self.source / "plain.jpg", format="JPEG")
        run_id = self.service.create_run(self.source)
        self.service.execute_import(run_id)
        collection = self.service.geojson(run_id)
        self.assertEqual(len(collection["features"]), 1)
        self.assertEqual(collection["features"][0]["properties"]["gps_source"], "exif")

    @unittest.skipUnless(HAS_CV2, "opencv-python not installed")
    def test_video_frames_extracted_with_srt_position(self) -> None:
        import cv2

        import numpy as np

        video_path = self.source / "clip.mp4"
        writer = cv2.VideoWriter(str(video_path), cv2.VideoWriter_fourcc(*"mp4v"), 10.0, (64, 64))
        self.assertTrue(writer.isOpened())
        for _ in range(30):
            writer.write(np.zeros((64, 64, 3), dtype=np.uint8))
        writer.release()
        (self.source / "clip.SRT").write_text(DJI_SRT, encoding="utf-8")
        run_id = self.service.create_run(self.source)
        manifest = self.service.execute_import(run_id, frame_stride=5)
        frames = [asset for asset in manifest["assets"] if asset["kind"] == "frame"]
        self.assertEqual(len(frames), 6)
        first = frames[0]
        self.assertEqual(first["video"]["fps"], 10.0)
        self.assertEqual(first["gps_source"], "srt_frame")
        self.assertIsNotNone(first["latitude"])


if __name__ == "__main__":
    unittest.main()
