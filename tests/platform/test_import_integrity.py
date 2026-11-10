from __future__ import annotations

from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from fastapi.testclient import TestClient
from PIL import Image

from examples.public_import.fixture import make_fixture
from lowalt_platform.api.routes import create_platform_app
from lowalt_platform.services.dji_metadata import extract_image_metadata, parse_srt, srt_nearest_position
from lowalt_platform.services.import_runner import ImportService
from lowalt_platform.settings import PlatformSettings
from scripts.reproduce_import import reproduce


class MetadataIntegrityTests(unittest.TestCase):
    def test_modern_bracketed_dji_fields_keep_relative_and_absolute_altitude_separate(self):
        records = parse_srt('1\r\n00:00:02,000 --> 00:00:02,500\r\n[latitude: -33] [longitude: 151] [rel_alt: 12 abs_alt: 80]\r\n')
        self.assertEqual(len(records), 1)
        self.assertEqual((records[0]['latitude'], records[0]['longitude']), (-33., 151.))
        self.assertEqual(records[0]['relative_altitude_m'], 12)
        self.assertEqual(records[0]['altitude_m'], 80)
        self.assertEqual(records[0]['altitude_reference'], 'srt_absolute_unspecified_datum')

    def test_height_is_not_sea_level_altitude(self):
        record = parse_srt('1\n00:00:00,000 --> 00:00:01,000\nlat: 30, lon: 120, H 7m')[0]
        self.assertEqual(record['relative_altitude_m'], 7)
        self.assertIsNone(record['altitude_m'])

    def test_invalid_or_incomplete_srt_gps_does_not_enter_map(self):
        for fields in ('lat: 91, lon: 120', 'lat: 30, lon: 181', '[latitude: 30]', 'GPS (120, 30, 5)', 'lat: 30e2, lon: 120'):
            record = parse_srt('1\n00:00:00,000 --> 00:00:01,000\n' + fields)[0]
            self.assertIsNone(record['latitude'])
            self.assertIsNone(record['longitude'])
            self.assertEqual(srt_nearest_position([record], 0), (None, 'none'))

    def test_invalid_timestamps_and_out_of_tolerance_are_not_matched(self):
        self.assertEqual(parse_srt('1\n00:90:00,000 --> 00:90:01,000\nlat: 30, lon: 120'), [])
        records = parse_srt('1\n00:00:00,000 --> 00:00:01,000\nlat: 30, lon: 120')
        self.assertEqual(srt_nearest_position(records, 5.01), (None, 'none'))
        self.assertEqual(srt_nearest_position(records, 2)[1], 'srt_nearest')
        self.assertEqual(srt_nearest_position(records, .3, tolerance=.2), (None, 'none'))
        self.assertEqual(srt_nearest_position(records, float('nan')), (None, 'none'))

    def test_body_clock_is_not_a_subtitle_timeline_and_reversed_interval_is_rejected(self):
        self.assertEqual(parse_srt('1\nGPS (30, 120, 10) clock 00:00:02,000'), [])
        self.assertEqual(parse_srt('1\n00:00:02,000 --> 00:00:01,000\nGPS (30, 120, 10)'), [])

    def test_overflowed_altitude_does_not_export_nonfinite_json(self):
        record = parse_srt('1\n00:00:00,000 --> 00:00:01,000\nlat: 30, lon: 120, ALT ' + '9' * 400 + 'm')[0]
        self.assertIsNone(record['altitude_m'])

    def test_exif_nested_timestamp_hemispheres_and_byte_altitude_reference(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            make_fixture(root)
            metadata = extract_image_metadata(root / 'positioned.jpg')
            self.assertEqual(metadata['captured_at'], '2026-01-02T03:04:05.125-05:00')
            self.assertEqual((metadata['longitude'], metadata['latitude']), (-70.5, -33.5))
            self.assertEqual(metadata['altitude_m'], -12.5)
            self.assertEqual(metadata['altitude_reference'], 'exif_sea_level')

    def test_malformed_exif_coordinates_and_date_stay_unknown(self):
        with TemporaryDirectory() as directory:
            path = Path(directory) / 'bad.jpg'
            for gps in ({1: 'N', 2: (30., 90., 0.), 3: 'E', 4: (120., 0., 0.)},
                        {2: (30., 0., 0.), 3: 'E', 4: (120., 0., 0.)},
                        {1: 'N', 2: (95., 0., 0.), 3: 'E', 4: (120., 0., 0.)}):
                exif = Image.Exif()
                exif[0x9003] = '2026:02:30 03:00:00'
                exif[0x8825] = gps
                Image.new('RGB', (16, 16)).save(path, exif=exif)
                metadata = extract_image_metadata(path)
                self.assertIsNone(metadata['latitude'])
                self.assertIsNone(metadata['longitude'])
                self.assertIsNone(metadata['captured_at'])
                self.assertIn('invalid_or_incomplete_exif_gps', metadata['warnings'])


class ImportIntegrityTests(unittest.TestCase):
    def setUp(self):
        self.temp = TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.source = self.root / 'source'
        self.source.mkdir()
        self.service = ImportService(self.root / 'imports', (), self.root)

    def tearDown(self):
        self.temp.cleanup()

    def test_rapid_run_creation_is_unique_and_pins_absolute_source(self):
        first = self.service.create_run(self.source)
        second = self.service.create_run(self.source)
        self.assertNotEqual(first, second)
        self.assertTrue(Path(self.service.run_detail(first)['source_dir']).is_absolute())

    def test_run_path_traversal_is_rejected(self):
        for run_id in ('..', '../source', str(self.source), 'run-../../source'):
            with self.assertRaises(KeyError):
                self.service.run_detail(run_id)

    def test_symlink_files_cannot_escape_allowed_source_roots(self):
        with TemporaryDirectory() as external:
            secret = Path(external) / 'outside.jpg'
            Image.new('RGB', (8, 8)).save(secret)
            (self.source / 'escape.jpg').symlink_to(secret)
            self.assertEqual(self.service.scan_source(self.source)['images'], [])
            run_id = self.service.create_run(self.source)
            self.assertEqual(self.service.execute_import(run_id)['assets'], [])

    def test_import_is_not_silently_reexecuted_over_review_or_analysis(self):
        run_id = self.service.create_run(self.source)
        self.service.execute_import(run_id)
        with self.assertRaises(ValueError):
            self.service.execute_import(run_id)

    def test_corrupt_media_is_partial_failure_not_success(self):
        (self.source / 'broken.jpg').write_bytes(b'not an image')
        run_id = self.service.create_run(self.source)
        manifest = self.service.execute_import(run_id)
        self.assertEqual(manifest['status'], 'partial')
        self.assertEqual(manifest['counts']['failed'], 1)
        self.assertEqual(manifest['assets'], [])

    def test_ambiguous_sidecar_is_not_guessed(self):
        make_fixture(self.source)
        (self.source / 'FLIGHT_A.srt').write_text((self.source / 'flight_a.SRT').read_text())
        run_id = self.service.create_run(self.source)
        manifest = self.service.execute_import(run_id, frame_stride=10)
        frames = [asset for asset in manifest['assets'] if asset['original_name'] == 'flight_a.avi']
        self.assertEqual(len(frames), 3)
        self.assertTrue(all(asset['gps_source'] == 'none' for asset in frames))
        self.assertTrue(any('ambiguous' in warning for warning in manifest['warnings']))

    def test_export_rejects_changed_evidence_and_cleans_failed_archive(self):
        Image.new('RGB', (8, 8)).save(self.source / 'a.jpg')
        run_id = self.service.create_run(self.source)
        manifest = self.service.execute_import(run_id)
        self.service.asset_path(run_id, manifest['assets'][0]['asset_id']).write_bytes(b'changed')
        with self.assertRaisesRegex(ValueError, 'checksum changed'):
            self.service.export_evidence(run_id)
        self.assertEqual(list((self.root / 'imports' / run_id).glob('evidence-*.zip')), [])

    def test_export_pending_run_is_blocked(self):
        run_id = self.service.create_run(self.source)
        with self.assertRaisesRegex(ValueError, 'finish'):
            self.service.export_evidence(run_id)

    def test_partial_gps_from_old_manifest_is_omitted_from_counts_and_geojson(self):
        run_id = self.service.create_run(self.source)
        manifest = self.service._read_manifest(run_id)
        manifest['assets'] = [{'asset_id': 'legacy', 'latitude': 30, 'longitude': None}]
        self.service._write_manifest(self.service._run_dir(run_id), manifest)
        self.assertEqual(self.service.geojson(run_id)['features'], [])
        self.assertEqual(self.service.run_detail(run_id)['georeferenced_assets'], 0)
        self.assertEqual(self.service.list_runs()[0].georeferenced_assets, 0)


class PublicHttpAcceptanceTests(unittest.TestCase):
    def test_public_scenario_validates_real_media_import_and_export(self):
        with TemporaryDirectory() as directory:
            report = reproduce(Path(directory) / 'acceptance')
            self.assertEqual(report['georeferenced'], 7)
            self.assertFalse(report['inference_executed'])

    def test_empty_project_starts_and_rejects_bad_stride_before_creating_runs(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            client = TestClient(create_platform_app(PlatformSettings.from_project(root)))
            with client:
                self.assertIn('id="map"', client.get('/').text)
                self.assertEqual(client.get('/api/platform/summary').json()['imagery']['count'], 0)
                self.assertEqual(client.get('/api/platform/imagery/blocks').json()['blocks'], [])
                for stride in (0, -1, 601, 1.5, True, '30', None):
                    self.assertEqual(client.post('/api/platform/imports', json={'source_dir': str(root), 'frame_stride': stride}).status_code, 422)
                self.assertEqual(client.get('/api/platform/imports').json()['runs'], [])
