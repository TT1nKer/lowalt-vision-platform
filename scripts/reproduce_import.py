"""Run a real CPU import and HTTP/export acceptance without model services.

Run from repository root: python -m scripts.reproduce_import --output runs/public-import
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import time
from zipfile import ZipFile
import io

from fastapi.testclient import TestClient

from examples.public_import.fixture import make_fixture
from lowalt_platform.api.routes import create_platform_app
from lowalt_platform.settings import PlatformSettings


def reproduce(output: Path) -> dict:
    output = output.resolve()
    # Never silently overwrite an earlier acceptance artifact.
    output.mkdir(parents=True, exist_ok=False)
    source = output / 'source'
    expected = make_fixture(source)
    settings = PlatformSettings.from_project(output)
    client = TestClient(create_platform_app(settings))
    with client:
        summary = client.get('/api/platform/summary').raise_for_status().json()
        assert summary['parking_model']['available'] is False
        assert summary['overview'] is None
        assert summary['parking_candidates']['total'] == 0
        assert client.get('/api/platform/candidates').json()['features'] == []
        assert len(client.post('/api/platform/imports/scan', json={'source_dir': str(source)}).raise_for_status().json()['videos']) == 3
        response = client.post('/api/platform/imports', json={'source_dir': str(source), 'name': 'SYNTHETIC public acceptance', 'frame_stride': 10})
        run_id = response.raise_for_status().json()['run_id']
        deadline = time.monotonic() + 30
        while True:
            manifest = client.get(f'/api/platform/imports/{run_id}').raise_for_status().json()
            if manifest['status'] in {'done', 'partial', 'error'}:
                break
            if time.monotonic() >= deadline:
                raise TimeoutError('import did not complete within 30 seconds')
            time.sleep(.02)
        assert manifest['status'] == 'done', manifest
        assert manifest['counts'] == {'images': 2, 'videos': 3, 'srt': 2, 'frames_extracted': 9, 'failed': 0}
        assert manifest['georeferenced_assets'] == 7
        assets = manifest['assets']
        for asset in assets:
            name = asset['original_name']
            assert asset['source_sha256'] == hashlib.sha256((source / name).read_bytes()).hexdigest()
            if name == 'positioned.jpg':
                assert [asset['longitude'], asset['latitude']] == expected[name]
                assert asset['captured_at'] == '2026-01-02T03:04:05.125-05:00'
                assert asset['altitude_m'] == -12.5
            elif name in {'flight_a.avi', 'flight_b.avi'}:
                video = asset['video']
                second = video['frame_index'] // 10
                assert video['time_seconds'] == second
                assert [asset['longitude'], asset['latitude']] == expected[name][second]
                assert video['srt_delta_seconds'] == 0
                assert Path(video['srt_name']).stem == Path(name).stem
                assert asset['gps_source'] == 'srt_frame'
            else:
                assert asset['latitude'] is None and asset['longitude'] is None
        collection = client.get(f'/api/platform/imports/{run_id}/geojson').raise_for_status().json()
        assert len(collection['features']) == 7
        by_id = {asset['asset_id']: asset for asset in assets}
        for feature in collection['features']:
            asset = by_id[feature['properties']['asset_id']]
            assert feature['geometry']['coordinates'] == [asset['longitude'], asset['latitude']]
            assert feature['properties']['review_status'] == 'unreviewed'
            assert feature['properties']['gsd_m'] is None
            assert client.get(f"/api/platform/imports/{run_id}/assets/{asset['asset_id']}/image").status_code == 200
        assert client.post(f'/api/platform/imports/{run_id}/analyze', json={}).status_code == 409
        exported = client.get(f'/api/platform/imports/{run_id}/export').raise_for_status()
        (output / 'evidence.zip').write_bytes(exported.content)
        (output / 'assets.geojson').write_text(json.dumps(collection, indent=2), encoding='utf-8')
        with ZipFile(io.BytesIO(exported.content)) as archive:
            packaged = json.loads(archive.read('manifest.json'))
            assert 'source_dir' not in packaged
            assert json.loads(archive.read('assets.geojson')) == collection
            for asset in packaged['assets']:
                assert 'source' not in asset
                payload = archive.read('assets/' + asset['file'])
                assert hashlib.sha256(payload).hexdigest() == asset['asset_sha256']
            for sidecar in packaged['sidecars']:
                assert hashlib.sha256(archive.read('sidecars/' + sidecar['file'])).hexdigest() == sidecar['sha256']
        assert not list((output / 'dji_imports' / run_id).glob('evidence-*.zip')), 'HTTP temporary ZIP not cleaned'
    report = {'scenario': 'synthetic_import_engineering_acceptance', 'run_id': run_id,
              'assets': 11, 'georeferenced': 7, 'unpositioned': 4,
              'inference_executed': False, 'accuracy_claim': None,
              'checks': ['same-stem SRT isolation', 'EXIF hemisphere/time/altitude',
                         'frame time and lon/lat order', 'unknown GPS retained',
                         'HTTP media', 'GeoJSON/ZIP equality', 'SHA256 evidence integrity',
                         'missing-model analysis blocked']}
    (output / 'verification.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=Path('runs/public-import'))
    args = parser.parse_args()
    print(json.dumps(reproduce(args.output), indent=2))


if __name__ == '__main__':
    main()
