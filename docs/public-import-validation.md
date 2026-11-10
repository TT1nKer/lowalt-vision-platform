# Public multi-flight import acceptance / 公开多航段导入验收

Validation date: 2026-11-10. Scope: public CPU engineering preview, Python 3.11/3.12.
The synthetic example is authored in this repository, carries no real flight or
private internship data, and requires no network access after dependencies are
installed. It is not an accuracy benchmark, annotation set or model inference.

## Reproduce

Follow the CPU quick start in [English](../README.md#quick-start) or
[中文](../README.zh-CN.md#运行). Run from the repository root:

```bash
python -m scripts.check_core
python -m scripts.reproduce_import --output runs/public-import
python platform_app.py --project-dir runs/public-import --port 7861
```

The acceptance script uses FastAPI's HTTP TestClient against the real import
service, background worker, Pillow EXIF parser, OpenCV MJPG decoder, GeoJSON
endpoint, media endpoint and ZIP exporter. No inference responses are mocked.
It fails on any discrepancy and refuses to reuse an output directory.

The map and engineering UI use the same API. Open `/engineering`, use “地图查看”
for the chosen run, and use “下载证据包” to download it. No external basemap is
required; an empty imagery catalog is valid. Frontend validation currently
covers contracts and JavaScript syntax; interactive visual acceptance was not
completed because the local browser attachment was unavailable.

## Fixed expectations

| Input | Expected result |
| --- | --- |
| `positioned.jpg` | EXIF longitude -70.5, latitude -33.5; 2026-01-02T03:04:05.125-05:00; altitude -12.5 m with EXIF sea-level reference |
| `unpositioned.jpg` | No GPS; an unrelated SRT never supplies still-photo GPS |
| `flight_a.avi` + `flight_a.SRT` | 3 sampled frames at 0/1/2 s; longitudes -79/-79.001/-79.002, latitudes 43/43.001/43.002; relative height 20 m kept separate from absolute altitude 120 m |
| `flight_b.avi` + `flight_b.srt` | 3 frames at 0/1/2 s; longitudes 151/151.001/151.002, latitudes -33/-33.001/-33.002; no reuse of flight A GPS |
| `unpaired.avi` | 3 frames without GPS despite two other sidecars in the directory |
| Missing model weights | Analyze request returns HTTP 409; no fake prediction |
| GeoJSON / ZIP | Exactly 7 points, `[longitude, latitude]`; same feature content in ZIP; all 11 asset hashes and both copied SRT hashes checked |

Outputs: `evidence.zip`, `assets.geojson`, `verification.json`, and a local
`dji_imports/<run>/manifest.json` with the full source path. The downloadable ZIP
removes the top-level source directory and per-asset absolute paths; it still
contains imagery, EXIF GPS, names, timestamps and other original media metadata.
It is an evidence bundle, not an anonymization tool. Source videos are identified
by SHA256 and are not bundled; sampled JPEGs and original SRT are bundled.
Checksums detect changes relative to the local import manifest, not authenticity
or a digital signature. HTTP temporary ZIP files are removed after delivery.

## Defects fixed and regression coverage

- Sidecars formerly shared one global position pool across all videos. Matching
  now requires exactly one case-insensitive same-stem sidecar. Missing or
  ambiguous matches stay unpositioned and produce a manifest warning.
- A nearest observation was labeled `srt_interpolated` without interpolation.
  It is now `srt_nearest`; `srt_frame` means a near observation within 1 second,
  not guaranteed frame-exact GPS. Both preserve the record index, subtitle time,
  absolute delta, source sidecar name and SHA256. Match tolerance is 5 seconds.
- Added modern bracketed DJI latitude/longitude fields and integer coordinates;
  invalid/incomplete/range-invalid pairs are never exported to a map.
- EXIF nested DateTimeOriginal, OffsetTimeOriginal, subseconds, S/W hemispheres,
  byte-valued altitude reference and malformed DMS/date handling are covered.
- Invalid frame strides are rejected with 422 before creating a run; corrupt
  media yields `partial` instead of a misleading all-success status.
- Rapid requests use unique run IDs. Re-execution cannot overwrite an existing
  imported run or its review/analysis state. Run IDs and symlink source files
  cannot escape configured project roots.
- Empty project startup no longer requires private candidate/overview files.
  The engineering page reports absent candidates/models without claiming that
  jobs completed or a model was used. Each run's map link selects that run.
- Exports retain unreviewed metadata observations, unknown scale, evidence hashes
  and sidecars. Changed evidence is rejected. Existing model-training and release
  approval gates remain enabled and unchanged.

## Test accounting

Baseline `cc4bfc3`, clean archive with the same CPU environment on Python 3.11.16:
178 discovered tests, **8 import errors and 7 skips**. Missing source packages
`uav_data` and `equipment_data` are absent from the published repository; this
work does not reconstruct their research algorithms from tests.

Published core: 189 tests, **182 pass and 7 skip** locally on Python 3.11.16.
This includes 19 new import-integrity/HTTP tests. A clean source archive with a
fresh project virtual environment also passes the suite and standalone case. Seven model tests skip because
`torch`/`transformers` are absent. The core runner names these eight historical
modules as outside the published scope (they are not counted as passing):

```text
test_dlp_adapter                 test_equipment_ontology
test_parking_e3_ingest           test_parking_e3_trajectory_ingest
test_trajectory_features        test_uav_data_schema
test_uav_physical_scale          test_uav_source_ingest
```

`python -m unittest discover -s tests -t .` still exposes those missing-source
errors. CI runs the declared core, frontend syntax check, and the standalone
acceptance on Python 3.11/3.12. CI artifacts are clearly labeled synthetic.
No performance, detection accuracy, real-user acceptance or GPU inference claim
is made.

## Supported boundary

- Constant-frame-rate video with a codec available to OpenCV. Relative frame
  time is `frame_index / nominal_fps`; VFR/decoder PTS alignment is not supported.
  FPS must be positive and finite. Default extraction cap: 480 frames/video.
- SRT forms: `GPS(latitude, longitude, optional_altitude)`, labeled `lat/lon`,
  and `[latitude: ...] [longitude: ...] [rel_alt: ... abs_alt: ...]`.
  The legacy GPS form explicitly uses **latitude first**. Longitude-first legacy
  variants cannot always be disambiguated; normalize them before import. There
  is no axis guessing, GPS interpolation, clock synchronization or RTK claim.
- EXIF times with a valid offset retain it; times without one remain local times
  with unknown timezone. Frame absolute capture time remains unknown.
- WGS84 camera point observations are not ground footprints or orthorectification.
  `H` / `rel_alt` is relative height; absolute SRT vertical datum is unspecified.
  Altitude alone cannot establish GSD, ground area, parking area or object size.
  `gsd_m` is null in exports; no physical-scale measurement is claimed here.
- All imported observations are `unreviewed`; model results remain candidates.
  The synthetic case never supplies training labels, review approval, a fixed
  evaluation set, or a release-gate bypass. Real model evaluation still needs
  licensed data and labels, immutable train/validation/test separation, review
  provenance, weights and hardware. None is supplied or fabricated by this case.
- The service binds localhost by default. It is a local filesystem application,
  not a hardened multi-user upload service. No server deployment is included.

中文摘要：本次完成的是可复现的数据与服务闭环；合成素材验证定位、时间、来源和导出一致性，
不证明模型准确率。8 个历史模块缺失源码、7 个模型测试跳过、变帧率视频与真实模型评估均明确保留为边界。
