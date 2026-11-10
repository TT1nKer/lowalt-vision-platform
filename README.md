# Low-Altitude Remote Sensing Vision Platform

**English** | [简体中文](README.zh-CN.md)

An extensible computer-vision platform for analyzing orthomosaics, UAV imagery,
and georeferenced video. It combines geospatial asset ingestion, model-assisted
candidate discovery, evidence-aware review, and lightweight-model training in
two independently deployable web applications.

The public CPU acceptance case is **multi-flight media import → GPS/time
provenance → map points → an integrity-checked evidence ZIP**. It runs with
synthetic, clearly marked fixtures and no private imagery, weights or remote
services. See [the validation record](docs/public-import-validation.md) for the
supported formats, exact checks and remaining limitations.

> Model detections are treated as candidates—not ground truth—until they pass
> the configured quality gates and human review.

## Applications

| Port | Application | Responsibility |
| --- | --- | --- |
| `7860` | Vision Task Console | General-purpose model-distillation workflow: define a target, discover candidates with SAM3, review semantics with Gemma, export segmentation/oriented-box datasets, and train lightweight YOLO models. |
| `7861` | Remote Sensing Analysis Platform | Orthomosaic map and candidate layers, DJI photo/video ingestion, EXIF/SRT geolocation, video frame extraction, SegFormer-based parking analysis, and an engineering workbench. |

## Core Capabilities

- **Parking-facility discovery:** run SegFormer-large-parking inference, classify
  evidence by strength (vehicle-row support, detected vehicles, or model-only
  candidates), and inspect results interactively on a map.
- **DJI asset ingestion:** retain image GPS, timestamp, and camera metadata;
  extract video frames at a configurable interval; geolocate frames from a
  matching SRT file; and add imported assets back to the map as point layers.
- **Extensible analysis services:** register independent, testable services in a
  capability catalog. Missing models or data are reported as unavailable rather
  than replaced with synthetic demo output.
- **Quality-controlled model distillation:** enforce dataset, review-provenance,
  fixed-test-set, and regression gates before a model is released.

## Architecture

```text
Orthomosaics / DJI photos / DJI video + SRT
                    |
                    v
       Metadata and geospatial ingestion
                    |
                    v
     Capability catalog and analysis services
          |                         |
          v                         v
 Parking candidate analysis   SAM3 + Gemma review
          |                         |
          v                         v
 Evidence-aware map layers    Training-data export
                                    |
                                    v
                           Lightweight YOLO models
```

## Repository Layout

```text
app.py                    Vision Task Console entry point (port 7860)
platform_app.py           Remote Sensing Platform entry point (port 7861)
console/                  Console configuration, SAM3/Gemma pipelines, quality and release gates
lowalt_platform/          Platform domain models, services, API, and frontend
parking/                  Built-in parking candidate and analysis capability
parking_map/              Parking-map research and evidence-fusion code
training/                 SegFormer, YOLO, and topology-model training/experiments
audits/                   One-off dataset and model-output audit tools
web/                      Vision Task Console frontend (HTML/CSS/JavaScript)
tests/                    Unit and contract test suite
scripts/                  Startup and operational helper scripts
profiles/                 Example task profiles
config.example.yaml       Example configuration
```

## Quick Start

Python **3.11 or 3.12** is supported for the published CPU path:

```bash
git clone https://github.com/TT1nKer/lowalt-vision-platform.git
cd lowalt-vision-platform
python3.11 -m venv .venv
# Windows: .venv\Scripts\activate
source .venv/bin/activate
python -m pip install -r requirements-core.lock
python -m scripts.check_core
python -m scripts.reproduce_import --output runs/public-import
python platform_app.py --project-dir runs/public-import --port 7861
```

Open [the map](http://127.0.0.1:7861/) or [the engineering
console](http://127.0.0.1:7861/engineering). The example produces 11 assets,
7 positioned camera observations, and 4 deliberately unpositioned assets.
It creates **zero parking candidates** and runs **no inference**. Without an
orthomosaic, the map has a plain background and fits the imported points.
The engineering console links each run to its own map and evidence download.

Outputs are `runs/public-import/evidence.zip`, `assets.geojson` and
`verification.json`. Choose a new output directory to repeat acceptance; the
script refuses to overwrite an existing directory. An empty project also starts
with `python platform_app.py --project-dir . --port 7861`; source directories
must be inside that project's data directory. The default no longer grants
access to entire Windows drives.

`requirements-core.txt` lists direct CPU dependencies; `requirements-core.lock`
records the exact resolved environment used by CI. The original
`requirements.txt` is the **optional model/research environment**, including
large dependencies; it is not needed by this example. For the task console,
install those dependencies in a separate environment, copy
`config.example.yaml` to `config.yaml`, configure the paths/endpoints, and run:

```bash
python app.py --project-dir . --config config.yaml --port 7860
```

Windows launch scripts remain available for configured model environments.

## Data, Models, and External Services

Large datasets and model weights are intentionally excluded from this
repository. Before enabling the associated capabilities, provide:

- an orthomosaic directory (defaults to `imagery/`; configure it through
  `paths.merged_dir` in `config.yaml` and the platform settings);
- SegFormer-large-parking weights at
  `models/segformer_large_parking/best_model.ckpt`;
- any lightweight YOLO weights required by the selected profile; and
- reachable SAM3 and Gemma service endpoints configured in `config.yaml`.

Features without their required data, weights, or services remain explicitly
unavailable.

## Testing

```bash
python -m scripts.check_core
```

The public CPU suite runs 189 tests: **182 pass, 7 skip** without
`torch`/`transformers`. It explicitly reports eight historical test modules
outside this scope because the public repository lacks their `uav_data` /
`equipment_data` source packages. They are not counted as passing or fixed.
`python -m unittest discover -s tests -t .` still exposes those eight import
errors. See [the full record](docs/public-import-validation.md).

[GitHub Actions](https://github.com/TT1nKer/lowalt-vision-platform/actions/workflows/core.yml)
runs the CPU suite and the standalone acceptance case on Python 3.11/3.12,
then uploads the synthetic evidence as an artifact. Optional model tests and
real inference accuracy remain outside the verified release scope.

## Design Principles

- Treat every unreviewed detection as a model candidate, not verified truth.
- Keep quality gates enabled by default; formal training requires a fixed test
  manifest, gold labels, and review provenance.
- Preserve timestamp, GPS/RTK, camera, and pose metadata during ingestion.
- Represent unknown values explicitly instead of inventing replacements.
- Build capabilities as independent, testable services registered through the
  platform catalog.
