# Low-Altitude Remote Sensing Vision Platform

**English** | [简体中文](README.zh-CN.md)

An extensible computer-vision platform for analyzing orthomosaics, UAV imagery,
and georeferenced video. It combines geospatial asset ingestion, model-assisted
candidate discovery, evidence-aware review, and lightweight-model training in
two independently deployable web applications.

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

Clone the repository and create a local configuration:

```bash
git clone https://github.com/TT1nKer/lowalt-vision-platform.git
cd lowalt-vision-platform
python -m pip install -r requirements.txt
cp config.example.yaml config.yaml
```

Update the paths and service endpoints in `config.yaml`, then start either web
application:

```bash
# Remote Sensing Analysis Platform
python platform_app.py --project-dir . --port 7861

# Vision Task Console
python app.py --project-dir . --config config.yaml --port 7860
```

On Windows, `start_platform.cmd` and `start_console.cmd` provide equivalent
launch shortcuts.

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

Run the complete test suite from the repository root:

```bash
python -m unittest discover -s tests -t .
```

Tests that exercise model inference require the optional `torch` and
`transformers` dependencies. They are skipped automatically when those
dependencies are unavailable; the remaining tests still run.

## Design Principles

- Treat every unreviewed detection as a model candidate, not verified truth.
- Keep quality gates enabled by default; formal training requires a fixed test
  manifest, gold labels, and review provenance.
- Preserve timestamp, GPS/RTK, camera, and pose metadata during ingestion.
- Represent unknown values explicitly instead of inventing replacements.
- Build capabilities as independent, testable services registered through the
  platform catalog.
