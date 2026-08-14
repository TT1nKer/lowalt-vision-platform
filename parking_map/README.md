# Parking map P1 / E1

This package establishes the geospatial and topological boundary for parking-map generation. It does not classify parking accuracy and does not train a model.

## Responsibilities

- `schema.py`: stable map object types, hierarchy, evidence and three-state review contract.
- `tile_georeference.py`: EPSG:4326W tile/pixel conversion and mask-to-Polygon/MultiPolygon conversion.
- `topology.py`: geometry validity and parent/child topology gates.
- `evidence_fusion.py`: subtracts building/road/vegetation/water exclusion geometry without reconnecting split regions.
- `aoi_evidence.py`: deterministic stratified AOI selection and evidence-package materialization.
- `image_evidence.py`: converts geographic geometry back to pixel masks and extracts line and cross-detector vehicle evidence.
- `evidence_gate.py`: applies the conservative `auto_accept` / `auto_reject` / `abstain` evidence contract.
- `map_builder.py`: preserves disconnected components while applying provisional exclusion layers.
- `e1_extraction.py`: records classical marking evidence and agreement between two independent detector runs.
- `sam_exclusions.py`: filters SAM building and public-road hypotheses as non-authoritative exclusions.
- `surface_hypothesis.py`: retains paved-surface components only when they intersect qualified local parking anchors.
- `e1_runner.py`: evaluates candidate evidence without treating teacher output as ground truth.
- `functional_zoning.py`: partitions facility pixels into conservative parking-band, between-band aisle, entrance and unknown masks, and measures tabular zone features.
- `e2_runner.py`: revalidates historical anchors after facility/exclusion clipping and rejects oversized marking-only unions.
- `e2_map_builder.py`: builds one AOI's hierarchical, abstaining geographic map.
- `e2_batch.py`: materializes AOI maps, overlays, anchor audits, feature JSONL and aggregate statistics.

Map truth is restricted to Polygon/MultiPolygon for area objects. BBox and OBB may be derived indexes, but the schema rejects them as truth geometry. Disconnected mask components remain separate polygons.

The current coordinate formula is sourced from `<PROJECT_ROOT>\rawdata\legacy\download_tiles.py` and matches the existing `pipeline1_sam3.py` implementation. Source metadata and unresolved fields are recorded under `data_registry/sources/`.

## Current E2 boundary

The 48-AOI E1 run produced 23 surface hypotheses. E2 retained parking-band evidence in 8 AOIs and between-band aisle evidence in 6; it produced no entrances because no candidate had both aisle and public-road adjacency. All generated features remain `abstain`. Training and automatic publication are blocked until an independent truth source is available and GSD-dependent pixel thresholds are converted to physical units.

Entry points are `parking_e1_evidence.py`, `parking_e1_exclusions.py`, `parking_e1_surface.py`, and `parking_e2_zoning.py`. Registered runs are under `data_registry/versions/`.

## E3 UAV sequence boundary

`uav_data/` owns the source/sequence/frame contract, directional GSD conversion, checksum-and-budget download gate, DLP adapter and trajectory features. `parking_e3_ingest.py` materializes independently sourced UAV evidence without changing E2 maps. The current DLP result validates one external scene only; it does not authorize a classifier or establish Shaoxing accuracy.
