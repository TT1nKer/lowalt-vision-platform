from __future__ import annotations

import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from parking.parking_direct_candidates import export_direct_candidates


def write_candidate(root: Path, aoi_id: str, x: float) -> dict:
    geometry = {
        "type": "Polygon",
        "coordinates": [[[x, 0], [x + 1, 0], [x + 1, 1], [x, 1], [x, 0]]],
    }
    directory = root / aoi_id
    directory.mkdir(parents=True)
    (directory / "mask_geometries.geojson").write_text(
        json.dumps({
            "type": "FeatureCollection",
            "aoi_id": aoi_id,
            "features": [{
                "type": "Feature",
                "geometry": geometry,
                "properties": {"source": "UTEL-UIUC/SegFormer-large-parking"},
            }],
        }),
        encoding="utf-8",
    )
    return geometry


class DirectParkingCandidateTests(unittest.TestCase):
    def test_vehicle_evidence_never_replaces_or_removes_segformer_geometry(self) -> None:
        # This catches the old behavior where candidates without vehicle rows disappeared.
        with TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            candidates = root / "candidates"
            expected_geometry = {
                aoi_id: write_candidate(candidates, aoi_id, index * 2)
                for index, aoi_id in enumerate(("segformer-only", "vehicle", "row"))
            }
            vehicle_manifest = root / "vehicles.json"
            vehicle_manifest.write_text(json.dumps({
                "aois": [{"aoi_id": "vehicle"}, {"aoi_id": "row"}],
            }), encoding="utf-8")
            band_manifest = root / "bands.json"
            band_manifest.write_text(json.dumps({
                "aois": [{"aoi_id": "row"}],
            }), encoding="utf-8")
            output = root / "direct_candidates.geojson"

            summary = export_direct_candidates(
                candidate_root=candidates,
                vehicle_manifest=vehicle_manifest,
                band_manifest=band_manifest,
                output_path=output,
            )

            features = json.loads(output.read_text(encoding="utf-8"))["features"]
            by_aoi = {feature["properties"]["aoi_id"]: feature for feature in features}
            self.assertEqual(set(by_aoi), set(expected_geometry))
            self.assertEqual(
                {aoi_id: feature["geometry"] for aoi_id, feature in by_aoi.items()},
                expected_geometry,
            )
            self.assertEqual(
                {aoi_id: feature["properties"]["support_level"] for aoi_id, feature in by_aoi.items()},
                {
                    "segformer-only": "segformer_only",
                    "vehicle": "vehicle_detected",
                    "row": "vehicle_row_supported",
                },
            )
            self.assertEqual(summary["total_candidates"], 3)
            self.assertEqual(summary["segformer_only"], 1)
            self.assertEqual(summary["vehicle_detected"], 1)
            self.assertEqual(summary["vehicle_row_supported"], 1)


if __name__ == "__main__":
    unittest.main()
