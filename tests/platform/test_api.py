from __future__ import annotations

import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from fastapi import HTTPException

from lowalt_platform.api.routes import create_platform_app
from lowalt_platform.settings import PlatformSettings


class PlatformApiTests(unittest.TestCase):
    def test_exposes_truthful_summary_filtered_candidates_and_media(self) -> None:
        with TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            images = root / "images"; masks = root / "masks"; web = root / "web"
            images.mkdir(); masks.mkdir(); web.mkdir()
            aoi_id = "block_z19_br0_bc0_r100_c200"
            (images / f"{aoi_id}.png").write_bytes(b"image-bytes")
            (masks / f"{aoi_id}_mask.png").write_bytes(b"mask-bytes")
            secondary = root / "secondary" / aoi_id
            secondary.mkdir(parents=True)
            (secondary / "vehicle.png").write_bytes(b"vehicle-mask")
            (secondary / "manifest.json").write_text(json.dumps({"aoi_id": aoi_id, "status": "completed", "evidence": {"vehicle": {"prompt": "vehicle", "target_count": 2, "mask": "vehicle.png", "targets": [{"bbox": [1, 2, 3, 4]}]}}}), encoding="utf-8")
            candidates = root / "candidates.geojson"
            candidates.write_text(json.dumps({"type": "FeatureCollection", "features": [{"type": "Feature", "geometry": {"type": "Polygon", "coordinates": [[[120, 30], [121, 30], [121, 31], [120, 31], [120, 30]]]}, "properties": {"aoi_id": aoi_id, "support_level": "vehicle_row_supported"}}]}), encoding="utf-8")
            summary = root / "summary.json"
            summary.write_text(json.dumps({"total_candidates": 1, "segformer_only": 0, "vehicle_detected": 0, "vehicle_row_supported": 1}), encoding="utf-8")
            overview = root / "overview.jpg"; overview.write_bytes(b"overview")
            overview_manifest = root / "overview.json"
            overview_manifest.write_text(json.dumps({"bounds": [119, 29, 122, 32], "image_count": 1}), encoding="utf-8")
            settings = PlatformSettings(project_root=root, candidate_geojson=candidates, candidate_summary=summary, image_root=images, mask_root=masks, secondary_analysis_root=secondary.parent, overview_image=overview, overview_manifest=overview_manifest, web_root=web)

            app = create_platform_app(settings)
            endpoints = {route.path: route.endpoint for route in app.routes if hasattr(route, "endpoint")}
            payload = endpoints["/api/platform/summary"]()
            self.assertEqual(payload["parking_candidates"]["total"], 1)
            self.assertEqual(payload["secondary_analysis"], {"completed": 1, "total": 1})
            self.assertEqual(payload["overview"]["bounds"], [119, 29, 122, 32])
            collection = endpoints["/api/platform/candidates"](west=119, south=29, east=122, north=32, support="vehicle_row_supported", limit=100)
            self.assertEqual(len(collection["features"]), 1)
            blocks = endpoints["/api/platform/imagery/blocks"](west=None, south=None, east=None, north=None, limit=10)
            self.assertEqual(len(blocks["blocks"]), 1)
            self.assertEqual(blocks["blocks"][0]["block_id"], aoi_id)
            block_media = endpoints["/api/platform/imagery/blocks/{block_id}"](aoi_id)
            self.assertEqual(Path(block_media.path).read_bytes(), b"image-bytes")
            evidence = endpoints["/api/platform/candidates/{aoi_id}/secondary"](aoi_id)
            self.assertEqual(evidence["evidence"]["vehicle"]["target_count"], 2)
            self.assertNotIn("targets", evidence["evidence"]["vehicle"])
            evidence_media = endpoints["/api/platform/candidates/{aoi_id}/secondary/{evidence_type}"](aoi_id, "vehicle")
            self.assertEqual(Path(evidence_media.path).read_bytes(), b"vehicle-mask")
            media = endpoints["/api/platform/media/image/{aoi_id}"](aoi_id)
            self.assertEqual(Path(media.path).read_bytes(), b"image-bytes")
            with self.assertRaises(HTTPException) as missing:
                endpoints["/api/platform/candidates/{aoi_id}"]("missing")
            self.assertEqual(missing.exception.status_code, 404)


if __name__ == "__main__":
    unittest.main()
