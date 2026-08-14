from __future__ import annotations

import unittest

from lowalt_platform.domain.models import AnalysisAsset, AssetKind, CandidateSummary, SupportLevel


class DomainModelTests(unittest.TestCase):
    def test_unknown_support_level_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            SupportLevel("unknown")

    def test_asset_without_location_remains_an_independent_scene(self) -> None:
        asset = AnalysisAsset(asset_id="video-1", kind=AssetKind.VIDEO, source_path="input/a.mp4")
        self.assertEqual(asset.spatial_state, "independent_scene")

    def test_candidate_summary_rejects_inconsistent_total(self) -> None:
        with self.assertRaises(ValueError):
            CandidateSummary(total=9, segformer_only=4, vehicle_detected=3, vehicle_row_supported=1)


if __name__ == "__main__":
    unittest.main()
