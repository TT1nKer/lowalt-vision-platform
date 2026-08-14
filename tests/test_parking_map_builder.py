import unittest

from parking_map.evidence_gate import ParkingEvidenceKind, ParkingEvidenceSignal
from parking_map.map_builder import build_parking_candidate
from parking_map.schema import ReviewState
from parking_map.topology import validate_geometry


def polygon(x1, y1, x2, y2):
    return {
        "type": "Polygon",
        "coordinates": [[[x1, y1], [x2, y1], [x2, y2], [x1, y2], [x1, y1]]],
    }


class ParkingMapBuilderTests(unittest.TestCase):
    def test_road_exclusion_splits_candidate_without_global_box_reconnection(self):
        result = build_parking_candidate(
            candidate_id="candidate-1",
            candidate_geometry=polygon(0, 0, 10, 10),
            exclusion_geometries=[polygon(4, -1, 6, 11)],
            signals=[],
        )

        self.assertEqual(result.geometry["type"], "MultiPolygon")
        self.assertEqual(result.component_count, 2)
        self.assertEqual(validate_geometry(result.geometry), ())
        self.assertEqual(result.decision.state, ReviewState.ABSTAIN)

    def test_fully_excluded_candidate_is_rejected_without_fabricated_geometry(self):
        result = build_parking_candidate(
            candidate_id="candidate-1",
            candidate_geometry=polygon(0, 0, 10, 10),
            exclusion_geometries=[polygon(-1, -1, 11, 11)],
            signals=[],
        )

        self.assertIsNone(result.geometry)
        self.assertEqual(result.component_count, 0)
        self.assertEqual(result.decision.state, ReviewState.AUTO_REJECT)
        self.assertEqual(result.decision.reasons, ("fully_excluded",))

    def test_provisional_full_exclusion_preserves_unknown_candidate(self):
        source = polygon(0, 0, 10, 10)

        result = build_parking_candidate(
            candidate_id="candidate-1",
            candidate_geometry=source,
            exclusion_geometries=[polygon(-1, -1, 11, 11)],
            signals=[],
            exclusions_are_authoritative=False,
        )

        self.assertEqual(result.geometry, source)
        self.assertEqual(result.component_count, 1)
        self.assertEqual(result.removed_fraction, 1.0)
        self.assertEqual(result.decision.state, ReviewState.ABSTAIN)
        self.assertEqual(result.decision.reasons, ("provisionally_fully_excluded",))

    def test_independent_evidence_is_evaluated_on_remaining_geometry(self):
        signals = [
            ParkingEvidenceSignal(
                ParkingEvidenceKind.EXCLUSION_COVERAGE,
                "complete-land-cover",
                "land-cover",
                1.0,
            ),
            ParkingEvidenceSignal(
                ParkingEvidenceKind.PARKING_MARKING,
                "marking-lines",
                "image-lines",
                0.8,
            ),
            ParkingEvidenceSignal(
                ParkingEvidenceKind.VEHICLE_ARRANGEMENT,
                "vehicle-grid",
                "vehicle-detector",
                0.8,
            ),
        ]

        result = build_parking_candidate(
            candidate_id="candidate-1",
            candidate_geometry=polygon(0, 0, 10, 10),
            exclusion_geometries=[polygon(4, 4, 6, 6)],
            signals=signals,
        )

        self.assertEqual(result.geometry["type"], "Polygon")
        self.assertEqual(len(result.geometry["coordinates"]), 2)
        self.assertEqual(result.decision.state, ReviewState.AUTO_ACCEPT)
        self.assertAlmostEqual(result.removed_fraction, 0.04)


if __name__ == "__main__":
    unittest.main()
