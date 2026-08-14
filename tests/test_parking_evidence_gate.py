import unittest

from parking_map.evidence_gate import (
    ParkingEvidenceKind,
    ParkingEvidenceSignal,
    decide_parking_candidate,
)
from parking_map.schema import ReviewState


def polygon(x1=0, y1=0, x2=10, y2=10):
    return {
        "type": "Polygon",
        "coordinates": [[[x1, y1], [x2, y1], [x2, y2], [x1, y2], [x1, y1]]],
    }


class ParkingEvidenceGateTests(unittest.TestCase):
    def test_sam_candidate_alone_abstains_regardless_of_candidate_confidence(self):
        decision = decide_parking_candidate(polygon(), [])

        self.assertEqual(decision.state, ReviewState.ABSTAIN)
        self.assertIn("missing_exclusion_coverage", decision.reasons)
        self.assertIn("insufficient_independent_support", decision.reasons)

    def test_independent_marking_and_vehicle_support_can_accept_after_exclusion_coverage(self):
        signals = [
            ParkingEvidenceSignal(
                ParkingEvidenceKind.EXCLUSION_COVERAGE,
                "land-cover-v1",
                "land-cover-teacher",
                0.9,
            ),
            ParkingEvidenceSignal(
                ParkingEvidenceKind.PARKING_MARKING,
                "hough-lines-v1",
                "classical-image-geometry",
                0.75,
            ),
            ParkingEvidenceSignal(
                ParkingEvidenceKind.VEHICLE_ARRANGEMENT,
                "coco-agreement-v1",
                "coco-vehicle-detectors",
                0.8,
            ),
        ]

        decision = decide_parking_candidate(polygon(), signals)

        self.assertEqual(decision.state, ReviewState.AUTO_ACCEPT)
        self.assertEqual(
            set(decision.qualified_support),
            {
                ParkingEvidenceKind.PARKING_MARKING,
                ParkingEvidenceKind.VEHICLE_ARRANGEMENT,
            },
        )

    def test_two_supports_from_same_source_group_do_not_self_validate(self):
        signals = [
            ParkingEvidenceSignal(
                ParkingEvidenceKind.EXCLUSION_COVERAGE,
                "sam-land-cover-v1",
                "sam3",
                0.95,
            ),
            ParkingEvidenceSignal(
                ParkingEvidenceKind.PARKING_MARKING,
                "sam-marking-v1",
                "sam3",
                0.9,
            ),
            ParkingEvidenceSignal(
                ParkingEvidenceKind.VEHICLE_ARRANGEMENT,
                "sam-vehicle-v1",
                "sam3",
                0.9,
            ),
        ]

        decision = decide_parking_candidate(polygon(), signals)

        self.assertEqual(decision.state, ReviewState.ABSTAIN)
        self.assertIn("support_not_independent", decision.reasons)

    def test_strong_non_parking_conflict_rejects_candidate(self):
        decision = decide_parking_candidate(
            polygon(),
            [
                ParkingEvidenceSignal(
                    ParkingEvidenceKind.NON_PARKING_CONFLICT,
                    "building-footprint-v1",
                    "municipal-map",
                    0.95,
                )
            ],
        )

        self.assertEqual(decision.state, ReviewState.AUTO_REJECT)
        self.assertEqual(decision.reasons, ("strong_non_parking_conflict",))

    def test_invalid_geometry_rejects_before_semantic_scoring(self):
        invalid = {
            "type": "Polygon",
            "coordinates": [[[0, 0], [10, 10], [10, 0], [0, 10], [0, 0]]],
        }

        decision = decide_parking_candidate(invalid, [])

        self.assertEqual(decision.state, ReviewState.AUTO_REJECT)
        self.assertIn("invalid_geometry:self_intersection", decision.reasons)


if __name__ == "__main__":
    unittest.main()
