import unittest

from parking_map.e1_runner import evaluate_aoi_candidates
from parking_map.schema import ReviewState


def polygon(x1, y1, x2, y2):
    return {
        "type": "Polygon",
        "coordinates": [[[x1, y1], [x2, y1], [x2, y2], [x1, y2], [x1, y1]]],
    }


class ParkingE1RunnerTests(unittest.TestCase):
    def test_legacy_accept_label_is_ignored_when_new_evidence_is_missing(self):
        candidates = {
            "type": "FeatureCollection",
            "features": [{
                "type": "Feature",
                "geometry": polygon(0, 0, 2, 2),
                "properties": {"target_id": "candidate-1", "label": "accept"},
            }],
        }

        result = evaluate_aoi_candidates("aoi-1", candidates, {"exclusions": [], "signals": {}})

        self.assertEqual(result["summary"][ReviewState.ABSTAIN.value], 1)
        self.assertEqual(
            result["features"][0]["properties"]["decision"],
            ReviewState.ABSTAIN.value,
        )
        self.assertNotIn("label", result["features"][0]["properties"])

    def test_runner_records_fully_excluded_candidate_without_emitting_geometry(self):
        candidates = {
            "type": "FeatureCollection",
            "features": [{
                "type": "Feature",
                "geometry": polygon(0, 0, 2, 2),
                "properties": {"target_id": "candidate-1"},
            }],
        }
        evidence = {"exclusions": [polygon(-1, -1, 3, 3)], "signals": {}}

        result = evaluate_aoi_candidates("aoi-1", candidates, evidence)

        self.assertEqual(result["features"], [])
        self.assertEqual(result["summary"][ReviewState.AUTO_REJECT.value], 1)
        self.assertEqual(result["decisions"][0]["reasons"], ["fully_excluded"])

    def test_runner_does_not_turn_teacher_exclusion_into_authoritative_rejection(self):
        candidates = {
            "type": "FeatureCollection",
            "features": [{
                "type": "Feature",
                "geometry": polygon(0, 0, 2, 2),
                "properties": {"target_id": "candidate-1"},
            }],
        }
        evidence = {
            "exclusions": [polygon(-1, -1, 3, 3)],
            "exclusions_authoritative": False,
            "signals": {},
        }

        result = evaluate_aoi_candidates("aoi-1", candidates, evidence)

        self.assertEqual(result["summary"][ReviewState.ABSTAIN.value], 1)
        self.assertEqual(len(result["features"]), 1)
        self.assertEqual(
            result["features"][0]["properties"]["reasons"],
            ["provisionally_fully_excluded"],
        )


if __name__ == "__main__":
    unittest.main()
