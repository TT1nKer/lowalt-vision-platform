import json
import tempfile
import unittest
from pathlib import Path

from audits.vehicle_label_audit import build_vehicle_inventory
from audits.vehicle_geometry_filter_audit import (
    classify_geometry_filter_evidence,
    review_label_for_evidence,
)
from audits.vehicle_prompt_overlap import count_pipeline_greedy_groups, profile_prompt_overlap


def write_result(path: Path, prompt: str, targets: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({
        "source_file": "frame.jpg",
        "text_prompt": prompt,
        "prompt_mode": "batch",
        "targets": targets,
    }), encoding="utf-8")


class VehicleLabelAuditTests(unittest.TestCase):
    def test_geometry_evidence_maps_to_actual_review_state_labels(self):
        self.assertEqual(review_label_for_evidence("direct_has_lines"), "accept")
        self.assertEqual(review_label_for_evidence("adjacent_to_lines"), "accept")
        self.assertEqual(review_label_for_evidence("no_line_evidence"), "reject")

    def test_geometry_evidence_separates_direct_and_adjacent_acceptance(self):
        targets = [
            {"target_id": "direct", "bbox": [0, 0, 10, 10]},
            {"target_id": "adjacent", "bbox": [15, 0, 25, 10]},
            {"target_id": "reject", "bbox": [100, 100, 110, 110]},
        ]
        geometry = {"direct": {"has_lines": True}}

        evidence = classify_geometry_filter_evidence(targets, geometry)

        self.assertEqual(evidence, {
            "direct": "direct_has_lines",
            "adjacent": "adjacent_to_lines",
            "reject": "no_line_evidence",
        })

    def test_inventory_separates_vehicle_and_mixed_merged_targets(self):
        with tempfile.TemporaryDirectory() as tmp:
            project = Path(tmp)
            batch = project / "sam3_runs" / "pklot_v1"
            (batch / "run_meta.json").parent.mkdir(parents=True)
            (batch / "run_meta.json").write_text(json.dumps({
                "prompt_mode": "batch",
                "prompts": ["parked car", "parking space"],
            }), encoding="utf-8")
            write_result(
                batch / "raw_prompts" / "parked_car" / "sam3_results" / "frame.json",
                "parked car",
                [{
                    "bbox": [10, 20, 30, 60],
                    "confidence": 0.8,
                    "class_name": "parked car",
                    "mask_file": "frame_t0.png",
                }],
            )
            index = batch / "merged" / "review_cache" / "target_index.jsonl"
            index.parent.mkdir(parents=True)
            rows = [
                {"target_id": "frame.jpg::0", "bbox": [10, 20, 30, 60],
                 "class_name": "parked car", "source_prompts": ["parked car"]},
                {"target_id": "frame.jpg::1", "bbox": [40, 20, 60, 60],
                 "class_name": "parking space",
                 "source_prompts": ["parked car", "parking space"]},
            ]
            index.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
            state_path = batch / "merged" / "review" / "target_state.json"
            state_path.parent.mkdir(parents=True)
            state_path.write_text(json.dumps({
                "frame.jpg::0": {"label": "accept", "source": "geometry_filter", "is_auto": True},
                "frame.jpg::1": {"label": "reject", "source": "human", "is_auto": False},
            }), encoding="utf-8")

            result = build_vehicle_inventory(project, batch, image_size=(100, 100))

            self.assertEqual(result["raw_prompts"]["parked car"]["targets"], 1)
            self.assertEqual(result["merged"]["vehicle_related"], 2)
            self.assertEqual(result["merged"]["vehicle_only"], 1)
            self.assertEqual(result["merged"]["mixed_with_non_vehicle"], 1)
            self.assertEqual(result["review"]["sources"], {"geometry_filter": 1, "human": 1})
            self.assertEqual(result["review"]["auto"], 1)

    def test_inventory_reports_invalid_and_edge_bboxes(self):
        with tempfile.TemporaryDirectory() as tmp:
            project = Path(tmp)
            batch = project / "sam3_runs" / "pklot_v1"
            (batch / "run_meta.json").parent.mkdir(parents=True)
            (batch / "run_meta.json").write_text("{}", encoding="utf-8")
            write_result(
                batch / "raw_prompts" / "parked_vehicle" / "sam3_results" / "frame.json",
                "parked vehicle",
                [
                    {"bbox": [0, 5, 20, 25], "confidence": 0.4, "mask_file": "a.png"},
                    {"bbox": [20, 20, 10, 30], "confidence": 0.2},
                ],
            )

            result = build_vehicle_inventory(project, batch, image_size=(100, 100))
            profile = result["raw_prompts"]["parked vehicle"]

            self.assertEqual(profile["edge_touching"], 1)
            self.assertEqual(profile["invalid_bbox"], 1)
            self.assertEqual(profile["mask_references"], 1)

    def test_overlap_profile_is_order_independent_and_reports_cross_prompt(self):
        candidates = [
            {"bbox": [0, 0, 10, 10], "prompt": "parked car"},
            {"bbox": [1, 0, 11, 10], "prompt": "parked vehicle"},
            {"bbox": [3, 0, 13, 10], "prompt": "vehicle in parking lot"},
            {"bbox": [30, 30, 40, 40], "prompt": "parked car"},
        ]

        forward = profile_prompt_overlap(candidates, (0.5, 0.8))
        reverse = profile_prompt_overlap(list(reversed(candidates)), (0.5, 0.8))

        self.assertEqual(forward, reverse)
        self.assertEqual(forward["0.5"]["groups"], 2)
        self.assertEqual(forward["0.5"]["cross_prompt_groups"], 1)
        self.assertEqual(forward["0.8"]["groups"], 3)

    def test_pipeline_greedy_grouping_is_order_sensitive(self):
        first = {"bbox": [0, 0, 10, 10], "confidence": 0.5}
        bridge = {"bbox": [1, 0, 11, 10], "confidence": 0.5}
        last = {"bbox": [3, 0, 13, 10], "confidence": 0.5}

        self.assertEqual(count_pipeline_greedy_groups([first, bridge, last]), 2)
        self.assertEqual(count_pipeline_greedy_groups([bridge, first, last]), 1)


if __name__ == "__main__":
    unittest.main()
