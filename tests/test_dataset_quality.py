import json
import os
import tempfile
import unittest

from console.dataset_quality import audit_yolo_dataset, build_recovery_plan, require_quality_pass


def _config(**overrides):
    quality = {
        "enforce": True,
        "require_fixed_test": False,
        "require_golden_manifest": False,
        "require_review_provenance": False,
        "min_test_images": 1,
        "max_object_area_ratio": 0.08,
        "max_oversized_fraction": 0.10,
        "max_aspect_ratio": 8.0,
        "max_extreme_aspect_fraction": 0.10,
        "max_duplicate_fraction": 0.10,
        "max_class_conflict_fraction": 0.10,
        "max_objects_per_image": 10,
    }
    quality.update(overrides)
    return {"yolo_obb": {"quality": quality}}


def _dataset(root, label_line, label_format="obb"):
    with open(os.path.join(root, "export_meta.json"), "w", encoding="utf-8") as handle:
        json.dump({"split_mode": "stable_hash", "format": label_format}, handle)
    for split in ("train", "val", "test"):
        image_dir = os.path.join(root, "images", split)
        label_dir = os.path.join(root, "labels", split)
        os.makedirs(image_dir)
        os.makedirs(label_dir)
        with open(os.path.join(image_dir, f"{split}.jpg"), "wb") as handle:
            handle.write(b"image")
        with open(os.path.join(label_dir, f"{split}.txt"), "w", encoding="utf-8") as handle:
            handle.write(label_line + "\n")


class DatasetQualityTests(unittest.TestCase):
    def test_clean_dataset_passes(self):
        with tempfile.TemporaryDirectory() as root:
            _dataset(root, "0 0.10 0.10 0.20 0.10 0.20 0.20 0.10 0.20")
            report = audit_yolo_dataset(root, _config(), project_dir=root)
            self.assertEqual(report["status"], "passed")
            self.assertEqual(report["summary"]["objects"], 3)
            self.assertTrue(os.path.isfile(os.path.join(root, "quality_report.json")))

    def test_segmentation_polygons_are_not_parsed_as_obb(self):
        with tempfile.TemporaryDirectory() as root:
            _dataset(
                root,
                "0 0.10 0.10 0.30 0.10 0.35 0.20 0.25 0.35 0.10 0.25",
                label_format="seg",
            )
            report = audit_yolo_dataset(root, _config(), project_dir=root)
            self.assertEqual(report["status"], "passed")
            self.assertEqual(report["label_format"], "seg")
            self.assertEqual(report["summary"]["objects"], 3)

    def test_oversized_boxes_block_training(self):
        with tempfile.TemporaryDirectory() as root:
            _dataset(root, "0 0.05 0.05 0.95 0.05 0.95 0.95 0.05 0.95")
            cfg = _config(max_oversized_fraction=0.0)
            report = audit_yolo_dataset(root, cfg, project_dir=root)
            self.assertEqual(report["status"], "failed")
            self.assertIn("OVERSIZED_OBJECTS", {item["code"] for item in report["errors"]})
            with self.assertRaisesRegex(RuntimeError, "数据质量门禁未通过"):
                require_quality_pass(root, cfg, project_dir=root)

    def test_missing_golden_manifest_is_a_blocker(self):
        with tempfile.TemporaryDirectory() as root:
            _dataset(root, "0 0.10 0.10 0.20 0.10 0.20 0.20 0.10 0.20")
            report = audit_yolo_dataset(
                root,
                _config(require_golden_manifest=True, golden_manifest="missing.json"),
                project_dir=root,
            )
            self.assertIn("GOLDEN_SET_NOT_APPROVED", {item["code"] for item in report["errors"]})

    def test_aabb_expansion_cannot_claim_obb_quality(self):
        with tempfile.TemporaryDirectory() as root:
            _dataset(root, "0 0.10 0.10 0.20 0.10 0.20 0.20 0.10 0.20")
            provenance = os.path.join(root, "source.json")
            with open(provenance, "w", encoding="utf-8") as handle:
                json.dump({
                    "source_dataset": "PKLot Roboflow export",
                    "conversion": "YOLO xywh AABB to four-corner OBB",
                }, handle)
            with open(os.path.join(root, "export_meta.json"), "w", encoding="utf-8") as handle:
                json.dump({"split_mode": "stable_hash", "source_provenance": provenance}, handle)
            report = audit_yolo_dataset(root, _config(), project_dir=root)
            codes = {item["code"] for item in report["errors"]}
            self.assertIn("AABB_TO_OBB_INVALID", codes)
            self.assertIn("SOURCE_DATA_NOT_GOLDEN", codes)

    def test_recovery_plan_reuses_existing_model(self):
        plan = build_recovery_plan({"summary": {
            "objects": 1000, "degenerate": 2, "duplicates": 8,
            "oversized": 10, "extreme_aspect": 20, "class_conflicts": 10,
        }})
        self.assertFalse(plan["full_retrain_required"])
        self.assertEqual(plan["source_model"], "train/weights/best.pt")
        self.assertEqual(plan["automatic_removals"], 10)
        self.assertEqual(plan["estimated_retained_fraction"], 0.95)


if __name__ == "__main__":
    unittest.main()
