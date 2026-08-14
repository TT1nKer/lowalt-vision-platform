import csv
import json
import os
import tempfile
import unittest
from pathlib import Path

from console.golden_review import (install_golden_package, read_review_rows,
                           save_review_row, validate_golden_package)


def _package(root: str, *, status="approved", reviewer="reviewer-a", approver="approver-b") -> Path:
    package = Path(root) / "quality" / "golden_review_v1"
    (package / "images").mkdir(parents=True)
    (package / "labels").mkdir()
    image = "sample.jpg"
    (package / "images" / image).write_bytes(b"image")
    (package / "labels" / "sample.txt").write_text(
        "0 0.1 0.1 0.2 0.1 0.2 0.2 0.1 0.2\n", encoding="utf-8"
    )
    with (package / "review.csv").open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=[
            "image", "review_status", "reviewer", "reviewed_at", "notes",
        ])
        writer.writeheader()
        writer.writerow({
            "image": image, "review_status": status, "reviewer": reviewer,
            "reviewed_at": "2026-07-28T11:00:00", "notes": "",
        })
    (package / "fixed_test_list.txt").write_text(image + "\n", encoding="utf-8")
    (package / "golden_test_manifest.json").write_text(json.dumps({
        "approved": True, "reviewer": reviewer, "approver": approver,
        "approved_at": "2026-07-28T12:00:00", "images": [image],
    }), encoding="utf-8")
    return package


class GoldenReviewTests(unittest.TestCase):
    def test_complete_package_passes(self):
        with tempfile.TemporaryDirectory() as root:
            report = validate_golden_package(str(_package(root)))
            self.assertEqual(report["status"], "passed")
            self.assertEqual(report["completed_count"], 1)
            self.assertIn("review.csv", report["artifact_sha256"])

    def test_pending_review_is_blocked(self):
        with tempfile.TemporaryDirectory() as root:
            report = validate_golden_package(str(_package(root, status="pending")))
            self.assertEqual(report["status"], "failed")
            self.assertIn("REVIEW_INCOMPLETE", {item["code"] for item in report["errors"]})

    def test_review_decision_is_atomic_and_preserves_identity(self):
        with tempfile.TemporaryDirectory() as root:
            package = _package(root, status="pending", reviewer="")
            saved = save_review_row(str(package), 0, {
                "review_status": "approved", "reviewer": "reviewer-a", "notes": "",
            })
            self.assertEqual(saved["image"], "sample.jpg")
            self.assertEqual(saved["review_status"], "approved")
            self.assertTrue(saved["reviewed_at"])
            self.assertEqual(read_review_rows(str(package))[0]["image"], "sample.jpg")

    def test_issue_requires_notes_and_uses_system_reviewer(self):
        with tempfile.TemporaryDirectory() as root:
            package = _package(root, status="pending", reviewer="")
            saved = save_review_row(str(package), 0, {
                "review_status": "approved", "notes": "",
            })
            self.assertEqual(saved["reviewer"], "local-review")
            with self.assertRaisesRegex(ValueError, "notes"):
                save_review_row(str(package), 0, {
                    "review_status": "needs_correction", "reviewer": "reviewer-a", "notes": "",
                })

    def test_four_eyes_violation_is_blocked(self):
        with tempfile.TemporaryDirectory() as root:
            report = validate_golden_package(str(_package(root, reviewer="same", approver="same")))
            self.assertIn("FOUR_EYES_VIOLATION", {item["code"] for item in report["errors"]})

    def test_install_requires_test_isolation(self):
        with tempfile.TemporaryDirectory() as root:
            package = _package(root)
            dataset = Path(root) / "dataset"
            for split in ("train", "val", "test"):
                (dataset / "images" / split).mkdir(parents=True)
            (dataset / "images" / "train" / "sample.jpg").write_bytes(b"leak")
            (dataset / "images" / "test" / "sample.jpg").write_bytes(b"image")
            (dataset / "export_meta.json").write_text("{}", encoding="utf-8")
            with self.assertRaisesRegex(RuntimeError, "未与训练/验证集隔离"):
                install_golden_package(str(package), root, str(dataset))


if __name__ == "__main__":
    unittest.main()
