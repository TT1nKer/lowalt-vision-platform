import csv
import tempfile
import unittest
from pathlib import Path

from console.acceptance_workflow import FIELDS, read_acceptance, save_acceptance_row, validate_acceptance
from console.release_gate import validate_release_approval


class AcceptanceWorkflowTests(unittest.TestCase):
    def _write(self, root, critical=0, complete=True):
        path = Path(root) / "acceptance.csv"
        with path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=FIELDS); writer.writeheader()
            for i in range(30):
                row = {k: "" for k in FIELDS}; row.update({"image": str(i), "comparison": str(i),
                    "occupied_tp": 10, "occupied_fp": 0, "occupied_fn": 0,
                    "empty_tp": 10, "empty_fp": 0, "empty_fn": 0,
                    "critical_fp": critical if i == 0 else 0,
                    "reviewer": "reviewer" if complete else "", "reviewed_at": "2026-07-28T12:00:00+08:00"})
                writer.writerow(row)
        return path

    def test_complete_high_quality_review_passes(self):
        with tempfile.TemporaryDirectory() as root:
            self.assertEqual(validate_acceptance(str(self._write(root)))["status"], "passed")

    def test_unsigned_or_critical_review_fails(self):
        with tempfile.TemporaryDirectory() as root:
            self.assertEqual(validate_acceptance(str(self._write(root, complete=False)))["status"], "failed")
            self.assertEqual(validate_acceptance(str(self._write(root, critical=1)))["status"], "failed")

    def test_row_save_is_atomic_and_only_complete_rows_count(self):
        with tempfile.TemporaryDirectory() as root:
            path = self._write(root, complete=False)
            save_acceptance_row(str(path), 0, {"reviewer": "reviewer"})
            self.assertEqual(validate_acceptance(str(path))["reviewed"], 0)
            values = {field: 0 for field in FIELDS[2:9]}
            values.update({"occupied_tp": 10, "empty_tp": 10, "reviewer": "reviewer"})
            saved = save_acceptance_row(str(path), 0, values)
            self.assertTrue(saved["reviewed_at"])
            self.assertEqual(validate_acceptance(str(path))["reviewed"], 1)
            self.assertEqual(read_acceptance(str(path))[0]["image"], "0")

    def test_approval_is_bound_to_current_evidence_and_independent(self):
        evidence = {"candidate_sha256": "model", "dataset_fingerprint": "data", "acceptance_sha256": "review"}
        pending = validate_release_approval({}, reviewers={"reviewer"}, **evidence)
        self.assertFalse(pending["valid"])
        self.assertFalse(validate_release_approval({"decision": "approved", "approver": "reviewer",
            "approved_at": "2026-07-28T12:00:00+00:00", **evidence}, reviewers={"reviewer"}, **evidence)["valid"])
        self.assertTrue(validate_release_approval({"decision": "approved", "approver": "approver",
            "approved_at": "2026-07-28T12:00:00+00:00", **evidence}, reviewers={"reviewer"}, **evidence)["valid"])


if __name__ == "__main__": unittest.main()
