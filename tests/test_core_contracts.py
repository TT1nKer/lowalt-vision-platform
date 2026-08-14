import os
import tempfile
import unittest
from types import SimpleNamespace

import app
from console.core import run_dirs
from console.pipeline2_review import NEGATIVE_LABELS, POSITIVE_LABELS, SKIP_LABELS
from console.pipeline3_yolo import _parse_yolo_metrics, export_yolo_obb


class CoreContractTests(unittest.TestCase):
    def test_review_label_semantics_are_disjoint(self):
        self.assertEqual(POSITIVE_LABELS, {"accept", "hard_positive"})
        self.assertEqual(NEGATIVE_LABELS, {"hard_negative", "empty_ok"})
        self.assertEqual(SKIP_LABELS, {"reject", "bad_mask", "needs_review"})
        self.assertFalse(POSITIVE_LABELS & NEGATIVE_LABELS)
        self.assertFalse(NEGATIVE_LABELS & SKIP_LABELS)

    def test_yolo_metric_parser_reads_cli_summary_row(self):
        metrics = _parse_yolo_metrics([
            "all        660      39439      0.7269      0.6724      0.7334      0.6018"
        ])
        self.assertEqual(metrics, {
            "precision": 0.7269,
            "recall": 0.6724,
            "mAP50": 0.7334,
            "mAP50-95": 0.6018,
        })

    def test_safe_join_blocks_parent_traversal(self):
        with tempfile.TemporaryDirectory() as base:
            child = app._safe_join(base, "nested", "image.png")
            self.assertEqual(os.path.commonpath([base, child]), os.path.abspath(base))
            with self.assertRaises(app.HTTPException):
                app._safe_join(base, "..", "secret.txt")

    def test_job_log_cursor_survives_ring_buffer_truncation(self):
        old_limit = app.LOG_KEEP
        app.LOG_KEEP = 3
        try:
            job = app.Job(id="test", action="noop")
            for i in range(5):
                job.add(str(i))
            self.assertEqual(job._log_total, 5)
            self.assertEqual(job._log_start, 2)
            self.assertEqual(len(job.log), 3)
        finally:
            app.LOG_KEEP = old_limit

    def test_job_parameter_validation(self):
        with tempfile.TemporaryDirectory() as project:
            images = os.path.join(project, "images")
            os.makedirs(images)
            old_args, old_cfg = app.ARGS, app.CFG
            app.ARGS = SimpleNamespace(project_dir=project, config_path="", app_dir=project)
            app.CFG = {"paths": {"merged_dir": images}}
            try:
                app._validate_job_params("p3_train", {"imgsz": 1024, "epochs": 10})
                with self.assertRaises(ValueError):
                    app._validate_job_params("p3_train", {"imgsz": 0, "epochs": 10})
                with self.assertRaises(ValueError):
                    app._validate_job_params("p3_predict", {"conf": 1.5})
                baseline = os.path.join(project, "baseline.pt")
                trained = os.path.join(project, "trained.pt")
                for path in (baseline, trained):
                    with open(path, "wb") as handle:
                        handle.write(b"weights")
                app._validate_job_params("p3_compare", {
                    "baseline": baseline, "trained": trained, "split": "test",
                })
                with self.assertRaises(ValueError):
                    app._validate_job_params("p3_compare", {
                        "baseline": baseline, "trained": baseline, "split": "test",
                    })
                with self.assertRaises(ValueError):
                    app._validate_job_params("p3_compare", {
                        "baseline": baseline, "trained": trained, "split": "invalid",
                    })
            finally:
                app.ARGS, app.CFG = old_args, old_cfg

    def test_job_parameters_support_current_and_legacy_payloads(self):
        current = app._extract_job_params({"action": "p3_train", "params": {"epochs": 20}})
        legacy = app._extract_job_params({"action": "p3_train", "epochs": 30})
        self.assertEqual(current, {"epochs": 20})
        self.assertEqual(legacy, {"epochs": 30})

    def test_model_test_routes_are_registered(self):
        paths = {route.path for route in app.APP.routes}
        self.assertIn("/api/test/models", paths)
        self.assertIn("/api/test/results", paths)
        self.assertIn("/api/test/image/{name:path}", paths)
        self.assertIn("/api/report/summary", paths)
        self.assertIn("/api/report/quality", paths)
        self.assertIn("/api/report/recovery", paths)
        self.assertIn("/api/report/golden", paths)
        self.assertIn("/golden-review", paths)
        self.assertIn("/api/golden-review", paths)
        self.assertIn("/api/golden-review/{index}", paths)
        self.assertIn("/recovery-comparison", paths)
        self.assertIn("/business-shadow", paths)
        self.assertIn("/business-shadow-calibrated", paths)
        self.assertIn("/space-classifier-shadow", paths)
        self.assertIn("/api/report/business-shadow", paths)
        self.assertIn("/business-acceptance", paths)
        self.assertIn("/api/business-acceptance", paths)
        self.assertIn("/api/job/cancel/{jid}", paths)

    def test_quality_report_defaults_to_official_recovery_candidate(self):
        with tempfile.TemporaryDirectory() as project:
            official = os.path.join(project, "quality", "pklot_official_recovery_v2")
            os.makedirs(official)
            with open(os.path.join(official, "quality_report.json"), "w", encoding="utf-8") as handle:
                import json
                json.dump({"status": "failed", "errors": [{"code": "GOLDEN_SET_NOT_APPROVED"}]}, handle)
            old_args, old_ds = app.ARGS, app.ds
            app.ARGS = SimpleNamespace(project_dir=project)
            app.ds = lambda: {"yolo": os.path.join(project, "legacy")}
            try:
                report = app.api_report_quality()
                self.assertEqual(report["artifact_role"], "official_recovery_candidate")
                self.assertEqual([e["code"] for e in report["errors"]], ["GOLDEN_SET_NOT_APPROVED"])
            finally:
                app.ARGS, app.ds = old_args, old_ds

    def test_recovery_report_uses_official_candidate(self):
        import json
        with tempfile.TemporaryDirectory() as project:
            official = os.path.join(project, "quality", "pklot_official_recovery_v2")
            os.makedirs(official)
            report = {
                "status": "failed",
                "summary": {"images": 206, "objects": 10844},
                "errors": [{"code": "GOLDEN_SET_NOT_APPROVED"}],
            }
            with open(os.path.join(official, "quality_report.json"), "w", encoding="utf-8") as handle:
                json.dump(report, handle)
            old_args, old_ds = app.ARGS, app.ds
            app.ARGS = SimpleNamespace(project_dir=project)
            app.ds = lambda: {"yolo": os.path.join(project, "legacy")}
            try:
                recovery = app.api_report_recovery()
                candidate = recovery["candidate"]
                self.assertEqual(candidate["dir"], official)
                self.assertEqual(candidate["artifact_role"], "official_recovery_candidate")
                self.assertEqual(
                    [item["code"] for item in candidate["errors"]],
                    ["GOLDEN_SET_NOT_APPROVED"],
                )
            finally:
                app.ARGS, app.ds = old_args, old_ds

    def test_golden_review_uses_routed_preview_urls(self):
        with tempfile.TemporaryDirectory() as project:
            package = os.path.join(project, "quality", "golden_review_v2")
            os.makedirs(package)
            with open(os.path.join(package, "index.html"), "w", encoding="utf-8") as handle:
                handle.write('<html><header>review</header><img src="previews/sample.jpg"></html>')
            old_args = app.ARGS
            app.ARGS = SimpleNamespace(project_dir=project)
            try:
                response = app.golden_review_page()
                self.assertIn(
                    'src="/golden-review/previews/sample.jpg"',
                    response.body.decode("utf-8"),
                )
            finally:
                app.ARGS = old_args

    def test_model_test_uses_latest_run_and_limits_results(self):
        import json
        with tempfile.TemporaryDirectory() as project:
            yolo = os.path.join(project, "yolo")
            predict = os.path.join(yolo, "predict")
            latest = os.path.join(predict, "run_new")
            os.makedirs(latest)
            for i in range(4):
                with open(os.path.join(latest, f"{i}.jpg"), "wb") as handle:
                    handle.write(b"test")
            with open(os.path.join(predict, "latest.json"), "w", encoding="utf-8") as handle:
                json.dump({"dir": latest}, handle)
            old_ds = app.ds
            app.ds = lambda: {"yolo": yolo}
            try:
                result = app.api_test_results(limit=2)
                self.assertEqual(app._test_result_dir(), latest)
                self.assertEqual(result["total"], 4)
                self.assertEqual(len(result["files"]), 2)
                self.assertTrue(result["truncated"])
            finally:
                app.ds = old_ds

    def test_failed_empty_export_preserves_previous_dataset(self):
        with tempfile.TemporaryDirectory() as project:
            images = os.path.join(project, "images")
            os.makedirs(images)
            cfg = {
                "paths": {"merged_dir": images, "results_root": "results"},
                "sam3": {"text_prompt": "target", "prompt_mode": "joined"},
                "web": {"workers": 1},
                "yolo_obb": {"min_mask_area": 10, "class_names": ["target"]},
            }
            dirs = run_dirs(project, cfg)
            os.makedirs(os.path.dirname(dirs["index"]), exist_ok=True)
            item = {
                "target_id": "image.jpg::0", "image": "image.jpg", "target_index": 0,
                "class_name": "target", "confidence": 0.9, "bbox": [0, 0, 10, 10],
                "mask_file": None, "overlay_file": "overlay.jpg",
            }
            with open(dirs["index"], "w", encoding="utf-8") as handle:
                import json
                handle.write(json.dumps(item) + "\n")
            os.makedirs(dirs["yolo"], exist_ok=True)
            sentinel = os.path.join(dirs["yolo"], "keep.txt")
            with open(sentinel, "w", encoding="utf-8") as handle:
                handle.write("previous")
            with self.assertRaises(RuntimeError):
                export_yolo_obb(project, cfg, fmt="obb")
            self.assertTrue(os.path.isfile(sentinel))


if __name__ == "__main__":
    unittest.main()
