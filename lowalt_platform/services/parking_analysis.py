from __future__ import annotations

import json
import time
from pathlib import Path

PARKING_REPO = "UTEL-UIUC/SegFormer-large-parking"
DEFAULT_THRESHOLD = 0.5


class ParkingAnalysisService:
    """Run the existing SegFormer-large-parking model over imported assets.

    The model is loaded once per run and the results are written next to the
    import manifest, so every number stays an honest model prediction.
    """

    def __init__(
        self,
        import_service,
        checkpoint: Path | None,
        config_dir: Path | None,
        head_checkpoint: Path | None = None,
        repo: str = PARKING_REPO,
        threshold: float = DEFAULT_THRESHOLD,
    ):
        self._imports = import_service
        self._checkpoint = checkpoint
        self._config_dir = config_dir
        self._head_checkpoint = head_checkpoint
        self._repo = repo
        self._threshold = threshold

    def availability(self) -> tuple[bool, str | None]:
        if self._checkpoint is None or not Path(self._checkpoint).is_file():
            return False, "checkpoint 未配置或缺失"
        if self._config_dir is None or not (Path(self._config_dir) / "preprocessor_config.json").is_file():
            return False, "模型配置目录未配置或缺失"
        try:
            import torch  # noqa: F401
            import transformers  # noqa: F401

            import parkseg12k_infer  # noqa: F401
        except ImportError as exc:
            return False, f"依赖缺失: {exc}"
        return True, None

    def run(self, run_id: str, device: str = "auto") -> dict:
        from training import parkseg12k_infer
        import torch

        assets = self._imports.image_assets(run_id)
        if not assets:
            raise ValueError("该任务没有可分析的图片或抽帧")
        checkpoint_path = Path(self._checkpoint)
        config_dir = Path(self._config_dir)
        state = parkseg12k_infer.load_checkpoint_state(checkpoint_path, self._repo)
        processor = parkseg12k_infer.load_processor(config_dir)
        model = parkseg12k_infer.build_model(state, config_dir)
        if self._head_checkpoint is not None and Path(self._head_checkpoint).is_file():
            parkseg12k_infer.apply_decode_head_checkpoint(model, Path(self._head_checkpoint))
        device_obj = parkseg12k_infer.resolve_device(device)
        model = model.to(device_obj)

        run_dir = self._imports._run_dir(run_id)
        analysis_dir = run_dir / "analysis"
        analysis_dir.mkdir(exist_ok=True)
        self._imports._append_log(run_dir, f"parking analysis started (device={device_obj})")

        results = []
        for index, asset in enumerate(assets, start=1):
            asset_id = asset["asset_id"]
            image_path = self._imports.asset_path(run_id, asset_id)
            started = time.perf_counter()
            inference = parkseg12k_infer.run_inference(
                model,
                processor,
                device_obj,
                image_path,
                analysis_dir,
                self._threshold,
                save_diagnostics=True,
            )
            result = {
                "mask_file": f"{image_path.stem}_mask.png",
                "pred_file": f"{image_path.stem}_pred.png",
                "parking_fraction": inference["predicted_parking_fraction"],
                "mean_parking_probability": inference["mean_parking_probability"],
                "inference_seconds": inference["inference_seconds"],
                "analyzed_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
                "wall_seconds": round(time.perf_counter() - started, 3),
            }
            self._imports.set_parking_result(run_id, asset_id, result)
            results.append({"asset_id": asset_id, **result})
            if index % 10 == 0:
                self._imports._append_log(run_dir, f"parking analysis progress {index}/{len(assets)}")

        summary = {
            "model_repo": self._repo,
            "checkpoint": str(checkpoint_path),
            "head_checkpoint": str(self._head_checkpoint) if self._head_checkpoint else None,
            "device": str(device_obj),
            "threshold": self._threshold,
            "total_assets": len(assets),
            "analyzed": len(results),
            "results": results,
        }
        (analysis_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
        self._imports.set_parking_analysis(run_id, {"device": str(device_obj), "threshold": self._threshold, "analyzed": len(results), "total": len(assets)})
        self._imports._append_log(run_dir, f"parking analysis done: {len(results)} assets")
        return summary

    def ensure_available(self) -> None:
        available, reason = self.availability()
        if not available:
            raise RuntimeError(f"停车分析不可用: {reason}")
