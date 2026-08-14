from __future__ import annotations

import json
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, HTMLResponse
from pydantic import BaseModel

from lowalt_platform.services import ParkingCatalog, WmtsBlockCatalog
from lowalt_platform.services.import_runner import ImportService
from lowalt_platform.services.parking_analysis import ParkingAnalysisService
from lowalt_platform.settings import PlatformSettings


class ImportScanRequest(BaseModel):
    source_dir: str


class ImportCreateRequest(BaseModel):
    source_dir: str
    name: str | None = None
    frame_stride: int | None = None


class ImportAnalyzeRequest(BaseModel):
    device: str = "auto"


def _existing_file(path: Path, description: str) -> Path:
    if not path.is_file():
        raise HTTPException(404, detail=f"{description} is not available")
    return path


def _optional_bounds(
    west: float | None,
    south: float | None,
    east: float | None,
    north: float | None,
) -> tuple[float, float, float, float] | None:
    coordinates = (west, south, east, north)
    if any(value is not None for value in coordinates) and any(value is None for value in coordinates):
        raise HTTPException(400, detail="west, south, east and north must be provided together")
    return None if west is None else (west, south, east, north)


def create_platform_app(settings: PlatformSettings) -> FastAPI:
    catalog = ParkingCatalog.from_paths(
        settings.candidate_geojson,
        settings.candidate_summary,
        settings.image_root,
        settings.mask_root,
    )
    imagery = WmtsBlockCatalog(settings.image_root)
    overview = json.loads(settings.overview_manifest.read_text(encoding="utf-8"))
    app = FastAPI(title="低空遥感智能分析平台")
    import_service = ImportService(
        settings.import_root or settings.project_root / "dji_imports",
        settings.allowed_source_roots,
        settings.project_root,
    )
    parking_service = ParkingAnalysisService(
        import_service,
        checkpoint=settings.parking_checkpoint,
        config_dir=settings.parking_config_dir,
        head_checkpoint=settings.parking_head_checkpoint,
    )

    @app.get("/api/platform/summary")
    def summary() -> dict:
        counts = catalog.summary()
        completed_secondary_analyses = sum(
            1 for path in settings.secondary_analysis_root.glob("*/manifest.json") if path.is_file()
        )
        return {
            "project": "低空遥感智能分析平台",
            "imagery": {"source": "imagery WMTS", "count": overview["image_count"]},
            "overview": {"bounds": overview["bounds"], "image_url": "/api/platform/overview"},
            "parking_candidates": {
                "total": counts.total,
                "segformer_only": counts.segformer_only,
                "vehicle_detected": counts.vehicle_detected,
                "vehicle_row_supported": counts.vehicle_row_supported,
                "truth_status": "model_prediction_not_ground_truth",
            },
            "secondary_analysis": {
                "completed": completed_secondary_analyses,
                "total": counts.total,
            },
            "capabilities": [
                {"id": "parking", "name": "停车设施", "status": "candidate"},
                {"id": "dji_import", "name": "DJI 图片/视频导入", "status": "available"},
                {"id": "entities", "name": "车辆与人员", "status": "not_connected"},
                {"id": "equipment", "name": "施工设备", "status": "not_connected"},
                {"id": "waterway", "name": "河道与岸线", "status": "not_connected"},
            ],
        }

    @app.get("/api/platform/candidates")
    def candidates(
        west: float | None = None,
        south: float | None = None,
        east: float | None = None,
        north: float | None = None,
        support: str | None = None,
        limit: int = 2000,
    ) -> dict:
        bounds = _optional_bounds(west, south, east, north)
        support_levels = {item for item in (support or "").split(",") if item} or None
        try:
            features = catalog.query(bounds=bounds, support_levels=support_levels, limit=limit)
        except ValueError as exc:
            raise HTTPException(400, detail=str(exc)) from exc
        return {"type": "FeatureCollection", "features": features}

    @app.get("/api/platform/imagery/blocks")
    def imagery_blocks(
        west: float | None = None,
        south: float | None = None,
        east: float | None = None,
        north: float | None = None,
        limit: int = 256,
    ) -> dict:
        bounds = _optional_bounds(west, south, east, north)
        try:
            blocks = imagery.query(bounds, limit=limit)
        except ValueError as exc:
            raise HTTPException(400, detail=str(exc)) from exc
        return {"blocks": blocks}

    @app.get("/api/platform/imagery/blocks/{block_id}")
    def imagery_block(block_id: str) -> FileResponse:
        try:
            path = imagery.image_path(block_id)
        except KeyError as exc:
            raise HTTPException(404, detail=str(exc)) from exc
        return FileResponse(_existing_file(path, "WMTS block"), media_type="image/png")

    @app.get("/api/platform/candidates/{aoi_id}")
    def candidate_detail(aoi_id: str) -> dict:
        try:
            return catalog.detail(aoi_id)
        except KeyError as exc:
            raise HTTPException(404, detail=str(exc)) from exc

    @app.get("/api/platform/candidates/{aoi_id}/secondary")
    def candidate_secondary(aoi_id: str) -> dict:
        try:
            catalog.detail(aoi_id)
        except KeyError as exc:
            raise HTTPException(404, detail=str(exc)) from exc
        manifest_path = settings.secondary_analysis_root / aoi_id / "manifest.json"
        manifest = json.loads(_existing_file(manifest_path, "secondary analysis").read_text(encoding="utf-8"))
        return {
            "aoi_id": manifest["aoi_id"],
            "status": manifest["status"],
            "evidence": {
                evidence_type: {
                    "prompt": evidence.get("prompt"),
                    "target_count": evidence.get("target_count", 0),
                    "mask": evidence.get("mask"),
                }
                for evidence_type, evidence in manifest.get("evidence", {}).items()
            },
        }

    @app.get("/api/platform/candidates/{aoi_id}/secondary/{evidence_type}")
    def candidate_secondary_mask(aoi_id: str, evidence_type: str) -> FileResponse:
        manifest = candidate_secondary(aoi_id)
        evidence = manifest.get("evidence", {}).get(evidence_type)
        if not evidence or not evidence.get("mask"):
            raise HTTPException(404, detail="secondary evidence is not available")
        path = settings.secondary_analysis_root / aoi_id / evidence["mask"]
        return FileResponse(_existing_file(path, "secondary evidence mask"), media_type="image/png")

    @app.get("/api/platform/media/image/{aoi_id}")
    def candidate_image(aoi_id: str) -> FileResponse:
        try:
            path = catalog.image_path(aoi_id)
        except KeyError as exc:
            raise HTTPException(404, detail=str(exc)) from exc
        return FileResponse(_existing_file(path, "candidate image"), media_type="image/png")

    @app.get("/api/platform/media/mask/{aoi_id}")
    def candidate_mask(aoi_id: str) -> FileResponse:
        try:
            path = catalog.mask_path(aoi_id)
        except KeyError as exc:
            raise HTTPException(404, detail=str(exc)) from exc
        return FileResponse(_existing_file(path, "candidate mask"), media_type="image/png")

    @app.get("/api/platform/overview")
    def overview_image() -> FileResponse:
        return FileResponse(_existing_file(settings.overview_image, "overview image"), media_type="image/jpeg")

    @app.get("/")
    def index() -> HTMLResponse:
        path = settings.web_root / "index.html"
        return HTMLResponse(path.read_text(encoding="utf-8") if path.is_file() else "<h1>平台前端正在初始化</h1>")

    @app.get("/engineering")
    def engineering() -> HTMLResponse:
        path = settings.web_root / "engineering.html"
        return HTMLResponse(path.read_text(encoding="utf-8") if path.is_file() else "<h1>技术工作台正在初始化</h1>")

    @app.get("/legacy")
    def legacy() -> HTMLResponse:
        return HTMLResponse('<h1>历史系统</h1><p>旧服务仍运行在本机 7860 端口。</p>')

    # ---- DJI 图片/视频导入与分析 ------------------------------------------------
    @app.get("/api/platform/imports")
    def import_runs() -> dict:
        return {"runs": [run.__dict__ for run in import_service.list_runs()]}

    @app.post("/api/platform/imports/scan")
    def import_scan(request: ImportScanRequest) -> dict:
        try:
            return import_service.scan_source(request.source_dir)
        except ValueError as exc:
            raise HTTPException(400, detail=str(exc)) from exc

    @app.post("/api/platform/imports")
    def import_create(request: ImportCreateRequest) -> dict:
        try:
            run_id = import_service.create_run(request.source_dir, request.name)
        except ValueError as exc:
            raise HTTPException(400, detail=str(exc)) from exc
        import_service.start_import_thread(run_id, request.frame_stride or 30)
        return {"run_id": run_id}

    @app.get("/api/platform/imports/{run_id}")
    def import_detail(run_id: str) -> dict:
        try:
            return import_service.run_detail(run_id)
        except KeyError as exc:
            raise HTTPException(404, detail=str(exc)) from exc

    @app.post("/api/platform/imports/{run_id}/analyze")
    def import_analyze(run_id: str, request: ImportAnalyzeRequest | None = None) -> dict:
        try:
            import_service.run_detail(run_id)
        except KeyError as exc:
            raise HTTPException(404, detail=str(exc)) from exc
        available, reason = parking_service.availability()
        if not available:
            raise HTTPException(409, detail=f"停车分析不可用: {reason}")
        device = request.device if request else "auto"

        def _analysis() -> None:
            try:
                parking_service.run(run_id, device)
            except Exception as exc:  # noqa: BLE001
                try:
                    run_dir = import_service._run_dir(run_id)
                    import_service._append_log(run_dir, f"parking analysis thread failed: {exc}")
                    manifest = import_service._read_manifest(run_id)
                    manifest["parking_analysis"] = {"error": str(exc)}
                    import_service._write_manifest(run_dir, manifest)
                except Exception:  # noqa: BLE001
                    pass

        import_service.start_analysis_thread(run_id, _analysis)
        return {"run_id": run_id, "status": "started", "device": device}

    @app.get("/api/platform/imports/{run_id}/geojson")
    def import_geojson(run_id: str) -> dict:
        try:
            return import_service.geojson(run_id)
        except KeyError as exc:
            raise HTTPException(404, detail=str(exc)) from exc

    @app.get("/api/platform/imports/{run_id}/assets/{asset_id}/image")
    def import_asset_image(run_id: str, asset_id: str) -> FileResponse:
        try:
            path = import_service.asset_path(run_id, asset_id)
        except KeyError as exc:
            raise HTTPException(404, detail=str(exc)) from exc
        return FileResponse(path, media_type="image/jpeg")

    @app.get("/api/platform/imports/{run_id}/assets/{asset_id}/mask")
    def import_asset_mask(run_id: str, asset_id: str) -> FileResponse:
        try:
            path = import_service.analysis_artifact_path(run_id, asset_id, "mask.png")
        except KeyError as exc:
            raise HTTPException(404, detail=str(exc)) from exc
        return FileResponse(path, media_type="image/png")

    @app.get("/api/platform/imports/{run_id}/assets/{asset_id}/pred")
    def import_asset_pred(run_id: str, asset_id: str) -> FileResponse:
        try:
            path = import_service.analysis_artifact_path(run_id, asset_id, "pred.png")
        except KeyError as exc:
            raise HTTPException(404, detail=str(exc)) from exc
        return FileResponse(path, media_type="image/png")

    @app.get("/api/platform/imports/{run_id}/log")
    def import_log(run_id: str) -> dict:
        try:
            run_dir = import_service._run_dir(run_id)
        except KeyError as exc:
            raise HTTPException(404, detail=str(exc)) from exc
        log_path = run_dir / "run.log"
        lines = log_path.read_text(encoding="utf-8").splitlines()[-200:] if log_path.is_file() else []
        return {"lines": lines}

    @app.get("/static/{name:path}")
    def static_file(name: str) -> FileResponse:
        root = settings.web_root.resolve()
        path = (root / name).resolve()
        if root not in path.parents or not path.is_file():
            raise HTTPException(404, detail="static file not found")
        return FileResponse(path)

    return app
