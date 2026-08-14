from __future__ import annotations

import json
import shutil
import threading
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from lowalt_platform.services import dji_metadata

DEFAULT_FRAME_STRIDE = 30
DEFAULT_MAX_FRAMES_PER_VIDEO = 480


@dataclass(frozen=True)
class ImportRunView:
    run_id: str
    name: str
    source_dir: str
    status: str
    created_at: str
    images: int
    videos: int
    srt: int
    assets: int
    georeferenced_assets: int
    parking_analyzed: int
    error: str | None = None


def _now() -> str:
    return datetime.now().isoformat(timespec="seconds")


class ImportService:
    def __init__(self, import_root: Path, allowed_source_roots: tuple[Path, ...], project_root: Path):
        self._import_root = Path(import_root)
        self._allowed_roots = tuple(Path(root) for root in allowed_source_roots)
        self._project_root = Path(project_root).resolve()
        self._import_root.mkdir(parents=True, exist_ok=True)

    # -- source access control -------------------------------------------------
    def is_allowed_source(self, source_dir: str | Path) -> bool:
        try:
            resolved = Path(source_dir).resolve()
        except OSError:
            return False
        if not resolved.is_dir():
            return False
        try:
            resolved.relative_to(self._project_root)
            return True
        except ValueError:
            pass
        for root in self._allowed_roots:
            try:
                resolved.relative_to(root.resolve())
                return True
            except (ValueError, OSError):
                continue
        return False

    def allowed_roots_text(self) -> str:
        roots = [str(self._project_root), *(str(root) for root in self._allowed_roots)]
        return "；".join(roots)

    # -- scanning ---------------------------------------------------------------
    def scan_source(self, source_dir: str | Path) -> dict:
        if not self.is_allowed_source(source_dir):
            raise ValueError(f"目录不在允许范围内。允许：{self.allowed_roots_text()}")
        root = Path(source_dir)
        images = sorted(path for path in root.iterdir() if path.is_file() and path.suffix.lower() in dji_metadata.IMAGE_SUFFIXES)
        videos = sorted(path for path in root.iterdir() if path.is_file() and path.suffix.lower() in dji_metadata.VIDEO_SUFFIXES)
        srts = sorted(path for path in root.iterdir() if path.is_file() and path.suffix.lower() in dji_metadata.SRT_SUFFIXES)
        return {
            "source_dir": str(root),
            "images": [{"name": path.name, "size": path.stat().st_size} for path in images],
            "videos": [{"name": path.name, "size": path.stat().st_size} for path in videos],
            "srt": [{"name": path.name, "size": path.stat().st_size} for path in srts],
        }

    # -- run lifecycle -----------------------------------------------------------
    def create_run(self, source_dir: str | Path, name: str | None = None) -> str:
        if not self.is_allowed_source(source_dir):
            raise ValueError(f"目录不在允许范围内。允许：{self.allowed_roots_text()}")
        run_id = "run-" + datetime.now().strftime("%Y%m%d-%H%M%S")
        run_dir = self._import_root / run_id
        run_dir.mkdir(parents=True)
        manifest = {
            "run_id": run_id,
            "name": (name or Path(source_dir).name).strip() or Path(source_dir).name,
            "source_dir": str(Path(source_dir)),
            "created_at": _now(),
            "status": "pending",
            "frame_stride": DEFAULT_FRAME_STRIDE,
            "max_frames_per_video": DEFAULT_MAX_FRAMES_PER_VIDEO,
            "counts": {"images": 0, "videos": 0, "srt": 0, "frames_extracted": 0, "failed": 0},
            "assets": [],
            "parking_analysis": None,
            "error": None,
        }
        self._write_manifest(run_dir, manifest)
        self._append_log(run_dir, "run created")
        return run_id

    def _run_dir(self, run_id: str) -> Path:
        run_dir = self._import_root / run_id
        if not run_dir.is_dir():
            raise KeyError(f"unknown import run: {run_id}")
        return run_dir

    @staticmethod
    def _write_manifest(run_dir: Path, manifest: dict) -> None:
        target = run_dir / "manifest.json"
        temporary = run_dir / ".manifest.json.tmp"
        temporary.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
        temporary.replace(target)

    @staticmethod
    def _append_log(run_dir: Path, line: str) -> None:
        with (run_dir / "run.log").open("a", encoding="utf-8") as handle:
            handle.write(f"{_now()} {line}\n")

    def _read_manifest(self, run_id: str) -> dict:
        return json.loads((self._run_dir(run_id) / "manifest.json").read_text(encoding="utf-8"))

    # -- import execution ----------------------------------------------------------
    def execute_import(self, run_id: str, frame_stride: int = DEFAULT_FRAME_STRIDE, max_frames_per_video: int = DEFAULT_MAX_FRAMES_PER_VIDEO) -> dict:
        run_dir = self._run_dir(run_id)
        manifest = self._read_manifest(run_id)
        source_dir = Path(manifest["source_dir"])
        if not 1 <= frame_stride <= 600:
            raise ValueError("frame stride must be between 1 and 600")
        if not 1 <= max_frames_per_video <= 2000:
            raise ValueError("max frames per video must be between 1 and 2000")
        manifest["status"] = "running"
        manifest["frame_stride"] = frame_stride
        manifest["max_frames_per_video"] = max_frames_per_video
        self._write_manifest(run_dir, manifest)
        self._append_log(run_dir, f"import started (stride={frame_stride})")

        assets_dir = run_dir / "assets"
        assets_dir.mkdir(exist_ok=True)
        assets: list[dict] = []
        counts = {"images": 0, "videos": 0, "srt": 0, "frames_extracted": 0, "failed": 0}

        srt_records: list[dict] = []
        for srt_path in sorted(source_dir.iterdir()):
            if srt_path.is_file() and srt_path.suffix.lower() in dji_metadata.SRT_SUFFIXES:
                try:
                    srt_records.extend(dji_metadata.parse_srt(srt_path.read_text(encoding="utf-8", errors="replace")))
                    counts["srt"] += 1
                except OSError:
                    counts["failed"] += 1

        try:
            for image_path in sorted(path for path in source_dir.iterdir() if path.is_file() and path.suffix.lower() in dji_metadata.IMAGE_SUFFIXES):
                try:
                    counts["images"] += 1
                    asset_id = f"img_{counts['images']:05d}"
                    target = assets_dir / f"{asset_id}{image_path.suffix.lower()}"
                    shutil.copy2(image_path, target)
                    metadata = dji_metadata.extract_image_metadata(target)
                    asset = self._build_image_asset(asset_id, target, metadata, srt_records)
                    assets.append(asset)
                except Exception as exc:  # noqa: BLE001 - one bad file must not kill the run
                    counts["failed"] += 1
                    self._append_log(run_dir, f"image {image_path.name} failed: {exc}")

            for video_path in sorted(path for path in source_dir.iterdir() if path.is_file() and path.suffix.lower() in dji_metadata.VIDEO_SUFFIXES):
                try:
                    counts["videos"] += 1
                    frame_assets = self._extract_frames(video_path, counts["videos"], assets_dir, srt_records, frame_stride, max_frames_per_video, run_dir)
                    assets.extend(frame_assets)
                    counts["frames_extracted"] += len(frame_assets)
                except Exception as exc:  # noqa: BLE001
                    counts["failed"] += 1
                    self._append_log(run_dir, f"video {video_path.name} failed: {exc}")
        except Exception as exc:  # noqa: BLE001
            manifest["status"] = "error"
            manifest["error"] = str(exc)
            manifest["counts"] = counts
            manifest["assets"] = assets
            self._write_manifest(run_dir, manifest)
            self._append_log(run_dir, f"import failed: {exc}")
            raise

        manifest["status"] = "done"
        manifest["counts"] = counts
        manifest["assets"] = assets
        manifest["error"] = None
        self._write_manifest(run_dir, manifest)
        self._append_log(run_dir, f"import done: {counts}")
        return manifest

    def _build_image_asset(self, asset_id: str, target: Path, metadata: dict, srt_records: list[dict]) -> dict:
        del srt_records  # SRT timestamps are relative to video start; not sound for stills
        latitude, longitude, altitude_m = metadata["latitude"], metadata["longitude"], metadata["altitude_m"]
        gps_source = "exif" if latitude is not None else "none"
        return {
            "asset_id": asset_id,
            "kind": "image",
            "source": str(target),
            "file": target.name,
            "width": metadata["width"],
            "height": metadata["height"],
            "captured_at": metadata["captured_at"],
            "latitude": latitude,
            "longitude": longitude,
            "altitude_m": altitude_m,
            "gps_source": gps_source,
            "camera": {"make": metadata["camera_make"], "model": metadata["camera_model"]},
            "video": None,
            "parking": None,
        }

    def _extract_frames(self, video_path: Path, video_index: int, assets_dir: Path, srt_records: list[dict], stride: int, max_frames: int, run_dir: Path) -> list[dict]:
        try:
            import cv2  # noqa: PLC0415 - optional heavy dependency
        except ImportError as exc:
            raise RuntimeError("opencv-python is required for video frame extraction") from exc
        capture = cv2.VideoCapture(str(video_path))
        if not capture.isOpened():
            raise RuntimeError(f"cannot open video: {video_path.name}")
        fps = float(capture.get(cv2.CAP_PROP_FPS) or 0)
        frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
        height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
        frames: list[dict] = []
        frame_number = 0
        for index in range(0, max(frame_count, 1), stride):
            if len(frames) >= max_frames:
                break
            capture.set(cv2.CAP_PROP_POS_FRAMES, index)
            ok, frame = capture.read()
            if not ok:
                continue
            frame_number += 1
            asset_id = f"frm_{video_index:03d}_{frame_number:05d}"
            target = assets_dir / f"{asset_id}.jpg"
            cv2.imwrite(str(target), frame, [cv2.IMWRITE_JPEG_QUALITY, 92])
            time_seconds = index / fps if fps > 0 else None
            latitude = longitude = altitude_m = None
            gps_source = "none"
            if time_seconds is not None:
                record, gps_source = dji_metadata.srt_nearest_position(srt_records, time_seconds)
                if record:
                    latitude, longitude, altitude_m = record["latitude"], record["longitude"], record["altitude_m"]
            frames.append(
                {
                    "asset_id": asset_id,
                    "kind": "frame",
                    "source": str(target),
                    "file": target.name,
                    "width": width,
                    "height": height,
                    "captured_at": None,
                    "latitude": latitude,
                    "longitude": longitude,
                    "altitude_m": altitude_m,
                    "gps_source": gps_source,
                    "camera": None,
                    "video": {
                        "name": video_path.name,
                        "frame_index": index,
                        "time_seconds": round(time_seconds, 3) if time_seconds is not None else None,
                        "fps": round(fps, 3) if fps else None,
                        "frame_count": frame_count,
                    },
                    "parking": None,
                }
            )
        capture.release()
        if not frames:
            self._append_log(run_dir, f"video {video_path.name}: no frames extracted")
        return frames

    # -- views ---------------------------------------------------------------------
    def list_runs(self) -> list[ImportRunView]:
        runs: list[ImportRunView] = []
        for run_dir in sorted(self._import_root.glob("run-*")):
            manifest_path = run_dir / "manifest.json"
            if not manifest_path.is_file():
                continue
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            assets = manifest.get("assets") or []
            runs.append(
                ImportRunView(
                    run_id=manifest["run_id"],
                    name=manifest.get("name", ""),
                    source_dir=manifest.get("source_dir", ""),
                    status=manifest.get("status", "unknown"),
                    created_at=manifest.get("created_at", ""),
                    images=manifest.get("counts", {}).get("images", 0),
                    videos=manifest.get("counts", {}).get("videos", 0),
                    srt=manifest.get("counts", {}).get("srt", 0),
                    assets=len(assets),
                    georeferenced_assets=sum(1 for asset in assets if asset.get("latitude") is not None),
                    parking_analyzed=int((manifest.get("parking_analysis") or {}).get("analyzed", 0) or 0),
                    error=manifest.get("error"),
                )
            )
        return runs

    def run_detail(self, run_id: str) -> dict:
        manifest = self._read_manifest(run_id)
        assets = manifest.get("assets") or []
        manifest["georeferenced_assets"] = sum(1 for asset in assets if asset.get("latitude") is not None)
        manifest["analyzed_assets"] = sum(1 for asset in assets if asset.get("parking"))
        return manifest

    def asset_path(self, run_id: str, asset_id: str) -> Path:
        manifest = self._read_manifest(run_id)
        asset = next((item for item in manifest.get("assets", []) if item["asset_id"] == asset_id), None)
        if asset is None:
            raise KeyError(f"unknown asset: {asset_id}")
        run_dir = self._run_dir(run_id)
        candidate = run_dir / "assets" / asset["file"]
        if not candidate.is_file():
            raise KeyError(f"asset file missing: {asset_id}")
        return candidate

    def analysis_artifact_path(self, run_id: str, asset_id: str, suffix: str) -> Path:
        manifest = self._read_manifest(run_id)
        asset = next((item for item in manifest.get("assets", []) if item["asset_id"] == asset_id), None)
        if asset is None:
            raise KeyError(f"unknown asset: {asset_id}")
        stem = Path(asset["file"]).stem
        run_dir = self._run_dir(run_id)
        candidate = run_dir / "analysis" / f"{stem}_{suffix}"
        if not candidate.is_file():
            raise KeyError(f"analysis artifact missing: {asset_id} ({suffix})")
        return candidate

    def geojson(self, run_id: str) -> dict:
        manifest = self._read_manifest(run_id)
        features = []
        for asset in manifest.get("assets", []):
            if asset.get("latitude") is None or asset.get("longitude") is None:
                continue
            parking = asset.get("parking") or {}
            features.append(
                {
                    "type": "Feature",
                    "geometry": {"type": "Point", "coordinates": [asset["longitude"], asset["latitude"]]},
                    "properties": {
                        "asset_id": asset["asset_id"],
                        "kind": asset["kind"],
                        "captured_at": asset.get("captured_at"),
                        "altitude_m": asset.get("altitude_m"),
                        "gps_source": asset.get("gps_source"),
                        "video": asset.get("video"),
                        "parking_fraction": parking.get("parking_fraction"),
                        "mask_available": bool(parking.get("mask_file")),
                    },
                }
            )
        return {"type": "FeatureCollection", "features": features}

    # -- helpers for analysis -------------------------------------------------------
    def image_assets(self, run_id: str) -> list[dict]:
        manifest = self._read_manifest(run_id)
        return [asset for asset in manifest.get("assets", []) if asset.get("kind") in {"image", "frame"}]

    def set_parking_result(self, run_id: str, asset_id: str, result: dict) -> None:
        run_dir = self._run_dir(run_id)
        manifest = self._read_manifest(run_id)
        for asset in manifest.get("assets", []):
            if asset["asset_id"] == asset_id:
                asset["parking"] = result
                break
        self._write_manifest(run_dir, manifest)

    def set_parking_analysis(self, run_id: str, analysis: dict) -> None:
        run_dir = self._run_dir(run_id)
        manifest = self._read_manifest(run_id)
        manifest["parking_analysis"] = analysis
        self._write_manifest(run_dir, manifest)

    # -- threading ----------------------------------------------------------------
    def start_import_thread(self, run_id: str, frame_stride: int = DEFAULT_FRAME_STRIDE, max_frames_per_video: int = DEFAULT_MAX_FRAMES_PER_VIDEO) -> None:
        def _run() -> None:
            try:
                self.execute_import(run_id, frame_stride, max_frames_per_video)
            except Exception as exc:  # noqa: BLE001
                run_dir = self._run_dir(run_id)
                manifest = self._read_manifest(run_id)
                manifest["status"] = "error"
                manifest["error"] = str(exc)
                self._write_manifest(run_dir, manifest)
                self._append_log(run_dir, f"import thread failed: {exc}")

        threading.Thread(target=_run, name=f"lowalt-import-{run_id}", daemon=True).start()

    def start_analysis_thread(self, run_id: str, analysis) -> None:
        threading.Thread(target=analysis, name=f"lowalt-parking-{run_id}", daemon=True).start()
