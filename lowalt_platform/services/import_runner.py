from __future__ import annotations

import hashlib
import json
import math
import re
import shutil
import threading
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from uuid import uuid4
from zipfile import ZIP_DEFLATED, ZipFile

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
    return datetime.now().isoformat(timespec="microseconds")


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

    def _allowed_file(self, path: Path) -> bool:
        # A symlink inside an allowed directory must not escape that directory.
        return self.is_allowed_source(path.resolve().parent)

    @staticmethod
    def _hash(path: Path) -> str:
        with path.open('rb') as handle:
            return hashlib.file_digest(handle, 'sha256').hexdigest()

    def allowed_roots_text(self) -> str:
        roots = [str(self._project_root), *(str(root) for root in self._allowed_roots)]
        return "；".join(roots)

    # -- scanning ---------------------------------------------------------------
    def scan_source(self, source_dir: str | Path) -> dict:
        if not self.is_allowed_source(source_dir):
            raise ValueError(f"目录不在允许范围内。允许：{self.allowed_roots_text()}")
        root = Path(source_dir).resolve()
        files = [path for path in root.iterdir() if path.is_file() and self._allowed_file(path)]
        images = sorted(path for path in files if path.suffix.lower() in dji_metadata.IMAGE_SUFFIXES)
        videos = sorted(path for path in files if path.suffix.lower() in dji_metadata.VIDEO_SUFFIXES)
        srts = sorted(path for path in files if path.suffix.lower() in dji_metadata.SRT_SUFFIXES)
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
        run_id = "run-" + datetime.now().strftime("%Y%m%d-%H%M%S") + "-" + uuid4().hex[:12]
        run_dir = self._import_root / run_id
        run_dir.mkdir(parents=True)
        manifest = {
            "run_id": run_id,
            "name": (name or Path(source_dir).name).strip() or Path(source_dir).name,
            "source_dir": str(Path(source_dir).resolve()),
            "schema_version": 2,
            "warnings": [],
            "sidecars": [],
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
        if not re.fullmatch(r'run-[A-Za-z0-9-]+', run_id):
            raise KeyError(f"unknown import run: {run_id}")
        run_dir = self._import_root / run_id
        if run_dir.is_symlink() or not run_dir.is_dir():
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
        if type(frame_stride) is not int or not 1 <= frame_stride <= 600:
            raise ValueError("frame stride must be between 1 and 600")
        if type(max_frames_per_video) is not int or not 1 <= max_frames_per_video <= 2000:
            raise ValueError("max frames per video must be between 1 and 2000")
        if manifest['status'] != 'pending':
            raise ValueError('only pending runs can be imported')
        if not self.is_allowed_source(source_dir):
            raise ValueError('source directory is no longer allowed or available')
        manifest["status"] = "running"
        manifest["frame_stride"] = frame_stride
        manifest["max_frames_per_video"] = max_frames_per_video
        self._write_manifest(run_dir, manifest)
        self._append_log(run_dir, f"import started (stride={frame_stride})")

        assets_dir = run_dir / "assets"
        assets_dir.mkdir(exist_ok=True)
        assets: list[dict] = []
        counts = {"images": 0, "videos": 0, "srt": 0, "frames_extracted": 0, "failed": 0}

        files = sorted(path for path in source_dir.iterdir() if path.is_file() and self._allowed_file(path))
        sidecars: dict[str, list[dict]] = {}
        evidence_dir = run_dir / 'sidecars'
        evidence_dir.mkdir(exist_ok=True)
        for srt_path in (path for path in files if path.suffix.lower() in dji_metadata.SRT_SUFFIXES):
            try:
                target = evidence_dir / srt_path.name
                shutil.copy2(srt_path, target)
                entry = {'name': srt_path.name, 'file': target.name, 'sha256': self._hash(target)}
                entry['records'] = dji_metadata.parse_srt(target.read_text(encoding='utf-8-sig', errors='replace'))
                sidecars.setdefault(srt_path.stem.casefold(), []).append(entry)
                manifest['sidecars'].append({key: value for key, value in entry.items() if key != 'records'})
                counts['srt'] += 1
            except OSError as exc:
                counts['failed'] += 1
                manifest['warnings'].append(f'sidecar {srt_path.name} failed: {exc}')

        try:
            for image_path in sorted(path for path in files if path.suffix.lower() in dji_metadata.IMAGE_SUFFIXES):
                try:
                    counts["images"] += 1
                    asset_id = f"img_{counts['images']:05d}"
                    target = assets_dir / f"{asset_id}{image_path.suffix.lower()}"
                    shutil.copy2(image_path, target)
                    metadata = dji_metadata.extract_image_metadata(target)
                    asset = self._build_image_asset(asset_id, target, metadata)
                    asset['original_name'] = image_path.name
                    asset['source_sha256'] = self._hash(target)
                    asset['asset_sha256'] = asset['source_sha256']
                    assets.append(asset)
                except Exception as exc:  # noqa: BLE001 - one bad file must not kill the run
                    counts["failed"] += 1
                    manifest["warnings"].append(f"image {image_path.name} failed ({type(exc).__name__}); see local run.log")
                    self._append_log(run_dir, f"image {image_path.name} failed: {exc}")

            for video_path in sorted(path for path in files if path.suffix.lower() in dji_metadata.VIDEO_SUFFIXES):
                try:
                    counts["videos"] += 1
                    matches = sidecars.get(video_path.stem.casefold(), [])
                    sidecar = matches[0] if len(matches) == 1 else None
                    if sidecar is None:
                        reason = 'ambiguous' if matches else 'missing'
                        manifest['warnings'].append(f'{video_path.name}: {reason} same-stem SRT; frames remain unpositioned')
                    video_hash = self._hash(video_path)
                    frame_assets = self._extract_frames(video_path, counts["videos"], assets_dir, sidecar['records'] if sidecar else [], frame_stride, max_frames_per_video, run_dir)
                    for asset in frame_assets:
                        asset['original_name'] = video_path.name
                        asset['source_sha256'] = video_hash
                        asset['asset_sha256'] = self._hash(assets_dir / asset['file'])
                        asset['video']['srt_name'] = sidecar['name'] if sidecar else None
                        asset['video']['srt_sha256'] = sidecar['sha256'] if sidecar else None
                    assets.extend(frame_assets)
                    counts["frames_extracted"] += len(frame_assets)
                except Exception as exc:  # noqa: BLE001
                    counts["failed"] += 1
                    manifest["warnings"].append(f"video {video_path.name} failed ({type(exc).__name__}); see local run.log")
                    self._append_log(run_dir, f"video {video_path.name} failed: {exc}")
        except Exception as exc:  # noqa: BLE001
            manifest["status"] = "error"
            manifest["error"] = str(exc)
            manifest["counts"] = counts
            manifest["assets"] = assets
            self._write_manifest(run_dir, manifest)
            self._append_log(run_dir, f"import failed: {exc}")
            raise

        manifest["status"] = "partial" if counts["failed"] else "done"
        manifest["counts"] = counts
        manifest["assets"] = assets
        manifest["error"] = None
        self._write_manifest(run_dir, manifest)
        self._append_log(run_dir, f"import done: {counts}")
        return manifest

    def _build_image_asset(self, asset_id: str, target: Path, metadata: dict) -> dict:
        latitude, longitude, altitude_m = metadata["latitude"], metadata["longitude"], metadata["altitude_m"]
        gps_source = "exif" if dji_metadata.valid_position(latitude, longitude) else "none"
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
            "altitude_reference": metadata.get('altitude_reference'),
            "metadata_warnings": metadata.get('warnings', []),
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
        try:
            return self._read_frames(cv2, capture, video_path, video_index, assets_dir, srt_records, stride, max_frames, run_dir)
        finally:
            capture.release()

    def _read_frames(self, cv2, capture, video_path, video_index, assets_dir, srt_records, stride, max_frames, run_dir) -> list[dict]:
        if not capture.isOpened():
            raise RuntimeError(f"cannot open video: {video_path.name}")
        fps = float(capture.get(cv2.CAP_PROP_FPS) or 0)
        if not math.isfinite(fps) or fps <= 0:
            raise RuntimeError(f'invalid video FPS: {video_path.name}; cannot align SRT')
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
            if not cv2.imwrite(str(target), frame, [cv2.IMWRITE_JPEG_QUALITY, 92]):
                raise RuntimeError(f'cannot write extracted frame: {target.name}')
            time_seconds = index / fps if fps > 0 else None
            latitude = longitude = altitude_m = None
            gps_source = "none"
            record = None
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
                    "altitude_reference": record.get('altitude_reference') if record else None,
                    "relative_altitude_m": record.get('relative_altitude_m') if record else None,
                    "camera": None,
                    "video": {
                        "name": video_path.name,
                        "frame_index": index,
                        "time_seconds": round(time_seconds, 6) if time_seconds is not None else None,
                        "fps": fps,
                        "frame_count": frame_count,
                        "srt_record_index": record.get('record_index') if record else None,
                        "srt_time_seconds": record['time_seconds'] if record else None,
                        "srt_delta_seconds": round(abs(record['time_seconds'] - time_seconds), 6) if record else None,
                        "timestamp_method": 'frame_index_over_fps',
                    },
                    "parking": None,
                }
            )
        if not frames:
            raise RuntimeError(f"video {video_path.name}: no frames extracted")
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
                    georeferenced_assets=sum(1 for asset in assets if dji_metadata.valid_position(asset.get("latitude"), asset.get("longitude"))),
                    parking_analyzed=int((manifest.get("parking_analysis") or {}).get("analyzed", 0) or 0),
                    error=manifest.get("error"),
                )
            )
        return sorted(runs, key=lambda run: run.created_at)

    def run_detail(self, run_id: str) -> dict:
        manifest = self._read_manifest(run_id)
        assets = manifest.get("assets") or []
        manifest["georeferenced_assets"] = sum(1 for asset in assets if dji_metadata.valid_position(asset.get("latitude"), asset.get("longitude")))
        manifest["analyzed_assets"] = sum(1 for asset in assets if asset.get("parking"))
        return manifest

    def asset_path(self, run_id: str, asset_id: str) -> Path:
        manifest = self._read_manifest(run_id)
        asset = next((item for item in manifest.get("assets", []) if item["asset_id"] == asset_id), None)
        if asset is None:
            raise KeyError(f"unknown asset: {asset_id}")
        run_dir = self._run_dir(run_id)
        candidate = run_dir / "assets" / asset["file"]
        if candidate.resolve().parent != (run_dir / "assets").resolve() or candidate.is_symlink() or not candidate.is_file():
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
        if candidate.resolve().parent != (run_dir / "analysis").resolve() or candidate.is_symlink() or not candidate.is_file():
            raise KeyError(f"analysis artifact missing: {asset_id} ({suffix})")
        return candidate

    def geojson(self, run_id: str) -> dict:
        manifest = self._read_manifest(run_id)
        features = []
        for asset in manifest.get("assets", []):
            if not dji_metadata.valid_position(asset.get("latitude"), asset.get("longitude")):
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
                        "altitude_reference": asset.get("altitude_reference"),
                        "relative_altitude_m": asset.get("relative_altitude_m"),
                        "original_name": asset.get("original_name"),
                        "source_sha256": asset.get("source_sha256"),
                        "truth_status": 'model_prediction_not_ground_truth' if parking else 'metadata_observation_not_ground_truth',
                        "review_status": 'unreviewed',
                        "gsd_m": None,
                        "gps_source": asset.get("gps_source"),
                        "asset_sha256": asset.get("asset_sha256"),
                        "video": asset.get("video"),
                        "parking_fraction": parking.get("parking_fraction"),
                        "mask_available": bool(parking.get("mask_file")),
                    },
                }
            )
        return {"type": "FeatureCollection", "features": features}

    def export_evidence(self, run_id: str) -> Path:
        manifest = self.run_detail(run_id)
        if manifest['status'] not in {'done', 'partial'}:
            raise ValueError('import must finish before evidence export')
        run_dir = self._run_dir(run_id)
        public_manifest = json.loads(json.dumps(manifest))
        public_manifest.pop('source_dir', None)
        for asset in public_manifest['assets']:
            asset.pop('source', None)
        target = run_dir / f'evidence-{uuid4().hex}.zip'
        try:
            with ZipFile(target, 'w', compression=ZIP_DEFLATED) as archive:
                archive.writestr('manifest.json', json.dumps(public_manifest, ensure_ascii=False, indent=2, allow_nan=False))
                archive.writestr('assets.geojson', json.dumps(self.geojson(run_id), ensure_ascii=False, indent=2, allow_nan=False))
                archive.writestr('README.txt', 'Imported metadata evidence, not ground truth or model validation.\n'
                                 'Coordinates: WGS84 longitude, latitude; point positions are camera locations, not ground footprints.\n'
                                 'Video times use frame index / nominal FPS (constant frame rate only). No GPS interpolation.\n'
                                 'Source videos are not bundled; source_sha256 identifies them. Stills/frames and SRT are included.\n'
                                 'gsd_m is unknown: do not infer area/length from camera altitude. Review/release gates still apply.\n')
                for asset in manifest['assets']:
                    path = self.asset_path(run_id, asset['asset_id'])
                    if asset.get('asset_sha256') and self._hash(path) != asset['asset_sha256']:
                        raise ValueError(f"asset checksum changed: {asset['asset_id']}")
                    archive.write(path, 'assets/' + path.name)
                for sidecar in manifest.get('sidecars', []):
                    path = run_dir / 'sidecars' / sidecar['file']
                    if path.name != sidecar['file'] or path.is_symlink() or self._hash(path) != sidecar['sha256']:
                        raise ValueError('sidecar checksum or path changed')
                    archive.write(path, 'sidecars/' + path.name)
        except Exception:
            target.unlink(missing_ok=True)
            raise
        return target

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
