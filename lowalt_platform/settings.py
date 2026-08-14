from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class PlatformSettings:
    project_root: Path
    candidate_geojson: Path
    candidate_summary: Path
    image_root: Path
    mask_root: Path
    secondary_analysis_root: Path
    overview_image: Path
    overview_manifest: Path
    web_root: Path
    import_root: Path | None = None
    allowed_source_roots: tuple[Path, ...] = ()
    parking_checkpoint: Path | None = None
    parking_config_dir: Path | None = None
    parking_head_checkpoint: Path | None = None

    @classmethod
    def from_project(cls, project_root: Path) -> "PlatformSettings":
        root = project_root.resolve()
        direct_layers = root / "quality" / "parking_direct_first_layers"
        candidate_roots = tuple(
            Path(drive)
            for drive in ("D:/", "E:/", "F:/", "G:/", "H:/", "C:/Users/sx")
            if Path(drive).anchor
        )
        return cls(
            project_root=root,
            candidate_geojson=direct_layers / "parking_facility_candidates.geojson",
            candidate_summary=direct_layers / "summary.json",
            image_root=root / "imagery",
            mask_root=root / "quality" / "parkseg_imagery_masks",
            secondary_analysis_root=root / "quality" / "parking_secondary_sam3",
            overview_image=root / "quality" / "platform_v1" / "overview.jpg",
            overview_manifest=root / "quality" / "platform_v1" / "overview.json",
            web_root=root / "lowalt_platform" / "web",
            import_root=root / "dji_imports",
            allowed_source_roots=candidate_roots,
            parking_checkpoint=root / "models" / "segformer_large_parking" / "best_model.ckpt",
            parking_config_dir=root / "models" / "nvidia_mit_b5_config",
            parking_head_checkpoint=root / "models" / "segformer_large_parking_finetuned_head" / "decode_head.pt",
        )
