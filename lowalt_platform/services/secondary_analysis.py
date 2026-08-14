from __future__ import annotations

import base64
from collections.abc import Callable
from io import BytesIO
import json
import os
from pathlib import Path

from PIL import Image, ImageChops


SECONDARY_PROMPTS = {
    "vehicle": "vehicle, car, truck, bus",
    "parking_marking": "white painted parking stall divider line",
    "internal_aisle": "parking lot internal driving aisle",
    "building": "building, roof",
    "vegetation": "tree, shrub, vegetation",
}


Sam3Client = Callable[[Path, str], dict]


def _atomic_json(path: Path, payload: dict) -> None:
    temporary_path = path.with_name(f".{path.name}.tmp")
    temporary_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    os.replace(temporary_path, path)


class ParkingSecondaryAnalyzer:
    def __init__(self, client: Sam3Client, output_root: Path, *, save_instance_masks: bool = False):
        self._client = client
        self._output_root = output_root
        self._save_instance_masks = save_instance_masks

    def analyze(
        self,
        aoi_id: str,
        image_path: Path,
        *,
        candidate_mask_path: Path | None = None,
    ) -> dict:
        output_directory = self._output_root / aoi_id
        manifest_path = output_directory / "manifest.json"
        if manifest_path.is_file():
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            if manifest.get("status") == "completed":
                return {**manifest, "resumed": True}

        output_directory.mkdir(parents=True, exist_ok=True)
        candidate_mask = None
        if candidate_mask_path is not None:
            with Image.open(candidate_mask_path) as source_candidate_mask:
                candidate_mask = source_candidate_mask.convert("L").point(lambda value: 255 if value > 0 else 0)
        evidence = {}
        for evidence_type, prompt in SECONDARY_PROMPTS.items():
            response = self._client(image_path, prompt)
            if response.get("error"):
                raise RuntimeError(f"SAM3 {evidence_type} failed: {response['error']}")
            targets = response.get("result", {}).get("targets") or []
            mask_path = output_directory / f"{evidence_type}.png"
            instance_directory = output_directory / f"{evidence_type}_instances"
            if self._save_instance_masks:
                instance_directory.mkdir(exist_ok=True)
            combined_mask = None
            mask_count = 0
            for target in targets:
                encoded_mask = target.get("single_mask")
                if encoded_mask:
                    mask_bytes = base64.b64decode(encoded_mask)
                    if self._save_instance_masks:
                        instance_path = instance_directory / f"{mask_count:03d}.png"
                        instance_path.write_bytes(mask_bytes)
                    with Image.open(BytesIO(mask_bytes)) as source_mask:
                        mask = source_mask.convert("L")
                        if candidate_mask is not None:
                            if mask.size != candidate_mask.size:
                                raise ValueError(f"candidate mask size differs from SAM3 mask for {aoi_id}")
                            mask = ImageChops.multiply(mask, candidate_mask)
                            if self._save_instance_masks:
                                mask.save(instance_path, format="PNG")
                        combined_mask = mask.copy() if combined_mask is None else ImageChops.lighter(combined_mask, mask)
                    mask_count += 1
            if combined_mask is not None:
                combined_mask.save(mask_path, format="PNG")
            evidence[evidence_type] = {
                "prompt": prompt,
                "target_count": len(targets),
                "mask": mask_path.name if combined_mask is not None else None,
                "instance_mask_count": mask_count,
                "targets": [
                    {
                        "bbox": target.get("bbox"),
                        "confidence": target.get("confidence"),
                        "class_name": target.get("class_name"),
                    }
                    for target in targets
                ],
            }
            if combined_mask is None:
                with Image.open(image_path) as source_image:
                    Image.new("L", source_image.size, 0).save(mask_path, format="PNG")

        manifest = {
            "aoi_id": aoi_id,
            "source_image": str(image_path),
            "candidate_mask": str(candidate_mask_path) if candidate_mask_path else None,
            "status": "completed",
            "evidence": evidence,
            "resumed": False,
        }
        _atomic_json(manifest_path, manifest)
        return manifest
