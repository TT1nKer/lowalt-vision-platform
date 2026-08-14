from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

try:
    from huggingface_hub import hf_hub_download
    from transformers import (
        SegformerConfig,
        SegformerForSemanticSegmentation,
        SegformerImageProcessor,
    )
except Exception as exc:  # pragma: no cover - dependency failure is environment-specific
    raise RuntimeError(
        "parkseg12k_infer requires huggingface_hub and transformers. "
        "Install them in the project environment before running."
    ) from exc


MODEL_REVISION = "74a2d1cb9a71c855d0922bd4659c6fc469d5c648"
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


def load_checkpoint_state(checkpoint_path: Path | None, repo: str) -> dict[str, torch.Tensor]:
    resolved_path = checkpoint_path or Path(
        hf_hub_download(repo, "best_model.ckpt", revision=MODEL_REVISION)
    )
    checkpoint = torch.load(resolved_path, map_location="cpu", weights_only=False)
    state = checkpoint.get("state_dict") if isinstance(checkpoint, dict) else None
    if not isinstance(state, dict):
        raise ValueError(f"checkpoint does not contain state_dict: {resolved_path}")

    model_state = {
        key.removeprefix("model."): value
        for key, value in state.items()
        if key.startswith("model.")
    }
    if not model_state:
        raise ValueError(f"checkpoint has no model.* parameters: {resolved_path}")
    return model_state


def load_processor(config_dir: Path) -> SegformerImageProcessor:
    settings = json.loads((config_dir / "preprocessor_config.json").read_text(encoding="utf-8"))
    return SegformerImageProcessor(
        do_resize=settings["do_resize"],
        size=settings["size"],
        resample=settings["resample"],
        do_normalize=settings["do_normalize"],
        image_mean=settings["image_mean"],
        image_std=settings["image_std"],
    )


def preprocess_image(image: Image.Image, processor: SegformerImageProcessor) -> torch.Tensor:
    return processor(images=image, return_tensors="pt")["pixel_values"]


def build_model(
    state: dict[str, torch.Tensor],
    config_dir: Path,
) -> SegformerForSemanticSegmentation:
    config = SegformerConfig.from_pretrained(config_dir, local_files_only=True)
    config.num_labels = 2
    config.id2label = {0: "background", 1: "parking"}
    config.label2id = {"background": 0, "parking": 1}
    model = SegformerForSemanticSegmentation(config)
    model.load_state_dict(state, strict=True)
    return model.eval()


def apply_decode_head_checkpoint(
    model: SegformerForSemanticSegmentation,
    checkpoint_path: Path,
) -> SegformerForSemanticSegmentation:
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if not isinstance(checkpoint, dict):
        raise ValueError(f"decode-head checkpoint is invalid: {checkpoint_path}")
    if checkpoint.get("base_model_revision") != MODEL_REVISION:
        raise ValueError("decode-head checkpoint base model revision does not match")
    decode_head = checkpoint.get("decode_head")
    if not isinstance(decode_head, dict):
        raise ValueError(f"decode-head checkpoint has no state: {checkpoint_path}")
    model.decode_head.load_state_dict(decode_head, strict=True)
    return model


def resolve_device(requested: str) -> torch.device:
    if requested != "auto":
        return torch.device(requested)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def run_inference(
    model: SegformerForSemanticSegmentation,
    processor: SegformerImageProcessor,
    device: torch.device,
    image_path: Path,
    output_dir: Path,
    threshold: float,
    *,
    save_diagnostics: bool = True,
) -> dict[str, float | int | str]:
    with Image.open(image_path) as source:
        image = source.convert("RGB")
        image_array = np.asarray(image, dtype=np.uint8)
    height, width = image_array.shape[:2]
    pixel_values = preprocess_image(image, processor).to(device)

    started_at = time.perf_counter()
    with torch.inference_mode():
        logits = model(pixel_values=pixel_values).logits
        parking_probability = torch.softmax(logits, dim=1)[:, 1:2]
        resized_probability = F.interpolate(
            parking_probability,
            size=(height, width),
            mode="bilinear",
            align_corners=False,
        )[0, 0]
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    inference_seconds = time.perf_counter() - started_at

    probability = resized_probability.float().cpu().numpy()
    parking_mask = probability >= threshold
    overlay = image_array.copy()
    green = np.zeros_like(overlay)
    green[:, :, 1] = parking_mask.astype(np.uint8) * 255
    overlay = (overlay * 0.72 + green * 0.28).astype(np.uint8)

    mask_path = output_dir / f"{image_path.stem}_mask.png"
    temporary_mask_path = output_dir / f".{image_path.stem}_mask.tmp.png"
    Image.fromarray(parking_mask.astype(np.uint8) * 255).save(
        temporary_mask_path,
        format="PNG",
    )
    temporary_mask_path.replace(mask_path)
    if save_diagnostics:
        Image.fromarray(overlay).save(output_dir / f"{image_path.stem}_pred.png")
        np.save(
            output_dir / f"{image_path.stem}_parking_prob.npy",
            probability.astype(np.float32),
        )

    predicted_pixels = int(parking_mask.sum())
    return {
        "image": str(image_path),
        "width": width,
        "height": height,
        "mean_parking_probability": float(probability.mean()),
        "max_parking_probability": float(probability.max()),
        "predicted_parking_pixels": predicted_pixels,
        "predicted_parking_fraction": predicted_pixels / float(width * height),
        "inference_seconds": inference_seconds,
    }


def select_pending_images(
    image_paths: list[Path],
    output_dir: Path,
    *,
    skip_existing: bool,
) -> tuple[list[Path], int]:
    if not skip_existing:
        return image_paths, 0
    pending = []
    skipped = 0
    for image_path in image_paths:
        mask_path = output_dir / f"{image_path.stem}_mask.png"
        try:
            with Image.open(mask_path) as mask:
                mask.verify()
            skipped += 1
        except (FileNotFoundError, OSError, ValueError):
            pending.append(image_path)
    return pending, skipped


def run_folder(
    image_dir: Path,
    output_dir: Path,
    checkpoint_path: Path | None,
    config_dir: Path,
    repo: str,
    device_name: str,
    threshold: float,
    max_images: int | None,
    head_checkpoint: Path | None = None,
    masks_only: bool = False,
    skip_existing: bool = False,
) -> list[dict[str, float | int | str]]:
    image_paths = sorted(path for path in image_dir.iterdir() if path.suffix.lower() in IMAGE_SUFFIXES)
    if max_images is not None:
        image_paths = image_paths[:max_images]
    if not image_paths:
        raise ValueError(f"no input images in {image_dir}")

    output_dir.mkdir(parents=True, exist_ok=True)
    total_images = len(image_paths)
    image_paths, skipped_existing = select_pending_images(
        image_paths,
        output_dir,
        skip_existing=skip_existing,
    )

    device = resolve_device(device_name)
    if image_paths:
        state = load_checkpoint_state(checkpoint_path, repo)
        processor = load_processor(config_dir)
        model = build_model(state, config_dir)
        if head_checkpoint is not None:
            apply_decode_head_checkpoint(model, head_checkpoint)
        model = model.to(device)

    results = [
        run_inference(
            model,
            processor,
            device,
            image_path,
            output_dir,
            threshold,
            save_diagnostics=not masks_only,
        )
        for image_path in image_paths
    ]
    summary = {
        "model_repo": repo,
        "model_revision": MODEL_REVISION,
        "checkpoint": str(checkpoint_path) if checkpoint_path else "huggingface_cache",
        "head_checkpoint": str(head_checkpoint) if head_checkpoint else None,
        "config_dir": str(config_dir),
        "device": str(device),
        "threshold": threshold,
        "masks_only": masks_only,
        "total_images": total_images,
        "processed_images": len(results),
        "skipped_existing": skipped_existing,
        "results": results,
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return results


def parse_args() -> argparse.Namespace:
    project_root = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(
        description="Run UTEL-UIUC SegFormer-large-parking on local images."
    )
    parser.add_argument("--image-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--checkpoint",
        type=Path,
        help="Local best_model.ckpt; downloads the pinned revision when omitted.",
    )
    parser.add_argument(
        "--config-dir",
        type=Path,
        default=project_root / "models" / "nvidia_mit_b5_config",
    )
    parser.add_argument("--head-checkpoint", type=Path)
    parser.add_argument("--repo", default="UTEL-UIUC/SegFormer-large-parking")
    parser.add_argument("--device", default="auto", help="auto, cuda, mps, or cpu")
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--max-images", type=int)
    parser.add_argument("--masks-only", action="store_true")
    parser.add_argument("--skip-existing", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not 0.0 <= args.threshold <= 1.0:
        raise ValueError("--threshold must be between 0 and 1")
    results = run_folder(
        image_dir=args.image_dir,
        output_dir=args.output_dir,
        checkpoint_path=args.checkpoint,
        config_dir=args.config_dir,
        repo=args.repo,
        device_name=args.device,
        threshold=args.threshold,
        max_images=args.max_images,
        head_checkpoint=args.head_checkpoint,
        masks_only=args.masks_only,
        skip_existing=args.skip_existing,
    )
    for result in results:
        print(json.dumps(result, ensure_ascii=False))


if __name__ == "__main__":
    main()
