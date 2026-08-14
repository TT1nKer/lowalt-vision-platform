from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
import math
from pathlib import Path
import random
import time

import numpy as np
from PIL import Image
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

@dataclass(frozen=True)
class TrainingExample:
    image_path: Path
    label_path: Path
    positive_example: bool


def prepare_label(mask: np.ndarray, *, positive_example: bool) -> np.ndarray:
    if mask.ndim != 2:
        raise ValueError("training mask must be grayscale")
    if positive_example:
        return (mask >= 128).astype(np.int64)
    invalid_values = set(np.unique(mask).tolist()) - {0, 255}
    if invalid_values:
        raise ValueError(f"partial negative label contains unsupported values: {invalid_values}")
    return mask.astype(np.int64)


def freeze_for_head_finetune(model: nn.Module) -> int:
    if not hasattr(model, "segformer") or not hasattr(model, "decode_head"):
        raise ValueError("model must expose segformer and decode_head")
    for parameter in model.parameters():
        parameter.requires_grad = False
    for parameter in model.decode_head.parameters():
        parameter.requires_grad = True
    return sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)


def _load_examples(
    positive_root: Path,
    hard_negative_manifest: Path,
    *,
    max_positive: int,
    hard_negative_fraction: float,
) -> tuple[list[TrainingExample], int, int]:
    positive_images = sorted((positive_root / "rgb").glob("*.jpg"))[:max_positive]
    positives = [
        TrainingExample(
            image_path=image_path,
            label_path=positive_root / "masks" / f"{image_path.stem}.png",
            positive_example=True,
        )
        for image_path in positive_images
    ]
    if not positives or any(not example.label_path.exists() for example in positives):
        raise ValueError("positive RGB/mask pairs are missing")

    hard_document = json.loads(hard_negative_manifest.read_text(encoding="utf-8"))
    hard_negatives = [
        TrainingExample(
            image_path=Path(record["image"]),
            label_path=Path(record["label"]),
            positive_example=False,
        )
        for record in hard_document.get("examples") or []
    ]
    if not hard_negatives or any(
        not example.image_path.exists() or not example.label_path.exists()
        for example in hard_negatives
    ):
        raise ValueError("hard-negative image/label pairs are missing")
    if not 0 < hard_negative_fraction < 1:
        raise ValueError("hard_negative_fraction must be between zero and one")

    target_hard_count = math.ceil(
        len(positives) * hard_negative_fraction / (1.0 - hard_negative_fraction)
    )
    repeats = math.ceil(target_hard_count / len(hard_negatives))
    mixed_hard_negatives = (hard_negatives * repeats)[:target_hard_count]
    return positives + mixed_hard_negatives, len(positives), len(mixed_hard_negatives)


class ParkingFineTuneDataset(Dataset):
    def __init__(self, examples: list[TrainingExample], processor: object) -> None:
        self.examples = examples
        self.processor = processor
        size = processor.size
        self.target_size = (int(size["width"]), int(size["height"]))

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        example = self.examples[index]
        with Image.open(example.image_path) as source:
            image = source.convert("RGB")
            pixel_values = self.processor(images=image, return_tensors="pt")["pixel_values"][0]
        with Image.open(example.label_path) as source:
            resized = source.convert("L").resize(self.target_size, Image.Resampling.NEAREST)
            label = prepare_label(np.asarray(resized), positive_example=example.positive_example)
        return {
            "pixel_values": pixel_values,
            "labels": torch.from_numpy(label),
        }


def _atomic_torch_save(document: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_suffix(path.suffix + ".tmp")
    try:
        torch.save(document, temporary_path)
        temporary_path.replace(path)
    finally:
        temporary_path.unlink(missing_ok=True)


def run_finetune(
    *,
    positive_root: Path,
    hard_negative_manifest: Path,
    base_checkpoint: Path,
    config_dir: Path,
    output_directory: Path,
    device_name: str,
    max_positive: int = 512,
    hard_negative_fraction: float = 0.25,
    max_steps: int = 300,
    learning_rate: float = 5e-5,
    seed: int = 20260809,
) -> dict:
    import parkseg12k_infer as inference

    if max_positive <= 0 or max_steps <= 0 or learning_rate <= 0:
        raise ValueError("sample, step and learning-rate values must be positive")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    examples, positive_count, hard_negative_count = _load_examples(
        positive_root,
        hard_negative_manifest,
        max_positive=max_positive,
        hard_negative_fraction=hard_negative_fraction,
    )
    processor = inference.load_processor(config_dir)
    dataset = ParkingFineTuneDataset(examples, processor)
    generator = torch.Generator().manual_seed(seed)
    loader = DataLoader(dataset, batch_size=1, shuffle=True, num_workers=0, generator=generator)

    device = inference.resolve_device(device_name)
    model = inference.build_model(
        inference.load_checkpoint_state(base_checkpoint, ""),
        config_dir,
    )
    model.config.semantic_loss_ignore_index = 255
    trainable_parameters = freeze_for_head_finetune(model)
    model.to(device).train()
    optimizer = torch.optim.AdamW(
        (parameter for parameter in model.parameters() if parameter.requires_grad),
        lr=learning_rate,
        weight_decay=0.01,
    )
    amp_enabled = device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=amp_enabled)

    losses = []
    iterator = iter(loader)
    started_at = time.perf_counter()
    for _step in range(max_steps):
        try:
            batch = next(iterator)
        except StopIteration:
            iterator = iter(loader)
            batch = next(iterator)
        pixel_values = batch["pixel_values"].to(device)
        labels = batch["labels"].to(device)
        optimizer.zero_grad(set_to_none=True)
        with torch.amp.autocast("cuda", dtype=torch.float16, enabled=amp_enabled):
            loss = model(pixel_values=pixel_values, labels=labels).loss
        if not torch.isfinite(loss):
            raise RuntimeError(f"non-finite training loss at step {len(losses) + 1}")
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.decode_head.parameters(), max_norm=1.0)
        scaler.step(optimizer)
        scaler.update()
        losses.append(float(loss.detach().cpu()))

    elapsed_seconds = time.perf_counter() - started_at
    output_directory.mkdir(parents=True, exist_ok=True)
    head_checkpoint = output_directory / "decode_head.pt"
    _atomic_torch_save({
        "base_model_revision": inference.MODEL_REVISION,
        "decode_head": {
            key: value.detach().cpu()
            for key, value in model.decode_head.state_dict().items()
        },
    }, head_checkpoint)
    summary = {
        "base_checkpoint": str(base_checkpoint),
        "base_model_revision": inference.MODEL_REVISION,
        "positive_examples": positive_count,
        "hard_negative_examples_after_repeat": hard_negative_count,
        "hard_negative_fraction": hard_negative_fraction,
        "max_steps": max_steps,
        "learning_rate": learning_rate,
        "trainable_parameters": trainable_parameters,
        "device": str(device),
        "first_20_loss_mean": float(np.mean(losses[:20])),
        "last_20_loss_mean": float(np.mean(losses[-20:])),
        "elapsed_seconds": elapsed_seconds,
        "head_checkpoint": str(head_checkpoint),
    }
    (output_directory / "training_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return summary


def main() -> None:
    project_root = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description="Fine-tune only the ParkSeg12k SegFormer decode head")
    parser.add_argument("--positive-root", type=Path, required=True)
    parser.add_argument("--hard-negative-manifest", type=Path, required=True)
    parser.add_argument("--base-checkpoint", type=Path, required=True)
    parser.add_argument("--config-dir", type=Path, default=project_root / "models" / "nvidia_mit_b5_config")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--max-positive", type=int, default=512)
    parser.add_argument("--hard-negative-fraction", type=float, default=0.25)
    parser.add_argument("--max-steps", type=int, default=300)
    parser.add_argument("--learning-rate", type=float, default=5e-5)
    args = parser.parse_args()
    summary = run_finetune(
        positive_root=args.positive_root,
        hard_negative_manifest=args.hard_negative_manifest,
        base_checkpoint=args.base_checkpoint,
        config_dir=args.config_dir,
        output_directory=args.output_dir,
        device_name=args.device,
        max_positive=args.max_positive,
        hard_negative_fraction=args.hard_negative_fraction,
        max_steps=args.max_steps,
        learning_rate=args.learning_rate,
    )
    print(json.dumps(summary, ensure_ascii=False))


if __name__ == "__main__":
    main()
