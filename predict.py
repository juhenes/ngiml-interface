from __future__ import annotations

import argparse
import json
from pathlib import Path

from src import run_inference


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run NGIML inference on a single RGB image.",
    )
    parser.add_argument(
        "--checkpoint",
        required=True,
        help="Path to a .pt checkpoint file.",
    )
    parser.add_argument(
        "--image",
        required=True,
        help="Path to the RGB image to analyze.",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Directory to save outputs. Defaults to inference_only/outputs/<image-stem>/",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=None,
        help="Binary threshold. Defaults to checkpoint threshold or 0.5.",
    )
    parser.add_argument(
        "--normalization-mode",
        choices=("imagenet", "zero_one"),
        default=None,
        help="Override normalization mode. Defaults to checkpoint metadata or imagenet.",
    )
    parser.add_argument(
        "--resize-max-side",
        type=int,
        default=None,
        help="Optional resize cap for the shorter image side before center-crop preprocessing.",
    )
    parser.add_argument(
        "--crop-size",
        type=int,
        default=None,
        help="Resize-keep-aspect center-crop size. Defaults to the checkpoint input size or 448.",
    )
    parser.add_argument(
        "--device",
        default=None,
        help="Torch device string, for example cpu or cuda:0. Defaults to auto-detect.",
    )
    return parser


def _resolve_output_dir(root: Path, image_path: Path, output_dir: str | None) -> Path:
    if output_dir:
        return Path(output_dir).expanduser().resolve()
    return (root / "outputs" / image_path.stem).resolve()


def main() -> int:
    parser = _build_parser()
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parent
    checkpoint_path = Path(args.checkpoint).expanduser().resolve()
    image_path = Path(args.image).expanduser().resolve()
    output_dir = _resolve_output_dir(project_root, image_path, args.output_dir)

    if not checkpoint_path.exists():
        parser.error(f"Checkpoint not found: {checkpoint_path}")
    if not image_path.exists():
        parser.error(f"Image not found: {image_path}")

    result = run_inference(
        checkpoint_path=checkpoint_path,
        image_path=image_path,
        output_dir=output_dir,
        threshold=args.threshold,
        normalization_mode=args.normalization_mode,
        resize_max_side=args.resize_max_side,
        crop_size=args.crop_size,
        device=args.device,
    )

    payload = {
        "checkpoint_path": result["checkpoint_path"],
        "image_path": result["image_path"],
        "device": result["device"],
        "threshold": result["threshold"],
        "normalization_mode": result["normalization_mode"],
        "inference_mode": result["inference_mode"],
        "crop_size": result["crop_size"],
        "probability_mean": float(result["probability"].mean().item()),
        "probability_max": float(result["probability"].max().item()),
        "predicted_positive_ratio": float(result["binary"].mean().item()),
        "output_dir": result["output_dir"],
        "saved_paths": result["saved_paths"],
    }
    print(json.dumps(payload, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
