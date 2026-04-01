from __future__ import annotations

import json
import logging
import re
from copy import deepcopy
from pathlib import Path
from typing import Any, Sequence

import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F

from .model.backbones.efficientnet_backbone import EfficientNetBackboneConfig
from .model.backbones.residual_noise_branch import ResidualNoiseConfig
from .model.backbones.swin_backbone import SwinBackboneConfig
from .model.feature_fusion import FeatureFusionConfig
from .model.hybrid_ngiml import (
    HybridNGIML,
    HybridNGIMLConfig,
    HybridNGIMLOptimizerConfig,
    OptimizerGroupConfig,
)
from .model.unet_decoder import UNetDecoderConfig

_LOG = logging.getLogger(__name__)

DEFAULT_HF_REPO_ID = "juhenes/ngiml"
AVAILABLE_HF_CHECKPOINTS = (
    "casia-effnet.pt",
    "casia-effnet+noise.pt",
    "casia-effnet+swin.pt",
    "casia-full.pt",
    "casia-swin.pt",
    "casia-swin+noise.pt",
)

def _require_matplotlib():
    try:
        import matplotlib.pyplot as plt
    except Exception as exc:
        raise ImportError('matplotlib is required for plotting. Install it with pip install matplotlib.') from exc
    return plt


def build_default_model_config() -> HybridNGIMLConfig:
    return HybridNGIMLConfig(
        efficientnet=EfficientNetBackboneConfig(pretrained=True),
        swin=SwinBackboneConfig(
            model_name="swin_tiny_patch4_window7_224",
            pretrained=True,
            input_size=448,
        ),
        residual=ResidualNoiseConfig(num_kernels=3, base_channels=32, num_stages=4),
        fusion=FeatureFusionConfig(fusion_channels=(64, 128, 192, 256)),
        decoder=UNetDecoderConfig(decoder_channels=None, out_channels=1, per_stage_heads=True),
        optimizer=HybridNGIMLOptimizerConfig(
            efficientnet=OptimizerGroupConfig(lr=1e-5, weight_decay=1.5e-4),
            swin=OptimizerGroupConfig(lr=5e-6, weight_decay=1e-4),
            residual=OptimizerGroupConfig(lr=2.5e-4, weight_decay=2e-4),
            fusion=OptimizerGroupConfig(lr=1.2e-4, weight_decay=2e-4),
            decoder=OptimizerGroupConfig(lr=1.8e-4, weight_decay=2e-4),
        ),
        use_low_level=True,
        use_context=True,
        use_residual=True,
    )


def _coerce_optimizer_config(value: Any) -> HybridNGIMLOptimizerConfig:
    default_opt = build_default_model_config().optimizer
    if value is None:
        return default_opt
    if isinstance(value, HybridNGIMLOptimizerConfig):
        return value
    if not isinstance(value, dict):
        raise TypeError("Optimizer config must be HybridNGIMLOptimizerConfig or dict")

    def _coerce_group(group_value: Any, default_group: OptimizerGroupConfig) -> OptimizerGroupConfig:
        if isinstance(group_value, OptimizerGroupConfig):
            return group_value
        if group_value is None:
            return default_group
        if isinstance(group_value, dict):
            return OptimizerGroupConfig(**group_value)
        raise TypeError("Optimizer group config must be OptimizerGroupConfig or dict")

    betas_raw = value.get("betas", default_opt.betas)
    if isinstance(betas_raw, list):
        betas = tuple(float(v) for v in betas_raw)
    else:
        betas = tuple(betas_raw)

    return HybridNGIMLOptimizerConfig(
        efficientnet=_coerce_group(value.get("efficientnet"), default_opt.efficientnet),
        swin=_coerce_group(value.get("swin"), default_opt.swin),
        residual=_coerce_group(value.get("residual"), default_opt.residual),
        fusion=_coerce_group(value.get("fusion"), default_opt.fusion),
        decoder=_coerce_group(value.get("decoder"), default_opt.decoder),
        betas=betas,
        eps=float(value.get("eps", default_opt.eps)),
        freeze_backbone_epochs=int(value.get("freeze_backbone_epochs", default_opt.freeze_backbone_epochs)),
    )


def coerce_model_config(value: Any) -> HybridNGIMLConfig:
    default_model = build_default_model_config()
    if value is None:
        return default_model
    if isinstance(value, HybridNGIMLConfig):
        return value
    if not isinstance(value, dict):
        raise TypeError("Model config must be HybridNGIMLConfig or dict")

    efficientnet = value.get("efficientnet", default_model.efficientnet)
    swin = value.get("swin", default_model.swin)
    residual = value.get("residual", default_model.residual)
    fusion = value.get("fusion", default_model.fusion)
    decoder = value.get("decoder", default_model.decoder)
    optimizer = value.get("optimizer", default_model.optimizer)

    return HybridNGIMLConfig(
        efficientnet=efficientnet if isinstance(efficientnet, EfficientNetBackboneConfig) else EfficientNetBackboneConfig(**efficientnet),
        swin=swin if isinstance(swin, SwinBackboneConfig) else SwinBackboneConfig(**swin),
        residual=residual if isinstance(residual, ResidualNoiseConfig) else ResidualNoiseConfig(**residual),
        fusion=fusion if isinstance(fusion, FeatureFusionConfig) else FeatureFusionConfig(**fusion),
        decoder=decoder if isinstance(decoder, UNetDecoderConfig) else UNetDecoderConfig(**decoder),
        optimizer=_coerce_optimizer_config(optimizer),
        use_low_level=bool(value.get("use_low_level", default_model.use_low_level)),
        use_context=bool(value.get("use_context", default_model.use_context)),
        use_residual=bool(value.get("use_residual", default_model.use_residual)),
        enable_residual_attention=bool(value.get("enable_residual_attention", default_model.enable_residual_attention)),
        enable_low_level_residual_attention=bool(
            value.get("enable_low_level_residual_attention", default_model.enable_low_level_residual_attention)
        ),
        enable_context_residual_attention=bool(
            value.get("enable_context_residual_attention", default_model.enable_context_residual_attention)
        ),
        residual_attention_init_scale=float(
            value.get("residual_attention_init_scale", default_model.residual_attention_init_scale)
        ),
        gradient_checkpointing=bool(value.get("gradient_checkpointing", default_model.gradient_checkpointing)),
        flash_attention=bool(value.get("flash_attention", default_model.flash_attention)),
        xformers=bool(value.get("xformers", default_model.xformers)),
    )


def _infer_fusion_channels_from_state_dict(model_state: dict[str, Any]) -> tuple[int, ...] | None:
    stage_channels: dict[int, int] = {}
    pattern = re.compile(r"^fusion\.stages\.(\d+)\.projections\.[^.]+\.weight$")
    for key, tensor in model_state.items():
        match = pattern.match(key)
        if not match or not isinstance(tensor, torch.Tensor):
            continue
        stage_channels[int(match.group(1))] = int(tensor.shape[0])
    if not stage_channels:
        return None
    return tuple(stage_channels[idx] for idx in sorted(stage_channels))


def build_model_config_from_checkpoint(checkpoint: dict[str, Any]) -> tuple[HybridNGIMLConfig, str]:
    model_cfg = build_default_model_config()
    train_config = checkpoint.get("train_config") if isinstance(checkpoint, dict) else None
    model_config = train_config.get("model_config") if isinstance(train_config, dict) else None
    if isinstance(model_config, dict):
        return coerce_model_config(model_config), "train_config.model_config"

    inferred_channels = _infer_fusion_channels_from_state_dict(checkpoint.get("model_state", {}))
    if inferred_channels:
        model_cfg.fusion.fusion_channels = inferred_channels
        return model_cfg, "state_dict.inferred_fusion_channels"

    return model_cfg, "defaults"


def disable_pretrained_backbones(model_cfg: HybridNGIMLConfig) -> HybridNGIMLConfig:
    cfg_out = deepcopy(model_cfg)
    try:
        cfg_out.efficientnet.pretrained = False
    except Exception:
        pass
    try:
        cfg_out.swin.pretrained = False
    except Exception:
        pass
    return cfg_out


def _normalize_profile_input_size(value: object) -> int | None:
    if isinstance(value, int):
        return int(value)
    if isinstance(value, (tuple, list)) and value:
        try:
            if len(value) >= 2:
                return int(max(value[-2], value[-1]))
            return int(value[0])
        except Exception:
            return None
    return None


def _resolve_checkpoint_profile_input_size(train_config: dict[str, Any], model_cfg: HybridNGIMLConfig) -> tuple[int, str]:
    train_value = _normalize_profile_input_size(train_config.get("input_size"))
    if train_value is not None:
        return train_value, "train_config.input_size"

    candidates = [
        getattr(getattr(model_cfg, "swin", None), "input_size", None),
        getattr(getattr(model_cfg, "efficientnet", None), "input_size", None),
    ]
    for candidate in candidates:
        resolved = _normalize_profile_input_size(candidate)
        if resolved is not None:
            return resolved, "model_config"

    return 448, "default"


def _dtype_name(value: torch.dtype | None) -> str:
    return str(value).replace("torch.", "") if isinstance(value, torch.dtype) else "none"


def _resolve_checkpoint_autocast_dtype(
    train_config: dict[str, Any],
    device: torch.device,
) -> tuple[torch.dtype | None, str]:
    precision_raw = str(train_config.get("precision", "") or "").strip().lower()
    amp_enabled = bool(train_config.get("amp", False))
    preferred: torch.dtype | None = None
    source = "checkpoint_precision"

    if precision_raw in {"bf16", "bfloat16"}:
        preferred = torch.bfloat16
    elif precision_raw in {"fp16", "float16", "half"}:
        preferred = torch.float16
    elif precision_raw in {"fp32", "float32", "32", "full", "none", "off", "disabled"}:
        preferred = None
    elif amp_enabled:
        preferred = torch.bfloat16 if (device.type == "cuda" and torch.cuda.is_bf16_supported()) else torch.float16
        source = "checkpoint_amp_fallback"

    if device.type != "cuda":
        return None, f"{source}:cpu"
    if preferred is torch.bfloat16 and not torch.cuda.is_bf16_supported():
        return torch.float16, f"{source}:bf16_unsupported_fallback_fp16"
    return preferred, source


def get_inference_autocast_dtype(model: HybridNGIML, device: torch.device) -> torch.dtype | None:
    dtype = getattr(model, "default_autocast_dtype", None)
    if not isinstance(dtype, torch.dtype):
        return None
    if device.type != "cuda":
        return None
    if dtype is torch.bfloat16 and not torch.cuda.is_bf16_supported():
        return torch.float16
    if dtype in {torch.float16, torch.bfloat16}:
        return dtype
    return None


def _load_state_dict_with_fallback(
    model: HybridNGIML,
    model_state: dict[str, Any],
) -> tuple[list[str], list[str], int]:
    try:
        missing, unexpected = model.load_state_dict(model_state, strict=False)
        return list(missing), list(unexpected), 0
    except RuntimeError:
        current_state = model.state_dict()
        compatible_state = {
            key: value
            for key, value in model_state.items()
            if key in current_state and hasattr(value, "shape") and current_state[key].shape == value.shape
        }
        skipped = int(len(model_state) - len(compatible_state))
        missing, unexpected = model.load_state_dict(compatible_state, strict=False)
        return list(missing), list(unexpected), skipped


def resolve_threshold_for_checkpoint(
    checkpoint_path: str | Path,
    checkpoint_epoch: int | None = None,
    fallback: float = 0.5,
) -> tuple[float, str]:
    checkpoint_path = Path(checkpoint_path)
    candidate_files = [
        checkpoint_path.parent / "best_threshold.json",
        checkpoint_path.parent.parent / "best_threshold.json",
    ]
    for candidate in candidate_files:
        if not candidate.exists():
            continue
        try:
            payload = json.loads(candidate.read_text(encoding="utf-8"))
            payload_ckpt = str(payload.get("checkpoint_path", ""))
            if payload_ckpt and Path(payload_ckpt).name == checkpoint_path.name:
                return float(payload.get("threshold", fallback)), f"{candidate.name}:matching_checkpoint"
            if checkpoint_epoch is not None and int(payload.get("epoch", -1)) == int(checkpoint_epoch):
                return float(payload.get("threshold", fallback)), f"{candidate.name}:matching_epoch"
        except Exception:
            continue

    metrics_candidates = [
        checkpoint_path.parent / "checkpoint_metrics.json",
        checkpoint_path.parent.parent / "checkpoint_metrics.json",
    ]
    for metrics_path in metrics_candidates:
        if not metrics_path.exists():
            continue
        try:
            payload = json.loads(metrics_path.read_text(encoding="utf-8"))
            if not isinstance(payload, list):
                continue
            by_path = next(
                (
                    record
                    for record in payload
                    if isinstance(record, dict)
                    and str(record.get("checkpoint_path", "")).endswith(checkpoint_path.name)
                    and record.get("val_threshold") is not None
                ),
                None,
            )
            if by_path is not None:
                return float(by_path["val_threshold"]), f"{metrics_path.name}:matching_path"
            if checkpoint_epoch is not None:
                by_epoch = next(
                    (
                        record
                        for record in reversed(payload)
                        if isinstance(record, dict)
                        and int(record.get("epoch", -1)) == int(checkpoint_epoch)
                        and record.get("val_threshold") is not None
                    ),
                    None,
                )
                if by_epoch is not None:
                    return float(by_epoch["val_threshold"]), f"{metrics_path.name}:matching_epoch"
        except Exception:
            continue

    for candidate in candidate_files:
        if not candidate.exists():
            continue
        try:
            payload = json.loads(candidate.read_text(encoding="utf-8"))
            return float(payload.get("threshold", fallback)), f"{candidate.name}:fallback"
        except Exception:
            continue

    return float(fallback), "fallback"


def resolve_normalization_mode_for_inference(
    manual_mode: str | None = None,
    checkpoint_train_config: dict[str, Any] | None = None,
    default_mode: str = "imagenet",
) -> str:
    if isinstance(manual_mode, str) and manual_mode.strip():
        mode = manual_mode.strip().lower()
        if mode in {"imagenet", "zero_one"}:
            return mode
        raise ValueError(f"Unsupported normalization mode: {manual_mode!r}")

    if isinstance(checkpoint_train_config, dict):
        mode = str(checkpoint_train_config.get("normalization_mode", "") or "").strip().lower()
        if mode in {"imagenet", "zero_one"}:
            return mode

    fallback = str(default_mode).strip().lower()
    return fallback if fallback in {"imagenet", "zero_one"} else "imagenet"


def load_model_from_checkpoint(
    checkpoint_path: str | Path,
    device: torch.device | None = None,
) -> tuple[HybridNGIML, torch.device, dict[str, Any]]:
    checkpoint_path = Path(checkpoint_path)
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    checkpoint_epoch = int(checkpoint.get("epoch", -1))
    train_config = checkpoint.get("train_config") or {}
    model_cfg, config_source = build_model_config_from_checkpoint(checkpoint)
    model_cfg = disable_pretrained_backbones(model_cfg)
    model = HybridNGIML(model_cfg)

    missing, unexpected, skipped_mismatched = _load_state_dict_with_fallback(model, checkpoint["model_state"])
    model = model.to(device)
    model.eval()

    resolved_threshold, threshold_source = resolve_threshold_for_checkpoint(
        checkpoint_path,
        checkpoint_epoch=checkpoint_epoch,
        fallback=0.5,
    )
    profile_input_size, profile_input_size_source = _resolve_checkpoint_profile_input_size(train_config, model_cfg)
    autocast_dtype, autocast_source = _resolve_checkpoint_autocast_dtype(train_config, device)
    precision_raw = str(train_config.get("precision", "") or "").strip().lower() or "unset"

    info = {
        "epoch": checkpoint_epoch,
        "missing_keys": len(missing),
        "unexpected_keys": len(unexpected),
        "skipped_mismatched_keys": int(skipped_mismatched),
        "config_source": str(config_source),
        "fusion_channels": tuple(int(value) for value in model.cfg.fusion.fusion_channels),
        "default_threshold": float(resolved_threshold),
        "threshold_source": str(threshold_source),
        "input_size": int(profile_input_size),
        "input_size_source": str(profile_input_size_source),
        "resize_max_side": int(train_config.get("resize_max_side", 0) or 0),
        "runtime_precision": precision_raw,
        "inference_autocast_dtype": _dtype_name(autocast_dtype),
        "inference_autocast_source": autocast_source,
        "train_config": train_config,
    }
    setattr(model, "default_threshold", float(info["default_threshold"]))
    setattr(model, "default_runtime_precision", precision_raw)
    setattr(model, "default_autocast_dtype", autocast_dtype)
    return model, device, info


def resolve_huggingface_checkpoint_filename(checkpoint_name: str) -> str:
    normalized = str(checkpoint_name).strip()
    if not normalized:
        raise ValueError("checkpoint_name must be a non-empty string")
    if normalized.endswith(".pt"):
        return normalized
    return f"{normalized}.pt"


def download_checkpoint_from_huggingface(
    checkpoint_name: str,
    *,
    repo_id: str = DEFAULT_HF_REPO_ID,
    cache_dir: str | Path | None = None,
) -> Path:
    try:
        from huggingface_hub import hf_hub_download
    except Exception as exc:
        raise ImportError(
            "huggingface_hub is required for downloading checkpoints. Install it with pip install huggingface_hub."
        ) from exc

    filename = resolve_huggingface_checkpoint_filename(checkpoint_name)
    if filename not in AVAILABLE_HF_CHECKPOINTS:
        available = ", ".join(AVAILABLE_HF_CHECKPOINTS)
        raise ValueError(f"Unsupported checkpoint {filename!r}. Expected one of: {available}")

    target_dir = Path(cache_dir) if cache_dir is not None else Path.cwd() / "checkpoints_cache"
    target_dir.mkdir(parents=True, exist_ok=True)
    downloaded = hf_hub_download(
        repo_id=repo_id,
        filename=filename,
        local_dir=str(target_dir),
    )
    return Path(downloaded).resolve()


def load_rgb_image(image_path: str | Path) -> torch.Tensor:
    image = Image.open(image_path).convert("RGB")
    image_np = np.asarray(image, dtype=np.float32) / 255.0
    return torch.from_numpy(image_np).permute(2, 0, 1).contiguous()


def _normalize(image: torch.Tensor, mode: str) -> torch.Tensor:
    if mode == "zero_one":
        return image
    if mode == "imagenet":
        mean = torch.tensor([0.485, 0.456, 0.406], device=image.device, dtype=image.dtype).view(3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225], device=image.device, dtype=image.dtype).view(3, 1, 1)
        return (image - mean) / std
    return image


def normalize_image_for_inference(image: torch.Tensor, normalization_mode: str = "imagenet") -> torch.Tensor:
    image = image.float().clamp(0.0, 1.0)
    return _normalize(image, str(normalization_mode).strip().lower())


def compute_residual_noise(image: torch.Tensor) -> torch.Tensor:
    if image.ndim != 3 or image.shape[0] != 3:
        raise ValueError(f"Expected CHW image with 3 channels, got shape {tuple(image.shape)}")
    x = image.float().unsqueeze(0)
    x_pad = F.pad(x, (2, 2, 2, 2), mode="reflect")
    blurred = F.avg_pool2d(x_pad, kernel_size=5, stride=1)
    residual = (x - blurred).squeeze(0)
    mean = residual.mean(dim=(1, 2), keepdim=True)
    std = residual.std(dim=(1, 2), keepdim=True).clamp_min(1e-6)
    return ((residual - mean) / std).to(torch.float32)


def _model_uses_residual_noise(model: HybridNGIML) -> bool:
    cfg = getattr(model, "cfg", None)
    return bool(getattr(cfg, "use_residual", True)) and (getattr(model, "noise", None) is not None)


def _select_output_head(outputs: Sequence[torch.Tensor]) -> torch.Tensor:
    if not outputs:
        raise ValueError("Model returned empty predictions list")
    return outputs[0]


def resize_image_for_inference(
    image: torch.Tensor,
    resize_max_side: int | None = None,
) -> tuple[torch.Tensor, tuple[int, int]]:
    cap = int(resize_max_side or 0)
    original_hw = (int(image.shape[-2]), int(image.shape[-1]))
    if cap <= 0:
        return image, original_hw

    h, w = original_hw
    short_side = min(h, w)
    if short_side <= 0 or short_side <= cap:
        return image, original_hw

    scale = float(cap) / float(short_side)
    new_h = max(1, int(round(h * scale)))
    new_w = max(1, int(round(w * scale)))
    resized = F.interpolate(
        image.unsqueeze(0),
        size=(new_h, new_w),
        mode="bilinear",
        align_corners=False,
    )[0]
    return resized, original_hw


def _resize_probability_to_original(probability: torch.Tensor, original_hw: tuple[int, int]) -> torch.Tensor:
    if tuple(probability.shape[-2:]) == tuple(original_hw):
        return probability
    resized = F.interpolate(
        probability.unsqueeze(0).unsqueeze(0),
        size=original_hw,
        mode="bilinear",
        align_corners=False,
    )[0, 0]
    return resized.clamp(0.0, 1.0)


def predict_probability_map(
    model: HybridNGIML,
    image: torch.Tensor,
    device: torch.device,
    normalization_mode: str = "imagenet",
) -> torch.Tensor:
    normalized = normalize_image_for_inference(image, normalization_mode=normalization_mode)
    x = normalized.unsqueeze(0).to(device)
    residual_noise = None
    if _model_uses_residual_noise(model):
        residual_noise = compute_residual_noise(image).unsqueeze(0).to(device)

    autocast_dtype = get_inference_autocast_dtype(model, device)
    use_amp = device.type == "cuda" and autocast_dtype is not None
    with torch.no_grad():
        with torch.autocast(device_type="cuda", dtype=autocast_dtype or torch.float16, enabled=use_amp):
            outputs = model(x, target_size=image.shape[-2:], residual_noise=residual_noise)
            logits = _select_output_head(outputs)
            probability = torch.sigmoid(logits)[0, 0].detach().cpu()
    return probability


def resolve_center_crop_size(
    checkpoint_info: dict[str, Any] | None,
    crop_size: int | None = None,
) -> int:
    default_crop_size = 448
    if isinstance(checkpoint_info, dict):
        default_crop_size = int(checkpoint_info.get("input_size") or default_crop_size)
    return max(32, int(crop_size or default_crop_size))


def resize_keep_aspect_center_crop(
    image: torch.Tensor,
    crop_size: int,
) -> tuple[torch.Tensor, dict[str, int]]:
    h, w = int(image.shape[-2]), int(image.shape[-1])
    target = max(1, int(crop_size))
    scale = max(float(target) / float(max(h, 1)), float(target) / float(max(w, 1)))
    resized_h = max(target, int(round(h * scale)))
    resized_w = max(target, int(round(w * scale)))

    resized = F.interpolate(
        image.unsqueeze(0),
        size=(resized_h, resized_w),
        mode="bilinear",
        align_corners=False,
    )[0]

    top = max(0, (resized_h - target) // 2)
    left = max(0, (resized_w - target) // 2)
    cropped = resized[:, top : top + target, left : left + target]
    metadata = {
        "original_height": h,
        "original_width": w,
        "resized_height": resized_h,
        "resized_width": resized_w,
        "crop_top": top,
        "crop_left": left,
        "crop_size": target,
    }
    return cropped.contiguous(), metadata


def restore_probability_from_center_crop(
    probability: torch.Tensor,
    transform: dict[str, int],
) -> torch.Tensor:
    resized_h = int(transform["resized_height"])
    resized_w = int(transform["resized_width"])
    top = int(transform["crop_top"])
    left = int(transform["crop_left"])
    original_h = int(transform["original_height"])
    original_w = int(transform["original_width"])

    canvas = torch.zeros((resized_h, resized_w), dtype=probability.dtype)
    canvas[top : top + probability.shape[-2], left : left + probability.shape[-1]] = probability
    return _resize_probability_to_original(canvas, (original_h, original_w))


def overlay_prediction_on_image(image: torch.Tensor, probability: torch.Tensor) -> np.ndarray:
    image_np = image.detach().cpu().clamp(0.0, 1.0).permute(1, 2, 0).numpy()
    prob_np = probability.detach().cpu().clamp(0.0, 1.0).numpy()
    alpha = 0.45 * prob_np[..., None]
    red = np.array([1.0, 0.0, 0.0], dtype=np.float32)
    return np.clip(image_np * (1.0 - alpha) + red * alpha, 0.0, 1.0)


def run_inference(
    checkpoint_path: str | Path,
    image_path: str | Path,
    *,
    output_dir: str | Path | None = None,
    threshold: float | None = None,
    normalization_mode: str | None = None,
    resize_max_side: int | None = None,
    crop_size: int | None = None,
    device: str | torch.device | None = None,
) -> dict[str, Any]:
    resolved_device = torch.device(device) if device is not None else None
    model, runtime_device, checkpoint_info = load_model_from_checkpoint(checkpoint_path, device=resolved_device)

    original_image = load_rgb_image(image_path)
    if resize_max_side is not None:
        working_image, _ = resize_image_for_inference(
            original_image,
            resize_max_side=resize_max_side,
        )
    else:
        working_image = original_image
    resolved_normalization = resolve_normalization_mode_for_inference(
        manual_mode=normalization_mode,
        checkpoint_train_config=checkpoint_info.get("train_config"),
        default_mode="imagenet",
    )
    resolved_threshold = float(
        checkpoint_info["default_threshold"] if threshold is None else threshold
    )
    resolved_crop_size = resolve_center_crop_size(
        checkpoint_info,
        crop_size=crop_size,
    )
    prepared_image, crop_transform = resize_keep_aspect_center_crop(
        working_image,
        crop_size=resolved_crop_size,
    )

    probability = predict_probability_map(
        model,
        prepared_image,
        runtime_device,
        normalization_mode=resolved_normalization,
    ).clamp(0.0, 1.0)
    preview_probability = probability.clone()
    preview_binary = (preview_probability >= resolved_threshold).float()
    preview_overlay = overlay_prediction_on_image(prepared_image, preview_probability)
    probability = restore_probability_from_center_crop(probability, crop_transform)
    binary = (probability >= resolved_threshold).float()
    overlay = overlay_prediction_on_image(original_image, probability)

    result = {
        "checkpoint_path": str(Path(checkpoint_path).resolve()),
        "image_path": str(Path(image_path).resolve()),
        "device": str(runtime_device),
        "threshold": resolved_threshold,
        "normalization_mode": resolved_normalization,
        "original_image": original_image,
        "working_image": prepared_image,
        "preview_probability": preview_probability,
        "preview_binary": preview_binary,
        "preview_overlay": preview_overlay,
        "probability": probability,
        "binary": binary,
        "overlay": overlay,
        "checkpoint_info": checkpoint_info,
        "inference_mode": "resize_keep_aspect_center_crop",
        "crop_size": resolved_crop_size,
        "output_dir": str(Path(output_dir).resolve()) if output_dir is not None else None,
        "saved_paths": None,
    }

    if output_dir is not None:
        result["saved_paths"] = save_result(result, output_dir)
    return result


def run_huggingface_inference(
    checkpoint_name: str,
    image_path: str | Path,
    *,
    output_dir: str | Path | None = None,
    threshold: float | None = None,
    normalization_mode: str | None = None,
    resize_max_side: int | None = None,
    crop_size: int | None = None,
    device: str | torch.device | None = None,
    repo_id: str = DEFAULT_HF_REPO_ID,
    cache_dir: str | Path | None = None,
) -> dict[str, Any]:
    checkpoint_path = download_checkpoint_from_huggingface(
        checkpoint_name,
        repo_id=repo_id,
        cache_dir=cache_dir,
    )
    return run_inference(
        checkpoint_path=checkpoint_path,
        image_path=image_path,
        output_dir=output_dir,
        threshold=threshold,
        normalization_mode=normalization_mode,
        resize_max_side=resize_max_side,
        crop_size=crop_size,
        device=device,
    )


def run_inference_with_model(
    model: HybridNGIML,
    runtime_device: torch.device,
    checkpoint_info: dict[str, Any],
    *,
    checkpoint_path: str | Path,
    image_path: str | Path,
    output_dir: str | Path | None = None,
    threshold: float | None = None,
    normalization_mode: str | None = None,
    resize_max_side: int | None = None,
    crop_size: int | None = None,
) -> dict[str, Any]:
    original_image = load_rgb_image(image_path)
    if resize_max_side is not None:
        working_image, _ = resize_image_for_inference(
            original_image,
            resize_max_side=resize_max_side,
        )
    else:
        working_image = original_image
    resolved_normalization = resolve_normalization_mode_for_inference(
        manual_mode=normalization_mode,
        checkpoint_train_config=checkpoint_info.get("train_config"),
        default_mode="imagenet",
    )
    resolved_threshold = float(
        checkpoint_info["default_threshold"] if threshold is None else threshold
    )
    resolved_crop_size = resolve_center_crop_size(
        checkpoint_info,
        crop_size=crop_size,
    )
    prepared_image, crop_transform = resize_keep_aspect_center_crop(
        working_image,
        crop_size=resolved_crop_size,
    )

    probability = predict_probability_map(
        model,
        prepared_image,
        runtime_device,
        normalization_mode=resolved_normalization,
    ).clamp(0.0, 1.0)
    preview_probability = probability.clone()
    preview_binary = (preview_probability >= resolved_threshold).float()
    preview_overlay = overlay_prediction_on_image(prepared_image, preview_probability)
    probability = restore_probability_from_center_crop(probability, crop_transform)
    binary = (probability >= resolved_threshold).float()
    overlay = overlay_prediction_on_image(original_image, probability)

    result = {
        "checkpoint_path": str(Path(checkpoint_path).resolve()),
        "image_path": str(Path(image_path).resolve()),
        "device": str(runtime_device),
        "threshold": resolved_threshold,
        "normalization_mode": resolved_normalization,
        "original_image": original_image,
        "working_image": prepared_image,
        "preview_probability": preview_probability,
        "preview_binary": preview_binary,
        "preview_overlay": preview_overlay,
        "probability": probability,
        "binary": binary,
        "overlay": overlay,
        "checkpoint_info": checkpoint_info,
        "inference_mode": "resize_keep_aspect_center_crop",
        "crop_size": resolved_crop_size,
        "output_dir": str(Path(output_dir).resolve()) if output_dir is not None else None,
        "saved_paths": None,
    }

    if output_dir is not None:
        result["saved_paths"] = save_result(result, output_dir)
    return result
def plot_result(result: dict[str, Any]) -> tuple[Any, Any]:
    plt = _require_matplotlib()
    image_np = result["original_image"].detach().cpu().permute(1, 2, 0).numpy()
    prob_np = result["probability"].detach().cpu().numpy()
    bin_np = result["binary"].detach().cpu().numpy()
    overlay_np = result["overlay"]

    fig, axes = plt.subplots(1, 4, figsize=(18, 5))
    axes[0].imshow(image_np)
    axes[0].set_title("Image")
    axes[0].axis("off")

    axes[1].imshow(prob_np, cmap="magma", vmin=0.0, vmax=1.0)
    axes[1].set_title("Probability")
    axes[1].axis("off")

    axes[2].imshow(bin_np, cmap="gray", vmin=0.0, vmax=1.0)
    axes[2].set_title(f"Binary (t={result['threshold']:.2f})")
    axes[2].axis("off")

    axes[3].imshow(overlay_np)
    axes[3].set_title("Overlay")
    axes[3].axis("off")

    fig.tight_layout()
    return fig, axes


def _save_image(path: Path, array: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(array).save(path)


def save_result(result: dict[str, Any], output_dir: str | Path) -> dict[str, str]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    image_np = (result["original_image"].detach().cpu().clamp(0.0, 1.0).permute(1, 2, 0).numpy() * 255.0).astype(np.uint8)
    preview_image_np = (result["working_image"].detach().cpu().clamp(0.0, 1.0).permute(1, 2, 0).numpy() * 255.0).astype(np.uint8)
    preview_prob_np = (result["preview_probability"].detach().cpu().clamp(0.0, 1.0).numpy() * 255.0).astype(np.uint8)
    preview_bin_np = (result["preview_binary"].detach().cpu().clamp(0.0, 1.0).numpy() * 255.0).astype(np.uint8)
    preview_overlay_np = (np.clip(result["preview_overlay"], 0.0, 1.0) * 255.0).astype(np.uint8)
    prob_np = (result["probability"].detach().cpu().clamp(0.0, 1.0).numpy() * 255.0).astype(np.uint8)
    bin_np = (result["binary"].detach().cpu().clamp(0.0, 1.0).numpy() * 255.0).astype(np.uint8)
    overlay_np = (np.clip(result["overlay"], 0.0, 1.0) * 255.0).astype(np.uint8)

    image_path = output_dir / "input_rgb.png"
    preview_image_path = output_dir / "preview_input_rgb.png"
    preview_probability_path = output_dir / "preview_probability_map.png"
    preview_binary_path = output_dir / "preview_binary_mask.png"
    preview_overlay_path = output_dir / "preview_overlay.png"
    probability_path = output_dir / "probability_map.png"
    binary_path = output_dir / "binary_mask.png"
    overlay_path = output_dir / "overlay.png"
    metadata_path = output_dir / "prediction.json"

    _save_image(image_path, image_np)
    _save_image(preview_image_path, preview_image_np)
    _save_image(preview_probability_path, preview_prob_np)
    _save_image(preview_binary_path, preview_bin_np)
    _save_image(preview_overlay_path, preview_overlay_np)
    _save_image(probability_path, prob_np)
    _save_image(binary_path, bin_np)
    _save_image(overlay_path, overlay_np)

    payload = {
        "checkpoint_path": result["checkpoint_path"],
        "image_path": result["image_path"],
        "device": result["device"],
        "threshold": result["threshold"],
        "normalization_mode": result["normalization_mode"],
        "inference_mode": result.get("inference_mode", "single_pass"),
        "crop_size": result.get("crop_size"),
        "probability_mean": float(result["probability"].mean().item()),
        "probability_max": float(result["probability"].max().item()),
        "predicted_positive_ratio": float(result["binary"].mean().item()),
        "checkpoint_info": {
            key: value
            for key, value in result["checkpoint_info"].items()
            if key != "train_config"
        },
    }
    metadata_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    return {
        "image_path": str(image_path.resolve()),
        "preview_image_path": str(preview_image_path.resolve()),
        "preview_probability_path": str(preview_probability_path.resolve()),
        "preview_binary_path": str(preview_binary_path.resolve()),
        "preview_overlay_path": str(preview_overlay_path.resolve()),
        "probability_path": str(probability_path.resolve()),
        "binary_path": str(binary_path.resolve()),
        "overlay_path": str(overlay_path.resolve()),
        "metadata_path": str(metadata_path.resolve()),
    }


