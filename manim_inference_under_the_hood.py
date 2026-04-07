from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image
from manim import *

# ---------------------------------------------------------------------------
# Utility helpers
# ---------------------------------------------------------------------------


def _to_uint8(image: np.ndarray) -> np.ndarray:
    return (np.clip(image, 0.0, 1.0) * 255.0).astype(np.uint8)


def _resize(image: np.ndarray, h: int, w: int) -> np.ndarray:
    pil = Image.fromarray(_to_uint8(image))
    return np.asarray(
        pil.resize((int(w), int(h)), Image.Resampling.BILINEAR),
        dtype=np.float32,
    ) / 255.0


def _chw_to_hwc(image_chw) -> np.ndarray:
    return image_chw.detach().cpu().clamp(0.0, 1.0).permute(1, 2, 0).numpy()


def _grayscale(image_hwc: np.ndarray) -> np.ndarray:
    return (
        0.299 * image_hwc[..., 0]
        + 0.587 * image_hwc[..., 1]
        + 0.114 * image_hwc[..., 2]
    ).astype(np.float32)


def _residual_to_rgb(residual_chw) -> np.ndarray:
    residual = residual_chw.detach().cpu().float()
    mean = residual.mean(dim=(1, 2), keepdim=True)
    std = residual.std(dim=(1, 2), keepdim=True).clamp_min(1e-6)
    norm = (residual - mean) / std
    vis = (0.5 + 0.20 * norm).clamp(0.0, 1.0)
    return vis.permute(1, 2, 0).numpy()


def _feature_tensor_to_rgb(
    feature_bchw,
    target_hw: tuple[int, int],
    tint: tuple[float, float, float] | None = None,
) -> np.ndarray:
    feat = feature_bchw[0].detach().cpu().float()
    fmap = feat.abs().mean(dim=0)
    fmap = (fmap - fmap.min()) / (fmap.max() - fmap.min() + 1e-6)
    gray = fmap.numpy()
    rgb = np.repeat(gray[..., None], 3, axis=-1)
    if tint is not None:
        tint_arr = np.array(tint, dtype=np.float32).reshape(1, 1, 3)
        rgb = np.clip(0.45 * rgb + 0.55 * (gray[..., None] * tint_arr), 0.0, 1.0)
    h, w = target_hw
    return _resize(rgb, h, w)


def _single_map_to_rgb(
    map_hw: np.ndarray,
    tint: tuple[float, float, float] | None = None,
    symmetric: bool = False,
) -> np.ndarray:
    arr = np.asarray(map_hw, dtype=np.float32)
    if symmetric:
        scale = float(np.max(np.abs(arr))) + 1e-6
        norm = arr / scale
        rgb = np.zeros((*arr.shape, 3), dtype=np.float32)
        rgb[..., 0] = np.clip(norm, 0.0, 1.0)
        rgb[..., 2] = np.clip(-norm, 0.0, 1.0)
        rgb[..., 1] = 0.2 * (1.0 - np.abs(norm))
        return np.clip(rgb, 0.0, 1.0)

    arr = np.abs(arr)
    arr = (arr - arr.min()) / (arr.max() - arr.min() + 1e-6)
    rgb = np.repeat(arr[..., None], 3, axis=-1)
    if tint is not None:
        tint_arr = np.array(tint, dtype=np.float32).reshape(1, 1, 3)
        rgb = np.clip(0.45 * rgb + 0.55 * (arr[..., None] * tint_arr), 0.0, 1.0)
    return rgb


def _heatmap(prob: np.ndarray) -> np.ndarray:
    p = np.clip(prob, 0.0, 1.0)
    r = np.clip(1.8 * p, 0.0, 1.0)
    g = np.clip(1.8 * (p - 0.25), 0.0, 1.0)
    b = np.clip(1.2 * (1.0 - p), 0.0, 1.0) * 0.65
    return np.stack([r, g, b], axis=-1)


def _load_real_artifacts(checkpoint_path: Path, image_path: Path) -> dict:
    import torch
    import torch.nn.functional as F
    from src.runtime import (
        compute_residual_noise,
        finalize_probability_for_inference_mode,
        get_inference_autocast_dtype,
        load_model_from_checkpoint,
        load_rgb_image,
        normalize_image_for_inference,
        overlay_prediction_on_image,
        prepare_image_for_inference_mode,
        resolve_center_crop_size,
        resolve_normalization_mode_for_inference,
    )

    model, device, checkpoint_info = load_model_from_checkpoint(
        checkpoint_path, device=torch.device("cpu")
    )
    original_image = load_rgb_image(image_path)
    crop_size = resolve_center_crop_size(checkpoint_info, crop_size=None)
    prepared_image, mode_transform = prepare_image_for_inference_mode(
        original_image, crop_size=crop_size
    )
    normalization_mode = resolve_normalization_mode_for_inference(
        manual_mode=None,
        checkpoint_train_config=checkpoint_info.get("train_config"),
        default_mode="imagenet",
    )
    normalized = normalize_image_for_inference(
        prepared_image, normalization_mode=normalization_mode
    )

    x = normalized.unsqueeze(0).to(device)
    prepared_residual = None
    residual_noise = None
    if bool(getattr(model.cfg, "use_residual", True)) and (
        getattr(model, "noise", None) is not None
    ):
        prepared_residual = compute_residual_noise(prepared_image)
        residual_noise = prepared_residual.unsqueeze(0).to(device)

    with torch.no_grad():
        autocast_dtype = get_inference_autocast_dtype(model, device)
        use_amp = device.type == "cuda" and autocast_dtype is not None
        with torch.autocast(
            device_type="cuda",
            dtype=autocast_dtype or torch.float16,
            enabled=use_amp,
        ):
            backbone = model._extract_features(x, residual_noise=residual_noise)
            fusion_inputs = {}
            if model.cfg.use_low_level and backbone.get("low_level") is not None:
                fusion_inputs["low_level"] = backbone["low_level"]
            if model.cfg.use_context and backbone.get("context") is not None:
                fusion_inputs["context"] = backbone["context"]
            if model.cfg.use_residual and backbone.get("residual") is not None:
                fusion_inputs["residual"] = backbone["residual"]
            fused_features = model.fusion(fusion_inputs, target_size=None)
            preds = model(
                x,
                target_size=prepared_image.shape[-2:],
                residual_noise=residual_noise,
            )
            probability_pre = torch.sigmoid(preds[0])[0, 0].detach().cpu().clamp(0.0, 1.0)

    probability, inference_mode = finalize_probability_for_inference_mode(
        probability_pre, mode_transform
    )
    output_image = prepared_image if inference_mode == "center_crop_keep" else original_image
    threshold = float(checkpoint_info.get("default_threshold", 0.5))
    binary = (probability >= threshold).float()
    overlay = overlay_prediction_on_image(output_image, probability)

    target_hw = (int(prepared_image.shape[-2]), int(prepared_image.shape[-1]))
    eff_feats = backbone.get("low_level") or []
    swin_feats = backbone.get("context") or []
    noise_feats = backbone.get("residual") or []
    fused_feats = fused_features or []

    eff_imgs = [
        _feature_tensor_to_rgb(t, target_hw, tint=(0.35, 0.7, 1.0))
        for t in eff_feats
    ]
    swin_imgs = [
        _feature_tensor_to_rgb(t, target_hw, tint=(0.45, 1.0, 0.45))
        for t in swin_feats
    ]
    noise_imgs = [
        _feature_tensor_to_rgb(t, target_hw, tint=(1.0, 0.62, 0.35))
        for t in noise_feats
    ]
    fused_imgs = [
        _feature_tensor_to_rgb(t, target_hw, tint=(0.95, 0.95, 0.95))
        for t in fused_feats
    ]
    fused_labels = [f"S{i + 1}" for i in range(len(fused_imgs))]

    decoder_stage_imgs: list[np.ndarray] = []
    decoder_stage_labels: list[str] = []
    decoder_head_imgs: list[np.ndarray] = []
    decoder_head_labels: list[str] = []

    if fused_feats:
        dec = model.decoder
        projected = [proj(feat) for proj, feat in zip(dec.skip_projections, fused_feats)]

        if getattr(dec, "enable_edge_guidance", False):
            gray = x.mean(dim=1, keepdim=True) if x.shape[1] > 1 else x
            sobel_x = dec.sobel_x.to(dtype=x.dtype, device=x.device)
            sobel_y = dec.sobel_y.to(dtype=x.dtype, device=x.device)
            grad_x = F.conv2d(gray, sobel_x, padding=1)
            grad_y = F.conv2d(gray, sobel_y, padding=1)
            edge_mag = torch.sqrt(grad_x**2 + grad_y**2 + 1e-6)
            edge_proj = dec.edge_proj(edge_mag)
            if edge_proj.shape[-2:] != projected[0].shape[-2:]:
                edge_proj = F.interpolate(
                    edge_proj,
                    size=projected[0].shape[-2:],
                    mode="bilinear",
                    align_corners=False,
                )
            projected[0] = projected[0] + edge_proj

        x_dec = dec.bottleneck(projected[-1])
        decoder_stage_labels.append("Bottleneck")
        decoder_stage_imgs.append(
            _feature_tensor_to_rgb(x_dec, target_hw, tint=(0.95, 0.85, 1.0))
        )

        if dec.cfg.per_stage_heads:
            predictions: list = [None] * len(projected)
            predictions[-1] = dec.predictors[-1](x_dec)
        else:
            predictions = []

        for idx in range(len(projected) - 2, -1, -1):
            skip = projected[idx]
            x_dec = F.interpolate(
                x_dec,
                size=skip.shape[-2:],
                mode="bilinear",
                align_corners=False,
            )
            x_dec = torch.cat([x_dec, skip], dim=1)
            x_dec = dec.decode_blocks[idx](x_dec)
            decoder_stage_labels.append(f"Decode S{idx + 1}")
            decoder_stage_imgs.append(
                _feature_tensor_to_rgb(x_dec, target_hw, tint=(0.90, 0.80, 1.0))
            )
            if dec.cfg.per_stage_heads:
                predictions[idx] = dec.predictors[idx](x_dec)

        if dec.cfg.per_stage_heads:
            out_preds = [pred for pred in predictions if pred is not None]
            if dec.use_dropout and out_preds:
                out_preds[0] = dec.dropout(out_preds[0])
            if out_preds:
                coarse_logits = out_preds[1] if len(out_preds) > 1 else None
                out_preds[0] = dec._refine_small_detail_logits(
                    out_preds[0],
                    x_dec,
                    projected[0],
                    coarse_logits=coarse_logits,
                )
                out_preds[0] = dec._refine_final_logits(out_preds[0])
            for i, logit in enumerate(out_preds):
                p = torch.sigmoid(logit[0, 0]).detach().cpu().numpy()
                decoder_head_imgs.append(_heatmap(p))
                decoder_head_labels.append(f"Head S{i + 1}")

    prob_np = probability.detach().cpu().numpy()
    if prepared_residual is None:
        prepared_residual = compute_residual_noise(prepared_image)

    x_vis = prepared_image.unsqueeze(0).to(torch.float32)
    hp_vis = prepared_residual.unsqueeze(0).to(torch.float32)
    c, h, w = int(x_vis.shape[1]), int(x_vis.shape[2]), int(x_vis.shape[3])

    if getattr(model, "noise", None) is not None and hasattr(model.noise, "srm_kernels"):
        base_kernels = model.noise.srm_kernels.detach().to(
            dtype=x_vis.dtype, device=x_vis.device
        )
    else:
        base_kernels = torch.tensor(
            [
                [
                    [0, 0, 0, 0, 0],
                    [0, -1, 2, -1, 0],
                    [0, 2, -4, 2, 0],
                    [0, -1, 2, -1, 0],
                    [0, 0, 0, 0, 0],
                ],
                [
                    [-1, 2, -2, 2, -1],
                    [2, -6, 8, -6, 2],
                    [-2, 8, -12, 8, -2],
                    [2, -6, 8, -6, 2],
                    [-1, 2, -2, 2, -1],
                ],
                [
                    [0, 0, 0, 0, 0],
                    [0, 1, -2, 1, 0],
                    [0, -2, 4, -2, 0],
                    [0, 1, -2, 1, 0],
                    [0, 0, 0, 0, 0],
                ],
            ],
            dtype=x_vis.dtype,
            device=x_vis.device,
        )
        base_kernels = base_kernels / base_kernels.abs().sum(dim=(1, 2), keepdim=True)
        base_kernels = base_kernels.view(3, 1, 5, 5)

    kernels = base_kernels.repeat(c, 1, 1, 1).contiguous()

    srm_rgb = F.conv2d(x_vis, kernels, padding=2, groups=c)
    srm_hp = F.conv2d(hp_vis, kernels, padding=2, groups=c)
    srm_avg = 0.5 * (srm_rgb + srm_hp)

    srm_rgb_by_filter = srm_rgb[0].view(c, 3, h, w).abs().mean(dim=0).cpu().numpy()
    hp_map = hp_vis[0].abs().mean(dim=0).cpu().numpy()
    avg_map = srm_avg[0].view(c, 3, h, w).abs().mean(dim=0).mean(dim=0).cpu().numpy()

    srm_filter_imgs = [
        _single_map_to_rgb(srm_rgb_by_filter[idx], tint=(1.0, 0.75, 0.35))
        for idx in range(3)
    ]
    hp_for_avg_img = _single_map_to_rgb(hp_map, tint=(0.45, 0.75, 1.0))
    srm_avg_img = _single_map_to_rgb(avg_map, tint=(1.0, 0.62, 0.35))

    return {
        "original_hwc": _chw_to_hwc(original_image),
        "prepared_hwc": _chw_to_hwc(prepared_image),
        "residual_hwc": _residual_to_rgb(prepared_residual),
        "eff_imgs": eff_imgs,
        "swin_imgs": swin_imgs,
        "noise_imgs": noise_imgs,
        "fused_imgs": fused_imgs,
        "fused_labels": fused_labels,
        "decoder_stage_imgs": decoder_stage_imgs,
        "decoder_stage_labels": decoder_stage_labels,
        "decoder_head_imgs": decoder_head_imgs,
        "decoder_head_labels": decoder_head_labels,
        "srm_filter_imgs": srm_filter_imgs,
        "hp_for_avg_img": hp_for_avg_img,
        "srm_avg_img": srm_avg_img,
        "prob_heat": _heatmap(prob_np),
        "binary_rgb": np.repeat(binary.detach().cpu().numpy()[..., None], 3, axis=-1),
        "overlay_hwc": np.clip(overlay, 0.0, 1.0),
        "crop_size": crop_size,
        "normalization_mode": normalization_mode,
        "threshold": threshold,
    }


# ---------------------------------------------------------------------------
# Scene
# ---------------------------------------------------------------------------


class NGIMLInferenceUnderTheHood(Scene):
    FRAME_W = 14.22
    FRAME_H = 8.0
    SAFE_X = 6.2
    TITLE_Y = 3.3
    CONTENT_TOP = 2.7

    def _panel(self, title: str, image: np.ndarray, width: float = 2.8) -> Group:
        img = ImageMobject(_to_uint8(image))
        img.set_width(width)
        lbl = Text(title, font_size=14).next_to(img, UP, buff=0.14)
        frame = RoundedRectangle(
            width=img.width + 0.12,
            height=img.height + 0.12,
            corner_radius=0.07,
        ).move_to(img)
        return Group(frame, img, lbl)

    def _stage_strip(
        self,
        title: str,
        images: list[np.ndarray],
        width_each: float = 0.82,
        gap: float = 0.08,
        max_width: float = 5.6,
    ) -> Group:
        n = max(1, len(images))
        est_width = n * width_each + (n - 1) * gap
        if est_width > max_width:
            width_each = max(0.55, (max_width - (n - 1) * gap) / n)

        panels = Group(
            *[self._panel(f"S{i + 1}", im, width=width_each) for i, im in enumerate(images)]
        )
        panels.arrange(RIGHT, buff=gap)
        head = Text(title, font_size=15).next_to(panels, UP, buff=0.16)
        return Group(head, panels)

    def _labeled_strip(
        self,
        title_str: str,
        labels: list[str],
        images: list[np.ndarray],
        width_each: float = 0.72,
        gap: float = 0.08,
        max_width: float = 8.8,
    ) -> Group:
        n = max(1, len(images))
        est_width = n * width_each + (n - 1) * gap
        if est_width > max_width:
            width_each = max(0.46, (max_width - (n - 1) * gap) / n)

        panels = Group(
            *[self._panel(lbl, im, width=width_each) for lbl, im in zip(labels, images)]
        )
        panels.arrange(RIGHT, buff=gap)
        head = Text(title_str, font_size=15).next_to(panels, UP, buff=0.16)
        return Group(head, panels)

    def _h_arrow(self, left_obj, right_obj, buff: float = 0.08) -> Arrow:
        start = np.array([left_obj.get_right()[0], left_obj.get_center()[1], 0])
        end = np.array([right_obj.get_left()[0], right_obj.get_center()[1], 0])
        return Arrow(
            start,
            end,
            buff=buff,
            stroke_width=3.2,
            max_tip_length_to_length_ratio=0.12,
        )

    def _v_arrow(self, top_obj, bottom_obj, buff: float = 0.08) -> Arrow:
        start = np.array([top_obj.get_center()[0], top_obj.get_bottom()[1], 0])
        end = np.array([bottom_obj.get_center()[0], bottom_obj.get_top()[1], 0])
        return Arrow(
            start,
            end,
            buff=buff,
            stroke_width=3.2,
            max_tip_length_to_length_ratio=0.12,
        )

    # ------------------------------------------------------------------
    # Chapter 1 — Preparation
    # ------------------------------------------------------------------
    def _chapter1(self, data: dict) -> None:
        title = Text("Preparation", font_size=30).move_to(UP * self.TITLE_Y)
        subtitle = Text(
            f"input=auto_preprocessed | normalization={data['normalization_mode']}",
            font_size=18,
        ).next_to(title, DOWN, buff=0.12)
        self.play(Write(title), FadeIn(subtitle))

        panel_w = 3.2
        gap = 0.9
        total = 2 * panel_w + gap
        x_left = -total / 2 + panel_w / 2
        y_row = UP * (self.CONTENT_TOP - 2.5)

        p1 = self._panel("Input", data["prepared_hwc"], width=panel_w).move_to(
            x_left * RIGHT + y_row
        )
        p2 = self._panel("High-Pass Residual", data["residual_hwc"], width=panel_w).move_to(
            (x_left + panel_w + gap) * RIGHT + y_row
        )

        arr1 = self._h_arrow(p1, p2, buff=0.10)

        self.play(FadeIn(p1))
        self.play(GrowArrow(arr1), FadeIn(p2))
        self.wait(0.7)
        self.play(FadeOut(Group(title, subtitle, p1, p2, arr1)))

    # ------------------------------------------------------------------
    # Chapter 2 — Feature Routing + Noise-Guided Attention
    # ------------------------------------------------------------------
    def _chapter2(self, data: dict) -> None:
        title = Text("Backbone Feature Extraction", font_size=28).move_to(
            UP * self.TITLE_Y
        )
        self.play(Write(title))

        # Spacious 3-column layout
        row_a, row_b, row_c = 1.95, 0.35, -1.35
        x_input = -5.4
        x_mid = -1.8
        x_avg = -3.7
        x_strip_center = 3.95
        strip_max_w = 3.7

        # Inputs
        rgb_in = self._panel("RGB", data["prepared_hwc"], width=1.00).move_to(
            RIGHT * x_input + UP * 1.45
        )
        hp_in = self._panel("HP Residual", data["residual_hwc"], width=1.00).move_to(
            RIGHT * x_input + DOWN * 1.55
        )

        def _enc_box(label: str, y: float):
            box = RoundedRectangle(width=1.9, height=0.72, corner_radius=0.08).move_to(
                RIGHT * x_mid + UP * y
            )
            txt = Text(label, font_size=17).move_to(box)
            return box, txt

        eff_box, eff_txt = _enc_box("EffNet", row_a)
        swin_box, swin_txt = _enc_box("Swin", row_b)
        noise_box, noise_txt = _enc_box("Noise", row_c)

        self.play(FadeIn(rgb_in), FadeIn(hp_in))
        self.play(
            FadeIn(eff_box), FadeIn(eff_txt),
            FadeIn(swin_box), FadeIn(swin_txt),
            FadeIn(noise_box), FadeIn(noise_txt),
        )

        # SRM branch placed lower-left to avoid collisions
        srm_panels = Group(
            *[
                self._panel(f"SRM {i + 1}", data["srm_filter_imgs"][i], width=0.38)
                for i in range(min(3, len(data.get("srm_filter_imgs", []))))
            ]
        )
        srm_panels.arrange(RIGHT, buff=0.08).move_to(RIGHT * -4.35 + DOWN * 0.01)

        avg_panel = self._panel("Avg(SRM, HP)", data["srm_avg_img"], width=0.56).move_to(
            RIGHT * x_avg + DOWN * 1.35
        )

        a_rgb_eff = self._h_arrow(rgb_in, eff_box, buff=0.08)
        a_rgb_swin = self._h_arrow(rgb_in, swin_box, buff=0.08)

        a_rgb_srm = Arrow(
            start=np.array([rgb_in[0].get_center()[0], rgb_in[0].get_bottom()[1], 0]),
            end=np.array([srm_panels.get_center()[0], srm_panels.get_top()[1], 0]),
            buff=0.08,
            stroke_width=3.2,
            max_tip_length_to_length_ratio=0.12,
        )

        a_srm_avg = Arrow(
            start=np.array([srm_panels.get_center()[0], srm_panels.get_bottom()[1], 0]),
            end=np.array([avg_panel[0].get_center()[0], avg_panel[0].get_top()[1], 0]),
            buff=0.08,
            stroke_width=3.2,
            max_tip_length_to_length_ratio=0.12,
        )

        a_hp_avg = Arrow(
            start=np.array([hp_in[0].get_right()[0], hp_in[0].get_center()[1], 0]),
            end=np.array([avg_panel[0].get_left()[0], avg_panel[0].get_center()[1], 0]),
            buff=0.06,
            stroke_width=3.2,
            max_tip_length_to_length_ratio=0.12,
        )

        a_avg_noise = Arrow(
            start=np.array([avg_panel[0].get_right()[0], avg_panel[0].get_center()[1], 0]),
            end=np.array([noise_box.get_left()[0], noise_box.get_center()[1], 0]),
            buff=0.06,
            stroke_width=3.2,
            max_tip_length_to_length_ratio=0.12,
        )

        self.play(GrowArrow(a_rgb_eff), GrowArrow(a_rgb_swin), GrowArrow(a_rgb_srm))
        self.play(FadeIn(srm_panels))
        for p in srm_panels:
            self.play(Indicate(p, scale_factor=1.03), run_time=0.12)
        self.play(GrowArrow(a_srm_avg))
        self.play(GrowArrow(a_hp_avg), FadeIn(avg_panel))
        self.play(Indicate(avg_panel, scale_factor=1.05), run_time=0.35)
        self.play(GrowArrow(a_avg_noise))

        # Feature strips
        def _make_strip(title_str, imgs, y):
            s = self._stage_strip(
                title_str,
                imgs,
                width_each=0.54,
                max_width=strip_max_w,
                gap=0.06,
            )
            s.move_to(RIGHT * x_strip_center + UP * y)
            return s

        eff_strip = _make_strip("EffNet Features", data["eff_imgs"], row_a)
        swin_strip = _make_strip("Swin Features", data["swin_imgs"], row_b)
        noise_strip = _make_strip("Noise Features", data["noise_imgs"], row_c)

        strips_group = Group(eff_strip, swin_strip, noise_strip).scale(0.95)

        eff_arr = self._h_arrow(eff_box, eff_strip, buff=0.10)
        swin_arr = self._h_arrow(swin_box, swin_strip, buff=0.10)
        noise_arr = self._h_arrow(noise_box, noise_strip, buff=0.10)

        self.play(GrowArrow(noise_arr), FadeIn(noise_strip))
        for panel in noise_strip[1]:
            self.play(Indicate(panel, scale_factor=1.03), run_time=0.12)

        # Attention demo, moved lower for breathing room
        attention_group = Group()
        if data["eff_imgs"] and data["noise_imgs"]:
            idx = min(1, len(data["eff_imgs"]) - 1, len(data["noise_imgs"]) - 1)
            eff_s = data["eff_imgs"][idx]
            noise_s = data["noise_imgs"][idx]

            attn = np.clip((_grayscale(noise_s) - 0.5) * 2.0, -1.0, 1.0)
            attn_vis = np.zeros((*attn.shape, 3), dtype=np.float32)
            attn_vis[..., 0] = np.clip(attn, 0.0, 1.0)
            attn_vis[..., 2] = np.clip(-attn, 0.0, 1.0)
            attn_vis[..., 1] = 0.2 * (1.0 - np.abs(attn))
            modulated = np.clip(eff_s * (1.0 + 0.95 * attn[..., None]), 0.0, 1.0)

            attn_y = -2.75
            panel_w = 0.92
            panels_attn = [
                self._panel(lbl, im, width=panel_w)
                for lbl, im in zip(
                    [f"Noise S{idx+1}", f"A_noise S{idx}", f"Eff S{idx}", "Eff*(1+A)"],
                    [noise_s, attn_vis, eff_s, modulated],
                )
            ]
            attn_row = Group(*panels_attn).arrange(RIGHT, buff=0.16).move_to(UP * attn_y)

            n_panel, a_panel, e_panel, o_panel = panels_attn
            caption = Text(
                "Noise produces attention that modulates EffNet features",
                font_size=13,
            ).next_to(attn_row, UP * 1.1, buff=0.12)

            n2a = self._h_arrow(n_panel, a_panel, buff=0.04)
            a2e = self._h_arrow(a_panel, e_panel, buff=0.04)
            e2o = self._h_arrow(e_panel, o_panel, buff=0.04)

            self.play(FadeIn(caption))
            self.play(FadeIn(n_panel))
            self.play(GrowArrow(n2a), FadeIn(a_panel))
            self.play(Indicate(a_panel, scale_factor=1.05), run_time=0.35)
            self.play(GrowArrow(a2e), FadeIn(e_panel))
            self.play(GrowArrow(e2o), FadeIn(o_panel))
            self.play(Indicate(o_panel, scale_factor=1.05), run_time=0.5)

            attention_group = Group(caption, attn_row, n2a, a2e, e2o)

        self.play(FadeOut(attention_group))
        self.play(GrowArrow(eff_arr), FadeIn(eff_strip))
        for panel in eff_strip[1]:
            self.play(Indicate(panel, scale_factor=1.03), run_time=0.12)

        self.play(GrowArrow(swin_arr), FadeIn(swin_strip))
        for panel in swin_strip[1]:
            self.play(Indicate(panel, scale_factor=1.03), run_time=0.12)

        self.wait(0.5)
        self.play(
            FadeOut(
                Group(
                    title,
                    rgb_in,
                    hp_in,
                    eff_box,
                    eff_txt,
                    swin_box,
                    swin_txt,
                    noise_box,
                    noise_txt,
                    srm_panels,
                    avg_panel,
                    a_rgb_eff,
                    a_rgb_swin,
                    a_rgb_srm,
                    a_srm_avg,
                    a_hp_avg,
                    a_avg_noise,
                    noise_strip,
                    noise_arr,
                    eff_strip,
                    eff_arr,
                    swin_strip,
                    swin_arr,
                )
            )
        )

    # ------------------------------------------------------------------
    # Chapter 3 — Fusion -> Decoder -> Prediction
    # ------------------------------------------------------------------
    def _chapter3(self, data: dict) -> None:
        title = Text("Fusion -> Decoder -> Prediction", font_size=28).move_to(
            UP * self.TITLE_Y
        )
        self.play(Write(title))

        fused_strip = self._labeled_strip(
            "Fused Multi-scale Features",
            data["fused_labels"],
            data["fused_imgs"],
            width_each=0.58,
            max_width=6.8,
            gap=0.07,
        ).move_to(UP * 2.15)

        # Reverse decoder presentation order so Fusion->Decoder arrows stay straight.
        decoder_stage_labels = list(reversed(data["decoder_stage_labels"]))
        decoder_stage_imgs = list(reversed(data["decoder_stage_imgs"]))

        dec_stage_strip = self._labeled_strip(
            "Decoder Stages",
            decoder_stage_labels,
            decoder_stage_imgs,
            width_each=0.50,
            max_width=6.8,
            gap=0.07,
        ).move_to(UP * 0.75)

        dec_head_strip = self._labeled_strip(
            "Per-stage Heads (Sigmoid)",
            data["decoder_head_labels"],
            data["decoder_head_imgs"],
            width_each=0.44,
            max_width=6.0,
            gap=0.07,
        ).move_to(DOWN * 0.95)

        chapter3_core = Group(fused_strip, dec_stage_strip, dec_head_strip).scale(0.95)

        fused_panels = fused_strip[1]
        stage_panels = dec_stage_strip[1]
        head_panels = dec_head_strip[1]

        def _parse_scale(label: str, prefix: str) -> int | None:
            s = str(label).strip()
            if not s.startswith(prefix):
                return None
            try:
                return int(s.split(prefix, 1)[1])
            except Exception:
                return None

        fused_index_by_scale = {
            _parse_scale(lbl, "S"): idx
            for idx, lbl in enumerate(data["fused_labels"])
            if _parse_scale(lbl, "S") is not None
        }
        stage_index_by_scale = {
            _parse_scale(lbl, "Decode S"): idx
            for idx, lbl in enumerate(decoder_stage_labels)
            if _parse_scale(lbl, "Decode S") is not None
        }
        head_index_by_scale = {
            _parse_scale(lbl, "Head S"): idx
            for idx, lbl in enumerate(data["decoder_head_labels"])
            if _parse_scale(lbl, "Head S") is not None
        }
        bottleneck_index = next(
            (idx for idx, lbl in enumerate(decoder_stage_labels) if lbl == "Bottleneck"),
            None,
        )
        max_fused_scale = max(fused_index_by_scale.keys()) if fused_index_by_scale else None

        # Straight index-based Fusion -> Decoder arrows (no crossed/twisted links).
        n_fused_stage = min(len(fused_panels), len(stage_panels))
        arr_fused_to_stages = VGroup(*[
            self._v_arrow(fused_panels[i], stage_panels[i], buff=0.08)
            for i in range(n_fused_stage)
        ])

        # Semantic decoder -> head connections:
        # Head Sx should come from Decode Sx, and highest-scale head can come from Bottleneck.
        stage_to_head_arrows: list[Arrow] = []
        for scale, head_idx in sorted(head_index_by_scale.items()):
            stage_idx = stage_index_by_scale.get(scale)
            if stage_idx is None and max_fused_scale is not None and scale == max_fused_scale:
                stage_idx = bottleneck_index
            if stage_idx is None:
                continue
            stage_to_head_arrows.append(
                self._v_arrow(
                    stage_panels[stage_idx],
                    head_panels[head_idx],
                    buff=0.08,
                )
            )
        arr_stages_to_heads = VGroup(*stage_to_head_arrows)

        # optional: decoder flow left->right or right->left depending on your ordering
        # if your decoder labels are [D4, D3, D2, D1], use this:
        n = min(len(stage_panels), len(head_panels))
        arr_decoder_flow = VGroup(*[
            self._h_arrow(stage_panels[i], stage_panels[i + 1], buff=0.05)
            for i in range(n - 1)
        ])

        self.play(FadeIn(fused_strip))
        for panel in fused_panels:
            self.play(Indicate(panel, scale_factor=1.03), run_time=0.10)

        self.play(
            LaggedStart(*[GrowArrow(a) for a in arr_fused_to_stages], lag_ratio=0.08),
            FadeIn(dec_stage_strip),
        )

        for a in arr_decoder_flow:
            self.play(GrowArrow(a), run_time=0.12)

        for panel in stage_panels:
            self.play(Indicate(panel, scale_factor=1.03), run_time=0.10)

        self.play(
            LaggedStart(*[GrowArrow(a) for a in arr_stages_to_heads], lag_ratio=0.08),
            FadeIn(dec_head_strip),
        )

        for panel in head_panels:
            self.play(Indicate(panel, scale_factor=1.03), run_time=0.09)

        # Bottom outputs
        out_y = -2.95
        out_w = 1.5

        prob_panel = self._panel("Sigmoid Prob", data["prob_heat"], width=out_w)
        bin_panel = self._panel(f"Mask (t={data['threshold']:.2f})", data["binary_rgb"], width=out_w)
        ov_panel = self._panel("Overlay", data["overlay_hwc"], width=out_w)

        Group(prob_panel, bin_panel, ov_panel).arrange(RIGHT, buff=0.68).move_to(UP * out_y)

        # connect from the MAIN/FINEST head only, not from the whole strip center
        arr_heads_prob = self._v_arrow(head_panels[0], prob_panel, buff=0.10)
        arr_prob_bin = self._h_arrow(prob_panel, bin_panel, buff=0.06)
        arr_bin_ov = self._h_arrow(bin_panel, ov_panel, buff=0.06)

        self.play(GrowArrow(arr_heads_prob), FadeIn(prob_panel))
        self.play(GrowArrow(arr_prob_bin), FadeIn(bin_panel))
        self.play(GrowArrow(arr_bin_ov), FadeIn(ov_panel))

        self.wait(0.8)

        # Final comparison
        final_title = Text("Final: Input vs Output", font_size=24).move_to(
            UP * self.TITLE_Y
        )
        final_in = self._panel("Input", data["prepared_hwc"], width=3.1).move_to(
            LEFT * 2.15 + DOWN * 0.35
        )
        final_out = self._panel("Output", data["overlay_hwc"], width=3.1).move_to(
            RIGHT * 2.15 + DOWN * 0.35
        )
        final_arr = self._h_arrow(final_in, final_out, buff=0.08)

        self.play(
            FadeOut(
                Group(
                    title,
                    fused_strip,
                    dec_stage_strip,
                    dec_head_strip,
                    arr_fused_to_stages,
                    arr_decoder_flow,
                    arr_stages_to_heads,
                    prob_panel,
                    bin_panel,
                    ov_panel,
                    arr_heads_prob,
                    arr_prob_bin,
                    arr_bin_ov
                )
            )
        )
        self.play(Write(final_title))
        self.play(FadeIn(final_in), GrowArrow(final_arr), FadeIn(final_out))
        self.wait(1.5)

    # ------------------------------------------------------------------
    # Main entry point
    # ------------------------------------------------------------------
    def construct(self):
        checkpoint_path = Path("checkpoints/casia-full.pt")
        image_path = Path("sample.jpg")

        if not checkpoint_path.exists() or not image_path.exists():
            self.play(
                Write(
                    Text(
                        "Required files missing: checkpoints/casia-full.pt or sample.jpg",
                        font_size=26,
                    )
                )
            )
            self.wait(2)
            return

        try:
            data = _load_real_artifacts(checkpoint_path, image_path)
        except Exception as exc:
            self.play(
                FadeIn(
                    Text(
                        f"Failed to load tensors: {str(exc)[:120]}",
                        font_size=22,
                    )
                )
            )
            self.wait(2)
            return

        self._chapter1(data)
        self._chapter2(data)
        self._chapter3(data)