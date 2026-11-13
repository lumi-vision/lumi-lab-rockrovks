"""Spatiotemporal partial mask integration, outward spatial feathering, and upsampling (`mask_ops.py`).

Implements the confirmed Partial Mask specifications:
  - 4-1A: Analysis-resolution (`analysis_max_width = 640`) `uint8` `[0..255]` storage +
          bilinear (`cv2.INTER_LINEAR`) upsampling helper to `(orig_height, orig_width)`.
  - 4-2A: `+-0.25s` sliding temporal maximum (union) via O(T) van Herk bitwise OR on packed
          bit-planes + `0.20s` cosine boundary temporal ramp (plus `hold`/`safe` Half-Hold support).
  - 4-3A: One-sided outward spatial feathering preserving `W_t == 1.0` (`255`) strictly across
          all core hazard pixels (`M_core`) while decaying smoothly outward over `~1.75 deg`.
"""

from __future__ import annotations

import math
from typing import Sequence

import cv2
import numpy as np
import torch
import torch.nn.functional as F

from vdcorefs.config import ViewingGeometry
from vdcorefs.photometry import resolve_device
from vdcorefs.pipeline import mask_to_spans

from .config import (
     PartialMaskConfig,
    PartialTemporalScheduleMode,
    PartialVideoAnalysisResult,
)

# Exact IEC 61966-2-1 8-bit sRGB <-> linear light lookup tables for Formula A partial dimming
_SRGB_U8_NORM = np.arange(256, dtype=np.float64) / 255.0
_SRGB_TO_LIN_F32 = np.where(
    _SRGB_U8_NORM <= 0.04045,
    _SRGB_U8_NORM / 12.92,
    ((_SRGB_U8_NORM + 0.055) / 1.055) ** 2.4,
).astype(np.float32)

_LIN16_GRID = np.linspace(0.0, 1.0, 65536, dtype=np.float64)
_LIN16_SRGB = np.where(
    _LIN16_GRID <= 0.0031308,
    _LIN16_GRID * 12.92,
    1.055 * np.power(np.maximum(_LIN16_GRID, 1e-12), 1.0 / 2.4) - 0.055,
)
_LIN16_TO_SRGB_U8 = np.clip(np.rint(_LIN16_SRGB * 255.0), 0.0, 255.0).astype(np.uint8)


def pack_bool_mask_2d(mask: torch.Tensor | np.ndarray) -> np.ndarray:
    """Bit-pack a `(H, W)` boolean mask along axis `-1` into `(H, ceil(W / 8))` `uint8`."""
    if isinstance(mask, torch.Tensor):
        arr = mask.detach().to(dtype=torch.uint8, device="cpu").numpy()
    else:
        arr = np.asarray(mask, dtype=np.uint8)
    return np.packbits(arr, axis=-1)


def unpack_bool_masks_3d(packed: np.ndarray, width: int) -> np.ndarray:
    """Unpack `(B, H, ceil(W / 8))` `uint8` bit-planes into `(B, H, width)` `bool`."""
    if packed.shape[0] == 0:
        return np.zeros((0, packed.shape[1], width), dtype=bool)
    return np.unpackbits(packed, axis=-1)[..., :width].astype(bool)


def temporal_sliding_or_packed(packed_3d: np.ndarray, radius_frames: int) -> np.ndarray:
    """Exact 1D sliding-window bitwise OR over `[t - radius_frames, t + radius_frames]` in O(T) time.

    Uses the van Herk / Gil-Werman block prefix-suffix decomposition on `np.packbits` bit-planes.
    """
    n_frames = packed_3d.shape[0]
    if n_frames <= 1 or radius_frames <= 0:
        return packed_3d.copy()

    r = int(radius_frames)
    win = 2 * r + 1
    fwd = np.empty_like(packed_3d)
    bwd = np.empty_like(packed_3d)

    for i in range(n_frames):
        if i % win == 0:
            fwd[i] = packed_3d[i]
        else:
            np.bitwise_or(fwd[i - 1], packed_3d[i], out=fwd[i])

    for i in range(n_frames - 1, -1, -1):
        if (i + 1) % win == 0 or i == n_frames - 1:
            bwd[i] = packed_3d[i]
        else:
            np.bitwise_or(bwd[i + 1], packed_3d[i], out=bwd[i])

    out = np.empty_like(packed_3d)
    for t in range(n_frames):
        lo = 0 if t < r else (t - r)
        hi = (n_frames - 1) if (t + r >= n_frames) else (t + r)
        if (lo // win) < (hi // win):
            np.bitwise_or(bwd[lo], fwd[hi], out=out[t])
        elif lo % win == 0:
            out[t] = fwd[hi]
        else:
            out[t] = bwd[lo]
    return out


def temporal_bridge_spans_packed(
    packed_3d: np.ndarray,
    spans: Sequence[tuple[int, int]],
    max_gap_frames: int,
) -> np.ndarray:
    """Close internal per-pixel temporal gaps `<= max_gap_frames` inside active `spans` and prevent zero dropouts."""
    if packed_3d.shape[0] == 0 or not spans:
        return packed_3d.copy()

    out = packed_3d.copy()
    r_close = max(0, int(max_gap_frames) // 2)

    for s_idx, e_idx in spans:
        if e_idx < s_idx:
            continue
        seg = out[s_idx : e_idx + 1]
        seg_len = seg.shape[0]

        if r_close > 0 and seg_len >= 3:
            pad_shape = (r_close, seg.shape[1], seg.shape[2])
            zeros_pad = np.zeros(pad_shape, dtype=np.uint8)
            padded = np.concatenate([zeros_pad, seg, zeros_pad], axis=0)
            dilated = temporal_sliding_or_packed(padded, r_close)
            inv_dil = np.bitwise_not(dilated)
            inv_eroded = temporal_sliding_or_packed(inv_dil, r_close)
            closed = np.bitwise_not(inv_eroded)[r_close : r_close + seg_len]
            np.bitwise_or(seg, closed, out=seg)

        # Guarantee no frame inside an active hazard span has an all-zero mask if any frame in the span is non-zero
        non_empty = np.flatnonzero(np.any(seg != 0, axis=(1, 2)))
        if len(non_empty) > 0 and len(non_empty) < seg_len:
            for local_i in range(seg_len):
                if not np.any(seg[local_i]):
                    pos = int(np.searchsorted(non_empty, local_i))
                    if pos <= 0:
                        nearest = int(non_empty[0])
                    elif pos >= len(non_empty):
                        nearest = int(non_empty[-1])
                    else:
                        left_i = int(non_empty[pos - 1])
                        right_i = int(non_empty[pos])
                        nearest = left_i if (local_i - left_i) <= (right_i - local_i) else right_i
                    seg[local_i] = seg[nearest]

    return out


def clean_speckles_packed(
    packed_3d: np.ndarray,
    active_frame_mask: np.ndarray,
    width: int,
    min_cc_area_px: int = 80,
) -> np.ndarray:
    """Remove isolated `< min_cc_area_px` noise speckles from packed core masks (`4-a`).

    Preserves fine dot-matrix/checkerboard patterns via `5x5` micro-closing cluster check
    and guarantees non-empty frames never become all-zero.
    """
    out = packed_3d.copy()
    min_area = int(min_cc_area_px)
    if packed_3d.shape[0] == 0 or min_area <= 1:
        return out

    h = packed_3d.shape[1]
    total_px = h * width
    close_k = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))

    for idx in np.flatnonzero(active_frame_mask):
        if not np.any(out[idx]):
            continue
        m_u8 = np.unpackbits(out[idx], axis=-1)[..., :width]
        fg_px = int(m_u8.sum())
        if fg_px == 0 or fg_px >= total_px:
            continue

        n_lbl, labels, stats, _ = cv2.connectedComponentsWithStats(m_u8, connectivity=8)
        if n_lbl <= 2:
            continue

        areas = stats[:, cv2.CC_STAT_AREA]
        small_flags = areas[1:] < min_area
        if not np.any(small_flags):
            continue

        small_sum = int(areas[1:][small_flags].sum())
        if small_sum >= int(0.04 * total_px):
            # Dense constellation (e.g., fine checkerboard): cluster nearby dots before filtering
            closed = cv2.morphologyEx(m_u8, cv2.MORPH_CLOSE, close_k)
            n_c, lbl_c, stats_c, _ = cv2.connectedComponentsWithStats(closed, connectivity=8)
            if n_c <= 2:
                out[idx] = np.packbits(closed, axis=-1)
                continue
            areas_c = stats_c[:, cv2.CC_STAT_AREA]
            keep_lut = np.zeros(n_c, dtype=np.uint8)
            valid_c = areas_c[1:] >= min_area
            if np.any(valid_c):
                keep_lut[1:][valid_c] = 1
            else:
                keep_lut[1 + int(np.argmax(areas_c[1:]))] = 1
            cleaned = keep_lut[lbl_c]
        else:
            keep_lut = np.zeros(n_lbl, dtype=np.uint8)
            valid_fg = ~small_flags
            if np.any(valid_fg):
                keep_lut[1:][valid_fg] = 1
            else:
                keep_lut[1 + int(np.argmax(areas[1:]))] = 1
            cleaned = keep_lut[labels]

        out[idx] = np.packbits(cleaned, axis=-1)

    return out


def _box_blur_tch(x: torch.Tensor, r: int) -> torch.Tensor:
    """Exact O(N) separable 2D box filter on `(B, C, H, W)` tensor with replicate padding."""
    if r <= 0:
        return x
    k = 2 * r + 1
    x = F.avg_pool2d(F.pad(x, (r, r, 0, 0), mode="replicate"), kernel_size=(1, k), stride=1)
    x = F.avg_pool2d(F.pad(x, (0, 0, r, r), mode="replicate"), kernel_size=(k, 1), stride=1)
    return x


def _dilate_tch(x: torch.Tensor, r: int) -> torch.Tensor:
    """Exact separable square morphological dilation on `(B, C, H, W)` tensor."""
    if r <= 0:
        return x
    k = 2 * r + 1
    x = F.max_pool2d(F.pad(x, (r, r, 0, 0), mode="replicate"), kernel_size=(1, k), stride=1)
    x = F.max_pool2d(F.pad(x, (0, 0, r, r), mode="replicate"), kernel_size=(k, 1), stride=1)
    return x


def _erode_tch(x: torch.Tensor, r: int) -> torch.Tensor:
    return -_dilate_tch(-x, r)


def _morph_open_close_mid_and_span(
    x: torch.Tensor,
    r: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return morphological midpoint `0.5*(open + close)` and span `close - open`."""
    opened = _dilate_tch(_erode_tch(x, r), r)
    closed = _erode_tch(_dilate_tch(x, r), r)
    return 0.5 * (opened + closed), torch.clamp(closed - opened, min=0.0)


def _gaussian_blur_tch(x: torch.Tensor, sigma: float) -> torch.Tensor:
    """Separable 1D Gaussian blur on `(B, C, H, W)` tensor with replicate padding."""
    if sigma <= 0.15:
        return x
    r = max(1, int(math.ceil(2.8 * sigma)))
    coords = torch.arange(-r, r + 1, device=x.device, dtype=x.dtype)
    g1d = torch.exp(-0.5 * (coords / sigma) ** 2)
    g1d = g1d / g1d.sum()
    c = x.shape[1]
    kh = g1d.view(1, 1, 1, 2 * r + 1).expand(c, 1, 1, 2 * r + 1)
    kv = g1d.view(1, 1, 2 * r + 1, 1).expand(c, 1, 2 * r + 1, 1)
    x = F.conv2d(F.pad(x, (r, r, 0, 0), mode="replicate"), kh, groups=c)
    x = F.conv2d(F.pad(x, (0, 0, r, r), mode="replicate"), kv, groups=c)
    return x


def _color_guided_filter_tch(
    guide: torch.Tensor,
    src: torch.Tensor,
    r: int,
    eps: float,
) -> torch.Tensor:
    """Batched O(N) 3-channel Color Guided Filter (He et al., TPAMI 2013) on `(B, 3, H, W)`."""
    mI = _box_blur_tch(guide, r)
    mp = _box_blur_tch(src, r)
    mIp = _box_blur_tch(guide * src, r)
    cov_Ip = mIp - mI * mp

    Ir, Ig, Ib = guide[:, 0:1], guide[:, 1:2], guide[:, 2:3]
    mIr, mIg, mIb = mI[:, 0:1], mI[:, 1:2], mI[:, 2:3]

    v_rr = _box_blur_tch(Ir * Ir, r) - mIr * mIr + eps
    v_rg = _box_blur_tch(Ir * Ig, r) - mIr * mIg
    v_rb = _box_blur_tch(Ir * Ib, r) - mIr * mIb
    v_gg = _box_blur_tch(Ig * Ig, r) - mIg * mIg + eps
    v_gb = _box_blur_tch(Ig * Ib, r) - mIg * mIb
    v_bb = _box_blur_tch(Ib * Ib, r) - mIb * mIb + eps

    inv_rr = v_gg * v_bb - v_gb * v_gb
    inv_rg = v_rb * v_gb - v_rg * v_bb
    inv_rb = v_rg * v_gb - v_rb * v_gg
    inv_gg = v_rr * v_bb - v_rb * v_rb
    inv_gb = v_rb * v_rg - v_rr * v_gb
    inv_bb = v_rr * v_gg - v_rg * v_rg

    det = torch.clamp(v_rr * inv_rr + v_rg * inv_rg + v_rb * inv_rb, min=1e-12)
    c0, c1, c2 = cov_Ip[:, 0:1], cov_Ip[:, 1:2], cov_Ip[:, 2:3]

    a_r = (inv_rr * c0 + inv_rg * c1 + inv_rb * c2) / det
    a_g = (inv_rg * c0 + inv_gg * c1 + inv_gb * c2) / det
    a_b = (inv_rb * c0 + inv_gb * c1 + inv_bb * c2) / det

    b = mp - (a_r * mIr + a_g * mIg + a_b * mIb)
    mean_ab = _box_blur_tch(torch.cat([a_r, a_g, a_b, b], dim=1), r)

    q = (
        mean_ab[:, 0:1] * Ir
        + mean_ab[:, 1:2] * Ig
        + mean_ab[:, 2:3] * Ib
        + mean_ab[:, 3:4]
    )
    return torch.clamp(q, 0.0, 1.0)


def _quintic_smoothstep(t: torch.Tensor) -> torch.Tensor:
    """C2-continuous quintic smoothstep `6t^5 - 15t^4 + 10t^3` (`f'=f''=0` at `0` and `1`)."""
    tc = torch.clamp(t, 0.0, 1.0)
    return tc * tc * tc * (tc * (tc * 6.0 - 15.0) + 10.0)


@torch.no_grad()
def apply_guided_borderless_batch(
    core_masks_bool: torch.Tensor,
    guide_peak_rgb: torch.Tensor,
    guide_swing_rgb: torch.Tensor,
    is_flash_frame: torch.Tensor,
    geometry: ViewingGeometry,
    mask_cfg: PartialMaskConfig = PartialMaskConfig(),
) -> torch.Tensor:
    """Build object-contour-aligned (`1-b`), C2 borderless (`3-a`) soft weights `(B, H, W)` in `[0, 1]`.

    Uses:
      1. Morphological Open-Close midpoint (`0.5*(open + close)`) to flatten internal periodic
         stripes into uniform object interiors while preserving step silhouettes without blur,
      2. Morphological texture span (`close - open`) on pattern frames and temporal RGB swing
         (`max - min` over `+-r_win`) on flash frames to trim background polygon spillover while
         covering contiguous hazardous objects/halos up to their natural boundary,
      3. Batched O(N) 3-channel Color Guided Filtering + C2 Gaussian cushion + quintic smoothstep
         so there are zero polygonal borders or Mach-band kinks anywhere.
    """
    if core_masks_bool.numel() == 0:
        return core_masks_bool.to(torch.float32)

    bsz, h, w = core_masks_bool.shape
    core_f = core_masks_bool.to(torch.float32).unsqueeze(1)
    is_full = core_masks_bool.flatten(1).all(dim=1)
    is_empty = ~core_masks_bool.flatten(1).any(dim=1)

    ppd = float(geometry.px_per_deg_ring)
    r_struct = max(2, int(round(0.35 * ppd)))
    struct_rgb, _ = _morph_open_close_mid_and_span(guide_peak_rgb, r_struct)
    struct_rgb = _gaussian_blur_tch(struct_rgb, max(1.0, 0.18 * ppd))

    lum_peak = (
        0.299 * guide_peak_rgb[:, 0:1]
        + 0.587 * guide_peak_rgb[:, 1:2]
        + 0.114 * guide_peak_rgb[:, 2:3]
    )
    _, tex_fine = _morph_open_close_mid_and_span(lum_peak, max(3, int(round(0.42 * ppd))))
    _, tex_wide = _morph_open_close_mid_and_span(lum_peak, max(5, int(round(1.45 * ppd))))
    tex_act_fine = torch.clamp((tex_fine - 0.07) / 0.16, 0.0, 1.0)
    # Require crisp local edges (`r=2`) within `1.45 deg` for wide-bar pattern membership so
    # blurry background bokeh never triggers `tex_act_wide`, while wide `~1 cpd` wedges/bars do.
    crisp_grad = _dilate_tch(lum_peak, 2) - _erode_tch(lum_peak, 2)
    has_crisp_edge_nearby = _dilate_tch((crisp_grad >= 0.14).to(torch.float32), max(5, int(round(1.45 * ppd))))
    tex_act_wide = torch.clamp((tex_wide - 0.09) / 0.18, 0.0, 1.0) * has_crisp_edge_nearby
    tex_act_in = torch.maximum(tex_act_fine, tex_act_wide)
    lum_act = torch.clamp((lum_peak - 0.12) / 0.24, 0.0, 1.0)

    swing_max = guide_swing_rgb.amax(dim=1, keepdim=True)
    swing_act = torch.clamp((swing_max - 0.06) / 0.16, 0.0, 1.0)

    flash_flag = is_flash_frame.view(bsz, 1, 1, 1).to(torch.float32)
    act_in = flash_flag * torch.maximum(swing_act, 0.75 * lum_act) + (1.0 - flash_flag) * tex_act_in
    # Outward reach: expand into crisp/wide pattern bars on radial/grating boundaries, or active flash swing,
    # gated tightly by `tex_act_fine` on natural scenes unless crisp high-contrast edges surround the pixel.
    crisp_wide_gate = _erode_tch(_dilate_tch((crisp_grad >= 0.22).to(torch.float32), max(5, int(round(1.50 * ppd)))), max(5, int(round(1.50 * ppd))))
    act_out = flash_flag * swing_act + (1.0 - flash_flag) * torch.maximum(tex_act_fine, tex_act_wide * crisp_wide_gate)

    r_reach = max(1, int(round(max(0.0, float(mask_cfg.guided_reach_deg)) * ppd)))
    dil_core = _dilate_tch(core_f, r_reach)
    seed = torch.maximum(core_f * (0.22 + 0.78 * act_in), dil_core * (0.85 * act_out))

    sig_pre = max(1.5, float(mask_cfg.guided_pre_sigma_deg) * ppd)
    seed_soft = _gaussian_blur_tch(seed, sig_pre)

    guide_feat = flash_flag * (0.65 * struct_rgb + 0.35 * guide_swing_rgb) + (1.0 - flash_flag) * (
        0.75 * struct_rgb + 0.25 * tex_act_fine
    )
    r_gf = max(4, int(round(max(0.25, float(mask_cfg.guided_radius_deg)) * ppd)))
    eps = max(1e-5, float(mask_cfg.guided_eps))
    gf = _color_guided_filter_tch(guide_feat, seed_soft, r=r_gf, eps=eps)

    sig_post = max(1.5, float(mask_cfg.guided_post_sigma_deg) * ppd)
    smooth = _gaussian_blur_tch(gf, sig_post)

    knee = max(0.12, float(mask_cfg.guided_knee))
    w_out = _quintic_smoothstep(smooth / knee).squeeze(1)

    if is_full.any():
        w_out[is_full] = 1.0
    if is_empty.any():
        w_out[is_empty] = 0.0
    return w_out


@torch.no_grad()
def apply_outward_spatial_feather_batch(
    core_masks_bool: torch.Tensor,
    geometry: ViewingGeometry,
    mask_cfg: PartialMaskConfig = PartialMaskConfig(),
) -> torch.Tensor:
    """Apply spatial smoothing to `(B, H, W)` boolean core masks when no image guide is supplied."""
    if core_masks_bool.numel() == 0:
        return core_masks_bool.to(torch.float32)

    is_full = core_masks_bool.flatten(1).all(dim=1)
    is_empty = ~core_masks_bool.flatten(1).any(dim=1)
    core_f = core_masks_bool.to(torch.float32)
    x = core_f.unsqueeze(1)  # (B, 1, H, W)
    ppd = float(geometry.px_per_deg_ring)

    if mask_cfg.borderless_guided:
        sig = max(2.0, (float(mask_cfg.guided_pre_sigma_deg) + float(mask_cfg.guided_post_sigma_deg)) * 0.65 * ppd)
        smooth = _gaussian_blur_tch(x, sig)
        w_out = _quintic_smoothstep(smooth / max(0.15, float(mask_cfg.guided_knee))).squeeze(1)
        if is_full.any():
            w_out[is_full] = 1.0
        if is_empty.any():
            w_out[is_empty] = 0.0
        return w_out

    pad_core = max(0, int(mask_cfg.core_pad_px))
    feather_px = max(0.0, float(mask_cfg.feather_angle_deg) * ppd)
    r_box = max(1, int(round(feather_px / 6.0))) if feather_px > 0.0 else 0

    r_dilate = 3 * r_box + pad_core
    if r_dilate > 0:
        x = _dilate_tch(x, r_dilate)
    if r_box > 0:
        for _ in range(3):
            x = _box_blur_tch(x, r_box)

    soft = torch.clamp(torch.maximum(core_f, x.squeeze(1)), 0.0, 1.0)
    return soft


def build_soft_weight_masks_u8(
    packed_core: np.ndarray,
    active_frame_mask: np.ndarray,
    geometry: ViewingGeometry,
    fps: float,
    mask_cfg: PartialMaskConfig = PartialMaskConfig(),
    *,
    guide_rgb_u8: np.ndarray | None = None,
    flash_frame_mask: np.ndarray | None = None,
    device: str | torch.device | None = None,
    schedule_mode: PartialTemporalScheduleMode = "base",
    hold_run_tails: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    """Construct `(T, ana_h, ana_w)` `uint8` (`[0..255]`) soft masks and `(T,)` `active_mask_area`.

    Combines:
      1. Structure-guided C2 borderless spatial weighting (`1-b + 3-a`) when `borderless_guided=True`
         and `guide_rgb_u8` is available (or C2 smooth fallback otherwise),
      2. Gentle intra-run temporal stabilization across non-fullscreen active frames,
      3. `temporal_fade_sec` (`0.20s`) cosine temporal envelope smoothing at mask entry/exit,
      4. Optional `hold_run_tails` extension aligned with Formula A `base` / `hold` / `safe` release tails.
    """
    n_frames = packed_core.shape[0]
    ana_h, ana_w = geometry.height, geometry.width
    spatial_u8 = np.zeros((n_frames, ana_h, ana_w), dtype=np.uint8)
    if n_frames == 0:
        return spatial_u8, np.zeros(0, dtype=np.float32)

    active_indices = np.flatnonzero(active_frame_mask)
    if len(active_indices) == 0:
        return spatial_u8, np.zeros(n_frames, dtype=np.float32)

    clean_fps = float(fps) if (fps > 0.0 and math.isfinite(fps)) else 30.0
    r_win = max(2, int(round(mask_cfg.temporal_window_sec * clean_fps)))
    dev = resolve_device(device)
    batch_sz = max(1, int(mask_cfg.feather_batch_size))
    use_guided = bool(mask_cfg.borderless_guided) and (guide_rgb_u8 is not None) and (len(guide_rgb_u8) == n_frames)

    flash_flags = (
        np.asarray(flash_frame_mask, dtype=bool)
        if (flash_frame_mask is not None and len(flash_frame_mask) == n_frames)
        else np.zeros(n_frames, dtype=bool)
    )

    for start_i in range(0, len(active_indices), batch_sz):
        batch_idx = active_indices[start_i : start_i + batch_sz]
        core_np = unpack_bool_masks_3d(packed_core[batch_idx], ana_w)
        core_t = torch.from_numpy(core_np).to(dev)

        if use_guided:
            assert guide_rgb_u8 is not None
            b_len = len(batch_idx)
            gh, gw = guide_rgb_u8.shape[1], guide_rgb_u8.shape[2]
            peak_np = np.empty((b_len, gh, gw, 3), dtype=np.uint8)
            swing_np = np.empty((b_len, gh, gw, 3), dtype=np.uint8)
            b_flash = flash_flags[batch_idx]

            for k_local, f_idx in enumerate(batch_idx):
                if b_flash[k_local]:
                    lo_w = max(0, int(f_idx) - r_win)
                    hi_w = min(n_frames, int(f_idx) + r_win + 1)
                    win_slice = guide_rgb_u8[lo_w:hi_w]
                    p_max = win_slice.max(axis=0)
                    p_min = win_slice.min(axis=0)
                    peak_np[k_local] = p_max
                    swing_np[k_local] = p_max - p_min
                else:
                    peak_np[k_local] = guide_rgb_u8[f_idx]
                    swing_np[k_local] = 0

            peak_t = (
                torch.from_numpy(peak_np).to(dev, dtype=torch.float32).permute(0, 3, 1, 2)
                * (1.0 / 255.0)
            )
            swing_t = (
                torch.from_numpy(swing_np).to(dev, dtype=torch.float32).permute(0, 3, 1, 2)
                * (1.0 / 255.0)
            )
            if (gh, gw) != (ana_h, ana_w):
                peak_t = F.interpolate(peak_t, size=(ana_h, ana_w), mode="bilinear", align_corners=False)
                swing_t = F.interpolate(swing_t, size=(ana_h, ana_w), mode="bilinear", align_corners=False)

            flash_t = torch.from_numpy(b_flash).to(dev)
            soft_t = apply_guided_borderless_batch(
                core_t,
                peak_t,
                swing_t,
                flash_t,
                geometry=geometry,
                mask_cfg=mask_cfg,
            )
            u8_t = torch.clamp(torch.round(soft_t * 255.0), 0.0, 255.0).to(torch.uint8)
        else:
            soft_t = apply_outward_spatial_feather_batch(core_t, geometry=geometry, mask_cfg=mask_cfg)
            u8_t = torch.clamp(torch.round(soft_t * 255.0), 0.0, 255.0).to(torch.uint8)
            if not mask_cfg.borderless_guided:
                u8_t = torch.where(core_t, torch.full_like(u8_t, 255), u8_t)

        spatial_u8[batch_idx] = u8_t.cpu().numpy()

    # Gentle intra-run temporal stabilization (`[0.22, 0.56, 0.22]`) across adjacent partial active frames
    # while preserving exact `255` full-screen cuts (`min == 255`) at shot boundaries.
    if mask_cfg.borderless_guided and int(mask_cfg.temporal_smooth_radius) > 0:
        is_full_frame = np.zeros(n_frames, dtype=bool)
        is_full_frame[active_indices] = spatial_u8[active_indices].min(axis=(1, 2)) == 255
        partial_active = active_frame_mask & (~is_full_frame)
        for s_r, e_r in mask_to_spans(partial_active):
            if e_r - s_r + 1 < 3:
                continue
            seg_f = spatial_u8[s_r : e_r + 1].astype(np.float32)
            for _ in range(int(mask_cfg.temporal_smooth_radius)):
                prev_f = np.concatenate([seg_f[:1], seg_f[:-1]], axis=0)
                next_f = np.concatenate([seg_f[1:], seg_f[-1:]], axis=0)
                seg_f = 0.22 * prev_f + 0.56 * seg_f + 0.22 * next_f
            spatial_u8[s_r : e_r + 1] = np.clip(np.rint(seg_f), 0.0, 255.0).astype(np.uint8)

    fade_frames = (
        max(3, int(round(mask_cfg.temporal_fade_sec * clean_fps)))
        if mask_cfg.temporal_fade_sec > 0.0
        else 0
    )

    out_u8 = spatial_u8.copy()
    inactive_mask = ~active_frame_mask

    # 4-2A: Apply `0.20s` cosine temporal ramp strictly on outside-run transition frames
    # (`~active_frame_mask`) when `not hold_run_tails`, avoiding moving ghost trails inside active runs.
    if fade_frames > 0 and not hold_run_tails and np.any(inactive_mask):
        steps = np.arange(1, fade_frames + 1, dtype=np.float64)
        cos_weights = 0.5 * (1.0 + np.cos(math.pi * steps / float(fade_frames + 1)))

        for d, cw in enumerate(cos_weights, start=1):
            w_scale = float(cw)
            if w_scale <= 0.0:
                continue
            dst_fwd = inactive_mask[d:]
            if np.any(dst_fwd):
                scaled_fwd = np.rint(spatial_u8[:-d][dst_fwd].astype(np.float32) * w_scale).astype(np.uint8)
                out_u8[d:][dst_fwd] = np.maximum(out_u8[d:][dst_fwd], scaled_fwd)

            dst_bwd = inactive_mask[:-d]
            if np.any(dst_bwd):
                scaled_bwd = np.rint(spatial_u8[d:][dst_bwd].astype(np.float32) * w_scale).astype(np.uint8)
                out_u8[:-d][dst_bwd] = np.maximum(out_u8[:-d][dst_bwd], scaled_bwd)

    # Run-boundary tail alignment for Formula A `M = 1 - W_t * (1 - alpha_t)`
    if hold_run_tails and fade_frames > 0:
        runs = mask_to_spans(active_frame_mask)
        if schedule_mode in ("hold", "safe"):
            hold_frames = max(1, int(round(mask_cfg.half_hold_duration_sec * clean_fps)))
            exit_span = 2 * fade_frames + hold_frames
        else:
            exit_span = fade_frames

        for s_run, e_run in runs:
            start_anchor = spatial_u8[s_run]
            if np.any(start_anchor):
                entry_start = max(0, s_run - fade_frames)
                if entry_start < s_run:
                    np.maximum(
                        out_u8[entry_start:s_run],
                        start_anchor[None, :, :],
                        out=out_u8[entry_start:s_run],
                    )

            end_anchor = spatial_u8[e_run]
            if np.any(end_anchor):
                tail_end = min(n_frames, e_run + exit_span + 1)
                if e_run + 1 < tail_end:
                    np.maximum(
                        out_u8[e_run + 1 : tail_end],
                        end_anchor[None, :, :],
                        out=out_u8[e_run + 1 : tail_end],
                    )

    active_area = (out_u8.astype(np.float64).mean(axis=(1, 2)) / 255.0).astype(np.float32)
    return out_u8, active_area


def compute_4metrics_union_from_result(
    res: PartialVideoAnalysisResult,
    metric_threshold: float = 50.0,
    nomura_threshold: float = 0.50,
) -> np.ndarray:
    """Evaluate `(calibrated >= 50) | (ppr >= 50) | (nomura >= 0.50) | (vfm >= 50)` boolean frame mask."""
    df = res.frame_metrics_df
    if len(df) == 0:
        return np.zeros(0, dtype=bool)
    return (
        (df["risk_calibrated"].to_numpy(dtype=np.float64) >= metric_threshold)
        | (df["risk_ppr_pct"].to_numpy(dtype=np.float64) >= metric_threshold)
        | (df["risk_nomura"].to_numpy(dtype=np.float64) >= nomura_threshold)
        | (df["risk_vfm_weibull"].to_numpy(dtype=np.float64) >= metric_threshold)
    )


def build_mode_weight_masks(
    res: PartialVideoAnalysisResult,
    mode: PartialTemporalScheduleMode = "base",
    *,
    hold_run_tails: bool = True,
    mask_config: PartialMaskConfig | None = None,
    device: str | torch.device | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Build mode-matched `(weight_masks_u8, active_mask_area)` for `base`, `hold`, or `safe` from Pass 1."""
    if res.packed_hazards is None:
        raise ValueError(
            "`res.packed_hazards` is None; run `analyze_video(..., collect_masks=True)` in Pass 1."
        )

    cfg = mask_config or res.mask_config
    in_confirmed = res.frame_metrics_df["is_dangerous"].to_numpy(dtype=bool)

    if mode == "safe":
        over4 = compute_4metrics_union_from_result(
            res,
            metric_threshold=cfg.safe_metric_threshold,
            nomura_threshold=cfg.safe_nomura_threshold,
        )
        active_frames = in_confirmed | over4
        packed = res.packed_hazards.safe_core_packed
        flash_mask = res.packed_hazards.safe_flash_frame_mask
    else:
        active_frames = in_confirmed
        packed = res.packed_hazards.confirmed_core_packed
        flash_mask = res.packed_hazards.flash_frame_mask

    return build_soft_weight_masks_u8(
        packed_core=packed,
        active_frame_mask=active_frames,
        geometry=res.geometry,
        fps=res.fps,
        mask_cfg=cfg,
        guide_rgb_u8=res.packed_hazards.guide_rgb_u8,
        flash_frame_mask=flash_mask,
        device=device or res.device,
        schedule_mode=mode,
        hold_run_tails=hold_run_tails,
    )


def upsample_weight_mask(
    mask_u8: np.ndarray,
    target_width: int,
    target_height: int,
) -> np.ndarray:
    """Upsample `(ana_h, ana_w)` `uint8` `[0..255]` mask to `(target_height, target_width)` `float32` `[0, 1]`."""
    mx = int(mask_u8.max())
    if mx == 0:
        return np.zeros((target_height, target_width), dtype=np.float32)
    if int(mask_u8.min()) == 255:
        return np.ones((target_height, target_width), dtype=np.float32)
    if mask_u8.shape == (target_height, target_width):
        return mask_u8.astype(np.float32) * (1.0 / 255.0)
    up_u8 = cv2.resize(mask_u8, (target_width, target_height), interpolation=cv2.INTER_LINEAR)
    return up_u8.astype(np.float32) * (1.0 / 255.0)


def apply_partial_linear_dimming_bgr(
    bgr_u8: np.ndarray,
    alpha_lin: float,
    mask_u8: np.ndarray,
) -> np.ndarray:
    """Apply Formula A `M(y, x) = 1 - W(y, x) * (1 - alpha_lin)` in exact IEC 61966-2-1 linear light."""
    a = float(np.clip(alpha_lin, 0.0, 1.0))
    if a >= 0.9999:
        return bgr_u8
    mx = int(mask_u8.max())
    if mx == 0:
        return bgr_u8

    h, w = bgr_u8.shape[:2]
    if int(mask_u8.min()) == 255:
        # Uniform full-frame fast path via 256-entry sRGB LUT
        c_lin_dim = np.clip(_SRGB_TO_LIN_F32 * a, 0.0, 1.0)
        idx16 = np.rint(c_lin_dim * 65535.0).astype(np.int32)
        lut_u8 = _LIN16_TO_SRGB_U8[idx16]
        return cv2.LUT(bgr_u8, lut_u8)

    w_map = upsample_weight_mask(mask_u8, target_width=w, target_height=h)
    mult = (1.0 - w_map * (1.0 - a))[:, :, None]  # (H, W, 1) float32
    bgr_lin = _SRGB_TO_LIN_F32[bgr_u8]
    dim_lin = np.clip(bgr_lin * mult, 0.0, 1.0)
    idx16 = np.rint(dim_lin * 65535.0).astype(np.int32)
    return _LIN16_TO_SRGB_U8[idx16]


__all__ = [
    "pack_bool_mask_2d",
    "unpack_bool_masks_3d",
    "temporal_sliding_or_packed",
    "temporal_bridge_spans_packed",
    "clean_speckles_packed",
    "apply_guided_borderless_batch",
    "apply_outward_spatial_feather_batch",
    "build_soft_weight_masks_u8",
    "compute_4metrics_union_from_result",
    "build_mode_weight_masks",
    "upsample_weight_mask",
    "apply_partial_linear_dimming_bgr",
]
