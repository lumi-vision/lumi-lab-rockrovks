"""LUMI B2 Full-Screen Risk-Proportional + Red-Channel Separated Dimming Core Engine (`fullscreen_coreops_1112.py`).

Implements Prototype B2 on top of Stage 4 `vdcore`:
  - Channel-separated linear RGB transformation:
      `(G'_lin, B'_lin) = alpha_t * (G_lin, B_lin)`
      `R'_lin           = (alpha_t * beta_R,t) * R_lin = alpha_R,t * R_lin`
  - Brightness attenuation strength (Method B, `안 B-1` + `안 2-A`):
      Normalized linear interpolation over `risk_calibrated` (`R_t in [50, 100]`) across all active
      hazard intervals (`in_danger_any`):
        `s_t = clip((R_t - 50.0) / 50.0, 0.0, 1.0)`
        `alpha_risk[t] = alpha_at_50 - s_t * (alpha_at_50 - alpha_at_100)`
      with defaults `(alpha_at_50, alpha_at_100) = (0.50, 0.09)`.
  - Red-flash interval handling (Choice 2, `안 1-A` hybrid additional `R` attenuation):
      Preserves `G, B` at the `risk_calibrated`-proportional brightness `alpha_t` while applying
      additional multiplier `beta_R,t` to `R` on active red-hazard frames (`in_red_danger`):
        * `red_mode='hybrid'` (default, Method 1-B): combines saturation-ratio escape
          (`red_ratio' <= 0.72 < 0.80`, `gamma_ratio = alpha_t * q24_red_escape_beta`) and
          Method A excess ceiling (`R'_lin - G'_lin - B'_lin <= 0.058 < 0.0625`,
          `gamma_abs = 0.058 / q76_sat_red_excess`), choosing `max(gamma_ratio, gamma_abs)`
          and factoring out `alpha_t` (`beta_R = min(alpha_t, req_alpha_R) / alpha_t`).
        * `red_mode='abs'` (Method 1-A): uses `gamma_abs` strictly.
  - Temporal scheduling modes (`base`, `hold`, `safe`, unified with `A1`/`A2`/`B1`):
      * `base`: Confirmed 7-category PSE hazard intervals + symmetric `0.20s` cosine Fade-in/out.
      * `hold`: Confirmed 7-category PSE hazard intervals + 2-stage Half-Hold release on both
                `alpha_t` and `beta_R,t` (`0.20s` ramp to half -> `0.50s` hold -> `0.20s` full release).
      * `safe`: `(confirmed intervals) U (4 continuous risk metrics >= 50% union)`
                (`risk_calibrated >= 50 | risk_ppr_pct >= 50 | risk_nomura >= 0.5 | risk_vfm_weibull >= 50`)
                with 2-stage Half-Hold release and red suppression on active `sat_red_area > 25%` frames.
  - Original-resolution (`orig_width x orig_height`) `.mp4` encoding via a 3-channel `(1, 256, 3)`
    IEC 61966-2-1 sRGB `uint8` LUT paired with single-pass streaming `vdcore` closed-loop verification.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from pathlib import Path
import sys
from typing import Iterable, Literal

import cv2
import numpy as np
import torch

_THIS_DIR = Path(__file__).resolve().parent
if str(_THIS_DIR) not in sys.path:
    sys.path.insert(0, str(_THIS_DIR))

from vdcorefs import (
    DetectorCache,
    LABEL_RED_EXTENDED,
    LABEL_RED_FLASH,
    PSEConfig,
    VideoAnalysisResult,
    ViewingConfig,
    analyze_frame_stream,
    analyze_video,
    mask_to_spans,
)

B2Mode = Literal["base", "hold", "safe"]
RedMitigationMode = Literal["hybrid", "abs"]
VALID_B2_MODES: tuple[B2Mode, ...] = ("base", "hold", "safe")
VALID_RED_MODES: tuple[RedMitigationMode, ...] = ("hybrid", "abs")

# Default quantitative mitigation constants for B2 (Method B + Choice 2)
RISK_CALIBRATED_THRESHOLD: float = 50.0
RISK_CALIBRATED_MAX: float = 100.0
ALPHA_AT_RISK_50: float = 0.50         # Linear multiplier at R_t = 50.0 (sRGB ~75.8%)
ALPHA_AT_RISK_100: float = 0.09        # Linear multiplier at R_t = 100.0 (200 -> 18 cd/m^2, sRGB ~36.7%)
TARGET_SAFE_RED_EXCESS: float = 0.058  # < 0.0625 (#470000); quantizes to R <= 70 in 8-bit sRGB
TARGET_SAFE_RED_RATIO: float = 0.72    # < 0.80 saturation ratio threshold
MIN_ALPHA_FLOOR: float = 0.005
FADE_DURATION_SEC: float = 0.20
HALF_HOLD_DURATION_SEC: float = 0.50
SAFE_METRIC_THRESHOLD: float = 50.0    # for [0, 100] metrics (calibrated, ppr_pct, vfm_weibull)
SAFE_NOMURA_THRESHOLD: float = 0.50    # for raw risk_nomura e(t) in [0, 1]

# Precomputed exact 256-entry IEC 61966-2-1 8-bit sRGB -> linear light table (float64)
_SRGB_GRID_F64 = np.arange(256, dtype=np.float64) / 255.0
_SRGB_TO_LIN_F64 = np.where(
    _SRGB_GRID_F64 <= 0.04045,
    _SRGB_GRID_F64 / 12.92,
    ((_SRGB_GRID_F64 + 0.055) / 1.055) ** 2.4,
)


@dataclass(frozen=True, slots=True)
class B2Config:
    """Configuration parameters for B2 `risk_calibrated`-proportional + red-channel separated dimming."""

    mode: B2Mode = "base"
    red_mode: RedMitigationMode = "hybrid"
    alpha_at_risk_50: float = ALPHA_AT_RISK_50
    alpha_at_risk_100: float = ALPHA_AT_RISK_100
    risk_threshold: float = RISK_CALIBRATED_THRESHOLD
    risk_max: float = RISK_CALIBRATED_MAX
    target_safe_red_excess: float = TARGET_SAFE_RED_EXCESS
    target_safe_red_ratio: float = TARGET_SAFE_RED_RATIO
    min_alpha_floor: float = MIN_ALPHA_FLOOR
    fade_duration_sec: float = FADE_DURATION_SEC
    half_hold_duration_sec: float = HALF_HOLD_DURATION_SEC
    safe_metric_threshold: float = SAFE_METRIC_THRESHOLD
    safe_nomura_threshold: float = SAFE_NOMURA_THRESHOLD
    fourcc: str = "mp4v"


@dataclass(slots=True)
class B2DimmingResult:
    """Complete before/after verification result from `dim_video_fullscreen_b2`."""

    mode: B2Mode
    red_mode: RedMitigationMode
    config: B2Config
    input_video_path: str
    output_video_path: str
    orig_width: int
    orig_height: int
    analysis_width: int
    analysis_height: int
    fps: float
    total_frames: int
    before: VideoAnalysisResult
    after: VideoAnalysisResult
    alpha_schedule: np.ndarray
    raw_alpha_target: np.ndarray
    beta_red_schedule: np.ndarray
    alpha_red_schedule: np.ndarray
    in_danger_mask: np.ndarray
    in_red_mask: np.ndarray
    extra_safe_runs: list[tuple[int, int]]
    over_4metrics_before: np.ndarray
    over_4metrics_after: np.ndarray


def build_srgb_dimming_lut(alpha_lin: float) -> np.ndarray:
    """Build an exact 256-entry uint8 LUT applying `c_lin' = alpha_lin * c_lin` in linear light."""
    a = float(np.clip(alpha_lin, 0.0, 1.0))
    if a >= 0.9999:
        return np.arange(256, dtype=np.uint8)

    c_lin_dim = np.clip(_SRGB_TO_LIN_F64 * a, 0.0, 1.0)
    c_s_dim = np.where(
        c_lin_dim <= 0.0031308,
        c_lin_dim * 12.92,
        1.055 * np.power(np.maximum(c_lin_dim, 1e-12), 1.0 / 2.4) - 0.055,
    )
    return np.clip(np.rint(c_s_dim * 255.0), 0.0, 255.0).astype(np.uint8)


def build_b2_bgr_lut(alpha_lin: float, beta_red_lin: float = 1.0) -> np.ndarray:
    """Build an exact `(1, 256, 3)` uint8 BGR LUT (`B, G` -> `alpha_lin`, `R` -> `alpha_lin * beta_red_lin`)."""
    a_gb = float(np.clip(alpha_lin, 0.0, 1.0))
    b_r = float(np.clip(beta_red_lin, 0.0, 1.0))
    lut_gb = build_srgb_dimming_lut(a_gb)
    lut_r = lut_gb if b_r >= 0.9999 else build_srgb_dimming_lut(a_gb * b_r)

    lut_bgr = np.empty((1, 256, 3), dtype=np.uint8)
    lut_bgr[0, :, 0] = lut_gb
    lut_bgr[0, :, 1] = lut_gb
    lut_bgr[0, :, 2] = lut_r
    return lut_bgr


def apply_linear_dimming_b2_bgr(
    bgr_u8: np.ndarray,
    alpha_lin: float,
    beta_red_lin: float = 1.0,
) -> np.ndarray:
    """Apply channel-separated linear dimming (`alpha_lin` on G/B, `alpha_lin * beta_red_lin` on R) to BGR `uint8`."""
    if alpha_lin >= 0.9999 and beta_red_lin >= 0.9999:
        return bgr_u8
    if beta_red_lin >= 0.9999:
        return cv2.LUT(bgr_u8, build_srgb_dimming_lut(alpha_lin))
    return cv2.LUT(bgr_u8, build_b2_bgr_lut(alpha_lin, beta_red_lin))


@torch.no_grad()
def apply_linear_dimming_b2_srgb_tensor(
    bgr_or_rgb_255: torch.Tensor,
    alpha_lin: float,
    beta_red_lin: float = 1.0,
    *,
    channel_order: Literal["bgr", "rgb"] = "bgr",
) -> torch.Tensor:
    """Tensor helper applying exact IEC 61966-2-1 B2 channel-separated dimming on `[0, 255]` tensors."""
    if alpha_lin >= 0.9999 and beta_red_lin >= 0.9999:
        return bgr_or_rgb_255

    c_s = torch.clamp(bgr_or_rgb_255.to(torch.float32) / 255.0, 0.0, 1.0)
    c_lin = torch.where(
        c_s <= 0.04045,
        c_s / 12.92,
        torch.pow((c_s + 0.055) / 1.055, 2.4),
    )

    a_gb = float(np.clip(alpha_lin, 0.0, 1.0))
    a_r = a_gb * float(np.clip(beta_red_lin, 0.0, 1.0))
    c_lin_dim = torch.empty_like(c_lin)
    if channel_order == "bgr":
        c_lin_dim[..., 0] = torch.clamp(c_lin[..., 0] * a_gb, 0.0, 1.0)
        c_lin_dim[..., 1] = torch.clamp(c_lin[..., 1] * a_gb, 0.0, 1.0)
        c_lin_dim[..., 2] = torch.clamp(c_lin[..., 2] * a_r, 0.0, 1.0)
    else:
        c_lin_dim[..., 0] = torch.clamp(c_lin[..., 0] * a_r, 0.0, 1.0)
        c_lin_dim[..., 1] = torch.clamp(c_lin[..., 1] * a_gb, 0.0, 1.0)
        c_lin_dim[..., 2] = torch.clamp(c_lin[..., 2] * a_gb, 0.0, 1.0)

    c_s_dim = torch.where(
        c_lin_dim <= 0.0031308,
        c_lin_dim * 12.92,
        1.055 * torch.pow(torch.clamp(c_lin_dim, min=1e-8), 1.0 / 2.4) - 0.055,
    )
    out = torch.clamp(torch.round(c_s_dim * 255.0), 0.0, 255.0)
    return out.to(torch.uint8) if bgr_or_rgb_255.dtype == torch.uint8 else out


def compute_4metrics_union_mask(
    res: VideoAnalysisResult,
    metric_threshold: float = SAFE_METRIC_THRESHOLD,
    nomura_threshold: float = SAFE_NOMURA_THRESHOLD,
) -> np.ndarray:
    """Return boolean frame mask `(calibrated >= 50) | (ppr >= 50) | (nomura >= 0.5) | (vfm >= 50)`."""
    df = res.frame_metrics_df
    if len(df) == 0:
        return np.zeros(0, dtype=bool)
    return (
        (df["risk_calibrated"].to_numpy(dtype=np.float64) >= metric_threshold)
        | (df["risk_ppr_pct"].to_numpy(dtype=np.float64) >= metric_threshold)
        | (df["risk_nomura"].to_numpy(dtype=np.float64) >= nomura_threshold)
        | (df["risk_vfm_weibull"].to_numpy(dtype=np.float64) >= metric_threshold)
    )


def compute_risk_proportional_alpha(
    risk_calibrated: np.ndarray,
    *,
    alpha_at_50: float = ALPHA_AT_RISK_50,
    alpha_at_100: float = ALPHA_AT_RISK_100,
    risk_threshold: float = RISK_CALIBRATED_THRESHOLD,
    risk_max: float = RISK_CALIBRATED_MAX,
) -> np.ndarray:
    """Compute Method B (`안 B-1`) linear brightness multiplier from `risk_calibrated`.

    Formula:
        `s_t = clip((R_t - risk_threshold) / (risk_max - risk_threshold), 0.0, 1.0)`
        `alpha_risk[t] = alpha_at_50 - s_t * (alpha_at_50 - alpha_at_100)`
    """
    span = max(float(risk_max) - float(risk_threshold), 1e-6)
    r_vals = np.asarray(risk_calibrated, dtype=np.float64)
    s_norm = np.clip((r_vals - float(risk_threshold)) / span, 0.0, 1.0)
    return float(alpha_at_50) - s_norm * (float(alpha_at_50) - float(alpha_at_100))


def _stabilize_in_run_envelope(
    raw_targets: np.ndarray,
    runs: list[tuple[int, int]],
    fps: float,
) -> np.ndarray:
    """Sliding minimum envelope (`+-0.25s`) + odd-kernel edge moving average (`0.30s`) + safety re-clamp."""
    n_frames = len(raw_targets)
    stabilized = np.ones(n_frames, dtype=np.float64)
    win_radius = max(2, int(round(0.25 * fps)))
    k_base = max(3, int(round(0.30 * fps)) | 1)

    for s_run, e_run in runs:
        run_targets = raw_targets[s_run : e_run + 1]
        run_len = len(run_targets)
        env = np.empty(run_len, dtype=np.float64)
        for idx_local in range(run_len):
            w_s = max(0, idx_local - win_radius)
            w_e = min(run_len, idx_local + win_radius + 1)
            env[idx_local] = float(np.min(run_targets[w_s:w_e]))

        if run_len >= 3:
            k_smooth = min(run_len, k_base)
            if k_smooth % 2 == 0:
                k_smooth = max(3, k_smooth - 1) if k_smooth > 3 else 1
            if k_smooth >= 3:
                pad_s = k_smooth // 2
                padded = np.pad(env, (pad_s, pad_s), mode="edge")
                kernel = np.ones(k_smooth, dtype=np.float64) / float(k_smooth)
                smoothed_env = np.convolve(padded, kernel, mode="valid")[:run_len]
                final_run_val = np.minimum(smoothed_env, run_targets)
            else:
                final_run_val = env
        else:
            final_run_val = env

        stabilized[s_run : e_run + 1] = final_run_val

    return stabilized


def _apply_symmetric_cosine_fade(
    stabilized_schedule: np.ndarray,
    runs: list[tuple[int, int]],
    fps: float,
    fade_duration_sec: float,
) -> np.ndarray:
    """Base mode boundary transition: symmetric `0.20s` cosine Fade-in and Fade-out to `1.0`."""
    n_frames = len(stabilized_schedule)
    fade_frames = max(3, int(round(fade_duration_sec * fps)))
    out = np.copy(stabilized_schedule)

    steps = np.arange(1, fade_frames + 1, dtype=np.float64)
    cosine_w = 0.5 * (1.0 - np.cos(math.pi * steps / float(fade_frames + 1)))

    for s_run, e_run in runs:
        start_val = float(stabilized_schedule[s_run])
        end_val = float(stabilized_schedule[e_run])

        for step_idx, w in enumerate(cosine_w, start=1):
            f_in = s_run - step_idx
            if f_in >= 0:
                ramp_in = start_val + (1.0 - start_val) * float(w)
                out[f_in] = min(out[f_in], ramp_in)

            f_out = e_run + step_idx
            if f_out < n_frames:
                ramp_out = end_val + (1.0 - end_val) * float(w)
                out[f_out] = min(out[f_out], ramp_out)

    return out


def _apply_half_hold_cosine_fade(
    stabilized_schedule: np.ndarray,
    runs: list[tuple[int, int]],
    fps: float,
    fade_duration_sec: float,
    half_hold_duration_sec: float,
) -> np.ndarray:
    """Hold/Safe mode boundary transition: 2-stage Half-Hold release + smooth anchor Fade-in."""
    n_frames = len(stabilized_schedule)
    fade_frames = max(3, int(round(fade_duration_sec * fps)))
    hold_frames = max(1, int(round(half_hold_duration_sec * fps)))
    out = np.copy(stabilized_schedule)

    steps = np.arange(1, fade_frames + 1, dtype=np.float64)
    cosine_w = 0.5 * (1.0 - np.cos(math.pi * steps / float(fade_frames + 1)))

    # (A) Exit 2-stage release: end_val -> val_half -> hold -> 1.0
    for _, e_run in runs:
        end_val = float(stabilized_schedule[e_run])
        val_half = 0.5 * (end_val + 1.0)

        for step_idx, w in enumerate(cosine_w, start=1):
            f_idx = e_run + step_idx
            if f_idx >= n_frames:
                break
            ramp1 = end_val + (val_half - end_val) * float(w)
            out[f_idx] = min(out[f_idx], ramp1)

        hold_start = e_run + fade_frames + 1
        hold_end = e_run + fade_frames + hold_frames
        if hold_start < n_frames:
            sl_end = min(n_frames, hold_end + 1)
            out[hold_start:sl_end] = np.minimum(out[hold_start:sl_end], val_half)

        for step_idx, w in enumerate(cosine_w, start=1):
            f_idx = hold_end + step_idx
            if f_idx >= n_frames:
                break
            ramp2 = val_half + (1.0 - val_half) * float(w)
            out[f_idx] = min(out[f_idx], ramp2)

    # (B) Entry Fade-in anchored to current relaxed level (prevents upward jump before re-entry)
    for s_run, _ in runs:
        start_val = float(stabilized_schedule[s_run])
        anchor_idx = max(0, s_run - fade_frames - 1)
        entry_from_val = max(start_val, float(out[anchor_idx]))

        for step_idx, w in enumerate(cosine_w, start=1):
            f_idx = s_run - step_idx
            if f_idx < 0:
                break
            ramp_in = start_val + (entry_from_val - start_val) * float(w)
            out[f_idx] = min(out[f_idx], ramp_in)

    return out


def _smooth_runs_by_mode(
    raw_targets: np.ndarray,
    runs: list[tuple[int, int]],
    fps: float,
    cfg: B2Config,
) -> np.ndarray:
    """Stabilize in-interval targets and apply mode-specific (`base` vs `hold`/`safe`) boundary fade."""
    if not runs:
        return np.ones(len(raw_targets), dtype=np.float64)
    stabilized = _stabilize_in_run_envelope(raw_targets, runs, fps)
    if cfg.mode == "base":
        return _apply_symmetric_cosine_fade(stabilized, runs, fps, cfg.fade_duration_sec)
    return _apply_half_hold_cosine_fade(
        stabilized, runs, fps, cfg.fade_duration_sec, cfg.half_hold_duration_sec
    )


def compute_alpha_and_red_schedule_b2(
    res_before: VideoAnalysisResult,
    cfg: B2Config = B2Config(),
    pse: PSEConfig = PSEConfig(),
) -> tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    list[tuple[int, int]],
    np.ndarray,
]:
    """Compute per-frame `(alpha_sched, raw_alpha_target, beta_red_sched, alpha_red_sched)` for B2.

    Returns:
        `(alpha_sched, raw_alpha_target, beta_red_sched, alpha_red_sched,
          in_danger_any, in_red_danger, extra_safe_runs, over_4metrics)`
    """
    if cfg.mode not in VALID_B2_MODES:
        raise ValueError(f"Unsupported B2 mode {cfg.mode!r}; expected one of {VALID_B2_MODES}.")
    if cfg.red_mode not in VALID_RED_MODES:
        raise ValueError(f"Unsupported red_mode {cfg.red_mode!r}; expected one of {VALID_RED_MODES}.")
    if res_before.physical_stats is None:
        raise ValueError("`res_before.physical_stats` is required (`collect_physical_stats=True`).")
    if not (0.0 < cfg.alpha_at_risk_100 <= cfg.alpha_at_risk_50 <= 1.0):
        raise ValueError(
            f"Expected `0 < alpha_at_risk_100 <= alpha_at_risk_50 <= 1`, got "
            f"({cfg.alpha_at_risk_50}, {cfg.alpha_at_risk_100})."
        )
    if cfg.risk_max <= cfg.risk_threshold:
        raise ValueError(
            f"Expected `risk_max > risk_threshold`, got ({cfg.risk_threshold}, {cfg.risk_max})."
        )

    n_frames = res_before.total_frames
    if n_frames == 0:
        empty_f = np.zeros(0, dtype=np.float32)
        empty_b = np.zeros(0, dtype=bool)
        return empty_f, empty_f, empty_f, empty_f, empty_b, empty_b, [], empty_b

    df_before = res_before.frame_metrics_df
    risk_cal = df_before["risk_calibrated"].to_numpy(dtype=np.float64)
    sat_red_area = res_before.physical_stats["sat_red_area"]
    q76_red_exc = res_before.physical_stats["q76_sat_red_excess"]
    q24_esc_beta = res_before.physical_stats["q24_red_escape_beta"]

    in_interval_danger = df_before["is_dangerous"].to_numpy(dtype=bool)
    over_4metrics = compute_4metrics_union_mask(
        res_before,
        metric_threshold=cfg.safe_metric_threshold,
        nomura_threshold=cfg.safe_nomura_threshold,
    )

    red_interval_mask = (
        res_before.frame_masks[LABEL_RED_FLASH] | res_before.frame_masks[LABEL_RED_EXTENDED]
    )

    if cfg.mode == "safe":
        in_danger_any = in_interval_danger | over_4metrics
        in_red_danger = red_interval_mask | (in_danger_any & (sat_red_area > pse.flash_area_min))
        extra_safe_runs = mask_to_spans(in_danger_any & (~in_interval_danger))
    else:
        in_danger_any = in_interval_danger.copy()
        in_red_danger = red_interval_mask.copy()
        extra_safe_runs = []

    # 1) Global screen (`G, B` & base luminance) schedule `alpha_sched` via Method B (`risk_calibrated` proportional)
    alpha_target = np.ones(n_frames, dtype=np.float64)
    active_idx = np.flatnonzero(in_danger_any)
    if len(active_idx) > 0:
        req_alpha = compute_risk_proportional_alpha(
            risk_cal[active_idx],
            alpha_at_50=cfg.alpha_at_risk_50,
            alpha_at_100=cfg.alpha_at_risk_100,
            risk_threshold=cfg.risk_threshold,
            risk_max=cfg.risk_max,
        )
        alpha_target[active_idx] = np.clip(req_alpha, cfg.min_alpha_floor, 1.0)

    danger_runs = mask_to_spans(in_danger_any)
    alpha_sched_f64 = _smooth_runs_by_mode(alpha_target, danger_runs, res_before.fps, cfg)

    # 2) Red-channel additional multiplier `beta_red_schedule` (Choice 2, `안 1-A`):
    #    Stabilize the required effective red ceiling `req_alpha_red` in-run FIRST, then factor out
    #    `alpha_sched_f64` (`beta_R = min(alpha_t, stabilized_red_ceiling) / alpha_t`).
    red_ceiling_target = np.ones(n_frames, dtype=np.float64)
    red_idx = np.flatnonzero(in_red_danger)
    if len(red_idx) > 0:
        q_exc_all = q76_red_exc[red_idx].astype(np.float64)
        need_red_sub = q_exc_all > cfg.target_safe_red_excess
        if np.any(need_red_sub):
            sub_frames = red_idx[need_red_sub]
            cur_alpha = np.maximum(alpha_sched_f64[sub_frames], 1e-6)
            q_exc = np.maximum(q_exc_all[need_red_sub], 1e-6)

            # (a) Method A saturated-red excess ceiling (`R'_lin - G'_lin - B'_lin <= 0.058 < 0.0625`)
            gamma_abs = np.minimum(1.0, cfg.target_safe_red_excess / q_exc)

            if cfg.red_mode == "hybrid":
                # (b) Saturation ratio escape (`red_ratio' <= target_safe_red_ratio < 0.80`)
                q24_vals = q24_esc_beta[sub_frames].astype(np.float64)
                if abs(cfg.target_safe_red_ratio - 0.72) > 1e-6:
                    r0 = float(np.clip(cfg.target_safe_red_ratio, 0.05, 0.79))
                    ratio_scale = (r0 / (1.0 - r0)) / (0.72 / 0.28)
                    q24_vals = np.where(q24_vals < 0.9999, np.clip(q24_vals * ratio_scale, 0.0, 1.0), 1.0)
                gamma_ratio = cur_alpha * q24_vals
                req_alpha_red = np.maximum(gamma_ratio, gamma_abs)
            else:
                req_alpha_red = gamma_abs

            red_ceiling_target[sub_frames] = np.clip(
                np.minimum(1.0, req_alpha_red), cfg.min_alpha_floor, 1.0
            )

    red_runs = mask_to_spans(in_red_danger)
    if not red_runs:
        beta_red_sched_f64 = np.ones(n_frames, dtype=np.float64)
    else:
        stabilized_red_ceiling = _stabilize_in_run_envelope(red_ceiling_target, red_runs, res_before.fps)
        beta_red_in_run = np.ones(n_frames, dtype=np.float64)
        beta_red_in_run[red_idx] = np.clip(
            np.minimum(alpha_sched_f64[red_idx], stabilized_red_ceiling[red_idx])
            / np.maximum(alpha_sched_f64[red_idx], 1e-6),
            cfg.min_alpha_floor,
            1.0,
        )
        if cfg.mode == "base":
            beta_red_sched_f64 = _apply_symmetric_cosine_fade(
                beta_red_in_run, red_runs, res_before.fps, cfg.fade_duration_sec
            )
        else:
            beta_red_sched_f64 = _apply_half_hold_cosine_fade(
                beta_red_in_run,
                red_runs,
                res_before.fps,
                cfg.fade_duration_sec,
                cfg.half_hold_duration_sec,
            )

    alpha_sched = alpha_sched_f64.astype(np.float32)
    beta_red_sched = beta_red_sched_f64.astype(np.float32)
    alpha_red_sched = (alpha_sched_f64 * beta_red_sched_f64).astype(np.float32)

    return (
        alpha_sched,
        alpha_target.astype(np.float32),
        beta_red_sched,
        alpha_red_sched,
        in_danger_any,
        in_red_danger,
        extra_safe_runs,
        over_4metrics,
    )


def dim_video_fullscreen_b2(
    input_video_path: str | Path,
    output_video_path: str | Path,
    *,
    b2_config: B2Config | None = None,
    viewing: ViewingConfig | None = None,
    pse: PSEConfig | None = None,
    device: str | torch.device | None = None,
    detector_cache: DetectorCache | None = None,
    pass1_result: VideoAnalysisResult | None = None,
) -> B2DimmingResult:
    """Run 2-Pass B2 channel-separated dimming at original resolution with closed-loop verification.

    Args:
        pass1_result: Optional precomputed Pass 1 `VideoAnalysisResult` (with `physical_stats`)
                      allowing `all`-mode execution to run Pass 1 only once per video.
    """
    in_path = Path(input_video_path)
    out_path = Path(output_video_path)
    cfg = b2_config or B2Config()
    view_cfg = viewing or ViewingConfig()
    pse_cfg = pse or PSEConfig()
    cache = detector_cache if detector_cache is not None else {}

    # Pass 1: Extract exact `vdcore` hazard intervals, `risk_calibrated`, & physical stats
    if (
        pass1_result is not None
        and pass1_result.physical_stats is not None
        and "q24_red_escape_beta" in pass1_result.physical_stats
    ):
        res_before = pass1_result
    else:
        res_before = analyze_video(
            in_path,
            viewing=view_cfg,
            pse=pse_cfg,
            device=device,
            collect_physical_stats=True,
            detector_cache=cache,
        )

    (
        alpha_schedule,
        raw_alpha_target,
        beta_red_schedule,
        alpha_red_schedule,
        in_danger_mask,
        in_red_mask,
        extra_safe_runs,
        over4_before,
    ) = compute_alpha_and_red_schedule_b2(res_before, cfg=cfg, pse=pse_cfg)

    # Pass 2: Stream original-resolution frames, apply exact 3-channel LUT dimming, encode `.mp4`,
    #         and simultaneously re-verify through `vdcore.analyze_frame_stream`.
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cap = cv2.VideoCapture(str(in_path))
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open input video for Pass 2 encoding: {in_path}")

    orig_w = res_before.orig_width
    orig_h = res_before.orig_height
    fps = res_before.fps
    n_frames = res_before.total_frames
    timestamps_ms = res_before.frame_metrics_df["timestamp_ms"].to_numpy(dtype=np.float64)

    fourcc = cv2.VideoWriter_fourcc(*cfg.fourcc)
    writer = cv2.VideoWriter(str(out_path), fourcc, fps, (orig_w, orig_h))
    if not writer.isOpened():
        cap.release()
        raise RuntimeError(f"Failed to open VideoWriter for output path: {out_path}")

    try:
        def _dimmed_stream() -> Iterable[tuple[int, float, np.ndarray]]:
            idx = 0
            dt_ms = 1000.0 / max(fps, 1e-6)
            while idx < n_frames:
                ok, raw_bgr = cap.read()
                if not ok or raw_bgr is None:
                    break
                if raw_bgr.shape[1] != orig_w or raw_bgr.shape[0] != orig_h:
                    raw_bgr = cv2.resize(raw_bgr, (orig_w, orig_h), interpolation=cv2.INTER_AREA)

                alpha_t = float(alpha_schedule[idx])
                beta_r_t = float(beta_red_schedule[idx])
                dim_bgr = apply_linear_dimming_b2_bgr(raw_bgr, alpha_t, beta_r_t)
                writer.write(dim_bgr)

                t_ms = float(timestamps_ms[idx]) if idx < len(timestamps_ms) else (idx * dt_ms)
                yield idx, t_ms, dim_bgr
                idx += 1

        res_after = analyze_frame_stream(
            frame_iter=_dimmed_stream(),
            fps=fps,
            orig_width=orig_w,
            orig_height=orig_h,
            video_name=str(out_path),
            viewing=view_cfg,
            pse=pse_cfg,
            device=device,
            collect_physical_stats=True,
            detector_cache=cache,
        )
    finally:
        cap.release()
        writer.release()

    over4_after = compute_4metrics_union_mask(
        res_after,
        metric_threshold=cfg.safe_metric_threshold,
        nomura_threshold=cfg.safe_nomura_threshold,
    )

    return B2DimmingResult(
        mode=cfg.mode,
        red_mode=cfg.red_mode,
        config=cfg,
        input_video_path=str(in_path),
        output_video_path=str(out_path),
        orig_width=orig_w,
        orig_height=orig_h,
        analysis_width=res_before.analysis_width,
        analysis_height=res_before.analysis_height,
        fps=fps,
        total_frames=n_frames,
        before=res_before,
        after=res_after,
        alpha_schedule=alpha_schedule,
        raw_alpha_target=raw_alpha_target,
        beta_red_schedule=beta_red_schedule,
        alpha_red_schedule=alpha_red_schedule,
        in_danger_mask=in_danger_mask,
        in_red_mask=in_red_mask,
        extra_safe_runs=extra_safe_runs,
        over_4metrics_before=over4_before,
        over_4metrics_after=over4_after,
    )
