"""LUMI Full-Screen Uniform Linear Dimming Core Engine (`fullscreen_coreops_1112.py`).

Upgraded for Stage 4 `vdcore` compatibility, integrating all three A1 scheduling modes
into a single clean module without code duplication:
  - `base`: Mitigates confirmed 7-category PSE hazard intervals with symmetric 0.20s cosine Fade-in/out.
  - `hold`: Mitigates confirmed 7-category PSE hazard intervals with 2-stage Half-Hold release
            (0.20s ramp to `alpha_half = (alpha_end + 1)/2` -> 0.50s hold -> 0.20s full release)
            to suppress brightness pumping between closely spaced intervals.
  - `safe`: Mitigates `(confirmed intervals) U (4 continuous risk metrics >= 50% union)`
            (`risk_calibrated >= 50 | risk_ppr_pct >= 50 | risk_nomura >= 0.5 | risk_vfm_weibull >= 50`)
            with 2-stage Half-Hold release and red excess suppression on active `sat_red_area > 25%` frames.

Key Stage 4 mathematical updates:
  1. Unified luminance ceiling `TARGET_SAFE_LUM_CD = 18.0 cd/m^2` (`< 20.0 cd/m^2`), which simultaneously
     bounds peak luminance, spatial contrast (`Delta L <= L_max`), pattern hysteresis (`20 cd/m^2`),
     and in-interval temporal luminance steps without over-dimming non-pattern flashes to `16 cd/m^2`.
  2. Method A saturated-red excess ceiling `TARGET_SAFE_RED_EXCESS = 0.058` (`< 0.0625`, `#470000` floor)
     computed strictly over `red_ratio >= 0.80` pixels (`q76_sat_red_excess`), avoiding false darkening
     on bright white/yellow scenes and preserving ~6.4x more linear light than legacy `0.0090`.
  3. Original-resolution (`orig_width x orig_height`) `.mp4` encoding via exact 256-entry IEC 61966-2-1
     sRGB-linear-sRGB LUT transformation paired with single-pass streaming `vdcore` closed-loop verification.
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

A1Mode = Literal["base", "hold", "safe"]
VALID_A1_MODES: tuple[A1Mode, ...] = ("base", "hold", "safe")

# Default quantitative mitigation constants
TARGET_SAFE_LUM_CD: float = 18.0
TARGET_SAFE_RED_EXCESS: float = 0.058  # < 0.0625 (#470000); quantizes to R <= 70 in 8-bit sRGB
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
class A1Config:
    """Configuration parameters for A1 full-screen uniform linear dimming."""

    mode: A1Mode = "base"
    target_safe_lum_cd: float = TARGET_SAFE_LUM_CD
    target_safe_red_excess: float = TARGET_SAFE_RED_EXCESS
    min_alpha_floor: float = MIN_ALPHA_FLOOR
    fade_duration_sec: float = FADE_DURATION_SEC
    half_hold_duration_sec: float = HALF_HOLD_DURATION_SEC
    safe_metric_threshold: float = SAFE_METRIC_THRESHOLD
    safe_nomura_threshold: float = SAFE_NOMURA_THRESHOLD
    fourcc: str = "mp4v"


@dataclass(slots=True)
class A1DimmingResult:
    """Complete before/after verification result from `dim_video_fullscreen_a1`."""

    mode: A1Mode
    config: A1Config
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


def apply_linear_dimming_bgr(bgr_u8: np.ndarray, alpha_lin: float) -> np.ndarray:
    """Apply uniform linear-light dimming `alpha_lin` to an arbitrary-resolution `uint8` BGR frame."""
    if alpha_lin >= 0.9999:
        return bgr_u8
    lut = build_srgb_dimming_lut(alpha_lin)
    return cv2.LUT(bgr_u8, lut)


@torch.no_grad()
def apply_linear_dimming_srgb_tensor(rgb_255: torch.Tensor, alpha_lin: float) -> torch.Tensor:
    """Tensor helper applying IEC 61966-2-1 linear dimming on `[0, 255]` float/uint8 tensors."""
    if alpha_lin >= 0.9999:
        return rgb_255

    c_s = torch.clamp(rgb_255.to(torch.float32) / 255.0, 0.0, 1.0)
    c_lin = torch.where(
        c_s <= 0.04045,
        c_s / 12.92,
        torch.pow((c_s + 0.055) / 1.055, 2.4),
    )
    c_lin_dim = torch.clamp(c_lin * float(alpha_lin), 0.0, 1.0)
    c_s_dim = torch.where(
        c_lin_dim <= 0.0031308,
        c_lin_dim * 12.92,
        1.055 * torch.pow(torch.clamp(c_lin_dim, min=1e-8), 1.0 / 2.4) - 0.055,
    )
    out = torch.clamp(torch.round(c_s_dim * 255.0), 0.0, 255.0)
    return out.to(torch.uint8) if rgb_255.dtype == torch.uint8 else out


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


def _stabilize_in_run_envelope(
    alpha_target: np.ndarray,
    runs: list[tuple[int, int]],
    fps: float,
) -> np.ndarray:
    """Sliding minimum envelope (`+-0.25s`) + odd-kernel edge moving average (`0.30s`) + safety re-clamp."""
    n_frames = len(alpha_target)
    alpha_schedule = np.ones(n_frames, dtype=np.float64)
    win_radius = max(2, int(round(0.25 * fps)))
    k_base = max(3, int(round(0.30 * fps)) | 1)

    for s_run, e_run in runs:
        run_targets = alpha_target[s_run : e_run + 1]
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
                final_run_alpha = np.minimum(smoothed_env, run_targets)
            else:
                final_run_alpha = env
        else:
            final_run_alpha = env

        alpha_schedule[s_run : e_run + 1] = final_run_alpha

    return alpha_schedule


def _apply_symmetric_cosine_fade(
    alpha_schedule: np.ndarray,
    runs: list[tuple[int, int]],
    fps: float,
    fade_duration_sec: float,
) -> np.ndarray:
    """Base mode boundary transition: symmetric `0.20s` cosine Fade-in and Fade-out to `1.0`."""
    n_frames = len(alpha_schedule)
    fade_frames = max(3, int(round(fade_duration_sec * fps)))
    out = np.copy(alpha_schedule)

    steps = np.arange(1, fade_frames + 1, dtype=np.float64)
    cosine_w = 0.5 * (1.0 - np.cos(math.pi * steps / float(fade_frames + 1)))

    for s_run, e_run in runs:
        start_alpha = float(alpha_schedule[s_run])
        end_alpha = float(alpha_schedule[e_run])

        for step_idx, w in enumerate(cosine_w, start=1):
            f_in = s_run - step_idx
            if f_in >= 0:
                ramp_in = start_alpha + (1.0 - start_alpha) * float(w)
                out[f_in] = min(out[f_in], ramp_in)

            f_out = e_run + step_idx
            if f_out < n_frames:
                ramp_out = end_alpha + (1.0 - end_alpha) * float(w)
                out[f_out] = min(out[f_out], ramp_out)

    return out


def _apply_half_hold_cosine_fade(
    alpha_schedule: np.ndarray,
    runs: list[tuple[int, int]],
    fps: float,
    fade_duration_sec: float,
    half_hold_duration_sec: float,
) -> np.ndarray:
    """Hold/Safe mode boundary transition: 2-stage Half-Hold release + smooth anchor Fade-in."""
    n_frames = len(alpha_schedule)
    fade_frames = max(3, int(round(fade_duration_sec * fps)))
    hold_frames = max(1, int(round(half_hold_duration_sec * fps)))
    out = np.copy(alpha_schedule)

    steps = np.arange(1, fade_frames + 1, dtype=np.float64)
    cosine_w = 0.5 * (1.0 - np.cos(math.pi * steps / float(fade_frames + 1)))

    # (A) Exit 2-stage release: end_alpha -> alpha_half -> hold -> 1.0
    for _, e_run in runs:
        end_alpha = float(alpha_schedule[e_run])
        alpha_half = 0.5 * (end_alpha + 1.0)

        for step_idx, w in enumerate(cosine_w, start=1):
            f_idx = e_run + step_idx
            if f_idx >= n_frames:
                break
            ramp1 = end_alpha + (alpha_half - end_alpha) * float(w)
            out[f_idx] = min(out[f_idx], ramp1)

        hold_start = e_run + fade_frames + 1
        hold_end = e_run + fade_frames + hold_frames
        if hold_start < n_frames:
            sl_end = min(n_frames, hold_end + 1)
            out[hold_start:sl_end] = np.minimum(out[hold_start:sl_end], alpha_half)

        for step_idx, w in enumerate(cosine_w, start=1):
            f_idx = hold_end + step_idx
            if f_idx >= n_frames:
                break
            ramp2 = alpha_half + (1.0 - alpha_half) * float(w)
            out[f_idx] = min(out[f_idx], ramp2)

    # (B) Entry Fade-in anchored to current relaxed level (prevents upward jump before re-entry)
    for s_run, _ in runs:
        start_alpha = float(alpha_schedule[s_run])
        anchor_idx = max(0, s_run - fade_frames - 1)
        entry_from_alpha = max(start_alpha, float(out[anchor_idx]))

        for step_idx, w in enumerate(cosine_w, start=1):
            f_idx = s_run - step_idx
            if f_idx < 0:
                break
            ramp_in = start_alpha + (entry_from_alpha - start_alpha) * float(w)
            out[f_idx] = min(out[f_idx], ramp_in)

    return out


def compute_alpha_schedule_a1(
    res_before: VideoAnalysisResult,
    cfg: A1Config = A1Config(),
    pse: PSEConfig = PSEConfig(),
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[tuple[int, int]], np.ndarray]:
    """Compute per-frame linear dimming schedule `alpha_t` for `base`, `hold`, or `safe` mode.

    Returns:
        `(alpha_schedule, in_danger_any, in_red_danger, extra_safe_runs, over_4metrics)`
    """
    if cfg.mode not in VALID_A1_MODES:
        raise ValueError(f"Unsupported A1 mode {cfg.mode!r}; expected one of {VALID_A1_MODES}.")
    if res_before.physical_stats is None:
        raise ValueError("`res_before.physical_stats` is required (`collect_physical_stats=True`).")

    n_frames = res_before.total_frames
    if n_frames == 0:
        empty_f = np.zeros(0, dtype=np.float32)
        empty_b = np.zeros(0, dtype=bool)
        return empty_f, empty_b, empty_b, [], empty_b

    max_lum = res_before.physical_stats["max_lum_cd_m2"]
    sat_red_area = res_before.physical_stats["sat_red_area"]
    q76_red_exc = res_before.physical_stats["q76_sat_red_excess"]

    in_interval_danger = res_before.frame_metrics_df["is_dangerous"].to_numpy(dtype=bool)
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

    # 1) Frame-wise raw safety ceiling `alpha_target`
    alpha_target = np.ones(n_frames, dtype=np.float64)
    active_idx = np.flatnonzero(in_danger_any)
    if len(active_idx) > 0:
        alpha_lum = np.minimum(
            1.0, cfg.target_safe_lum_cd / np.maximum(max_lum[active_idx].astype(np.float64), 1e-6)
        )
        req_alpha = alpha_lum

        red_sub = in_red_danger[active_idx] & (q76_red_exc[active_idx] > cfg.target_safe_red_excess)
        if np.any(red_sub):
            q_vals = np.maximum(q76_red_exc[active_idx][red_sub].astype(np.float64), 1e-6)
            alpha_red = np.minimum(1.0, cfg.target_safe_red_excess / q_vals)
            req_alpha = req_alpha.copy()
            req_alpha[red_sub] = np.minimum(req_alpha[red_sub], alpha_red)

        alpha_target[active_idx] = np.clip(req_alpha, cfg.min_alpha_floor, 1.0)

    # 2) Contiguous run extraction & in-interval minimum-envelope stabilization
    runs = mask_to_spans(in_danger_any)
    if not runs:
        return (
            np.ones(n_frames, dtype=np.float32),
            in_danger_any,
            in_red_danger,
            extra_safe_runs,
            over_4metrics,
        )

    stabilized = _stabilize_in_run_envelope(alpha_target, runs, res_before.fps)

    # 3) Boundary transition smoothing (`base` symmetric fade vs `hold`/`safe` Half-Hold)
    if cfg.mode == "base":
        final_sched = _apply_symmetric_cosine_fade(
            stabilized, runs, res_before.fps, cfg.fade_duration_sec
        )
    else:
        final_sched = _apply_half_hold_cosine_fade(
            stabilized,
            runs,
            res_before.fps,
            cfg.fade_duration_sec,
            cfg.half_hold_duration_sec,
        )

    return (
        final_sched.astype(np.float32),
        in_danger_any,
        in_red_danger,
        extra_safe_runs,
        over_4metrics,
    )


def dim_video_fullscreen_a1(
    input_video_path: str | Path,
    output_video_path: str | Path,
    *,
    a1_config: A1Config | None = None,
    viewing: ViewingConfig | None = None,
    pse: PSEConfig | None = None,
    device: str | torch.device | None = None,
    detector_cache: DetectorCache | None = None,
    pass1_result: VideoAnalysisResult | None = None,
) -> A1DimmingResult:
    """Run 2-Pass A1 full-screen linear dimming at original resolution with closed-loop verification.

    Args:
        pass1_result: Optional precomputed Pass 1 `VideoAnalysisResult` (with `physical_stats`)
                      allowing `all`-mode execution to run Pass 1 only once per video.
    """
    in_path = Path(input_video_path)
    out_path = Path(output_video_path)
    cfg = a1_config or A1Config()
    view_cfg = viewing or ViewingConfig()
    pse_cfg = pse or PSEConfig()
    cache = detector_cache if detector_cache is not None else {}

    # Pass 1: Extract exact `vdcore` hazard intervals, 4 risk metrics, & physical stats
    if pass1_result is not None and pass1_result.physical_stats is not None:
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

    alpha_schedule, in_danger_mask, in_red_mask, extra_safe_runs, over4_before = (
        compute_alpha_schedule_a1(res_before, cfg=cfg, pse=pse_cfg)
    )

    # Pass 2: Stream original-resolution frames, apply exact LUT dimming, encode `.mp4`,
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
                dim_bgr = apply_linear_dimming_bgr(raw_bgr, alpha_t)
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

    return A1DimmingResult(
        mode=cfg.mode,
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
        in_danger_mask=in_danger_mask,
        in_red_mask=in_red_mask,
        extra_safe_runs=extra_safe_runs,
        over_4metrics_before=over4_before,
        over_4metrics_after=over4_after,
    )
