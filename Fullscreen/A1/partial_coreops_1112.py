"""LUMI Partial-Screen Spatiotemporal Linear Dimming Core Engine (`partial_coreops_1112.py`).

Built on top of `vdcoreps` (Pass 1 pixel-level hazard localization & soft mask synthesis) and
`vdcorefs` (Pass 2 closed-loop verification), integrating all three A1 scheduling modes into a
single clean module symmetric to `A1/fullscreen/fullscreen_coreops_1112.py`:
  - `base`: Mitigates confirmed 7-category PSE hazard intervals using Formula A
            `M(t, y, x) = 1 - W_t(y, x) * (1 - alpha_t)` with symmetric `0.20s` cosine Fade-in/out.
  - `hold`: Mitigates confirmed 7-category PSE hazard intervals with 2-stage Half-Hold release
            (`0.20s` ramp to `alpha_half = (alpha_end + 1)/2` -> `0.50s` hold -> `0.20s` release).
  - `safe`: Mitigates `(confirmed intervals) U (4 continuous risk metrics >= 50% union)`
            with 2-stage Half-Hold release and saturated-red suppression on active `sat_red_area > 25%` frames.

Key Partial Dimming formulations:
  1. Formula A in IEC 61966-2-1 linear light:
         c_lin'(t, y, x) = [1.0 - W_t(y, x) * (1.0 - alpha_t)] * c_lin(t, y, x)
     where `W_t(y, x) in [0, 1]` preserves `W_t == 1.0` strictly across core hazard pixels while
     feathering smoothly outward over `~0.40 deg` of visual angle, leaving safe background (`W_t == 0`)
     100% untouched.
  2. Masked-region safety ceilings (`1-2 (i)`): `alpha_t` is computed strictly from physical maxima
     inside the localized core hazard mask (`mask_max_lum_cd_m2 <= 18.0 cd/m^2` and
     `mask_max_sat_red_excess <= 0.058`), avoiding false darkening caused by safe bright/red pixels
     outside the hazard region.
  3. Exact spatial multiplier summaries: per-frame effective active dimmed area fraction
     `mean_W_t = mean_{y,x} W_t(y, x)` (when `alpha_t < 1`), full-frame mean linear multiplier
     `mean_M_t = 1.0 - mean_W_t * (1.0 - alpha_t)`, and full-frame mean perceptual sRGB multiplier
     `mean_{y,x} M(t, y, x)^(1/2.4)`.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
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

import vdcorefs
from vdcoreps import (
    LABEL_RED_EXTENDED,
    LABEL_RED_FLASH,
    PSEConfig,
    PartialDetectorCache,
    PartialMaskConfig,
    PartialVideoAnalysisResult,
    VideoAnalysisResult,
    ViewingConfig,
    analyze_video,
    apply_partial_linear_dimming_bgr,
    build_mode_weight_masks,
    compute_4metrics_union_from_result,
    mask_to_spans,
)

A1Mode = Literal["base", "hold", "safe"]
VALID_A1_MODES: tuple[A1Mode, ...] = ("base", "hold", "safe")

# Default quantitative mitigation & spatiotemporal mask constants
TARGET_SAFE_LUM_CD: float = 18.0
TARGET_SAFE_RED_EXCESS: float = 0.058  # < 0.0625 (#470000); quantizes to R <= 70 in 8-bit sRGB
MIN_ALPHA_FLOOR: float = 0.005
FADE_DURATION_SEC: float = 0.20
HALF_HOLD_DURATION_SEC: float = 0.50
TEMPORAL_WINDOW_SEC: float = 0.16
FEATHER_ANGLE_DEG: float = 0.40
SAFE_METRIC_THRESHOLD: float = 50.0    # for [0, 100] metrics (calibrated, ppr_pct, vfm_weibull)
SAFE_NOMURA_THRESHOLD: float = 0.50    # for raw risk_nomura e(t) in [0, 1]


@dataclass(frozen=True, slots=True)
class A1PartialConfig:
    """Configuration parameters for A1 partial-screen linear dimming (Formula A)."""

    mode: A1Mode = "base"
    target_safe_lum_cd: float = TARGET_SAFE_LUM_CD
    target_safe_red_excess: float = TARGET_SAFE_RED_EXCESS
    min_alpha_floor: float = MIN_ALPHA_FLOOR
    fade_duration_sec: float = FADE_DURATION_SEC
    half_hold_duration_sec: float = HALF_HOLD_DURATION_SEC
    temporal_window_sec: float = TEMPORAL_WINDOW_SEC
    feather_angle_deg: float = FEATHER_ANGLE_DEG
    borderless_guided: bool = True
    enforce_zero_residual: bool = True
    safe_metric_threshold: float = SAFE_METRIC_THRESHOLD
    safe_nomura_threshold: float = SAFE_NOMURA_THRESHOLD
    fourcc: str = "mp4v"

    def to_mask_config(self) -> PartialMaskConfig:
        """Build matching `PartialMaskConfig` for `vdcoreps`."""
        return PartialMaskConfig(
            temporal_window_sec=self.temporal_window_sec,
            temporal_fade_sec=self.fade_duration_sec,
            half_hold_duration_sec=self.half_hold_duration_sec,
            feather_angle_deg=self.feather_angle_deg,
            borderless_guided=self.borderless_guided,
            safe_metric_threshold=self.safe_metric_threshold,
            safe_nomura_threshold=self.safe_nomura_threshold,
        )


@dataclass(slots=True)
class A1PartialDimmingResult:
    """Complete before/after verification result from `dim_video_partial_a1`."""

    mode: A1Mode
    config: A1PartialConfig
    input_video_path: str
    output_video_path: str
    orig_width: int
    orig_height: int
    analysis_width: int
    analysis_height: int
    fps: float
    total_frames: int
    before: PartialVideoAnalysisResult
    after: VideoAnalysisResult
    alpha_schedule: np.ndarray
    active_dimmed_area: np.ndarray
    mean_linear_multiplier: np.ndarray
    mean_srgb_multiplier: np.ndarray
    in_danger_mask: np.ndarray
    in_red_mask: np.ndarray
    extra_safe_runs: list[tuple[int, int]]
    over_4metrics_before: np.ndarray
    over_4metrics_after: np.ndarray


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
    temporal_window_sec: float = TEMPORAL_WINDOW_SEC,
) -> np.ndarray:
    """Sliding minimum envelope (`+-temporal_window_sec`) + odd-kernel moving average (`0.30s`) + re-clamp."""
    n_frames = len(alpha_target)
    alpha_schedule = np.ones(n_frames, dtype=np.float64)
    win_radius = max(2, int(round(temporal_window_sec * fps)))
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
    fade_frames = max(3, int(round(fade_duration_sec * fps))) if fade_duration_sec > 0.0 else 0
    out = np.copy(alpha_schedule)
    if fade_frames <= 0:
        return out

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
    fade_frames = max(3, int(round(fade_duration_sec * fps))) if fade_duration_sec > 0.0 else 0
    hold_frames = max(1, int(round(half_hold_duration_sec * fps)))
    out = np.copy(alpha_schedule)
    if fade_frames <= 0:
        return out

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


def compute_alpha_schedule_partial_a1(
    res_before: PartialVideoAnalysisResult,
    cfg: A1PartialConfig = A1PartialConfig(),
    pse: PSEConfig = PSEConfig(),
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[tuple[int, int]], np.ndarray]:
    """Compute per-frame core dimming schedule `alpha_t` from masked-region physical stats.

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

    stats = res_before.physical_stats
    sat_red_area = stats["sat_red_area"]

    in_interval_danger = res_before.frame_metrics_df["is_dangerous"].to_numpy(dtype=bool)
    over_4metrics = compute_4metrics_union_from_result(
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
        mask_max_lum = stats["safe_mask_max_lum_cd_m2"]
        mask_max_red = stats["safe_mask_max_sat_red_excess"]
    else:
        in_danger_any = in_interval_danger.copy()
        in_red_danger = red_interval_mask.copy()
        extra_safe_runs = []
        mask_max_lum = stats["mask_max_lum_cd_m2"]
        mask_max_red = stats["mask_max_sat_red_excess"]

    # 1) Frame-wise masked-region safety ceiling `alpha_target`
    alpha_target = np.ones(n_frames, dtype=np.float64)
    active_idx = np.flatnonzero(in_danger_any)
    if len(active_idx) > 0:
        alpha_lum = np.minimum(
            1.0,
            cfg.target_safe_lum_cd / np.maximum(mask_max_lum[active_idx].astype(np.float64), 1e-6),
        )
        req_alpha = alpha_lum

        red_sub = in_red_danger[active_idx] & (mask_max_red[active_idx] > cfg.target_safe_red_excess)
        if np.any(red_sub):
            r_vals = np.maximum(mask_max_red[active_idx][red_sub].astype(np.float64), 1e-6)
            alpha_red = np.minimum(1.0, cfg.target_safe_red_excess / r_vals)
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

    stabilized = _stabilize_in_run_envelope(
        alpha_target,
        runs,
        res_before.fps,
        temporal_window_sec=max(0.25, cfg.temporal_window_sec),
    )

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


def compute_frame_multiplier_summaries(
    alpha_schedule: np.ndarray,
    weight_masks_u8: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute exact per-frame `(active_dimmed_area, mean_linear_multiplier, mean_srgb_multiplier)`.

    For Formula A `M(t, y, x) = 1.0 - W_t(y, x) * (1.0 - alpha_t)`:
      - When `alpha_t >= 0.9999` or `W_t == 0`: `active_dimmed_area = 0.0`, `mean_M = 1.0`.
      - When `alpha_t < 0.9999`:
          * `active_dimmed_area[t] = mean_{y,x} W_t(y, x)`
          * `mean_linear_multiplier[t] = 1.0 - active_dimmed_area[t] * (1.0 - alpha_t)`
          * `mean_srgb_multiplier[t] = mean_{y,x} [M(t, y, x)^(1 / 2.4)]` (exact 256-bin histogram)
    """
    n_frames = len(alpha_schedule)
    active_area = np.zeros(n_frames, dtype=np.float32)
    mean_lin = np.ones(n_frames, dtype=np.float32)
    mean_srgb = np.ones(n_frames, dtype=np.float32)
    if n_frames == 0 or weight_masks_u8.size == 0:
        return active_area, mean_lin, mean_srgb

    w_grid_256 = np.arange(256, dtype=np.float64) / 255.0
    n_px = float(weight_masks_u8.shape[1] * weight_masks_u8.shape[2])

    for t in range(n_frames):
        a_t = float(np.clip(alpha_schedule[t], 0.0, 1.0))
        if a_t >= 0.9999:
            continue
        m_u8 = weight_masks_u8[t]
        if int(m_u8.max()) == 0:
            continue

        counts = np.bincount(m_u8.ravel(), minlength=256).astype(np.float64)
        mean_w = float(np.dot(counts, w_grid_256) / n_px)
        m_vals = np.clip(1.0 - w_grid_256 * (1.0 - a_t), 0.0, 1.0)
        srgb_vals = np.power(m_vals, 1.0 / 2.4)

        active_area[t] = mean_w
        mean_lin[t] = 1.0 - mean_w * (1.0 - a_t)
        mean_srgb[t] = float(np.dot(counts, srgb_vals) / n_px)

    return active_area, mean_lin, mean_srgb


def dim_video_partial_a1(
    input_video_path: str | Path,
    output_video_path: str | Path,
    *,
    a1_config: A1PartialConfig | None = None,
    viewing: ViewingConfig | None = None,
    pse: PSEConfig | None = None,
    device: str | torch.device | None = None,
    partial_cache: PartialDetectorCache | None = None,
    verify_cache: vdcorefs.DetectorCache | None = None,
    pass1_result: PartialVideoAnalysisResult | None = None,
) -> A1PartialDimmingResult:
    """Run 2-Pass A1 partial-screen linear dimming at original resolution with closed-loop verification.

    Args:
        pass1_result: Optional precomputed Pass 1 `PartialVideoAnalysisResult` (with `physical_stats`
                      and `packed_hazards`), allowing `all`-mode execution to run Pass 1 once per video.
    """
    in_path = Path(input_video_path)
    out_path = Path(output_video_path)
    cfg = a1_config or A1PartialConfig()
    view_cfg = viewing or ViewingConfig()
    pse_cfg = pse or PSEConfig()
    mask_cfg = cfg.to_mask_config()
    p_cache = partial_cache if partial_cache is not None else {}
    v_cache = verify_cache if verify_cache is not None else {}

    # Pass 1: Extract exact 7-category hazard intervals, 4 risk metrics, soft/packed masks, & masked stats
    if (
        pass1_result is not None
        and pass1_result.physical_stats is not None
        and pass1_result.packed_hazards is not None
    ):
        res_before = pass1_result
    else:
        res_before = analyze_video(
            in_path,
            viewing=view_cfg,
            pse=pse_cfg,
            mask_config=mask_cfg,
            device=device,
            collect_physical_stats=True,
            collect_masks=True,
            detector_cache=p_cache,
        )

    alpha_schedule, in_danger_mask, in_red_mask, extra_safe_runs, over4_before = (
        compute_alpha_schedule_partial_a1(res_before, cfg=cfg, pse=pse_cfg)
    )

    weight_masks_u8, _ = build_mode_weight_masks(
        res_before,
        mode=cfg.mode,
        hold_run_tails=True,
        mask_config=mask_cfg,
        device=device,
    )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    orig_w = res_before.orig_width
    orig_h = res_before.orig_height
    fps = res_before.fps
    n_frames = res_before.total_frames
    timestamps_ms = res_before.frame_metrics_df["timestamp_ms"].to_numpy(dtype=np.float64)
    fourcc = cv2.VideoWriter_fourcc(*cfg.fourcc)

    def _encode_and_verify(
        sched_arr: np.ndarray,
        masks_arr_u8: np.ndarray,
    ) -> VideoAnalysisResult:
        cap = cv2.VideoCapture(str(in_path))
        if not cap.isOpened():
            raise RuntimeError(f"Failed to open input video for Pass 2 encoding: {in_path}")
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

                    alpha_t = float(sched_arr[idx])
                    dim_bgr = apply_partial_linear_dimming_bgr(
                        raw_bgr, alpha_t, masks_arr_u8[idx]
                    )
                    writer.write(dim_bgr)

                    t_ms = float(timestamps_ms[idx]) if idx < len(timestamps_ms) else (idx * dt_ms)
                    yield idx, t_ms, dim_bgr
                    idx += 1

            return vdcorefs.analyze_frame_stream(
                frame_iter=_dimmed_stream(),
                fps=fps,
                orig_width=orig_w,
                orig_height=orig_h,
                video_name=str(out_path),
                viewing=view_cfg,
                pse=pse_cfg,
                device=device,
                collect_physical_stats=True,
                detector_cache=v_cache,
            )
        finally:
            cap.release()
            writer.release()

    # Pass 2 + `(2-1)` Closed-Loop Borderless Safety Reinforcement (`0` residual intervals guarantee)
    res_after = _encode_and_verify(alpha_schedule, weight_masks_u8)

    if cfg.enforce_zero_residual and n_frames > 0 and len(res_after.intervals) > 0:
        fade_f = max(3, int(round(cfg.fade_duration_sec * fps))) if cfg.fade_duration_sec > 0.0 else 3
        pad_f = max(fade_f, int(round(0.35 * fps)))
        steps = np.arange(1, fade_f + 1, dtype=np.float64)
        cos_out = 0.5 * (1.0 + np.cos(math.pi * steps / float(fade_f + 1)))

        reinforce_specs = (
            (0.32, 0.90, 0.85, 0.72),
            (0.18, 1.50, 1.05, 0.68),
            (0.12, 2.20, 1.25, 0.60),
        )
        for step_i, (r_knee, r_reach, r_post, a_scale) in enumerate(reinforce_specs, start=1):
            if len(res_after.intervals) == 0:
                break

            res_mask = np.zeros(n_frames, dtype=bool)
            for rec in res_after.intervals:
                s_i = max(0, int(rec.start_frame) - pad_f)
                e_i = min(n_frames - 1, int(rec.end_frame) + pad_f)
                if e_i >= s_i:
                    res_mask[s_i : e_i + 1] = True

            if not np.any(res_mask):
                break

            step_cfg = replace(
                mask_cfg,
                guided_knee=min(mask_cfg.guided_knee, r_knee),
                guided_reach_deg=max(mask_cfg.guided_reach_deg, r_reach),
                guided_post_sigma_deg=max(mask_cfg.guided_post_sigma_deg, r_post),
            )
            if step_i == 3 and res_before.packed_hazards is not None:
                # Final borderless fallback: pure C2 Gaussian + quintic smoothstep without guide clipping
                orig_guide = res_before.packed_hazards.guide_rgb_u8
                try:
                    res_before.packed_hazards.guide_rgb_u8 = None
                    step_w_u8, _ = build_mode_weight_masks(
                        res_before,
                        mode=cfg.mode,
                        hold_run_tails=True,
                        mask_config=step_cfg,
                        device=device,
                    )
                finally:
                    res_before.packed_hazards.guide_rgb_u8 = orig_guide
            else:
                step_w_u8, _ = build_mode_weight_masks(
                    res_before,
                    mode=cfg.mode,
                    hold_run_tails=True,
                    mask_config=step_cfg,
                    device=device,
                )

            # Smoothly blend reinforced borderless weights and tighten `alpha_schedule` on residual spans
            res_spans = mask_to_spans(res_mask)
            alpha_next = alpha_schedule.astype(np.float64).copy()
            for s_sp, e_sp in res_spans:
                weight_masks_u8[s_sp : e_sp + 1] = np.maximum(
                    weight_masks_u8[s_sp : e_sp + 1],
                    step_w_u8[s_sp : e_sp + 1],
                )
                scaled_seg = np.maximum(cfg.min_alpha_floor, alpha_next[s_sp : e_sp + 1] * a_scale)
                alpha_next[s_sp : e_sp + 1] = np.minimum(alpha_next[s_sp : e_sp + 1], scaled_seg)

                sa = float(alpha_next[s_sp])
                ea = float(alpha_next[e_sp])
                for d, cw in enumerate(cos_out, start=1):
                    if s_sp - d >= 0:
                        blend_w = np.rint(
                            (1.0 - cw) * weight_masks_u8[s_sp - d].astype(np.float32)
                            + cw * np.maximum(weight_masks_u8[s_sp - d], step_w_u8[s_sp - d]).astype(np.float32)
                        ).astype(np.uint8)
                        weight_masks_u8[s_sp - d] = np.maximum(weight_masks_u8[s_sp - d], blend_w)
                        ramp_in = sa + (float(alpha_next[s_sp - d]) - sa) * (1.0 - float(cw))
                        alpha_next[s_sp - d] = min(alpha_next[s_sp - d], ramp_in)
                    if e_sp + d < n_frames:
                        blend_w = np.rint(
                            (1.0 - cw) * weight_masks_u8[e_sp + d].astype(np.float32)
                            + cw * np.maximum(weight_masks_u8[e_sp + d], step_w_u8[e_sp + d]).astype(np.float32)
                        ).astype(np.uint8)
                        weight_masks_u8[e_sp + d] = np.maximum(weight_masks_u8[e_sp + d], blend_w)
                        ramp_out = ea + (float(alpha_next[e_sp + d]) - ea) * (1.0 - float(cw))
                        alpha_next[e_sp + d] = min(alpha_next[e_sp + d], ramp_out)

            alpha_schedule = alpha_next.astype(np.float32)
            res_after = _encode_and_verify(alpha_schedule, weight_masks_u8)

    active_dimmed_area, mean_lin_mult, mean_srgb_mult = compute_frame_multiplier_summaries(
        alpha_schedule, weight_masks_u8
    )

    over4_after = compute_4metrics_union_mask(
        res_after,
        metric_threshold=cfg.safe_metric_threshold,
        nomura_threshold=cfg.safe_nomura_threshold,
    )

    return A1PartialDimmingResult(
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
        active_dimmed_area=active_dimmed_area,
        mean_linear_multiplier=mean_lin_mult,
        mean_srgb_multiplier=mean_srgb_mult,
        in_danger_mask=in_danger_mask,
        in_red_mask=in_red_mask,
        extra_safe_runs=extra_safe_runs,
        over_4metrics_before=over4_before,
        over_4metrics_after=over4_after,
    )
