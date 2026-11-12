"""Configuration, viewing geometry, and structured output types for `vdcorefs`."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Literal

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Output Hazard Labels
# ---------------------------------------------------------------------------

LABEL_LUM_FLASH: str = "휘도 플래시"
LABEL_INTERSECTION: str = "교집합"
LABEL_SCENE_TRANSITION: str = "빠른 장면 전환"
LABEL_RED_FLASH: str = "적색 플래시"
LABEL_LUM_EXTENDED: str = "휘도 연장"
LABEL_RED_EXTENDED: str = "적색 연장"
LABEL_STATIC_PATTERN: str = "정적 공간 패턴"
LABEL_SAFE: str = "안전"

ALL_HAZARD_LABELS: tuple[str, ...] = (
    LABEL_LUM_FLASH,
    LABEL_INTERSECTION,
    LABEL_SCENE_TRANSITION,
    LABEL_RED_FLASH,
    LABEL_LUM_EXTENDED,
    LABEL_RED_EXTENDED,
    LABEL_STATIC_PATTERN,
)

ViewingProfile = Literal["laptop", "desktop", "tv"]

_PROFILE_DEFAULTS: dict[str, tuple[float, float | None]] = {
    # profile -> (default_diagonal_inch, default_viewing_distance_cm or None for 3*H)
    "laptop": (15.6, 56.0),
    "desktop": (27.0, 70.0),
    "tv": (55.0, None),
}


# ---------------------------------------------------------------------------
# Viewing Geometry & PSE Quantitative Criteria Configurations
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class ViewingGeometry:
    """Physical screen dimensions and visual-angle mapping for an analysis frame."""

    width: int
    height: int
    width_cm: float
    height_cm: float
    viewing_distance_cm: float
    fov_w_deg: float
    fov_h_deg: float
    px_per_deg_x: float
    px_per_deg_y: float

    @property
    def area_deg2(self) -> float:
        return self.fov_w_deg * self.fov_h_deg

    @property
    def px_per_deg_ring(self) -> float:
        return math.sqrt(self.px_per_deg_x * self.px_per_deg_y)

    def pixels_per_degree(self, phi_rad: float | np.ndarray) -> float | np.ndarray:
        """Continuous pixel density p(phi) [px/deg] along scan angle phi [rad]."""
        cos_term = np.cos(phi_rad) / self.px_per_deg_x
        sin_term = np.sin(phi_rad) / self.px_per_deg_y
        denom = np.sqrt(cos_term * cos_term + sin_term * sin_term)
        if np.ndim(denom) == 0:
            return float(1.0 / max(float(denom), 1e-9))
        return 1.0 / np.maximum(denom, 1e-9)


@dataclass(frozen=True, slots=True)
class ViewingConfig:
    """Display and viewing-environment parameters (Stage 3 Section 1.1 & 1.3)."""

    profile: ViewingProfile = "laptop"
    diagonal_inch: float | None = None
    viewing_distance_cm: float | None = None
    l_max_cd_m2: float = 200.0
    analysis_max_width: int = 640

    def resolve_geometry(self, width: int, height: int) -> ViewingGeometry:
        if width <= 0 or height <= 0:
            raise ValueError(f"Frame dimensions must be positive, got {(width, height)}.")
        default_diag, default_dist = _PROFILE_DEFAULTS.get(self.profile, (15.6, 56.0))
        diag_inch = float(self.diagonal_inch if self.diagonal_inch is not None else default_diag)
        diag_px = math.hypot(width, height)
        width_cm = 2.54 * diag_inch * (width / diag_px)
        height_cm = 2.54 * diag_inch * (height / diag_px)

        if self.viewing_distance_cm is not None:
            dist_cm = float(self.viewing_distance_cm)
        elif default_dist is not None:
            dist_cm = default_dist
        else:
            dist_cm = 3.0 * height_cm

        fov_w_deg = math.degrees(2.0 * math.atan(width_cm / (2.0 * dist_cm)))
        fov_h_deg = math.degrees(2.0 * math.atan(height_cm / (2.0 * dist_cm)))
        return ViewingGeometry(
            width=width,
            height=height,
            width_cm=width_cm,
            height_cm=height_cm,
            viewing_distance_cm=dist_cm,
            fov_w_deg=fov_w_deg,
            fov_h_deg=fov_h_deg,
            px_per_deg_x=width / max(fov_w_deg, 1e-6),
            px_per_deg_y=height / max(fov_h_deg, 1e-6),
        )


@dataclass(frozen=True, slots=True)
class PSEConfig:
    """Quantitative PSE thresholds established in Stage 3 Sections 1-3."""

    # Photometric & Michelson thresholds (ITU-R BT.1702-3)
    delta_lum_min_cd_m2: float = 20.0
    dark_lum_limit_cd_m2: float = 160.0
    bright_michelson_min: float = 1.0 / 17.0

    # Chromaticity & Saturated Red (ISO 9241-391 + WCAG 2.0 Method A dark-red floor)
    uv_denom_floor: float = 1e-4
    d65_u_prime: float = 0.1978
    d65_v_prime: float = 0.4683
    sat_red_ratio_min: float = 0.80
    sat_red_excess_min: float = 0.0625  # 20 / 320 -> #470000 pure red floor
    delta_uv_min: float = 0.20

    # Flash area & opposing transition counts
    flash_area_min: float = 0.25        # strictly > 0.25
    scene_area_min: float = 0.80        # >= 0.80
    flash_min_transitions: int = 7      # > 3 flashes/s -> >= 7 opposing transitions/1s
    scene_min_transitions: int = 4      # > 3 cuts/s -> >= 4 opposing transitions/1s

    # High-frame-rate temporal windows (Jordan & Vanderheiden 2024 + ITU-R BT.1702-3)
    max_transition_ms: float = 66.8     # captures 2-frame ramps at 29.97/30 fps
    sync_window_ms: float = 20.0        # spatial area union window
    ultrafast_merge_ms: float = 15.0    # >= 65 Hz same-polarity oscillation merge
    window_1s_ms: float = 1000.05       # 1.0 s sliding window
    safe_gap_50hz_ms: float = 359.9     # 9 frames @ 25 fps = 360.0 ms
    safe_gap_default_ms: float = 333.3  # 10 frames @ 30 fps = 333.33 ms

    # EA IRIS 5-second extended flash parameters
    extended_band_min_transitions: int = 4
    extended_band_max_transitions: int = 6
    extended_window_sec: float = 5.0
    extended_active_sec: float = 4.0

    # Static spatial pattern parameters (ITU-R BT.1702-3 + Wilkins et al. 2005)
    pattern_area_min: float = 0.40      # strictly > 0.40
    pattern_hysteresis_cd_m2: float = 20.0
    pattern_cpd_min: float = 1.0
    pattern_cpd_max: float = 10.0
    pattern_min_pairs: int = 6          # > 5 light-dark pairs -> >= 6 pairs (>= 11 half-cycles)
    pattern_min_duration_sec: float = 0.5
    pattern_fft_energy_ratio_min: float = 0.35


@dataclass(frozen=True, slots=True)
class FPSWindowSpec:
    """Discrete frame-window conversions derived from video frame rate (Stage 3 Section 1.4)."""

    fps: float
    dt_ms: float
    k_trans: int
    k_sync: int
    k_merge: int
    is_50hz_family: bool
    safe_gap_ms: float
    w_1s: int
    w_5s: int
    w_4s: int
    n_static_0p5s: int

    @classmethod
    def from_fps(cls, fps: float, pse: PSEConfig = PSEConfig()) -> FPSWindowSpec:
        clean_fps = float(fps) if (fps is not None and math.isfinite(fps) and fps > 0.0) else 30.0
        dt_ms = 1000.0 / clean_fps
        k_trans = max(1, int(math.floor((pse.max_transition_ms / 1000.0) * clean_fps + 1e-6)))
        k_sync = 1 + int(math.floor((pse.sync_window_ms / 1000.0) * clean_fps + 1e-6))
        k_merge = int(math.floor((pse.ultrafast_merge_ms / 1000.0) * clean_fps + 1e-6))
        is_50hz = (abs(clean_fps - 25.0) < 0.5) or (abs(clean_fps - 50.0) < 0.5)
        safe_gap_ms = pse.safe_gap_50hz_ms if is_50hz else pse.safe_gap_default_ms
        return cls(
            fps=clean_fps,
            dt_ms=dt_ms,
            k_trans=k_trans,
            k_sync=k_sync,
            k_merge=k_merge,
            is_50hz_family=is_50hz,
            safe_gap_ms=safe_gap_ms,
            w_1s=max(1, int(round(1.0 * clean_fps))),
            w_5s=max(1, int(round(pse.extended_window_sec * clean_fps))),
            w_4s=max(1, int(round(pse.extended_active_sec * clean_fps))),
            n_static_0p5s=max(1, int(math.ceil(pse.pattern_min_duration_sec * clean_fps - 1e-6))),
        )


# ---------------------------------------------------------------------------
# Event & Analysis Result Dataclasses
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class TransitionEvent:
    """Single committed opposing luminance or saturated-red transition event."""

    ref_frame: int
    frame_idx: int
    t_ref_ms: float
    t_ms: float
    polarity: int  # +1 (brightening / to-red) or -1 (darkening / from-red)
    area: float
    mean_magnitude: float


@dataclass(frozen=True, slots=True)
class IntervalRecord:
    """Detected contiguous hazard interval for one of the 7 output labels."""

    label: str
    start_frame: int
    end_frame: int
    start_sec: float
    end_sec: float
    duration_sec: float
    peak_risk_calibrated: float
    peak_risk_ppr_pct: float
    peak_risk_nomura: float
    peak_risk_vfm_weibull: float


@dataclass(slots=True)
class VideoAnalysisResult:
    """Complete structured output from `analyze_video` / `analyze_frames`."""

    video_path: str
    fps: float
    total_frames: int
    orig_width: int
    orig_height: int
    analysis_width: int
    analysis_height: int
    device: str
    geometry: ViewingGeometry
    fps_spec: FPSWindowSpec
    is_dangerous: bool
    active_labels: list[str]
    intervals: list[IntervalRecord]
    intervals_by_label: dict[str, list[tuple[int, int]]]
    frame_masks: dict[str, np.ndarray]
    frame_metrics_df: pd.DataFrame
    intervals_df: pd.DataFrame
    physical_stats: dict[str, np.ndarray] | None = None
