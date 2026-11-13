"""Configuration, partial-mask parameters, and structured output types for `vdcoreps`.

Follows Structure B: re-exports unchanged PSE quantitative configurations and label constants
from `vdcorefs.config`, and extends them with partial-screen spatiotemporal mask settings
and `PartialVideoAnalysisResult`.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

import cv2
import numpy as np

from vdcorefs.config import (
    ALL_HAZARD_LABELS,
    FPSWindowSpec,
    IntervalRecord,
    LABEL_INTERSECTION,
    LABEL_LUM_EXTENDED,
    LABEL_LUM_FLASH,
    LABEL_RED_EXTENDED,
    LABEL_RED_FLASH,
    LABEL_SAFE,
    LABEL_SCENE_TRANSITION,
    LABEL_STATIC_PATTERN,
    PSEConfig,
    TransitionEvent,
    VideoAnalysisResult,
    ViewingConfig,
    ViewingGeometry,
    ViewingProfile,
)

PartialTemporalScheduleMode = Literal["base", "hold", "safe"]


@dataclass(frozen=True, slots=True)
class PartialMaskConfig:
    """Spatiotemporal partial hazard mask parameters (4-1A, 4-2A, 4-3A + tight localization).

    Attributes:
        temporal_window_sec: Sliding temporal max-union half-window radius in seconds
            for holding localized flash pixels across consecutive ON/OFF half-cycles
            (default `0.16`, covering `~0.360s` broadcast safe gap without motion smearing).
        temporal_fade_sec: Temporal cosine ramp duration at active mask entry/exit boundaries
            in seconds (default `0.20`).
        half_hold_duration_sec: Half-Hold plateau duration in seconds when generating masks
            aligned with `hold` or `safe` scheduling modes (default `0.50`).
        extended_bridge_sec: Maximum internal plateau gap in seconds bridged strictly inside
            extended hazard intervals (`휘도 연장`, `적색 연장`).
        feather_angle_deg: Narrow one-sided outward spatial feathering width in degrees of
            visual angle outside the `W = 1.0` core hazard mask (default `0.40`).
        core_pad_px: Inner `W = 1.0` safety collar in analysis pixels around core hazard pixels
            so `INTER_LINEAR` upsampling + `INTER_AREA` verification never leaks at boundaries.
        min_cc_area_px: Minimum connected-component area in analysis-grid pixels retained after
            micro-closing, eliminating isolated single-pixel/speckle noise inflation.
        flash_min_reversals_1s: Minimum per-pixel opposing transitions within a `1.0s` window
            required for partial flash core membership (`1-a`, default `4`).
        flash_residual_area_cap: Maximum residual unmasked transition area allowed on a
            committed flash event (`< 0.25` ITU-R threshold) before relaxing reversal count.
        fullscreen_scene_transition: When `True` (`1-1`), applies full-screen dimming
            (`W_t == 1.0`) across `raw_scene_transition` (`빠른 장면 전환` U `교집합`) frames.
        pattern_close_scale: Morphological closing radius scale relative to
            `StaticPatternDetector.pool_radius` for filling internal grating gaps.
        safe_metric_threshold: Threshold for `[0, 100]` continuous risk metrics in `safe` mode.
        safe_nomura_threshold: Threshold for `[0, 1]` `risk_nomura` in `safe` mode.
        feather_batch_size: Frame batch size for GPU/CPU spatial outward feathering.
    """

    temporal_window_sec: float = 0.16
    temporal_fade_sec: float = 0.20
    half_hold_duration_sec: float = 0.50
    extended_bridge_sec: float = 1.00
    feather_angle_deg: float = 0.40
    core_pad_px: int = 1
    min_cc_area_px: int = 80
    flash_min_reversals_1s: int = 4
    flash_residual_area_cap: float = 0.18
    fullscreen_scene_transition: bool = True
    pattern_close_scale: float = 0.50
    borderless_guided: bool = True
    guided_radius_deg: float = 1.15
    guided_eps: float = 0.005
    guided_pre_sigma_deg: float = 0.60
    guided_post_sigma_deg: float = 0.70
    guided_reach_deg: float = 0.60
    guided_knee: float = 0.48
    temporal_smooth_radius: int = 1
    safe_metric_threshold: float = 50.0
    safe_nomura_threshold: float = 0.50
    feather_batch_size: int = 64


@dataclass(slots=True)
class PackedHazardBuffers:
    """Bit-packed (`np.packbits(..., axis=-1)`) analysis-grid core masks for multi-mode reuse.

    Allows `A1/partial` (`base`, `hold`, `safe`) to construct mode-matched, structure-guided
    borderless spatiotemporal soft masks from a single Pass 1 video decode.
    """

    analysis_height: int
    analysis_width: int
    confirmed_core_packed: np.ndarray
    safe_core_packed: np.ndarray
    guide_rgb_u8: np.ndarray | None = None
    flash_frame_mask: np.ndarray | None = None
    safe_flash_frame_mask: np.ndarray | None = None


@dataclass(slots=True)
class PartialVideoAnalysisResult(VideoAnalysisResult):
    """Extended `VideoAnalysisResult` with partial-screen soft masks and masked physical stats.

    Inherits 100% of `vdcorefs.VideoAnalysisResult` fields for full backward compatibility,
    while adding analysis-resolution `uint8` (`[0..255]`) soft weight masks `W_t(y, x)`,
    per-frame active mask area ratios, and compact packed core masks.
    """

    mask_config: PartialMaskConfig = field(default_factory=PartialMaskConfig)
    weight_masks_u8: np.ndarray | None = None
    active_mask_area: np.ndarray | None = None
    packed_hazards: PackedHazardBuffers | None = None

    def get_frame_weight_mask(
        self,
        frame_idx: int,
        *,
        orig_resolution: bool = False,
    ) -> np.ndarray:
        """Return normalized `float32` soft mask `W_t` in `[0.0, 1.0]` for `frame_idx`."""
        if self.weight_masks_u8 is None:
            raise ValueError("`weight_masks_u8` is None (`collect_masks=False`).")
        if frame_idx < 0 or frame_idx >= self.total_frames:
            raise IndexError(f"Frame index {frame_idx} out of range [0, {self.total_frames}).")

        mask_u8 = self.weight_masks_u8[frame_idx]
        target_w = self.orig_width if orig_resolution else self.analysis_width
        target_h = self.orig_height if orig_resolution else self.analysis_height

        mx = int(mask_u8.max())
        if mx == 0:
            return np.zeros((target_h, target_w), dtype=np.float32)
        if int(mask_u8.min()) == 255:
            return np.ones((target_h, target_w), dtype=np.float32)
        if mask_u8.shape == (target_h, target_w):
            return mask_u8.astype(np.float32) * (1.0 / 255.0)

        up_u8 = cv2.resize(mask_u8, (target_w, target_h), interpolation=cv2.INTER_LINEAR)
        return up_u8.astype(np.float32) * (1.0 / 255.0)
