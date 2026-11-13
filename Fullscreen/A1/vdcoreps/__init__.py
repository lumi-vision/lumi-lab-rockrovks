"""LUMI Partial-Screen Photosensitive Epilepsy (PSE) Detection & Mask Core Package (`vdcoreps`).

Implements Structure B on top of `vdcorefs`:
  - Re-exports unchanged PSE quantitative configurations, photometric converters, and continuous
    risk metrics from `vdcorefs` so binary 7-category hazard verdicts and 4 risk metrics remain
    100% identical.
  - Provides pixel-level hazard localization across temporal luminance flashes, saturated-red
    flashes, and static spatial patterns (`Close(M_contrast) U M_shell U M_wave`), integrated
    with `+-0.25s` temporal max-pooling, `0.20s` cosine boundary smoothing, and one-sided
    outward spatial feathering (`W_t == 1.0` strictly preserved on core hazard pixels).
"""

from __future__ import annotations

from pathlib import Path as _Path
import sys as _sys

_PKG_PARENT = str(_Path(__file__).resolve().parent.parent)
if _PKG_PARENT not in _sys.path:
    _sys.path.insert(0, _PKG_PARENT)

from vdcorefs.metrics import (
    NomuraTemporalFilter,
    PixelDiagnosticTracker,
    compute_ppr_and_calibrated_series,
    compute_vfm_weibull_series,
)
from vdcorefs.photometry import PhotometricConverter, resolve_device

from .config import (
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
    PackedHazardBuffers,
    PartialMaskConfig,
    PartialTemporalScheduleMode,
    PartialVideoAnalysisResult,
    TransitionEvent,
    VideoAnalysisResult,
    ViewingConfig,
    ViewingGeometry,
    ViewingProfile,
)
from .mask_ops import (
    apply_guided_borderless_batch,
    apply_outward_spatial_feather_batch,
    apply_partial_linear_dimming_bgr,
    build_mode_weight_masks,
    build_soft_weight_masks_u8,
    clean_speckles_packed,
    compute_4metrics_union_from_result,
    pack_bool_mask_2d,
    temporal_bridge_spans_packed,
    temporal_sliding_or_packed,
    unpack_bool_masks_3d,
    upsample_weight_mask,
)
from .pipeline import (
    PartialDetectorCache,
    analyze_frame_stream,
    analyze_frames,
    analyze_video,
    evaluate_video_pse_risk,
    mask_to_spans,
)
from .spatial_mask import (
    PartialStaticPatternDetector,
    extract_hysteresis_extrema_1d,
    filter_static_pattern_runs,
    morphological_close_2d,
    separable_max_min_2d,
)
from .temporal_mask import (
    PartialTemporalTransitionEngine,
    build_flash_boolean_mask,
    build_iris_extended_mask,
    merge_ultrafast_transitions,
    trailing_1s_event_stats,
)

# Alias matching `vdcorefs.DetectorCache` naming convention for downstream convenience
DetectorCache = PartialDetectorCache

__all__ = [
    # Labels & constants
    "LABEL_LUM_FLASH",
    "LABEL_INTERSECTION",
    "LABEL_SCENE_TRANSITION",
    "LABEL_RED_FLASH",
    "LABEL_LUM_EXTENDED",
    "LABEL_RED_EXTENDED",
    "LABEL_STATIC_PATTERN",
    "LABEL_SAFE",
    "ALL_HAZARD_LABELS",
    "ViewingProfile",
    "PartialTemporalScheduleMode",
    # Config & result types
    "ViewingGeometry",
    "ViewingConfig",
    "PSEConfig",
    "FPSWindowSpec",
    "TransitionEvent",
    "IntervalRecord",
    "VideoAnalysisResult",
    "PartialMaskConfig",
    "PackedHazardBuffers",
    "PartialVideoAnalysisResult",
    # Photometry & device
    "resolve_device",
    "PhotometricConverter",
    # Temporal & spatial localization engines
    "PartialTemporalTransitionEngine",
    "merge_ultrafast_transitions",
    "build_flash_boolean_mask",
    "trailing_1s_event_stats",
    "build_iris_extended_mask",
    "PartialStaticPatternDetector",
    "morphological_close_2d",
    "extract_hysteresis_extrema_1d",
    "separable_max_min_2d",
    "filter_static_pattern_runs",
    # Continuous metrics
    "PixelDiagnosticTracker",
    "NomuraTemporalFilter",
    "compute_vfm_weibull_series",
    "compute_ppr_and_calibrated_series",
    # Spatiotemporal mask operations & Formula A helpers
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
    # Pipeline entry points
    "DetectorCache",
    "PartialDetectorCache",
    "mask_to_spans",
    "analyze_frame_stream",
    "analyze_frames",
    "analyze_video",
    "evaluate_video_pse_risk",
]
