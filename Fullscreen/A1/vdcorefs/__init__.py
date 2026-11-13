"""LUMI Fullscreen Photosensitive Epilepsy (PSE) Detection Core Package (`vdcorefs`).

Public facade re-exporting configurations, photometric/temporal/spatial/metric engines,
and high-level video/frame analysis entry points.
"""

from __future__ import annotations

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
    TransitionEvent,
    VideoAnalysisResult,
    ViewingConfig,
    ViewingGeometry,
    ViewingProfile,
)
from .metrics import (
    NomuraTemporalFilter,
    PixelDiagnosticTracker,
    compute_ppr_and_calibrated_series,
    compute_vfm_weibull_series,
)
from .photometry import PhotometricConverter, resolve_device
from .pipeline import (
    DetectorCache,
    analyze_frame_stream,
    analyze_frames,
    analyze_video,
    evaluate_video_pse_risk,
    mask_to_spans,
)
from .spatial import (
    StaticPatternDetector,
    extract_hysteresis_extrema_1d,
    filter_static_pattern_runs,
    separable_max_min_2d,
)
from .temporal import (
    TemporalTransitionEngine,
    build_flash_boolean_mask,
    build_iris_extended_mask,
    merge_ultrafast_transitions,
    trailing_1s_event_stats,
)

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
    # Config & result types
    "ViewingGeometry",
    "ViewingConfig",
    "PSEConfig",
    "FPSWindowSpec",
    "TransitionEvent",
    "IntervalRecord",
    "VideoAnalysisResult",
    # Photometry & device
    "resolve_device",
    "PhotometricConverter",
    # Temporal engine
    "TemporalTransitionEngine",
    "merge_ultrafast_transitions",
    "build_flash_boolean_mask",
    "trailing_1s_event_stats",
    "build_iris_extended_mask",
    # Spatial engine
    "StaticPatternDetector",
    "extract_hysteresis_extrema_1d",
    "separable_max_min_2d",
    "filter_static_pattern_runs",
    # Continuous metrics
    "PixelDiagnosticTracker",
    "NomuraTemporalFilter",
    "compute_vfm_weibull_series",
    "compute_ppr_and_calibrated_series",
    # Pipeline entry points
    "DetectorCache",
    "mask_to_spans",
    "analyze_frame_stream",
    "analyze_frames",
    "analyze_video",
    "evaluate_video_pse_risk",
]
