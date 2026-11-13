"""End-to-end streaming analysis pipeline and public API functions (`analyze_video`, `analyze_frames`)."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Iterable, Sequence

import cv2
import numpy as np
import pandas as pd
import torch

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
    VideoAnalysisResult,
    ViewingConfig,
)
from .metrics import (
    NomuraTemporalFilter,
    PixelDiagnosticTracker,
    compute_ppr_and_calibrated_series,
    compute_vfm_weibull_series,
)
from .photometry import PhotometricConverter, resolve_device
from .spatial import StaticPatternDetector, filter_static_pattern_runs
from .temporal import (
    TemporalTransitionEngine,
    build_flash_boolean_mask,
    build_iris_extended_mask,
    merge_ultrafast_transitions,
    trailing_1s_event_stats,
)


def mask_to_spans(mask: np.ndarray) -> list[tuple[int, int]]:
    """Extract inclusive `(start_frame, end_frame)` pairs from a 1D boolean mask."""
    if len(mask) == 0 or not np.any(mask):
        return []
    padded = np.concatenate(([False], mask, [False]))
    edges = np.flatnonzero(padded[1:] != padded[:-1])
    return [(int(edges[i]), int(edges[i + 1] - 1)) for i in range(0, len(edges), 2)]


def _resize_for_analysis(bgr: np.ndarray, max_width: int) -> np.ndarray:
    h, w = bgr.shape[:2]
    if max_width <= 0 or w <= max_width:
        return bgr
    scale = max_width / float(w)
    new_w = max_width
    new_h = max(2, int(round(h * scale)))
    return cv2.resize(bgr, (new_w, new_h), interpolation=cv2.INTER_AREA)


DetectorCache = dict[
    tuple[int, int, str], tuple[PhotometricConverter, StaticPatternDetector]
]


def _run_analysis_stream(
    frame_iter: Iterable[tuple[int, float, np.ndarray]],
    fps: float,
    orig_width: int,
    orig_height: int,
    video_label: str,
    viewing: ViewingConfig,
    pse: PSEConfig,
    device: torch.device,
    collect_physical_stats: bool,
    detector_cache: DetectorCache | None = None,
) -> VideoAnalysisResult:
    if orig_width > viewing.analysis_max_width > 0:
        scale = viewing.analysis_max_width / float(orig_width)
        ana_w = viewing.analysis_max_width
        ana_h = max(2, int(round(orig_height * scale)))
    else:
        ana_w, ana_h = orig_width, orig_height

    geometry = viewing.resolve_geometry(ana_w, ana_h)
    fps_spec = FPSWindowSpec.from_fps(fps, pse)

    cache_key = (ana_w, ana_h, str(device))
    if detector_cache is not None and cache_key in detector_cache:
        converter, pattern_detector = detector_cache[cache_key]
        pattern_detector.reset()
    else:
        converter = PhotometricConverter(l_max_cd_m2=viewing.l_max_cd_m2, pse=pse, device=device)
        pattern_detector = StaticPatternDetector(geometry=geometry, pse=pse, device=device)
        if detector_cache is not None:
            detector_cache[cache_key] = (converter, pattern_detector)

    temporal_engine = TemporalTransitionEngine(fps_spec=fps_spec, pse=pse)
    diag_tracker = PixelDiagnosticTracker(
        height=ana_h, width=ana_w, w_1s=fps_spec.w_1s, pse=pse, device=device
    )
    nomura_filter = NomuraTemporalFilter(fps=fps_spec.fps, device=device)

    timestamps_list: list[float] = []
    mean_y_rel_list: list[float] = []
    lum_step_area_list: list[float] = []
    red_step_area_list: list[float] = []
    raw_pat_bool_list: list[bool] = []
    pat_area_list: list[float] = []
    pat_pairs_list: list[int] = []
    pat_cpd_list: list[float] = []
    diag_25_list: list[int] = []
    nomura_raw_list: list[float] = []
    nomura_scaled_list: list[float] = []

    phys_mean_lum: list[float] = []
    phys_max_lum: list[float] = []
    phys_sat_red_frac: list[float] = []
    phys_q76_red_lin: list[float] = []
    phys_q76_sat_red_excess: list[float] = []
    phys_q24_red_escape_beta: list[float] = []

    with torch.no_grad():
        for frame_idx, t_ms, raw_bgr in frame_iter:
            bgr = _resize_for_analysis(raw_bgr, viewing.analysis_max_width)
            rgb_lin, lum, u_prime, v_prime, is_sat_red = converter.convert_bgr(bgr)

            lum_area, red_area = temporal_engine.update(
                frame_idx=frame_idx,
                t_ms=t_ms,
                lum=lum,
                u_prime=u_prime,
                v_prime=v_prime,
                is_sat_red=is_sat_red,
            )
            is_pat, pat_area, pat_pairs, pat_cpd = pattern_detector.inspect_frame(lum)
            diag_cnt = diag_tracker.update(lum)
            nom_raw, nom_scaled = nomura_filter.update(rgb_lin)

            mean_y_rel = float((lum.mean() / max(viewing.l_max_cd_m2, 1e-6)).item())

            timestamps_list.append(t_ms)
            mean_y_rel_list.append(mean_y_rel)
            lum_step_area_list.append(lum_area)
            red_step_area_list.append(red_area)
            raw_pat_bool_list.append(is_pat)
            pat_area_list.append(pat_area)
            pat_pairs_list.append(pat_pairs)
            pat_cpd_list.append(pat_cpd)
            diag_25_list.append(diag_cnt)
            nomura_raw_list.append(nom_raw)
            nomura_scaled_list.append(nom_scaled)

            if collect_physical_stats:
                r_lin = rgb_lin[..., 0]
                g_lin = rgb_lin[..., 1]
                b_lin = rgb_lin[..., 2]
                gb_lin = g_lin + b_lin
                rgb_sum = torch.clamp(r_lin + gb_lin, min=1e-6)
                sat_ratio_ok = (r_lin / rgb_sum) >= pse.sat_red_ratio_min
                sat_red_excess = torch.where(
                    sat_ratio_ok,
                    torch.clamp(r_lin - gb_lin, min=0.0),
                    torch.zeros_like(r_lin),
                )
                red_escape_beta = torch.where(
                    is_sat_red,
                    torch.clamp((0.72 / 0.28) * gb_lin / torch.clamp(r_lin, min=1e-6), 0.0, 1.0),
                    torch.ones_like(r_lin),
                )
                p_stats = torch.stack(
                    [
                        lum.mean(),
                        lum.max(),
                        is_sat_red.float().mean(),
                        torch.quantile(r_lin.reshape(-1), 0.76),
                        torch.quantile(sat_red_excess.reshape(-1), 0.76),
                        torch.quantile(red_escape_beta.reshape(-1), 0.24),
                    ]
                ).tolist()
                phys_mean_lum.append(float(p_stats[0]))
                phys_max_lum.append(float(p_stats[1]))
                phys_sat_red_frac.append(float(p_stats[2]))
                phys_q76_red_lin.append(float(p_stats[3]))
                phys_q76_sat_red_excess.append(float(p_stats[4]))
                phys_q24_red_escape_beta.append(float(p_stats[5]))

    total_frames = len(timestamps_list)
    timestamps_ms = np.asarray(timestamps_list, dtype=np.float64)
    mean_y_rel_arr = np.asarray(mean_y_rel_list, dtype=np.float32)
    raw_pat_mask = np.asarray(raw_pat_bool_list, dtype=bool)
    pat_areas_arr = np.asarray(pat_area_list, dtype=np.float32)
    pat_pairs_arr = np.asarray(pat_pairs_list, dtype=np.int32)
    pat_cpds_arr = np.asarray(pat_cpd_list, dtype=np.float32)
    diag_25_arr = np.asarray(diag_25_list, dtype=np.int32)
    nomura_raw_arr = np.asarray(nomura_raw_list, dtype=np.float32)
    nomura_scaled_arr = np.asarray(nomura_scaled_list, dtype=np.float32)

    # Merge >=65 Hz (<=15 ms same-polarity) ultra-fast bursts before interval evaluation
    lum_events = merge_ultrafast_transitions(
        temporal_engine.lum_flash_state.events, pse.ultrafast_merge_ms
    )
    scene_events = merge_ultrafast_transitions(
        temporal_engine.scene_cut_state.events, pse.ultrafast_merge_ms
    )
    red_events = merge_ultrafast_transitions(
        temporal_engine.red_flash_state.events, pse.ultrafast_merge_ms
    )

    # Raw boolean hazard masks
    b_lf = build_flash_boolean_mask(
        total_frames=total_frames,
        timestamps_ms=timestamps_ms,
        events=lum_events,
        min_transitions=pse.flash_min_transitions,
        spacing_stride=2,
        safe_gap_ms=fps_spec.safe_gap_ms,
        window_1s_ms=pse.window_1s_ms,
    )
    b_st = build_flash_boolean_mask(
        total_frames=total_frames,
        timestamps_ms=timestamps_ms,
        events=scene_events,
        min_transitions=pse.scene_min_transitions,
        spacing_stride=1,
        safe_gap_ms=fps_spec.safe_gap_ms,
        window_1s_ms=pse.window_1s_ms,
    )
    b_rf = build_flash_boolean_mask(
        total_frames=total_frames,
        timestamps_ms=timestamps_ms,
        events=red_events,
        min_transitions=pse.flash_min_transitions,
        spacing_stride=2,
        safe_gap_ms=fps_spec.safe_gap_ms,
        window_1s_ms=pse.window_1s_ms,
    )

    lum_c1s, lum_a1s, lum_m1s = trailing_1s_event_stats(timestamps_ms, lum_events, pse.window_1s_ms)
    scn_c1s, scn_a1s, scn_m1s = trailing_1s_event_stats(timestamps_ms, scene_events, pse.window_1s_ms)
    red_c1s, red_a1s, red_m1s = trailing_1s_event_stats(timestamps_ms, red_events, pse.window_1s_ms)

    b_le_raw = build_iris_extended_mask(
        total_frames=total_frames,
        timestamps_ms=timestamps_ms,
        events=lum_events,
        counts_1s=lum_c1s,
        fps_spec=fps_spec,
        pse=pse,
    )
    b_re_raw = build_iris_extended_mask(
        total_frames=total_frames,
        timestamps_ms=timestamps_ms,
        events=red_events,
        counts_1s=red_c1s,
        fps_spec=fps_spec,
        pse=pse,
    )
    b_sp = filter_static_pattern_runs(raw_pat_mask, fps_spec.n_static_0p5s)

    # Set-algebra decomposition into the 7 structured output hazard labels (Stage 3 Section 0.2)
    final_masks: dict[str, np.ndarray] = {
        LABEL_LUM_FLASH: b_lf & (~b_st),
        LABEL_INTERSECTION: b_lf & b_st,
        LABEL_SCENE_TRANSITION: b_st & (~b_lf),
        LABEL_RED_FLASH: b_rf,
        LABEL_LUM_EXTENDED: b_le_raw & (~b_lf),
        LABEL_RED_EXTENDED: b_re_raw & (~b_rf),
        LABEL_STATIC_PATTERN: b_sp,
    }

    any_danger = np.zeros(total_frames, dtype=bool)
    for m in final_masks.values():
        any_danger |= m

    # Continuous risk metrics (Candidates A, B, C, D)
    risk_ppr_arr, risk_cal_arr = compute_ppr_and_calibrated_series(
        total_frames=total_frames,
        geometry=geometry,
        pse=pse,
        fps_spec=fps_spec,
        lum_counts_1s=lum_c1s,
        lum_areas_1s=lum_a1s,
        lum_mags_1s=lum_m1s,
        scene_counts_1s=scn_c1s,
        scene_areas_1s=scn_a1s,
        scene_mags_1s=scn_m1s,
        red_counts_1s=red_c1s,
        red_areas_1s=red_a1s,
        red_mags_1s=red_m1s,
        pat_areas=pat_areas_arr,
        pat_pairs=pat_pairs_arr,
        pat_cpds=pat_cpds_arr,
        any_danger_mask=any_danger,
        b_le=final_masks[LABEL_LUM_EXTENDED],
        b_re=final_masks[LABEL_RED_EXTENDED],
        b_sp=b_sp,
    )
    risk_vfm_arr = compute_vfm_weibull_series(
        mean_y_rel=mean_y_rel_arr,
        fps=fps_spec.fps,
        geometry=geometry,
        l_max_cd_m2=viewing.l_max_cd_m2,
    )

    intervals_by_label: dict[str, list[tuple[int, int]]] = {}
    interval_records: list[IntervalRecord] = []
    dt_sec = 1.0 / fps_spec.fps

    for label in ALL_HAZARD_LABELS:
        spans = mask_to_spans(final_masks[label])
        intervals_by_label[label] = spans
        for s_idx, e_idx in spans:
            s_sec = float(timestamps_ms[s_idx] / 1000.0)
            e_sec = float(timestamps_ms[e_idx] / 1000.0 + dt_sec)
            sl = slice(s_idx, e_idx + 1)
            interval_records.append(
                IntervalRecord(
                    label=label,
                    start_frame=s_idx,
                    end_frame=e_idx,
                    start_sec=round(s_sec, 4),
                    end_sec=round(e_sec, 4),
                    duration_sec=round(e_sec - s_sec, 4),
                    peak_risk_calibrated=round(float(np.max(risk_cal_arr[sl])), 2),
                    peak_risk_ppr_pct=round(float(np.max(risk_ppr_arr[sl])), 2),
                    peak_risk_nomura=round(float(np.max(nomura_raw_arr[sl])), 4),
                    peak_risk_vfm_weibull=round(float(np.max(risk_vfm_arr[sl])), 2),
                )
            )

    interval_records.sort(key=lambda rec: (rec.start_frame, rec.end_frame, rec.label))
    active_labels = [lbl for lbl in ALL_HAZARD_LABELS if intervals_by_label[lbl]]
    is_dangerous = bool(active_labels)

    frame_label_strs: list[str] = []
    for idx in range(total_frames):
        lbls = [lbl for lbl in ALL_HAZARD_LABELS if final_masks[lbl][idx]]
        frame_label_strs.append("+".join(lbls) if lbls else LABEL_SAFE)

    frame_metrics_df = pd.DataFrame(
        {
            "frame": np.arange(total_frames, dtype=np.int32),
            "timestamp_ms": np.round(timestamps_ms, 3),
            "hazard_labels": frame_label_strs,
            "is_dangerous": any_danger,
            "lum_step_area": np.round(lum_step_area_list, 4),
            "red_step_area": np.round(red_step_area_list, 4),
            "lum_transitions_1s": lum_c1s,
            "scene_transitions_1s": scn_c1s,
            "red_transitions_1s": red_c1s,
            "pattern_area": np.round(pat_areas_arr, 4),
            "pattern_pairs": pat_pairs_arr,
            "pattern_cpd": np.round(pat_cpds_arr, 3),
            "diag_transitions_25pct": diag_25_arr,
            "risk_ppr_pct": np.round(risk_ppr_arr, 3),
            "risk_calibrated": np.round(risk_cal_arr, 3),
            "risk_nomura": np.round(nomura_raw_arr, 5),
            "risk_nomura_scaled": np.round(nomura_scaled_arr, 3),
            "risk_vfm_weibull": np.round(risk_vfm_arr, 3),
        }
    )

    file_stem = Path(video_label).name
    if interval_records:
        intervals_df = pd.DataFrame(
            [
                {
                    "파일명": file_stem,
                    "위험 구분": rec.label,
                    "시작 프레임": rec.start_frame,
                    "끝 프레임": rec.end_frame,
                    "시작 시각(s)": rec.start_sec,
                    "끝 시각(s)": rec.end_sec,
                    "지속 시간(s)": rec.duration_sec,
                    "최대 risk_calibrated": rec.peak_risk_calibrated,
                    "최대 risk_ppr_pct": rec.peak_risk_ppr_pct,
                    "최대 risk_nomura": rec.peak_risk_nomura,
                    "최대 risk_vfm_weibull": rec.peak_risk_vfm_weibull,
                }
                for rec in interval_records
            ]
        )
    else:
        intervals_df = pd.DataFrame(
            [
                {
                    "파일명": file_stem,
                    "위험 구분": LABEL_SAFE,
                    "시작 프레임": -1,
                    "끝 프레임": -1,
                    "시작 시각(s)": 0.0,
                    "끝 시각(s)": 0.0,
                    "지속 시간(s)": 0.0,
                    "최대 risk_calibrated": round(
                        float(np.max(risk_cal_arr)) if total_frames else 0.0, 2
                    ),
                    "최대 risk_ppr_pct": round(
                        float(np.max(risk_ppr_arr)) if total_frames else 0.0, 2
                    ),
                    "최대 risk_nomura": round(
                        float(np.max(nomura_raw_arr)) if total_frames else 0.0, 4
                    ),
                    "최대 risk_vfm_weibull": round(
                        float(np.max(risk_vfm_arr)) if total_frames else 0.0, 2
                    ),
                }
            ]
        )

    all_masks = dict(final_masks)
    all_masks.update(
        {
            "raw_lum_flash": b_lf,
            "raw_scene_transition": b_st,
            "raw_lum_extended": b_le_raw,
            "raw_red_extended": b_re_raw,
            "raw_static_pattern": raw_pat_mask,
        }
    )

    physical_stats: dict[str, np.ndarray] | None = None
    if collect_physical_stats:
        physical_stats = {
            "mean_lum_cd_m2": np.asarray(phys_mean_lum, dtype=np.float32),
            "max_lum_cd_m2": np.asarray(phys_max_lum, dtype=np.float32),
            "sat_red_area": np.asarray(phys_sat_red_frac, dtype=np.float32),
            "q76_red_lin": np.asarray(phys_q76_red_lin, dtype=np.float32),
            "q76_sat_red_excess": np.asarray(phys_q76_sat_red_excess, dtype=np.float32),
            "q24_red_escape_beta": np.asarray(phys_q24_red_escape_beta, dtype=np.float32),
        }

    return VideoAnalysisResult(
        video_path=str(video_label),
        fps=fps_spec.fps,
        total_frames=total_frames,
        orig_width=orig_width,
        orig_height=orig_height,
        analysis_width=ana_w,
        analysis_height=ana_h,
        device=str(device),
        geometry=geometry,
        fps_spec=fps_spec,
        is_dangerous=is_dangerous,
        active_labels=active_labels,
        intervals=interval_records,
        intervals_by_label=intervals_by_label,
        frame_masks=all_masks,
        frame_metrics_df=frame_metrics_df,
        intervals_df=intervals_df,
        physical_stats=physical_stats,
    )


def analyze_frame_stream(
    frame_iter: Iterable[tuple[int, float, np.ndarray]],
    fps: float,
    orig_width: int,
    orig_height: int,
    *,
    video_name: str = "<stream>",
    viewing: ViewingConfig | None = None,
    pse: PSEConfig | None = None,
    device: str | torch.device | None = None,
    collect_physical_stats: bool = False,
    detector_cache: DetectorCache | None = None,
) -> VideoAnalysisResult:
    """Analyze a streaming iterable of `(frame_idx, timestamp_ms, bgr_u8)` tuples."""
    view_cfg = viewing or ViewingConfig()
    pse_cfg = pse or PSEConfig()
    dev = resolve_device(device)
    clean_fps = float(fps) if (fps > 0.0 and math.isfinite(fps)) else 30.0
    return _run_analysis_stream(
        frame_iter=frame_iter,
        fps=clean_fps,
        orig_width=orig_width,
        orig_height=orig_height,
        video_label=video_name,
        viewing=view_cfg,
        pse=pse_cfg,
        device=dev,
        collect_physical_stats=collect_physical_stats,
        detector_cache=detector_cache,
    )


def analyze_frames(
    frames_bgr: Sequence[np.ndarray],
    fps: float,
    *,
    timestamps_ms: Sequence[float] | None = None,
    video_name: str = "<memory>",
    viewing: ViewingConfig | None = None,
    pse: PSEConfig | None = None,
    device: str | torch.device | None = None,
    collect_physical_stats: bool = False,
) -> VideoAnalysisResult:
    """Analyze an in-memory sequence of BGR uint8 frames."""
    if not frames_bgr:
        raise ValueError("`frames_bgr` must contain at least one frame.")
    view_cfg = viewing or ViewingConfig()
    pse_cfg = pse or PSEConfig()
    dev = resolve_device(device)

    h, w = frames_bgr[0].shape[:2]
    clean_fps = float(fps) if (fps > 0.0 and math.isfinite(fps)) else 30.0
    dt_ms = 1000.0 / clean_fps

    def _gen() -> Iterable[tuple[int, float, np.ndarray]]:
        for idx, frame in enumerate(frames_bgr):
            t_ms = float(timestamps_ms[idx]) if timestamps_ms is not None else (idx * dt_ms)
            yield idx, t_ms, frame

    return _run_analysis_stream(
        frame_iter=_gen(),
        fps=clean_fps,
        orig_width=w,
        orig_height=h,
        video_label=video_name,
        viewing=view_cfg,
        pse=pse_cfg,
        device=dev,
        collect_physical_stats=collect_physical_stats,
    )


def analyze_video(
    video_path: str | Path,
    *,
    viewing: ViewingConfig | None = None,
    pse: PSEConfig | None = None,
    device: str | torch.device | None = None,
    use_container_timestamps: bool = False,
    collect_physical_stats: bool = False,
    detector_cache: DetectorCache | None = None,
) -> VideoAnalysisResult:
    """Analyze a video file on disk and return full PSE intervals, masks, and 4 risk metrics."""
    path_obj = Path(video_path)
    if not path_obj.exists():
        raise FileNotFoundError(f"Video file not found: {path_obj}")

    cap = cv2.VideoCapture(str(path_obj))
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video file: {path_obj}")

    try:
        fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
        if not math.isfinite(fps) or fps <= 0.0:
            fps = 30.0
        orig_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
        orig_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)

        first_ok, first_frame = cap.read()
        if not first_ok or first_frame is None:
            raise RuntimeError(f"Video contains no readable frames: {path_obj}")
        if orig_w <= 0 or orig_h <= 0:
            orig_h, orig_w = first_frame.shape[:2]

        view_cfg = viewing or ViewingConfig()
        pse_cfg = pse or PSEConfig()
        dev = resolve_device(device)
        dt_ms = 1000.0 / fps

        def _video_gen() -> Iterable[tuple[int, float, np.ndarray]]:
            yield 0, 0.0, first_frame
            idx = 1
            prev_t = 0.0
            while True:
                ok, frame = cap.read()
                if not ok or frame is None:
                    break
                if use_container_timestamps:
                    pos_ms = float(cap.get(cv2.CAP_PROP_POS_MSEC) or 0.0)
                    t_ms = pos_ms if pos_ms > prev_t else (idx * dt_ms)
                else:
                    t_ms = idx * dt_ms
                prev_t = t_ms
                yield idx, t_ms, frame
                idx += 1

        return _run_analysis_stream(
            frame_iter=_video_gen(),
            fps=fps,
            orig_width=orig_w,
            orig_height=orig_h,
            video_label=str(path_obj),
            viewing=view_cfg,
            pse=pse_cfg,
            device=dev,
            collect_physical_stats=collect_physical_stats,
            detector_cache=detector_cache,
        )
    finally:
        cap.release()


def evaluate_video_pse_risk(
    video_path: str | Path,
    *,
    viewing: ViewingConfig | None = None,
    pse: PSEConfig | None = None,
    device: str | torch.device | None = None,
) -> pd.DataFrame:
    """One-call helper returning the structured interval summary DataFrame for `video_path`."""
    return analyze_video(video_path, viewing=viewing, pse=pse, device=device).intervals_df
