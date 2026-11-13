"""End-to-end partial-screen PSE analysis pipeline (`vdcoreps/pipeline.py`).

Follows Structure B: reuses unchanged photometric, continuous-metric, and interval post-processing
components from `vdcorefs` so all 7 binary hazard categories and 4 continuous risk metrics remain
100% identical to `vdcorefs`, while producing analysis-grid soft spatial masks `W_t(y, x)` (`uint8`),
compact multi-mode core buffers, and masked-region physical statistics.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Iterable, Sequence

import cv2
import numpy as np
import pandas as pd
import torch

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
    ViewingConfig,
)
from vdcorefs.metrics import (
    NomuraTemporalFilter,
    PixelDiagnosticTracker,
    compute_ppr_and_calibrated_series,
    compute_vfm_weibull_series,
)
from vdcorefs.photometry import PhotometricConverter, resolve_device
from vdcorefs.pipeline import _resize_for_analysis, mask_to_spans
from vdcorefs.spatial import filter_static_pattern_runs
from vdcorefs.temporal import (
    build_flash_boolean_mask,
    build_iris_extended_mask,
    merge_ultrafast_transitions,
    trailing_1s_event_stats,
)

from .config import (
    PackedHazardBuffers,
    PartialMaskConfig,
    PartialVideoAnalysisResult,
)
from .mask_ops import (
    build_soft_weight_masks_u8,
    clean_speckles_packed,
    pack_bool_mask_2d,
    temporal_bridge_spans_packed,
    temporal_sliding_or_packed,
    unpack_bool_masks_3d,
)
from .spatial_mask import PartialStaticPatternDetector
from .temporal_mask import (
    PartialTemporalTransitionEngine,
    RedPixelReversalTracker,
    build_committed_flash_core_packed,
    forward_window_or_packed,
    merge_ultrafast_transitions_with_masks,
)

PartialDetectorCache = dict[
    tuple[int, int, str], tuple[PhotometricConverter, PartialStaticPatternDetector]
]


def _extract_masked_frame_stats(
    mask_bool: np.ndarray,
    lum_f16: np.ndarray,
    red_exc_f16: np.ndarray | None,
) -> tuple[float, float, float]:
    """Compute `(mask_max_lum, mask_max_sat_red_excess, mask_q76_sat_red_excess)` on `mask_bool`."""
    if not np.any(mask_bool):
        return 0.0, 0.0, 0.0

    lum_vals = lum_f16[mask_bool]
    max_lum = float(lum_vals.max())

    if red_exc_f16 is None:
        return max_lum, 0.0, 0.0

    red_vals = red_exc_f16[mask_bool]
    if red_vals.size == 0:
        return max_lum, 0.0, 0.0
    max_red = float(red_vals.max())
    if max_red <= 0.0:
        return max_lum, 0.0, 0.0
    q76_red = float(np.quantile(red_vals.astype(np.float32), 0.76))
    return max_lum, max_red, q76_red


def _run_partial_analysis_stream(
    frame_iter: Iterable[tuple[int, float, np.ndarray]],
    fps: float,
    orig_width: int,
    orig_height: int,
    video_label: str,
    viewing: ViewingConfig,
    pse: PSEConfig,
    mask_cfg: PartialMaskConfig,
    device: torch.device,
    collect_physical_stats: bool,
    collect_masks: bool,
    detector_cache: PartialDetectorCache | None = None,
) -> PartialVideoAnalysisResult:
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
        pattern_detector = PartialStaticPatternDetector(geometry=geometry, pse=pse, device=device)
        if detector_cache is not None:
            detector_cache[cache_key] = (converter, pattern_detector)

    temporal_engine = PartialTemporalTransitionEngine(fps_spec=fps_spec, pse=pse)
    diag_tracker = PixelDiagnosticTracker(
        height=ana_h, width=ana_w, w_1s=fps_spec.w_1s, pse=pse, device=device
    )
    red_rev_tracker = (
        RedPixelReversalTracker(
            height=ana_h, width=ana_w, w_1s=fps_spec.w_1s, pse=pse, device=device
        )
        if collect_masks
        else None
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

    frame_to_pos: dict[int, int] = {}
    lum_trail_p_list: list[np.ndarray] = []
    lum_trail_f_list: list[np.ndarray] = []
    red_trail_p_list: list[np.ndarray] = []
    red_trail_f_list: list[np.ndarray] = []
    raw_pat_packed_list: list[np.ndarray] = []
    raw_bright_packed_list: list[np.ndarray] = []
    raw_sat_red_packed_dict: dict[int, np.ndarray] = {}
    guide_rgb_list: list[np.ndarray] = []
    guide_half_res: bool = False

    lum_f16_list: list[np.ndarray] = []
    red_exc_f16_dict: dict[int, np.ndarray] = {}

    need_pixel_stats_buf = collect_masks and collect_physical_stats
    min_rev_1s = max(2, int(mask_cfg.flash_min_reversals_1s))

    with torch.no_grad():
        for frame_idx, t_ms, raw_bgr in frame_iter:
            pos_idx = len(timestamps_list)
            frame_to_pos[frame_idx] = pos_idx

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
            diag_cnt = diag_tracker.update(lum)

            if collect_masks:
                is_pat, pat_area, pat_pairs, pat_cpd, pat_mask = (
                    pattern_detector.inspect_frame_with_mask(
                        lum, close_scale=mask_cfg.pattern_close_scale
                    )
                )
                assert red_rev_tracker is not None
                red_rc = red_rev_tracker.update(u_prime, v_prime, is_sat_red)
                lum_rc = diag_tracker._running_counts

                lum_trail_p_list.append(pack_bool_mask_2d(lum_rc >= min_rev_1s))
                lum_trail_f_list.append(pack_bool_mask_2d(lum_rc >= 2))
                red_trail_p_list.append(pack_bool_mask_2d(red_rc >= min_rev_1s))
                red_trail_f_list.append(pack_bool_mask_2d(red_rc >= 2))

                raw_pat_packed_list.append(pack_bool_mask_2d(pat_mask))
                raw_bright_packed_list.append(pack_bool_mask_2d(lum >= pse.delta_lum_min_cd_m2))
                if bool(is_sat_red.any().item()):
                    raw_sat_red_packed_dict[pos_idx] = pack_bool_mask_2d(is_sat_red)
                if mask_cfg.borderless_guided:
                    if not guide_half_res and pos_idx >= 2000 and ana_w > 320:
                        hw_s = (max(1, ana_w // 2), max(1, ana_h // 2))
                        guide_rgb_list = [
                            cv2.resize(g, hw_s, interpolation=cv2.INTER_AREA)
                            for g in guide_rgb_list
                        ]
                        guide_half_res = True
                    rgb_frame = bgr[:, :, ::-1]
                    if guide_half_res:
                        rgb_frame = cv2.resize(
                            rgb_frame,
                            (max(1, ana_w // 2), max(1, ana_h // 2)),
                            interpolation=cv2.INTER_AREA,
                        )
                    else:
                        rgb_frame = rgb_frame.copy()
                    guide_rgb_list.append(rgb_frame)
            else:
                is_pat, pat_area, pat_pairs, pat_cpd = pattern_detector.inspect_frame(lum)

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

                if need_pixel_stats_buf:
                    lum_f16_list.append(lum.to(torch.float16).cpu().numpy())
                    if bool(sat_ratio_ok.any().item()):
                        red_exc_f16_dict[pos_idx] = sat_red_excess.to(torch.float16).cpu().numpy()

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

    # Identical event merging & 7-category interval decomposition as `vdcorefs`
    if collect_masks:
        lum_events, lum_ev_masks = merge_ultrafast_transitions_with_masks(
            temporal_engine.lum_flash_state.events,
            temporal_engine.lum_flash_state.event_packed_masks,
            pse.ultrafast_merge_ms,
        )
        scene_events, scene_ev_masks = merge_ultrafast_transitions_with_masks(
            temporal_engine.scene_cut_state.events,
            temporal_engine.scene_cut_state.event_packed_masks,
            pse.ultrafast_merge_ms,
        )
        red_events, red_ev_masks = merge_ultrafast_transitions_with_masks(
            temporal_engine.red_flash_state.events,
            temporal_engine.red_flash_state.event_packed_masks,
            pse.ultrafast_merge_ms,
        )
    else:
        lum_events = merge_ultrafast_transitions(
            temporal_engine.lum_flash_state.events, pse.ultrafast_merge_ms
        )
        scene_events = merge_ultrafast_transitions(
            temporal_engine.scene_cut_state.events, pse.ultrafast_merge_ms
        )
        red_events = merge_ultrafast_transitions(
            temporal_engine.red_flash_state.events, pse.ultrafast_merge_ms
        )
        lum_ev_masks = []
        scene_ev_masks = []
        red_ev_masks = []

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
    scn_c1s, scn_a1s, scn_m1s = trailing_1s_event_stats(
        timestamps_ms, scene_events, pse.window_1s_ms
    )
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

    # -----------------------------------------------------------------------
    # Spatiotemporal Partial Hazard Mask Integration & Masked Physical Stats
    # -----------------------------------------------------------------------
    weight_masks_u8: np.ndarray | None = None
    active_mask_area: np.ndarray | None = None
    packed_hazards: PackedHazardBuffers | None = None

    mask_max_lum_arr = np.zeros(total_frames, dtype=np.float32)
    mask_max_red_arr = np.zeros(total_frames, dtype=np.float32)
    mask_q76_red_arr = np.zeros(total_frames, dtype=np.float32)
    safe_max_lum_arr = np.zeros(total_frames, dtype=np.float32)
    safe_max_red_arr = np.zeros(total_frames, dtype=np.float32)
    safe_q76_red_arr = np.zeros(total_frames, dtype=np.float32)

    if collect_masks:
        packed_w = (ana_w + 7) // 8
        if total_frames > 0:
            lum_trail_p = np.stack(lum_trail_p_list, axis=0)
            lum_trail_f = np.stack(lum_trail_f_list, axis=0)
            red_trail_p = np.stack(red_trail_p_list, axis=0)
            red_trail_f = np.stack(red_trail_f_list, axis=0)
            raw_pat_packed = np.stack(raw_pat_packed_list, axis=0)
            raw_bright_packed = np.stack(raw_bright_packed_list, axis=0)
        else:
            empty_p = np.zeros((0, ana_h, packed_w), dtype=np.uint8)
            lum_trail_p = empty_p
            lum_trail_f = empty_p
            red_trail_p = empty_p
            red_trail_f = empty_p
            raw_pat_packed = empty_p
            raw_bright_packed = empty_p

        del (
            lum_trail_p_list,
            lum_trail_f_list,
            red_trail_p_list,
            red_trail_f_list,
            raw_pat_packed_list,
            raw_bright_packed_list,
        )

        # Non-causal `1.0s` oscillation membership covering `[ref_frame, frame_idx]` (`1-a`)
        fwd_win = fps_spec.w_1s + fps_spec.k_trans
        lum_osc_p = forward_window_or_packed(lum_trail_p, fwd_win)
        lum_osc_f = forward_window_or_packed(lum_trail_f, fwd_win)
        red_osc_p = forward_window_or_packed(red_trail_p, fwd_win)
        red_osc_f = forward_window_or_packed(red_trail_f, fwd_win)
        del lum_trail_p, lum_trail_f, red_trail_p, red_trail_f

        lum_ev_core = build_committed_flash_core_packed(
            total_frames=total_frames,
            ana_height=ana_h,
            ana_width=ana_w,
            events=lum_events,
            event_masks=lum_ev_masks,
            frame_to_pos=frame_to_pos,
            osc_primary_packed=lum_osc_p,
            osc_fallback_packed=lum_osc_f,
            residual_area_cap=mask_cfg.flash_residual_area_cap,
        )
        red_ev_core = build_committed_flash_core_packed(
            total_frames=total_frames,
            ana_height=ana_h,
            ana_width=ana_w,
            events=red_events,
            event_masks=red_ev_masks,
            frame_to_pos=frame_to_pos,
            osc_primary_packed=red_osc_p,
            osc_fallback_packed=red_osc_f,
            residual_area_cap=mask_cfg.flash_residual_area_cap,
        )
        del lum_osc_p, red_osc_p

        r_win = max(2, int(round(mask_cfg.temporal_window_sec * fps_spec.fps)))
        safe_gap_frames = max(r_win * 2, int(math.ceil((fps_spec.safe_gap_ms / 1000.0) * fps_spec.fps)))
        ext_bridge_frames = max(safe_gap_frames, int(round(mask_cfg.extended_bridge_sec * fps_spec.fps)))
        r_pat_stab = min(r_win, max(1, int(round(0.06 * fps_spec.fps))))

        in_lum_flash_only = b_lf & (~b_st)
        in_lum_ext_only = final_masks[LABEL_LUM_EXTENDED]
        in_lum_partial = in_lum_flash_only | in_lum_ext_only
        in_red_flash_only = b_rf
        in_red_ext_only = final_masks[LABEL_RED_EXTENDED]
        in_red_temp = in_red_flash_only | in_red_ext_only
        in_pat_cat = b_sp

        # 1) Partial luminance flash & extended flash core mask (strictly category-separated)
        pooled_lum = temporal_sliding_or_packed(lum_ev_core, r_win)
        bridged_lum = temporal_bridge_spans_packed(
            pooled_lum, mask_to_spans(in_lum_flash_only), safe_gap_frames
        )
        if np.any(in_lum_ext_only):
            bridged_lum = temporal_bridge_spans_packed(
                bridged_lum, mask_to_spans(in_lum_ext_only), ext_bridge_frames
            )

        # 2) Partial saturated-red flash & extended flash core mask
        pooled_red = temporal_sliding_or_packed(red_ev_core, r_win)
        bridged_red = temporal_bridge_spans_packed(
            pooled_red, mask_to_spans(in_red_flash_only), safe_gap_frames
        )
        if np.any(in_red_ext_only):
            bridged_red = temporal_bridge_spans_packed(
                bridged_red, mask_to_spans(in_red_ext_only), ext_bridge_frames
            )
        for idx in np.flatnonzero(in_red_temp):
            if not np.any(bridged_red[idx]) and idx in raw_sat_red_packed_dict:
                bridged_red[idx] = raw_sat_red_packed_dict[idx]

        # 3) Static spatial pattern core mask (`2-a`, gated strictly to `b_sp`)
        pat_gated = np.zeros_like(raw_pat_packed)
        if np.any(raw_pat_mask):
            pat_gated[raw_pat_mask] = raw_pat_packed[raw_pat_mask]
        stab_pat = temporal_bridge_spans_packed(
            temporal_sliding_or_packed(pat_gated, r_pat_stab),
            mask_to_spans(in_pat_cat),
            r_pat_stab * 2,
        )

        # 4) Combine partial hazard masks & clean isolated speckles (`4-a`), then apply `1-1` full-screen scene cuts
        full_ones_2d = pack_bool_mask_2d(np.ones((ana_h, ana_w), dtype=bool))
        confirmed_core_packed = np.zeros_like(raw_pat_packed)
        if np.any(in_lum_partial):
            confirmed_core_packed[in_lum_partial] |= bridged_lum[in_lum_partial]
        if np.any(in_red_temp):
            confirmed_core_packed[in_red_temp] |= bridged_red[in_red_temp]
        if np.any(in_pat_cat):
            confirmed_core_packed[in_pat_cat] |= stab_pat[in_pat_cat]

        if np.any(b_st):
            if mask_cfg.fullscreen_scene_transition:
                confirmed_core_packed[b_st] = full_ones_2d[None, :, :]
            else:
                scn_core = build_committed_flash_core_packed(
                    total_frames=total_frames,
                    ana_height=ana_h,
                    ana_width=ana_w,
                    events=scene_events,
                    event_masks=scene_ev_masks,
                    frame_to_pos=frame_to_pos,
                    osc_primary_packed=lum_osc_f,
                    osc_fallback_packed=lum_osc_f,
                    residual_area_cap=0.70,
                )
                scn_bridged = temporal_bridge_spans_packed(
                    temporal_sliding_or_packed(scn_core, r_win),
                    mask_to_spans(b_st),
                    safe_gap_frames,
                )
                confirmed_core_packed[b_st] |= scn_bridged[b_st]

        partial_danger_only = any_danger & (~(b_st if mask_cfg.fullscreen_scene_transition else np.zeros_like(b_st)))
        confirmed_core_packed = clean_speckles_packed(
            confirmed_core_packed,
            active_frame_mask=partial_danger_only,
            width=ana_w,
            min_cc_area_px=mask_cfg.min_cc_area_px,
        )

        for idx in np.flatnonzero(any_danger):
            if not np.any(confirmed_core_packed[idx]):
                confirmed_core_packed[idx] = raw_bright_packed[idx]

        # 5) Safe-mode (`confirmed U over_4metrics`) core mask
        over_4metrics = (
            (risk_cal_arr >= mask_cfg.safe_metric_threshold)
            | (risk_ppr_arr >= mask_cfg.safe_metric_threshold)
            | (nomura_raw_arr >= mask_cfg.safe_nomura_threshold)
            | (risk_vfm_arr >= mask_cfg.safe_metric_threshold)
        )
        in_safe_any = any_danger | over_4metrics
        safe_core_packed = confirmed_core_packed.copy()

        extra_safe_mask = over_4metrics & (~any_danger)
        extra_safe_idx = np.flatnonzero(extra_safe_mask)
        if len(extra_safe_idx) > 0:
            r_safe = max(r_win, int(round(0.35 * fps_spec.fps)))
            local_safe_src = lum_ev_core | red_ev_core | raw_pat_packed
            # Include sub-threshold scene cuts on `extra_safe` if present
            scn_extra = np.zeros_like(local_safe_src)
            for ev in scene_events:
                pc = frame_to_pos.get(ev.frame_idx)
                if pc is None:
                    continue
                pr = frame_to_pos.get(ev.ref_frame, pc)
                lo_s, hi_s = max(0, min(pr, pc)), min(total_frames - 1, max(pr, pc))
                scn_extra[lo_s : hi_s + 1] = full_ones_2d[None, :, :]

            pooled_safe = temporal_sliding_or_packed(local_safe_src | scn_extra, r_safe)
            global_surge = (risk_vfm_arr >= mask_cfg.safe_metric_threshold) | (
                nomura_raw_arr >= mask_cfg.safe_nomura_threshold
            )

            for idx in extra_safe_idx:
                if scn_c1s[idx] > 0 and mask_cfg.fullscreen_scene_transition and np.any(scn_extra[idx]):
                    safe_core_packed[idx] = full_ones_2d.copy()
                    continue

                cand = pooled_safe[idx].copy()
                if global_surge[idx]:
                    np.bitwise_or(cand, lum_osc_f[idx] & raw_bright_packed[idx], out=cand)
                    if not np.any(cand):
                        cand = raw_bright_packed[idx].copy()
                if (
                    collect_physical_stats
                    and phys_sat_red_frac[idx] > pse.flash_area_min
                    and idx in raw_sat_red_packed_dict
                ):
                    np.bitwise_or(cand, red_osc_f[idx] & raw_sat_red_packed_dict[idx], out=cand)
                if not np.any(cand):
                    cand = raw_bright_packed[idx].copy()
                if not np.any(cand) and idx in raw_sat_red_packed_dict:
                    cand = raw_sat_red_packed_dict[idx].copy()
                safe_core_packed[idx] = cand

            safe_core_packed = clean_speckles_packed(
                safe_core_packed,
                active_frame_mask=extra_safe_mask,
                width=ana_w,
                min_cc_area_px=mask_cfg.min_cc_area_px,
            )
            safe_core_packed = temporal_bridge_spans_packed(
                safe_core_packed, mask_to_spans(extra_safe_mask), r_win * 2
            )

        del lum_ev_core, red_ev_core, lum_osc_f, red_osc_f, raw_pat_packed, raw_bright_packed
        raw_sat_red_packed_dict.clear()

        # 6) Compute exact per-frame masked physical statistics before releasing pixel buffers
        if need_pixel_stats_buf and total_frames > 0:
            for idx in range(total_frames):
                has_conf = bool(any_danger[idx]) and bool(np.any(confirmed_core_packed[idx]))
                has_safe = bool(in_safe_any[idx]) and bool(np.any(safe_core_packed[idx]))
                if not has_conf and not has_safe:
                    continue

                lum_arr = lum_f16_list[idx]
                red_arr = red_exc_f16_dict.get(idx)

                if has_conf:
                    m_conf = unpack_bool_masks_3d(confirmed_core_packed[idx : idx + 1], ana_w)[0]
                    ml, mr, mq = _extract_masked_frame_stats(m_conf, lum_arr, red_arr)
                    mask_max_lum_arr[idx] = ml
                    mask_max_red_arr[idx] = mr
                    mask_q76_red_arr[idx] = mq
                    if has_safe and np.array_equal(
                        confirmed_core_packed[idx], safe_core_packed[idx]
                    ):
                        safe_max_lum_arr[idx] = ml
                        safe_max_red_arr[idx] = mr
                        safe_q76_red_arr[idx] = mq
                        continue

                if has_safe:
                    m_safe = unpack_bool_masks_3d(safe_core_packed[idx : idx + 1], ana_w)[0]
                    sml, smr, smq = _extract_masked_frame_stats(m_safe, lum_arr, red_arr)
                    safe_max_lum_arr[idx] = sml
                    safe_max_red_arr[idx] = smr
                    safe_q76_red_arr[idx] = smq

        lum_f16_list.clear()
        red_exc_f16_dict.clear()

        guide_rgb_arr: np.ndarray | None = None
        if mask_cfg.borderless_guided and len(guide_rgb_list) == total_frames and total_frames > 0:
            guide_rgb_arr = np.stack(guide_rgb_list, axis=0)
        guide_rgb_list.clear()

        conf_flash_mask = in_lum_partial | in_red_temp
        safe_flash_mask = conf_flash_mask | (extra_safe_mask & (~raw_pat_mask))

        # 6) Build default soft `uint8` weight masks (`1-b + 3-a` borderless guided) & active area ratio
        weight_masks_u8, active_mask_area = build_soft_weight_masks_u8(
            packed_core=confirmed_core_packed,
            active_frame_mask=any_danger,
            geometry=geometry,
            fps=fps_spec.fps,
            mask_cfg=mask_cfg,
            guide_rgb_u8=guide_rgb_arr,
            flash_frame_mask=conf_flash_mask,
            device=device,
            schedule_mode="base",
            hold_run_tails=False,
        )
        packed_hazards = PackedHazardBuffers(
            analysis_height=ana_h,
            analysis_width=ana_w,
            confirmed_core_packed=confirmed_core_packed,
            safe_core_packed=safe_core_packed,
            guide_rgb_u8=guide_rgb_arr,
            flash_frame_mask=conf_flash_mask,
            safe_flash_frame_mask=safe_flash_mask,
        )

    physical_stats: dict[str, np.ndarray] | None = None
    if collect_physical_stats:
        area_stat = (
            active_mask_area
            if active_mask_area is not None
            else np.zeros(total_frames, dtype=np.float32)
        )
        physical_stats = {
            # Full-frame statistics (100% compatible with `vdcorefs`)
            "mean_lum_cd_m2": np.asarray(phys_mean_lum, dtype=np.float32),
            "max_lum_cd_m2": np.asarray(phys_max_lum, dtype=np.float32),
            "sat_red_area": np.asarray(phys_sat_red_frac, dtype=np.float32),
            "q76_red_lin": np.asarray(phys_q76_red_lin, dtype=np.float32),
            "q76_sat_red_excess": np.asarray(phys_q76_sat_red_excess, dtype=np.float32),
            "q24_red_escape_beta": np.asarray(phys_q24_red_escape_beta, dtype=np.float32),
            # Partial-mask interior statistics & active mask area ratio
            "mask_max_lum_cd_m2": mask_max_lum_arr,
            "mask_max_sat_red_excess": mask_max_red_arr,
            "mask_q76_sat_red_excess": mask_q76_red_arr,
            "safe_mask_max_lum_cd_m2": safe_max_lum_arr,
            "safe_mask_max_sat_red_excess": safe_max_red_arr,
            "safe_mask_q76_sat_red_excess": safe_q76_red_arr,
            "active_mask_area": area_stat,
        }

    return PartialVideoAnalysisResult(
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
        mask_config=mask_cfg,
        weight_masks_u8=weight_masks_u8,
        active_mask_area=active_mask_area,
        packed_hazards=packed_hazards,
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
    mask_config: PartialMaskConfig | None = None,
    device: str | torch.device | None = None,
    collect_physical_stats: bool = True,
    collect_masks: bool = False,
    detector_cache: PartialDetectorCache | None = None,
) -> PartialVideoAnalysisResult:
    """Analyze a streaming iterable of `(frame_idx, timestamp_ms, bgr_u8)` tuples.

    Note: `collect_masks` defaults to `False` on streaming verification (`Pass 2`) to avoid
    unnecessary mask storage during post-mitigation closed-loop verification; pass
    `collect_masks=True` when streaming Pass 1 localization is desired.
    """
    view_cfg = viewing or ViewingConfig()
    pse_cfg = pse or PSEConfig()
    m_cfg = mask_config or PartialMaskConfig()
    dev = resolve_device(device)
    clean_fps = float(fps) if (fps > 0.0 and math.isfinite(fps)) else 30.0
    return _run_partial_analysis_stream(
        frame_iter=frame_iter,
        fps=clean_fps,
        orig_width=orig_width,
        orig_height=orig_height,
        video_label=video_name,
        viewing=view_cfg,
        pse=pse_cfg,
        mask_cfg=m_cfg,
        device=dev,
        collect_physical_stats=collect_physical_stats,
        collect_masks=collect_masks,
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
    mask_config: PartialMaskConfig | None = None,
    device: str | torch.device | None = None,
    collect_physical_stats: bool = True,
    collect_masks: bool = True,
) -> PartialVideoAnalysisResult:
    """Analyze an in-memory sequence of BGR `uint8` frames with partial hazard localization."""
    if not frames_bgr:
        raise ValueError("`frames_bgr` must contain at least one frame.")
    view_cfg = viewing or ViewingConfig()
    pse_cfg = pse or PSEConfig()
    m_cfg = mask_config or PartialMaskConfig()
    dev = resolve_device(device)

    h, w = frames_bgr[0].shape[:2]
    clean_fps = float(fps) if (fps > 0.0 and math.isfinite(fps)) else 30.0
    dt_ms = 1000.0 / clean_fps

    def _gen() -> Iterable[tuple[int, float, np.ndarray]]:
        for idx, frame in enumerate(frames_bgr):
            t_ms = float(timestamps_ms[idx]) if timestamps_ms is not None else (idx * dt_ms)
            yield idx, t_ms, frame

    return _run_partial_analysis_stream(
        frame_iter=_gen(),
        fps=clean_fps,
        orig_width=w,
        orig_height=h,
        video_label=video_name,
        viewing=view_cfg,
        pse=pse_cfg,
        mask_cfg=m_cfg,
        device=dev,
        collect_physical_stats=collect_physical_stats,
        collect_masks=collect_masks,
    )


def analyze_video(
    video_path: str | Path,
    *,
    viewing: ViewingConfig | None = None,
    pse: PSEConfig | None = None,
    mask_config: PartialMaskConfig | None = None,
    device: str | torch.device | None = None,
    use_container_timestamps: bool = False,
    collect_physical_stats: bool = True,
    collect_masks: bool = True,
    detector_cache: PartialDetectorCache | None = None,
) -> PartialVideoAnalysisResult:
    """Analyze a video file on disk, returning 7-category PSE intervals, soft 2D masks, and stats."""
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
        m_cfg = mask_config or PartialMaskConfig()
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

        return _run_partial_analysis_stream(
            frame_iter=_video_gen(),
            fps=fps,
            orig_width=orig_w,
            orig_height=orig_h,
            video_label=str(path_obj),
            viewing=view_cfg,
            pse=pse_cfg,
            mask_cfg=m_cfg,
            device=dev,
            collect_physical_stats=collect_physical_stats,
            collect_masks=collect_masks,
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
    return analyze_video(
        video_path,
        viewing=viewing,
        pse=pse,
        device=device,
        collect_physical_stats=False,
        collect_masks=False,
    ).intervals_df
