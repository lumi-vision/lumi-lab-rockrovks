"""
LUMI 전체 화면 밝기 보정(Full-Screen Dimming) 핵심 엔진 모듈 — 단일 프레임 50% 초과 전면 보정(Safe) 버전
(fullscreen_coreops_safe_1108.py)
========================================================================================
[역할 및 기존 Hold 버전과의 차이점]
1. 기존 버전(fullscreen_coreops_1108.py / fullscreen_coreops_hold_1108.py):
   - 1단계(단일 프레임 임계치 >= 50%)와 2단계(시간적 점멸 횟수 또는 최소 지속 시간)를 모두 통과하여
     최종 확정된 '위험 구간'에 대해서만 밝기를 감소시킵니다.
2. 본 Safe 비교 버전(fullscreen_coreops_safe_1108.py):
   - 최종 확정된 '위험 구간(점멸 구간 내부 프레임 전체 포함)'뿐만 아니라,
     시간 조건 미달로 구간 판정에서 제외되었더라도 **단일 프레임 최대 위험도(prob1)가 임계점(50.0%) 이상인 모든 프레임**을
     보정 대상(합집합)으로 포함하여 전체 화면 밝기를 줄입니다.
   - 위험 구간 및 50% 초과 구간이 종료될 때는 선형 물리 휘도 기준 절반 완화 지점
     alpha_half = (alpha_danger + 1.0) / 2.0 에서 HALF_HOLD_DURATION_SEC(0.50초) 동안 유지한 뒤
     100%(1.0)로 완전히 해제하는 2단계 Half-Hold 스무딩을 동일하게 적용합니다.
========================================================================================
"""

import os
import sys
import math
import cv2
import numpy as np
import pandas as pd
import torch

_CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
if _CURRENT_DIR not in sys.path:
    sys.path.insert(0, _CURRENT_DIR)

from vdcoreops_v2 import (
    MAX_SIDE,
    evaluate_single_frame_all_metrics,
    detect_all_trigger_intervals,
    summarize_intervals_to_df,
    analyze_video_frames,
)
from fullscreen_coreops_1108 import (
    TARGET_SAFE_LUM_CD,
    TARGET_SAFE_DELTA_L,
    TARGET_SAFE_RED_LIN,
    MIN_ALPHA_FLOOR,
    FADE_DURATION_SEC,
    apply_linear_dimming_to_srgb,
)
from fullscreen_coreops_hold_1108 import HALF_HOLD_DURATION_SEC

SINGLE_FRAME_SAFE_THRESHOLD = 50.0  # 단일 프레임 최대 위험도(prob1 %) 보정 기준 임계점


def compute_smoothed_alpha_schedule_safe_method_a(
    frame_stats: list[dict],
    trigger_intervals: dict[str, list[tuple[int, int]]],
    orig_history: dict,
    orig_max_prob1: list[float] | np.ndarray,
    fps: float,
    fade_duration_sec: float = FADE_DURATION_SEC,
    half_hold_duration_sec: float = HALF_HOLD_DURATION_SEC,
    single_frame_threshold: float = SINGLE_FRAME_SAFE_THRESHOLD,
) -> tuple[np.ndarray, list[tuple[int, int]]]:
    """
    [방식 A + 선택 1 + (최종 위험 구간 U 단일 프레임 50% 초과 전체) + 절반 완화 Hold 스무딩]
    1. 보정 대상 프레임(in_danger_any):
       - (a) 5대 트리거의 시계열 위험 구간(trigger_intervals) 내부의 모든 프레임
       - (b) 단일 프레임 최대 위험도(orig_max_prob1[t])가 50.0% 이상인 모든 프레임
       의 합집합(Union)으로 설정합니다.
    2. 해당되는 모든 프레임에 대해 물리 휘도 <= 18 cd/m^2, 국소 명암 대비 <= 16 cd/m^2,
       적색 위험 시 선형 적색 <= 0.0090이 되도록 안전 상한 감쇠율 alpha_target[t]를 산출합니다.
    3. 구간 내부 최소 포락선 스무딩 후, 구간 종료 시 절반 완화(alpha_half) 상태로 0.5초 유지했다가
       100%로 해제하는 2단계 Half-Hold 코사인 스무딩을 적용합니다.
    """
    n_frames = len(frame_stats)
    if n_frames == 0:
        return np.array([], dtype=np.float32), []

    alpha_target = np.ones(n_frames, dtype=np.float64)
    in_interval_danger = np.zeros(n_frames, dtype=bool)
    in_danger_any = np.zeros(n_frames, dtype=bool)
    in_red_danger = np.zeros(n_frames, dtype=bool)

    # (a) 시계열 최종 위험 구간 반영
    for trig_name, iv_list in trigger_intervals.items():
        for s_f, e_f in iv_list:
            s_c = max(0, min(n_frames - 1, s_f))
            e_c = max(0, min(n_frames - 1, e_f))
            in_interval_danger[s_c : e_c + 1] = True
            in_danger_any[s_c : e_c + 1] = True
            if trig_name == "적색 플래시":
                in_red_danger[s_c : e_c + 1] = True

    # (b) 단일 프레임 위험도 50% 초과 프레임 합집합 반영
    red_danger_list = orig_history.get("red_flash_danger", [False] * n_frames)
    red_prob1_list = orig_history.get("red_flash_prob1", [0.0] * n_frames)
    for t in range(n_frames):
        if float(orig_max_prob1[t]) >= single_frame_threshold:
            in_danger_any[t] = True
        if red_danger_list[t] or float(red_prob1_list[t]) >= single_frame_threshold:
            in_red_danger[t] = True

    # 시계열 위험 구간 외에 단일 프레임 50% 초과로 추가 보정된 구간 추출 (그래프 표시용)
    extra_single_frame_runs = []
    idx_e = 0
    while idx_e < n_frames:
        if in_danger_any[idx_e] and not in_interval_danger[idx_e]:
            s_ex = idx_e
            while idx_e + 1 < n_frames and (in_danger_any[idx_e + 1] and not in_interval_danger[idx_e + 1]):
                idx_e += 1
            extra_single_frame_runs.append((s_ex, idx_e))
        idx_e += 1

    # 1) 보정 대상 프레임별 필요 감쇠율(alpha_target) 계산
    for t in range(n_frames):
        if not in_danger_any[t]:
            continue
        st = frame_stats[t]
        alpha_lum = min(1.0, TARGET_SAFE_LUM_CD / max(st["max_lum"], 1e-6))
        alpha_contrast = min(1.0, TARGET_SAFE_DELTA_L / max(st["max_delta_l"], 1e-6))
        req_alpha = min(alpha_lum, alpha_contrast)

        if in_red_danger[t] and st["q76_red_lin"] > TARGET_SAFE_RED_LIN:
            alpha_red = TARGET_SAFE_RED_LIN / max(st["q76_red_lin"], 1e-6)
            req_alpha = min(req_alpha, alpha_red)

        alpha_target[t] = max(MIN_ALPHA_FLOOR, min(1.0, req_alpha))

    # 2) 연속된 보정 구간 추출 및 구간 내부 안정화(Minimum Envelope Smoothing)
    alpha_schedule = np.ones(n_frames, dtype=np.float64)
    win_radius = max(2, int(round(0.25 * fps)))

    i = 0
    danger_merged_runs = []
    while i < n_frames:
        if in_danger_any[i]:
            s_run = i
            while i + 1 < n_frames and in_danger_any[i + 1]:
                i += 1
            danger_merged_runs.append((s_run, i))
        i += 1

    for s_run, e_run in danger_merged_runs:
        run_targets = alpha_target[s_run : e_run + 1]
        env = np.copy(run_targets)
        for idx_local in range(len(run_targets)):
            w_s = max(0, idx_local - win_radius)
            w_e = min(len(run_targets), idx_local + win_radius + 1)
            env[idx_local] = np.min(run_targets[w_s:w_e])

        if len(env) >= 3:
            k_smooth = min(len(env), max(3, int(round(0.30 * fps)) | 1))
            pad_s = k_smooth // 2
            padded = np.pad(env, (pad_s, pad_s), mode="edge")
            kernel = np.ones(k_smooth, dtype=np.float64) / float(k_smooth)
            smoothed_env = np.convolve(padded, kernel, mode="valid")[: len(env)]
            final_run_alpha = np.minimum(smoothed_env, run_targets)
        else:
            final_run_alpha = env

        alpha_schedule[s_run : e_run + 1] = final_run_alpha

    # 3) 구간 종료 후 2단계 절반 완화(Half-Hold) 및 진입부 코사인 스무딩
    fade_frames = max(3, int(round(fade_duration_sec * fps)))
    hold_frames = max(1, int(round(half_hold_duration_sec * fps)))
    alpha_with_hold = np.copy(alpha_schedule)

    # (3-A) 각 보정 구간 종료 직후 [1단계 절반 완화 -> 절반 완화 0.5초 유지 -> 2단계 100% 완전 해제]
    for s_run, e_run in danger_merged_runs:
        end_alpha = float(alpha_schedule[e_run])
        alpha_half = 0.5 * (end_alpha + 1.0)

        # 1단계 완화: [e_run + 1, e_run + fade_frames] 동안 end_alpha -> alpha_half 코사인 상승
        for step in range(1, fade_frames + 1):
            f_idx = e_run + step
            if f_idx >= n_frames:
                break
            u = float(step) / float(fade_frames + 1)
            cosine_w = 0.5 * (1.0 - math.cos(math.pi * u))
            ramp1_val = end_alpha + (alpha_half - end_alpha) * cosine_w
            alpha_with_hold[f_idx] = min(alpha_with_hold[f_idx], ramp1_val)

        # 절반 완화 유지(Hold): [e_run + fade_frames + 1, e_run + fade_frames + hold_frames] 동안 alpha_half 유지
        hold_start = e_run + fade_frames + 1
        hold_end = e_run + fade_frames + hold_frames
        for f_idx in range(hold_start, min(n_frames, hold_end + 1)):
            alpha_with_hold[f_idx] = min(alpha_with_hold[f_idx], alpha_half)

        # 2단계 완전 해제: [hold_end + 1, hold_end + fade_frames] 동안 alpha_half -> 1.0 코사인 복원
        for step in range(1, fade_frames + 1):
            f_idx = hold_end + step
            if f_idx >= n_frames:
                break
            u = float(step) / float(fade_frames + 1)
            cosine_w = 0.5 * (1.0 - math.cos(math.pi * u))
            ramp2_val = alpha_half + (1.0 - alpha_half) * cosine_w
            alpha_with_hold[f_idx] = min(alpha_with_hold[f_idx], ramp2_val)

    # (3-B) 각 보정 구간 진입 직전의 Fade-in
    for s_run, e_run in danger_merged_runs:
        start_alpha = float(alpha_schedule[s_run])
        anchor_idx = max(0, s_run - fade_frames - 1)
        entry_from_alpha = max(start_alpha, float(alpha_with_hold[anchor_idx]))

        for step in range(1, fade_frames + 1):
            f_idx = s_run - step
            if f_idx < 0:
                break
            u = float(step) / float(fade_frames + 1)
            cosine_w = 0.5 * (1.0 - math.cos(math.pi * u))
            ramp_in_val = start_alpha + (entry_from_alpha - start_alpha) * cosine_w
            alpha_with_hold[f_idx] = min(alpha_with_hold[f_idx], ramp_in_val)

    return alpha_with_hold.astype(np.float32), extra_single_frame_runs


@torch.no_grad()
def dim_video_fullscreen_safe_method_a(
    input_video_path: str,
    output_video_path: str,
    device: str = "laptop",
    display_area: int = 15,
    max_side: int = MAX_SIDE,
    max_safe_gap_ms: float = 334.0,
    fade_duration_sec: float = FADE_DURATION_SEC,
    half_hold_duration_sec: float = HALF_HOLD_DURATION_SEC,
    single_frame_threshold: float = SINGLE_FRAME_SAFE_THRESHOLD,
) -> dict:
    """
    단일 동영상을 입력받아:
    1) [1패스] vdcoreops_v2의 공용 analyze_video_frames(return_physical_stats=True)로
       원본 5대 위험 지표·구간 및 물리 휘도 통계를 추출합니다.
    2) [감쇠율 계산] 최종 위험 구간 + 단일 프레임 위험도 50% 이상인 모든 프레임을 합쳐
       방식 A + 선택 1 + 절반 완화(Half-Hold) 스케줄 alpha_t를 산출합니다.
    3) [2패스 저장 및 동시 재검증] alpha_t >= 0.9999인 무보정 프레임은 1패스 측정값을 재사용하고,
       보정된 프레임만 재평가하여 보정 동영상(.mp4)과 검증 결과를 반환합니다.
    """
    # Safe 버전은 기존 단일 프레임 기준 prob1(%) 로직을 그대로 유지합니다 (apply_temporal_prob1=False)
    analysis = analyze_video_frames(
        video_path=input_video_path,
        device=device,
        display_area=display_area,
        max_side=max_side,
        max_safe_gap_ms=max_safe_gap_ms,
        return_physical_stats=True,
        apply_temporal_prob1=False,
    )
    meta = analysis["meta"]
    proc_w, proc_h = meta["proc_w"], meta["proc_h"]
    fps = meta["fps"]
    cpd_spec = analysis["cpd_spec"]
    scan_lines = analysis["scan_lines"]
    orig_history = analysis["history"]
    trigger_intervals = analysis["trigger_intervals"]
    orig_trigger_specs = analysis["trigger_specs"]
    orig_frame_stats = analysis["frame_stats"]
    orig_max_prob1_per_frame = analysis["max_prob1_per_frame"]
    n_frames = len(orig_frame_stats)

    df_before = summarize_intervals_to_df(orig_trigger_specs, fps=fps)

    # 최종 위험 구간 + 단일 프레임 50% 초과 전체 보정 스케줄 산출
    alpha_schedule, extra_single_frame_runs = compute_smoothed_alpha_schedule_safe_method_a(
        frame_stats=orig_frame_stats,
        trigger_intervals=trigger_intervals,
        orig_history=orig_history,
        orig_max_prob1=orig_max_prob1_per_frame,
        fps=fps,
        fade_duration_sec=fade_duration_sec,
        half_hold_duration_sec=half_hold_duration_sec,
        single_frame_threshold=single_frame_threshold,
    )

    # 2패스: 보정 영상 저장 및 보정 후 지표 동시 측정
    os.makedirs(os.path.dirname(os.path.abspath(output_video_path)), exist_ok=True)
    cap_in = cv2.VideoCapture(input_video_path)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(output_video_path, fourcc, fps, (proc_w, proc_h))
    device_t = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    dimmed_history = {
        "lum_flash_danger": [], "lum_flash_prob1": [],
        "red_flash_danger": [], "red_flash_prob1": [],
        "sustained_danger": [], "sustained_prob1": [],
        "stripe_danger": [], "stripe_prob1": [],
        "hc_danger": [], "hc_prob1": [],
    }
    dimmed_frame_stats = []
    dimmed_max_prob1_per_frame = []

    frame_idx = 0
    while True:
        ret, frame_bgr = cap_in.read()
        if not ret or frame_bgr is None or frame_idx >= n_frames:
            break

        if frame_bgr.shape[1] != proc_w or frame_bgr.shape[0] != proc_h:
            frame_bgr = cv2.resize(frame_bgr, (proc_w, proc_h), interpolation=cv2.INTER_AREA)

        alpha_t = float(alpha_schedule[frame_idx])

        if alpha_t >= 0.9999:
            writer.write(frame_bgr)
            for k in dimmed_history:
                dimmed_history[k].append(orig_history[k][frame_idx])
            dimmed_frame_stats.append(orig_frame_stats[frame_idx])
            dimmed_max_prob1_per_frame.append(orig_max_prob1_per_frame[frame_idx])
        else:
            img_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            rgb_t = torch.from_numpy(img_rgb).to(device=device_t, dtype=torch.float32)
            rgb_dimmed_t = apply_linear_dimming_to_srgb(rgb_t, alpha_lin=alpha_t)

            rgb_dimmed_np = rgb_dimmed_t.detach().cpu().numpy().astype(np.uint8)
            bgr_dimmed_np = cv2.cvtColor(rgb_dimmed_np, cv2.COLOR_RGB2BGR)
            writer.write(bgr_dimmed_np)

            res_dim = evaluate_single_frame_all_metrics(
                rgb_dimmed_t,
                scan_lines,
                cpd_min=cpd_spec["cpd_min"],
                cpd_max=cpd_spec["cpd_max"],
                return_physical_stats=True,
            )
            for k in dimmed_history:
                dimmed_history[k].append(res_dim[k])
            dimmed_frame_stats.append(res_dim["physical_stats"])

            max_p1_dim = max(
                res_dim["lum_flash_prob1"],
                res_dim["red_flash_prob1"],
                res_dim["sustained_prob1"],
                res_dim["stripe_prob1"],
                res_dim["hc_prob1"],
            )
            dimmed_max_prob1_per_frame.append(float(max_p1_dim))

        frame_idx += 1

    cap_in.release()
    writer.release()

    dimmed_trigger_intervals, dimmed_trigger_specs = detect_all_trigger_intervals(
        history=dimmed_history,
        fps=fps,
        max_safe_gap_ms=max_safe_gap_ms,
        apply_temporal_prob1=False,
    )
    df_after = summarize_intervals_to_df(dimmed_trigger_specs, fps=fps)

    all_orig_intervals = []
    for trig_name, iv_list in trigger_intervals.items():
        for s_f, e_f in iv_list:
            all_orig_intervals.append((s_f, e_f, trig_name))

    return {
        "meta": meta,
        "output_video_path": output_video_path,
        "df_before": df_before,
        "df_after": df_after,
        "trigger_intervals_before": trigger_intervals,
        "trigger_intervals_after": dimmed_trigger_intervals,
        "all_orig_intervals": all_orig_intervals,
        "extra_single_frame_runs": extra_single_frame_runs,
        "alpha_schedule": alpha_schedule,
        "orig_frame_stats": orig_frame_stats,
        "dimmed_frame_stats": dimmed_frame_stats,
        "orig_max_prob1": np.array(orig_max_prob1_per_frame, dtype=np.float32),
        "dimmed_max_prob1": np.array(dimmed_max_prob1_per_frame, dtype=np.float32),
    }
