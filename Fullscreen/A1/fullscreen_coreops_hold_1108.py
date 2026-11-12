"""
LUMI 전체 화면 밝기 보정(Full-Screen Dimming) 핵심 엔진 모듈 — 절반 완화 유지(Half-Hold) 버전
(fullscreen_coreops_hold_1108.py)
========================================================================================
[신경과학 및 국제 표준 기반 밝기 감소 근거 (방식 A + 선택 1 + 절반 완화 Hold 후 해제 스무딩)]
1. 물리 휘도 및 명암 차이 안전 역치 (Fisher, Harding, Wilkins et al., 2005; ITU-R BT.1702-3; Jordan & Vanderheiden, 2024):
   - 제1시각피질(V1)의 광과민성 발작(PPR) 동기화 과흥분은 물리 휘도 차이(Delta L)가 20 cd/m^2 이상이거나
     국소 고휘도 영역(>= 20 cd/m^2)이 넓게 점멸/지속될 때 발생합니다.
   - 위험 구간 프레임의 전체 선형 물리 휘도에 감쇠 계수 alpha_t in (0, 1]를 곱하면(L' = alpha_t * L),
     절대 휘도(L')와 픽셀 간 명암 차이(Delta L' = alpha_t * Delta L)가 모두 alpha_t배로 비례 축소됩니다.
   - 따라서 보정 후 프레임 내 최대 물리 휘도가 TARGET_SAFE_LUM_CD(18.0 cd/m^2 < 20.0 cd/m^2) 이하,
     최대 국소 명암 차이가 TARGET_SAFE_DELTA_L(16.0 cd/m^2 < 20.0 cd/m^2) 이하가 되도록 동적 감쇠율 alpha_t를 산출하면
     휘도 플래시, 지속 플래시, 줄무늬 패턴, 고대비 자극이 모두 발작 역치 미만의 약한 시각 자극으로 전환됩니다.
2. 적색 플래시 전체 밝기 감쇠 (선택 1):
   - 적색 플래시 위험 구간에서는 전체 화면 밝기 감소만으로 포화 적색(r_lin > 0.01) 면적이 25% 미만이 되도록
     상위 24% 지점의 r_lin을 0.0090 이하(8-bit R <= 25)로 낮추는 감쇠율을 적용합니다.
3. 2단계 절반 완화 유지(Half-Hold) 후 완전 해제 및 코사인 스무딩:
   - 위험 구간 종료 직후 밝기를 곧바로 100%(1.0)로 완전히 풀면, 짧은 간격(예: 0.5~0.6초)을 두고 다음 위험 구간이
     이어질 때 밝기가 0% -> 100% -> 0%로 급격히 출렁이는 펌핑(Pumping) 현상이 발생합니다.
   - 이를 방지하기 위해 위험 구간 종료 시:
     (1단계) FADE_DURATION_SEC(0.20초) 동안 선형 물리 휘도 기준 절반 완화 지점
             alpha_half = (alpha_danger + 1.0) / 2.0 까지만 코사인 곡선으로 완만하게 완화한 뒤,
     (유지)  HALF_HOLD_DURATION_SEC(기본 0.50초) 동안 alpha_half 상태를 유지합니다.
             이 유지 구간 도중 다음 위험 구간이 시작되면 100%로 튀어오르지 않고 alpha_half에서 곧바로 다음 위험 구간으로 부드럽게 재진입합니다.
     (2단계) 유지 시간 동안 후속 위험 구간이 없을 때만 FADE_DURATION_SEC(0.20초) 동안 alpha_half -> 1.0(100%)으로 완전히 해제합니다.
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

HALF_HOLD_DURATION_SEC = 0.50    # 위험 구간 종료 후 절반 완화(alpha_half) 상태로 유지하는 시간 (500ms)


def compute_smoothed_alpha_schedule_hold_method_a(
    frame_stats: list[dict],
    trigger_intervals: dict[str, list[tuple[int, int]]],
    fps: float,
    fade_duration_sec: float = FADE_DURATION_SEC,
    half_hold_duration_sec: float = HALF_HOLD_DURATION_SEC,
) -> np.ndarray:
    """
    [방식 A + 선택 1 + 절반 완화(Half-Relaxation Hold) 후 완전 해제 스무딩]
    1. 탐지된 위험 구간 내 프레임들에 대해 PSE 유발 역치 미만으로 낮추는 안전 상한 감쇠율 alpha_target[t]를 계산합니다.
    2. 위험 구간 내부에서 밝기가 들썩거리지 않도록 구간별 최소 포락선 스무딩(Minimum Envelope Smoothing)을 수행합니다.
    3. 위험 구간 종료 후에는 곧바로 1.0(100%)으로 풀지 않고:
       - [1단계 완화] fade_duration_sec(0.20초) 동안 alpha_half = (end_alpha + 1.0) / 2.0 까지만 코사인 복원
       - [절반 완화 유지] half_hold_duration_sec(0.50초) 동안 alpha_half 상태 유지
       - [2단계 완전 해제] 후속 위험 구간이 없을 경우 fade_duration_sec(0.20초) 동안 alpha_half -> 1.0 코사인 복원
       - 만약 절반 완화 유지 중(또는 2단계 해제 중) 다음 위험 구간이 다가오면, 현재 완화된 밝기에서 곧바로 다음 위험 구간의
         start_alpha로 부드럽게 코사인 재진입하여 인접 위험 구간 사이에서 100%로 튀어오르는 현상을 방지합니다.
    """
    n_frames = len(frame_stats)
    if n_frames == 0:
        return np.array([], dtype=np.float32)

    alpha_target = np.ones(n_frames, dtype=np.float64)
    in_danger_any = np.zeros(n_frames, dtype=bool)
    in_red_danger = np.zeros(n_frames, dtype=bool)

    for trig_name, iv_list in trigger_intervals.items():
        for s_f, e_f in iv_list:
            s_c = max(0, min(n_frames - 1, s_f))
            e_c = max(0, min(n_frames - 1, e_f))
            in_danger_any[s_c : e_c + 1] = True
            if trig_name == "적색 플래시":
                in_red_danger[s_c : e_c + 1] = True

    # 1) 위험 구간 내 프레임별 필요 감쇠율(alpha_target) 계산
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

    # 2) 연속된 위험 구간 추출 및 구간 내부 안정화(Minimum Envelope Smoothing)
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

    # 3) 위험 구간 종료 후 2단계 절반 완화(Half-Hold) 및 진입부 코사인 스무딩
    fade_frames = max(3, int(round(fade_duration_sec * fps)))
    hold_frames = max(1, int(round(half_hold_duration_sec * fps)))
    alpha_with_hold = np.copy(alpha_schedule)

    # (3-A) 각 위험 구간 종료 직후의 [1단계 절반 완화 -> 절반 완화 유지 -> 2단계 완전 해제] 포락선 적용
    for s_run, e_run in danger_merged_runs:
        end_alpha = float(alpha_schedule[e_run])
        # 선형 물리 휘도 배율 기준 절반 완화 지점: alpha_half = (alpha_danger + 1.0) / 2.0
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

    # (3-B) 각 위험 구간 진입 직전의 Fade-in:
    #       이전 구간의 절반 완화 유지(alpha_half) 중이면 alpha_half -> start_alpha로 부드럽게 재진입하고,
    #       평상시(1.0)였으면 1.0 -> start_alpha로 코사인 진입합니다.
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

    return alpha_with_hold.astype(np.float32)


@torch.no_grad()
def dim_video_fullscreen_hold_method_a(
    input_video_path: str,
    output_video_path: str,
    device: str = "laptop",
    display_area: int = 15,
    max_side: int = MAX_SIDE,
    max_safe_gap_ms: float = 334.0,
    fade_duration_sec: float = FADE_DURATION_SEC,
    half_hold_duration_sec: float = HALF_HOLD_DURATION_SEC,
) -> dict:
    """
    단일 동영상을 입력받아:
    1) [1패스] vdcoreops_v2의 공용 analyze_video_frames(return_physical_stats=True)로
       중복 연산 없이 5대 위험 지표·구간 및 물리 휘도 통계를 한 번에 추출합니다.
    2) [감쇠율 계산] 방식 A + 선택 1 + 2단계 절반 완화(Half-Hold) 후 해제 스무딩으로 프레임별 alpha_t를 산출합니다.
    3) [2패스 저장 및 동시 재검증] alpha_t >= 0.9999인 무보정 안전 프레임은 GPU 재평가를 생략하고
       1패스 측정값을 즉시 재사용하여 속도를 극대화합니다.
    """
    # 1패스: 원본 영상 통합 분석 (중복 GPU 연산 제거)
    analysis = analyze_video_frames(
        video_path=input_video_path,
        device=device,
        display_area=display_area,
        max_side=max_side,
        max_safe_gap_ms=max_safe_gap_ms,
        return_physical_stats=True,
    )
    meta = analysis["meta"]
    proc_w, proc_h = meta["proc_w"], meta["proc_h"]
    fps = meta["fps"]
    cpd_spec = analysis["cpd_spec"]
    scan_lines = analysis["scan_lines"]
    orig_raw_history = analysis["raw_history"]
    trigger_intervals = analysis["trigger_intervals"]
    orig_trigger_specs = analysis["trigger_specs"]
    orig_frame_stats = analysis["frame_stats"]
    orig_max_prob1_per_frame = analysis["max_prob1_per_frame"]
    n_frames = len(orig_frame_stats)

    df_before = summarize_intervals_to_df(orig_trigger_specs, fps=fps)

    # 2단계 절반 완화(Half-Hold) 스케줄 산출
    alpha_schedule = compute_smoothed_alpha_schedule_hold_method_a(
        frame_stats=orig_frame_stats,
        trigger_intervals=trigger_intervals,
        fps=fps,
        fade_duration_sec=fade_duration_sec,
        half_hold_duration_sec=half_hold_duration_sec,
    )

    # 2패스: 보정 영상 저장 및 보정 후 지표 동시 측정 (alpha_t == 1.0 프레임은 GPU 재평가 생략)
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

    frame_idx = 0
    while True:
        ret, frame_bgr = cap_in.read()
        if not ret or frame_bgr is None or frame_idx >= n_frames:
            break

        if frame_bgr.shape[1] != proc_w or frame_bgr.shape[0] != proc_h:
            frame_bgr = cv2.resize(frame_bgr, (proc_w, proc_h), interpolation=cv2.INTER_AREA)

        alpha_t = float(alpha_schedule[frame_idx])

        # [속도 최적화] 밝기 배율이 100%(>= 0.9999)인 안전 프레임은 원본과 100% 동일하므로
        # 불필요한 색공간 변환 및 GPU 재평가를 건너뛰고 1패스 단일 프레임 원시 측정값을 즉시 재사용합니다.
        if alpha_t >= 0.9999:
            writer.write(frame_bgr)
            for k in dimmed_history:
                dimmed_history[k].append(orig_raw_history[k][frame_idx])
            dimmed_frame_stats.append(orig_frame_stats[frame_idx])
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

        frame_idx += 1

    cap_in.release()
    writer.release()

    # 보정 후 시계열 위험 구간 재판정 및 시계열 통합 prob1(%) 산출
    dimmed_trigger_intervals, dimmed_trigger_specs = detect_all_trigger_intervals(
        history=dimmed_history,
        fps=fps,
        max_safe_gap_ms=max_safe_gap_ms,
        apply_temporal_prob1=True,
    )
    dimmed_max_prob1_per_frame = [
        float(
            max(
                dimmed_history["lum_flash_prob1"][i],
                dimmed_history["red_flash_prob1"][i],
                dimmed_history["sustained_prob1"][i],
                dimmed_history["stripe_prob1"][i],
                dimmed_history["hc_prob1"][i],
            )
        )
        for i in range(len(dimmed_frame_stats))
    ]
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
        "alpha_schedule": alpha_schedule,
        "orig_frame_stats": orig_frame_stats,
        "dimmed_frame_stats": dimmed_frame_stats,
        "orig_max_prob1": np.array(orig_max_prob1_per_frame, dtype=np.float32),
        "dimmed_max_prob1": np.array(dimmed_max_prob1_per_frame, dtype=np.float32),
    }
