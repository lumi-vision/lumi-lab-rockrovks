"""
LUMI 전체 화면 밝기 보정(Full-Screen Dimming) 핵심 엔진 모듈 (fullscreen_coreops_1108.py)
========================================================================================
[신경과학 및 국제 표준 기반 밝기 감소 근거 (방식 A + 선택 1 + Fade-in/out 스무딩)]
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
     상위 24% 지점의 r_lin을 0.0090 이하(8-bit 양자화 시 R <= 25)로 낮추는 감쇠율을 적용합니다.
3. 급격한 밝기 단차(인공 플래시) 방지를 위한 코사인 Fade-in / Fade-out (Jordan & Vanderheiden, 2024 Section 6.4):
   - 66ms 이내의 급격한 밝기 변화는 그 자체로 1회의 플래시 자극이 될 수 있으므로,
     위험 구간 진입 전 및 종료 후 FADE_DURATION_SEC(기본 0.20초 = 200ms) 동안 부드러운 코사인 경사(Ramp)로
     밝기를 서서히 낮추고 복원합니다.
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

# 신경과학 및 국제 표준(ITU-R BT.1702-3: 20 cd/m^2 기준) 안전 마진 상한선
TARGET_SAFE_LUM_CD = 18.0       # 위험 구간 보정 후 허용 최대 물리 휘도 (cd/m^2, 20.0 미만)
TARGET_SAFE_DELTA_L = 16.0      # 위험 구간 보정 후 허용 최대 국소 명암 대비 (cd/m^2, 20.0 미만)
TARGET_SAFE_RED_LIN = 0.0090    # 적색 플래시 구간(선택 1) 보정 후 8-bit 양자화 시 R<=25(r_lin<=0.01) 보장 상한
MIN_ALPHA_FLOOR = 0.005         # 화면이 완전한 블랙(0)으로 소실되는 것을 막기 위한 최소 선형 휘도 하한 (sRGB 약 10%)
FADE_DURATION_SEC = 0.20        # 위험 구간 진입·이탈 시 부드러운 밝기 전환 시간 (200ms > 66ms)


@torch.no_grad()
def apply_linear_dimming_to_srgb(rgb_255: torch.Tensor, alpha_lin: float) -> torch.Tensor:
    """
    단일 프레임(sRGB [0, 255])에 선형 물리 휘도 감쇠율 alpha_lin in (0, 1]을 적용한 뒤,
    표준 sRGB 역감마 변환(IEC 61966-2-1)을 거쳐 보정된 sRGB [0, 255] 텐서를 반환합니다.
    - 물리 휘도 L은 선형 RGB(c_lin)의 선형 결합이므로 c_lin' = alpha_lin * c_lin 으로 정확히 스케일링됩니다.
    """
    if alpha_lin >= 0.9999:
        return rgb_255

    c_s = torch.clamp(rgb_255 / 255.0, 0.0, 1.0)
    c_lin = torch.where(
        c_s <= 0.04045,
        c_s / 12.92,
        torch.pow((c_s + 0.055) / 1.055, 2.4)
    )

    # 선형 빛 에너지(물리 휘도) 공간에서 전체 화면 밝기 감쇠
    c_lin_dimmed = torch.clamp(c_lin * float(alpha_lin), 0.0, 1.0)

    # 선형 RGB -> sRGB 역변환 (감마 압축)
    c_s_dimmed = torch.where(
        c_lin_dimmed <= 0.0031308,
        c_lin_dimmed * 12.92,
        1.055 * torch.pow(torch.clamp(c_lin_dimmed, min=1e-8), 1.0 / 2.4) - 0.055
    )
    return torch.clamp(torch.round(c_s_dimmed * 255.0), 0.0, 255.0)


def compute_smoothed_alpha_schedule_method_a(
    frame_stats: list[dict],
    trigger_intervals: dict[str, list[tuple[int, int]]],
    fps: float,
    fade_duration_sec: float = FADE_DURATION_SEC,
) -> np.ndarray:
    """
    [방식 A + 선택 1 + 코사인 Fade-in/Fade-out 스무딩]
    1. 탐지된 각 위험 구간(휘도 플래시, 적색 플래시, 지속 플래시, 줄무늬 패턴, 고대비)에 속한 프레임들에 대해,
       PSE 유발 역치 미만으로 낮추기 위한 안전 상한 감쇠율 alpha_target[t]를 계산합니다.
    2. 위험 구간 내부에서 프레임마다 밝기가 들썩거리지 않도록 구간별 안정화(Minimum Envelope Smoothing)를 수행합니다.
    3. 위험 구간 진입 전과 종료 후 fade_duration_sec(200ms) 동안 코사인 경사(Ramp)를 적용하여
       급격한 밝기 단차로 인한 인공 플래시를 방지합니다.
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
        # (a) 물리 휘도 상한 기준: 최대 휘도를 TARGET_SAFE_LUM_CD(18 cd/m^2) 이하로 억제
        alpha_lum = min(1.0, TARGET_SAFE_LUM_CD / max(st["max_lum"], 1e-6))
        # (b) 국소 명암 대비 상한 기준: 최대 Delta L을 TARGET_SAFE_DELTA_L(16 cd/m^2) 이하로 억제
        alpha_contrast = min(1.0, TARGET_SAFE_DELTA_L / max(st["max_delta_l"], 1e-6))
        req_alpha = min(alpha_lum, alpha_contrast)

        # (c) 선택 1: 적색 플래시 위험 구간에서는 전체 밝기 감소만으로 sat_red_area <= 24%가 되도록 억제
        if in_red_danger[t] and st["q76_red_lin"] > TARGET_SAFE_RED_LIN:
            alpha_red = TARGET_SAFE_RED_LIN / max(st["q76_red_lin"], 1e-6)
            req_alpha = min(req_alpha, alpha_red)

        alpha_target[t] = max(MIN_ALPHA_FLOOR, min(1.0, req_alpha))

    # 2) 연속된 위험 구간 내부에서 밝기가 다시 튀어오르지 않도록 구간 내 안전 감쇠값 안정화
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

    # 3) 위험 구간 진입(Fade-in) 및 이탈(Fade-out) 코사인 스무딩 (기본 200ms)
    fade_frames = max(3, int(round(fade_duration_sec * fps)))
    alpha_with_fade = np.copy(alpha_schedule)

    for s_run, e_run in danger_merged_runs:
        start_alpha = alpha_schedule[s_run]
        end_alpha = alpha_schedule[e_run]

        # 진입부 Fade-in: [s_run - fade_frames, s_run - 1] 동안 1.0 -> start_alpha 코사인 감쇠
        for step in range(1, fade_frames + 1):
            f_idx = s_run - step
            if f_idx < 0:
                break
            u = float(step) / float(fade_frames + 1)
            cosine_w = 0.5 * (1.0 - math.cos(math.pi * u))
            ramp_val = start_alpha + (1.0 - start_alpha) * cosine_w
            alpha_with_fade[f_idx] = min(alpha_with_fade[f_idx], ramp_val)

        # 이탈부 Fade-out: [e_run + 1, e_run + fade_frames] 동안 end_alpha -> 1.0 코사인 복원
        for step in range(1, fade_frames + 1):
            f_idx = e_run + step
            if f_idx >= n_frames:
                break
            u = float(step) / float(fade_frames + 1)
            cosine_w = 0.5 * (1.0 - math.cos(math.pi * u))
            ramp_val = end_alpha + (1.0 - end_alpha) * cosine_w
            alpha_with_fade[f_idx] = min(alpha_with_fade[f_idx], ramp_val)

    return alpha_with_fade.astype(np.float32)


@torch.no_grad()
def dim_video_fullscreen_method_a(
    input_video_path: str,
    output_video_path: str,
    device: str = "laptop",
    display_area: int = 15,
    max_side: int = MAX_SIDE,
    max_safe_gap_ms: float = 334.0,
    fade_duration_sec: float = FADE_DURATION_SEC,
) -> dict:
    """
    단일 동영상을 입력받아:
    1) [1패스] vdcoreops_v2의 공용 analyze_video_frames(return_physical_stats=True)로
       중복 연산 없이 5대 위험 지표·구간 및 물리 휘도 통계를 한 번에 추출합니다.
    2) [감쇠율 계산] 방식 A(물리 휘도 안전 상한선) + 선택 1 + 코사인 Fade-in/out으로 프레임별 alpha_t를 산출합니다.
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

    # 2단계: 프레임별 선형 물리 휘도 감쇠율(alpha_schedule) 산출
    alpha_schedule = compute_smoothed_alpha_schedule_method_a(
        frame_stats=orig_frame_stats,
        trigger_intervals=trigger_intervals,
        fps=fps,
        fade_duration_sec=fade_duration_sec,
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
