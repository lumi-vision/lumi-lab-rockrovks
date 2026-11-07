import os
import math
import cv2
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

MAX_SIDE = 720  # 동영상 프레임 분석 시 최대 해상도 변 길이 (속도 및 정확도 균형)


@torch.no_grad()
def srgb_to_linear_and_luminance(rgb_255: torch.Tensor):
    """방식 A (ITU-R BT.1702-3 / WCAG 2.2): sRGB [0,255] -> 선형 RGB [0,1] 및 물리 휘도 L [0, 200] cd/m^2"""
    
    c_s = torch.clamp(rgb_255 / 255.0, 0.0, 1.0) # sRGB Color
    """0~255를 0~1로 스케일링"""
    
    c_lin = torch.where(
        c_s <= 0.04045,
        c_s / 12.92,
        torch.pow((c_s + 0.055) / 1.055, 2.4)
    ) # Linear Color
    """
    사람의 눈은 밝은 곳보다 어두운 곳의 미세한 밝기 변화에 훨씬 민감함.
    제한된 숫자(0~255)에 어두운 색조를 촘촘하게 저장하기 위해 실제 빛의 에너지를 그대로 표현 x
    => 원래 sRGB는 감마 압축 (1/2.2)승 __ 여기선 2.4로 쓰임.
    픽셀값 0~10 수준에서는 미분값이 너무 낮음 -> 0.04045 이하에서는 직선으로 변환
    0.055, 1.055는 미분가능하게 만들기 위한 숫자
    """
    y_rel = 0.2126 * c_lin[..., 0] + 0.7152 * c_lin[..., 1] + 0.0722 * c_lin[..., 2] # 상대 휘도
    """ITU-R에서 정한 인간 시각 밝기 가중치"""
    lum_cd = 200.0 * y_rel # 물리 휘도
    """ITU-R에서 표준 일반 디스플레이(SDR)에서 최대 흰색(y_rel = 1.0)의 물리 휘도를 200cd/m^2로 기준 삼는다"""
    return c_lin, lum_cd


def compute_viewing_angle_and_cpd_spec(
    img_w: int,
    img_h: int,
    device: str = "laptop",
    display_area: int = 15,
    cpd_min: float = 1.0,
    cpd_max: float = 10.0
) -> dict:
    """
    표준 시청거리 가정을 통해 디스플레이 시야각(Visual Angle)과 가로·세로 1도당 픽셀 수,
    그리고 위험 공간 주파수(1.0 ~ 10.0 cpd) 판정을 위한 기준을 정립합니다.
    - device: 'laptop' (표준 시청거리 56.0 cm) 또는 'desktop' (표준 시청거리 68.0 cm)
    - display_area: 디스플레이 대각선 크기 (inch, 기본값 15)
    """
    if device == "laptop":
        viewing_dist_cm = 56.0
    elif device == "desktop":
        viewing_dist_cm = 68.0
    else:
        raise ValueError(f"device 인자는 'laptop' 또는 'desktop'만 허용됩니다. (입력값: {device!r})")

    if int(display_area) <= 0:
        raise ValueError(f"display_area(디스플레이 크기 inch)는 양의 정수여야 합니다. (입력값: {display_area})")

    # 1. 16:9 표준 화면비 기준 물리적 가로/세로 길이(cm) 계산 (1 inch = 2.54 cm)
    diag_cm = float(int(display_area)) * 2.54
    aspect_diag = math.sqrt(16.0 ** 2 + 9.0 ** 2)
    screen_w_cm = diag_cm * (16.0 / aspect_diag)
    screen_h_cm = diag_cm * (9.0 / aspect_diag)

    # 2. 시야각 계산: 2*theta = 2 * arctan(x / (2 * d)) (3694790.pdf Appendix A)
    fov_w_deg = 2.0 * math.degrees(math.atan(screen_w_cm / (2.0 * viewing_dist_cm)))
    fov_h_deg = 2.0 * math.degrees(math.atan(screen_h_cm / (2.0 * viewing_dist_cm)))

    # 3. 가로·세로 시야각 1도당 픽셀 수 (가로는 가로대로, 세로는 세로대로 분리 보관)
    px_per_deg_w = float(img_w) / fov_w_deg
    px_per_deg_h = float(img_h) / fov_h_deg

    return {
        "device": device,
        "display_area_inch": int(display_area),
        "viewing_dist_cm": viewing_dist_cm,
        "fov_w_deg": fov_w_deg,
        "fov_h_deg": fov_h_deg,
        "px_per_deg_w": px_per_deg_w,
        "px_per_deg_h": px_per_deg_h,
        "cpd_min": cpd_min,
        "cpd_max": cpd_max,
    }


def open_video_and_get_info(video_path: str, max_side: int = MAX_SIDE):
    """동영상 파일을 단 1회 열어 VideoCapture 객체와 해상도·FPS·프레임 수 정보를 함께 반환합니다."""
    if not os.path.exists(video_path):
        raise FileNotFoundError(f"동영상 파일을 찾을 수 없습니다: {video_path}")
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"동영상을 열 수 없습니다: {video_path}")
    orig_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    orig_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = float(cap.get(cv2.CAP_PROP_FPS))
    if fps <= 0 or math.isnan(fps):
        fps = 30.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    if max(orig_h, orig_w) > max_side:
        scale = max_side / float(max(orig_h, orig_w))
        proc_w, proc_h = int(orig_w * scale), int(orig_h * scale)
    else:
        proc_w, proc_h = orig_w, orig_h

    meta = {
        "proc_w": proc_w,
        "proc_h": proc_h,
        "fps": fps,
        "total_frames": total_frames,
        "duration_sec": total_frames / fps if total_frames > 0 else 0.0,
    }
    return cap, meta


def build_video_spec_df(meta: dict, cpd_spec: dict) -> pd.DataFrame:
    """이미 추출된 비디오 메타데이터(meta)와 시야각 정보(cpd_spec)로 설정 요약 DataFrame을 생성합니다."""
    return pd.DataFrame([{
        "분석 해상도 (가로×세로)": f"{meta['proc_w']} × {meta['proc_h']}",
        "프레임 레이트 (FPS)": f"{meta['fps']:.2f} fps",
        "총 프레임 수 (재생 시간)": f"{meta['total_frames']} 프레임 ({meta['duration_sec']:.2f}s)",
        "기기 및 화면 크기": f"{cpd_spec['device']} ({cpd_spec['display_area_inch']} inch, 시청거리 {cpd_spec['viewing_dist_cm']:.0f}cm)",
        "디스플레이 시야각": f"{cpd_spec['fov_w_deg']:.2f}° × {cpd_spec['fov_h_deg']:.2f}°",
        "1°당 픽셀 수 (가로/세로)": f"{cpd_spec['px_per_deg_w']:.2f} / {cpd_spec['px_per_deg_h']:.2f} px/deg",
        "위험 공간주파수 범위": f"{cpd_spec['cpd_min']:.1f} ~ {cpd_spec['cpd_max']:.1f} cpd",
    }], index=["입력 영상 및 CPD 설정 정보"])


def count_opposing_stripe_pairs_1d_cpd(
    profile: np.ndarray,
    px_per_deg: float,
    cpd_min: float = 1.0,
    cpd_max: float = 10.0
) -> int:
    """
    1차원 휘도 프로파일(1 px 간격 샘플링)에서:
    1) ITU-R BT.1702-3 명암 대비 기준(>=20 cd/m^2 또는 Michelson > 1/17)을 만족하고,
    2) 인접한 봉우리-골짜기 간격이 공간 주파수 1.0 ~ 10.0 cpd 범위 안에 들어오는
       '연속된(끊기지 않은) 줄무늬 쌍(contiguous light-dark pairs)'의 최대 개수를 반환합니다.
    """
    if profile.size < 4:
        return 0

    extrema_indices = [0]
    extrema_vals = [float(profile[0])]
    direction = 0

    for idx in range(1, profile.size):
        v = float(profile[idx])
        diff = v - extrema_vals[-1]
        if direction >= 0 and diff > 0:
            if direction == 0:
                extrema_vals.append(v)
                extrema_indices.append(idx)
            else:
                extrema_vals[-1] = v
                extrema_indices[-1] = idx
            direction = 1
        elif direction <= 0 and diff < 0:
            if direction == 0:
                extrema_vals.append(v)
                extrema_indices.append(idx)
            else:
                extrema_vals[-1] = v
                extrema_indices[-1] = idx
            direction = -1
        elif direction == 1 and (extrema_vals[-1] - v) >= 5.0:
            extrema_vals.append(v)
            extrema_indices.append(idx)
            direction = -1
        elif direction == -1 and (v - extrema_vals[-1]) >= 5.0:
            extrema_vals.append(v)
            extrema_indices.append(idx)
            direction = 1

    max_contiguous_transitions = 0
    cur_transitions = 0

    for i in range(len(extrema_vals) - 1):
        l_high = max(extrema_vals[i], extrema_vals[i + 1])
        l_low = min(extrema_vals[i], extrema_vals[i + 1])
        cond_low = (l_low < 160.0) and ((l_high - l_low) >= 20.0)
        cond_high = (l_low >= 160.0) and (((l_high - l_low) / max(l_high + l_low, 1e-6)) > (1.0 / 17.0))
        contrast_ok = cond_low or cond_high

        half_cycle_px = float(extrema_indices[i + 1] - extrema_indices[i])
        full_cycle_px = 2.0 * max(half_cycle_px, 1e-6)
        local_cpd = px_per_deg / full_cycle_px
        cpd_ok = (cpd_min <= local_cpd <= cpd_max)

        if contrast_ok and cpd_ok:
            cur_transitions += 1
            if cur_transitions > max_contiguous_transitions:
                max_contiguous_transitions = cur_transitions
        else:
            cur_transitions = 0

    return max_contiguous_transitions // 2


def precompute_stripe_scan_lines(w: int, h: int, cpd_spec: dict) -> list:
    """동영상의 모든 프레임에서 동일한 스캔 좌표를 재사용할 수 있도록 60개 직선 좌표와 방향별 px_per_deg를 사전 계산합니다."""
    cy, cx = h / 2.0, w / 2.0
    radius = int(min(h, w) * 0.45)
    t_vals = np.arange(-radius, radius + 1, 1.0)
    fov_w_deg = cpd_spec["fov_w_deg"]
    fov_h_deg = cpd_spec["fov_h_deg"]

    scan_lines = []
    for deg in range(0, 180, 15):
        rad = math.radians(deg)
        px_per_deg_dir = 1.0 / math.sqrt((math.cos(rad) / (w / fov_w_deg)) ** 2 + (math.sin(rad) / (h / fov_h_deg)) ** 2)
        for offset in np.linspace(-radius * 0.5, radius * 0.5, 5):
            xs = np.clip(np.round(cx + t_vals * math.cos(rad) - offset * math.sin(rad)).astype(int), 0, w - 1) # 이 부분 설명 가능하게
            ys = np.clip(np.round(cy + t_vals * math.sin(rad) + offset * math.cos(rad)).astype(int), 0, h - 1) # 이 부분 설명 가능하게
            scan_lines.append((ys, xs, px_per_deg_dir))
    return scan_lines


@torch.no_grad()
def evaluate_single_frame_all_metrics(rgb_255: torch.Tensor, scan_lines: list, cpd_min: float = 1.0, cpd_max: float = 10.0) -> dict:
    """
    단일 비디오 프레임(rgb_255)에서 5대 위험 트리거의 (위험 여부 bool, 방식 1 확률 float)을 최적화된 파이프라인으로 계산합니다.
    - 안전 프레임(danger == False)에서는 불필요한 무거운 후속 연산 및 확률 계산을 조기 종료(Early Exit)합니다.
    - 국소 명암 대비(짧은 변 3% 커널)를 단 1회만 계산하여 '고대비(40%)'와 '줄무늬 패턴(25%)'이 공유합니다.
    """
    c_lin, lum = srgb_to_linear_and_luminance(rgb_255)
    h, w = lum.shape

    # -------------------------------------------------------------------------
    # A. 국소 고휘도 평가 (25% -> 휘도 플래시용 / 40% -> 지속 플래시용)
    # -------------------------------------------------------------------------
    l_base = torch.clamp(lum.mean(), min=1.0)
    target_lum = torch.clamp(l_base * 1.30, min=20.0)
    high_lum_mask = lum >= target_lum
    high_lum_area = float(high_lum_mask.float().mean().item())

    lum_flash_danger = high_lum_area > 0.25
    sustained_danger = high_lum_area > 0.40

    if lum_flash_danger:
        rel_ratio_map = torch.clamp((lum - l_base) / torch.clamp(l_base * 0.30, min=1e-6), min=0.0)
        lum_gate = float(torch.clamp(rel_ratio_map[high_lum_mask].mean() if high_lum_mask.any() else rel_ratio_map.max(), 0.0, 1.0).item()) # 이 부분 설명 가능하게
        eta_lum_25 = (high_lum_area / 0.25) * lum_gate
        lum_flash_prob1 = (eta_lum_25 ** 2 / (1.0 + eta_lum_25 ** 2)) * 100.0
        if sustained_danger:
            eta_lum_40 = (high_lum_area / 0.40) * lum_gate
            sustained_prob1 = (eta_lum_40 ** 2 / (1.0 + eta_lum_40 ** 2)) * 100.0
        else:
            sustained_prob1 = 0.0
    else:
        lum_flash_prob1 = 0.0
        sustained_prob1 = 0.0

    # -------------------------------------------------------------------------
    # B. 포화 적색 평가 (25% -> 적색 플래시용) => 이 부분 설명 가능하게
    # -------------------------------------------------------------------------
    r_lin, g_lin, b_lin = c_lin[..., 0], c_lin[..., 1], c_lin[..., 2]
    rgb_sum = r_lin + g_lin + b_lin
    red_ratio = torch.where(rgb_sum > 1e-4, r_lin / torch.clamp(rgb_sum, min=1e-6), torch.zeros_like(r_lin))
    sat_red_mask = (red_ratio >= 0.8) & (r_lin > 0.01)
    sat_red_area = float(sat_red_mask.float().mean().item())

    # 조기 종료: 포화 적색 면적이 25%를 초과할 때만 무거운 CIE 1976 색도 변환 및 quantile 정렬 수행
    if sat_red_area > 0.25:
        x_cie = 0.4124564 * r_lin + 0.3575761 * g_lin + 0.1804375 * b_lin
        y_cie = 0.2126729 * r_lin + 0.7151522 * g_lin + 0.0721750 * b_lin
        z_cie = 0.0193339 * r_lin + 0.1191920 * g_lin + 0.9503041 * b_lin
        denom = x_cie + 15.0 * y_cie + 3.0 * z_cie
        valid_chroma = denom > 1e-4
        non_red_mask = (~sat_red_mask) & valid_chroma

        if non_red_mask.any():
            u_prime = torch.where(valid_chroma, (4.0 * x_cie) / torch.clamp(denom, min=1e-6), torch.full_like(x_cie, 0.1978))
            v_prime = torch.where(valid_chroma, (9.0 * y_cie) / torch.clamp(denom, min=1e-6), torch.full_like(y_cie, 0.4683))
            u_red_mean = u_prime[sat_red_mask].mean()
            v_red_mean = v_prime[sat_red_mask].mean()
            dist_map = torch.sqrt((u_prime[non_red_mask] - u_red_mean) ** 2 + (v_prime[non_red_mask] - v_red_mean) ** 2)
            delta_uv = float(torch.quantile(dist_map, 0.90).item())
        else:
            delta_uv = 0.0

        red_flash_danger = delta_uv > 0.2
        if red_flash_danger:
            red_gate = float(torch.clamp(red_ratio[sat_red_mask].mean() / 0.8, 0.0, 1.0).item())
            chroma_gate = min(1.0, delta_uv / 0.2)
            eta_red_25 = (sat_red_area / 0.25) * red_gate * chroma_gate
            red_flash_prob1 = (eta_red_25 ** 2 / (1.0 + eta_red_25 ** 2)) * 100.0
        else:
            red_flash_prob1 = 0.0
    else:
        red_flash_danger = False
        red_flash_prob1 = 0.0

    # -------------------------------------------------------------------------
    # C & D 통합. 국소 명암 대비 1회 계산 (방식 A: 3% 커널 통일 -> 고대비 & 줄무늬 패턴 공유)
    # -------------------------------------------------------------------------
    lum_4d = lum.unsqueeze(0).unsqueeze(0)
    k_size = max(7, int(min(h, w) * 0.03) | 1)
    pad = k_size // 2
    l_high = F.max_pool2d(lum_4d, kernel_size=k_size, stride=1, padding=pad).squeeze()
    l_low = -F.max_pool2d(-lum_4d, kernel_size=k_size, stride=1, padding=pad).squeeze()
    delta_l = l_high - l_low
    michelson = delta_l / torch.clamp(l_high + l_low, min=1e-6)

    contrast_mask = torch.where(l_low < 160.0, delta_l >= 20.0, michelson > (1.0 / 17.0))
    contrast_ratio = torch.where(l_low < 160.0, delta_l / 20.0, michelson / (1.0 / 17.0))

    # 1) 고대비 영역 평가 (면적 임계치 40%)
    hc_area = float(contrast_mask.float().mean().item())
    hc_danger = hc_area > 0.40
    if hc_danger:
        hc_gate = float(torch.clamp(contrast_ratio[contrast_mask].mean(), 0.0, 1.0).item())
        eta_hc_40 = (hc_area / 0.40) * hc_gate
        hc_prob1 = (eta_hc_40 ** 2 / (1.0 + eta_hc_40 ** 2)) * 100.0
    else:
        hc_prob1 = 0.0

    # 2) 줄무늬 패턴 평가 (면적 임계치 25%, 1.0 ~ 10.0 cpd 연속 5쌍 이상)
    env_k = max(15, int(min(h, w) * 0.06) | 1)
    pattern_density = F.avg_pool2d(contrast_mask.float().unsqueeze(0).unsqueeze(0), kernel_size=env_k, stride=1, padding=env_k // 2).squeeze()
    pattern_region_mask = pattern_density >= 0.15
    pattern_area = float(pattern_region_mask.float().mean().item())

    # 조기 종료: 패턴 면적이 25%를 초과할 때만 60개 직선 스캔 수행 (5쌍 도달 시 즉시 break)
    if pattern_area > 0.25:
        lum_np = lum.detach().cpu().numpy()
        max_pairs = 0
        for ys, xs, px_per_deg_dir in scan_lines:
            pairs = count_opposing_stripe_pairs_1d_cpd(
                lum_np[ys, xs],
                px_per_deg=px_per_deg_dir,
                cpd_min=cpd_min,
                cpd_max=cpd_max
            )
            if pairs > max_pairs:
                max_pairs = pairs
                if max_pairs >= 5:
                    break

        stripe_danger = max_pairs >= 5
        if stripe_danger:
            mean_contrast_factor = float(torch.clamp(contrast_ratio[pattern_region_mask].mean(), 0.0, 1.0).item())
            pair_factor = min(1.0, max_pairs / 5.0)
            eta_stripe = (pattern_area / 0.25) * pair_factor * mean_contrast_factor
            stripe_prob1 = (eta_stripe ** 2 / (1.0 + eta_stripe ** 2)) * 100.0
        else:
            stripe_prob1 = 0.0
    else:
        stripe_danger = False
        stripe_prob1 = 0.0

    return {
        "lum_flash_danger": lum_flash_danger,
        "lum_flash_prob1": lum_flash_prob1,
        "red_flash_danger": red_flash_danger,
        "red_flash_prob1": red_flash_prob1,
        "sustained_danger": sustained_danger,
        "sustained_prob1": sustained_prob1,
        "stripe_danger": stripe_danger,
        "stripe_prob1": stripe_prob1,
        "hc_danger": hc_danger,
        "hc_prob1": hc_prob1,
    }


def _extract_danger_runs(is_danger_list: list[bool]) -> list[tuple[int, int]]:
    """불리언 리스트에서 연속된 True(위험 프레임) 묶음들의 (시작 프레임, 끝 프레임) 리스트를 반환합니다."""
    runs = []
    n = len(is_danger_list)
    i = 0
    while i < n:
        if is_danger_list[i]:
            start = i
            while i + 1 < n and is_danger_list[i + 1]:
                i += 1
            runs.append((start, i))
        i += 1
    return runs


def _merge_intervals(intervals: list[tuple[int, int]]) -> list[tuple[int, int]]:
    """겹치거나 인접한 (start_frame, end_frame) 구간들을 하나로 병합합니다."""
    if not intervals:
        return []
    sorted_iv = sorted(intervals, key=lambda x: (x[0], x[1]))
    merged = [sorted_iv[0]]
    for cur_s, cur_e in sorted_iv[1:]:
        prev_s, prev_e = merged[-1]
        if cur_s <= prev_e + 1:
            merged[-1] = (prev_s, max(prev_e, cur_e))
        else:
            merged.append((cur_s, cur_e))
    return merged


def detect_flashing_intervals(
    is_danger_list: list[bool],
    fps: float,
    min_pairs: int = 4,
    max_safe_gap_frames: int = 9,
    danger_runs: list[tuple[int, int]] | None = None
) -> list[tuple[int, int]]:
    """
    [방식 A - 점멸 탐지]
    연속된 위험 프레임 묶음(True)과 직후 이어지는 안전 프레임 묶음(False)을 1개의 '(위험, 안전) 쌍'으로 정의할 때,
    - 위험 프레임 묶음 사이의 안전 프레임 개수가 1 ~ max_safe_gap_frames(9프레임) 이하이고,
    - 1초(fps 프레임 길이 창) 안에 이러한 (위험, 안전) 쌍이 min_pairs(4번, > 3Hz) 이상 들어오는 구간을 검출합니다.
    """
    n_frames = len(is_danger_list)
    if danger_runs is None:
        danger_runs = _extract_danger_runs(is_danger_list)
    if len(danger_runs) < min_pairs:
        return []

    pairs = []
    for idx, (d_start, d_end) in enumerate(danger_runs):
        if d_end + 1 >= n_frames:
            continue
        next_d_start = danger_runs[idx + 1][0] if (idx + 1 < len(danger_runs)) else None
        gap_frames = (next_d_start - d_end - 1) if (next_d_start is not None) else (n_frames - 1 - d_end)
        pairs.append({
            "run_idx": idx,
            "d_start": d_start,
            "d_end": d_end,
            "first_safe_frame": d_end + 1,
            "gap_to_next_danger": gap_frames,
        })

    if len(pairs) < min_pairs:
        return []

    raw_intervals = []
    for k in range(len(pairs) - min_pairs + 1):
        window_pairs = pairs[k : k + min_pairs]
        gaps_ok = True
        for p_idx in range(min_pairs - 1):
            if (window_pairs[p_idx + 1]["run_idx"] != window_pairs[p_idx]["run_idx"] + 1) or (
                window_pairs[p_idx]["gap_to_next_danger"] > max_safe_gap_frames
            ):
                gaps_ok = False
                break
        if not gaps_ok:
            continue

        seq_start_frame = window_pairs[0]["d_start"]
        seq_pair_complete_frame = window_pairs[-1]["first_safe_frame"]
        elapsed_sec = (seq_pair_complete_frame - seq_start_frame) / float(fps)

        if elapsed_sec <= 1.0:
            raw_intervals.append((seq_start_frame, window_pairs[-1]["d_end"]))

    if not raw_intervals:
        return []
    sorted_iv = sorted(raw_intervals, key=lambda x: (x[0], x[1]))
    merged = [sorted_iv[0]]
    for cur_s, cur_e in sorted_iv[1:]:
        prev_s, prev_e = merged[-1]
        if cur_s <= prev_e + max_safe_gap_frames + 1:
            merged[-1] = (prev_s, max(prev_e, cur_e))
        else:
            merged.append((cur_s, cur_e))
    return merged


def detect_sustained_flash_intervals(
    is_danger_list: list[bool],
    fps: float,
    min_duration_sec: float = 1.0
) -> list[tuple[int, int]]:
    """
    [3. 지속 플래시 - 방식 A]
    국소 고휘도(40%) 안전 프레임(False)에서 위험 프레임(True)으로 전환(d_start >= 1)된 뒤,
    1.0초 이상 연속으로 위험 프레임이 지속되는 구간을 반환합니다.
    """
    danger_runs = _extract_danger_runs(is_danger_list)
    intervals = []
    for d_start, d_end in danger_runs:
        if d_start >= 1 and not is_danger_list[d_start - 1]:
            duration_sec = (d_end - d_start + 1) / float(fps)
            if duration_sec >= min_duration_sec:
                intervals.append((d_start, d_end))
    return _merge_intervals(intervals)


def detect_pattern_or_contrast_intervals(
    is_danger_list: list[bool],
    fps: float,
    min_duration_sec: float = 0.5,
    min_pairs: int = 4,
    max_safe_gap_frames: int = 9
) -> list[tuple[int, int]]:
    """
    [4. 줄무늬 패턴 (25%) & 5. 고대비 (40%) - 방식 A (논문 기준 통합)]
    _extract_danger_runs를 단 1회만 호출하여:
    1) 위험 프레임이 0.5초 이상 연속 지속되는 구간, 또는
    2) (위험 프레임, 안전 프레임) 쌍이 1초에 4번 이상(간격 9프레임 이하) 점멸하는 구간
    을 모두 탐지하여 병합 후 반환합니다.
    """
    danger_runs = _extract_danger_runs(is_danger_list)
    sustained_ivs = []
    for d_start, d_end in danger_runs:
        duration_sec = (d_end - d_start + 1) / float(fps)
        if duration_sec >= min_duration_sec:
            sustained_ivs.append((d_start, d_end))

    flash_ivs = detect_flashing_intervals(
        is_danger_list,
        fps=fps,
        min_pairs=min_pairs,
        max_safe_gap_frames=max_safe_gap_frames,
        danger_runs=danger_runs
    )

    return _merge_intervals(sustained_ivs + flash_ivs)


@torch.no_grad()
def evaluate_video_pse_risk(
    video_path: str,
    device: str = "laptop",
    display_area: int = 15,
    max_side: int = MAX_SIDE,
    return_spec_df: bool = False
) -> pd.DataFrame | tuple[pd.DataFrame, pd.DataFrame]:
    """
    동영상을 단 1회 열어 5대 위험 트리거(휘도 플래시, 적색 플래시, 지속 플래시, 줄무늬 패턴, 고대비)에 따른
    위험 구간 요약 데이터프레임을 반환합니다.
    - 컬럼: ['위험 구분', '위험 구간 (프레임)', '위험 구간 (시간)', '판정 근거 - 임계치 달성 정도']
    - return_spec_df=True 설정 시 (df_video_result, df_video_spec) 튜플을 반환합니다.
    """
    cap, meta = open_video_and_get_info(video_path, max_side=max_side)
    proc_w, proc_h = meta["proc_w"], meta["proc_h"]
    fps = meta["fps"]

    cpd_spec = compute_viewing_angle_and_cpd_spec(
        proc_w, proc_h, device=device, display_area=display_area, cpd_min=1.0, cpd_max=10.0
    )
    scan_lines = precompute_stripe_scan_lines(proc_w, proc_h, cpd_spec)
    device_t = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    history = {
        "lum_flash_danger": [], "lum_flash_prob1": [],
        "red_flash_danger": [], "red_flash_prob1": [],
        "sustained_danger": [], "sustained_prob1": [],
        "stripe_danger": [], "stripe_prob1": [],
        "hc_danger": [], "hc_prob1": [],
    }

    while True:
        ret, frame_bgr = cap.read()
        if not ret or frame_bgr is None:
            break
        img_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        if img_rgb.shape[1] != proc_w or img_rgb.shape[0] != proc_h:
            img_rgb = cv2.resize(img_rgb, (proc_w, proc_h), interpolation=cv2.INTER_AREA)

        rgb_t = torch.from_numpy(img_rgb).to(device=device_t, dtype=torch.float32)
        res = evaluate_single_frame_all_metrics(
            rgb_t, scan_lines, cpd_min=cpd_spec["cpd_min"], cpd_max=cpd_spec["cpd_max"]
        )
        for k in history:
            history[k].append(res[k])
    cap.release()

    trigger_specs = [
        (
            "휘도 플래시",
            history["lum_flash_danger"],
            history["lum_flash_prob1"],
            detect_flashing_intervals(history["lum_flash_danger"], fps=fps, min_pairs=4, max_safe_gap_frames=9)
        ),
        (
            "적색 플래시",
            history["red_flash_danger"],
            history["red_flash_prob1"],
            detect_flashing_intervals(history["red_flash_danger"], fps=fps, min_pairs=4, max_safe_gap_frames=9)
        ),
        (
            "지속 플래시",
            history["sustained_danger"],
            history["sustained_prob1"],
            detect_sustained_flash_intervals(history["sustained_danger"], fps=fps, min_duration_sec=1.0)
        ),
        (
            "줄무늬 패턴",
            history["stripe_danger"],
            history["stripe_prob1"],
            detect_pattern_or_contrast_intervals(history["stripe_danger"], fps=fps, min_duration_sec=0.5, min_pairs=4, max_safe_gap_frames=9)
        ),
        (
            "고대비",
            history["hc_danger"],
            history["hc_prob1"],
            detect_pattern_or_contrast_intervals(history["hc_danger"], fps=fps, min_duration_sec=0.5, min_pairs=4, max_safe_gap_frames=9)
        ),
    ]

    rows = []
    for trigger_name, danger_list, prob1_list, intervals in trigger_specs:
        for s_frame, e_frame in intervals:
            danger_probs = [
                prob1_list[f_idx]
                for f_idx in range(s_frame, e_frame + 1)
                if danger_list[f_idx]
            ]
            mean_prob1 = float(np.mean(danger_probs)) if danger_probs else 0.0

            start_sec = s_frame / float(fps)
            end_sec = e_frame / float(fps)
            rows.append({
                "_sort_frame": s_frame,
                "위험 구분": trigger_name,
                "위험 구간 (프레임)": f"{s_frame} ~ {e_frame}",
                "위험 구간 (시간)": f"{start_sec:05.2f}s ~ {end_sec:05.2f}s",
                "판정 근거 - 임계치 달성 정도": f"{mean_prob1:.2f}%"
            })

    rows.sort(key=lambda r: (r["_sort_frame"], r["위험 구분"]))
    for r in rows:
        r.pop("_sort_frame", None)

    columns = ["위험 구분", "위험 구간 (프레임)", "위험 구간 (시간)", "판정 근거 - 임계치 달성 정도"]
    df_video_result = pd.DataFrame(rows, columns=columns)

    if return_spec_df:
        df_video_spec = build_video_spec_df(meta, cpd_spec)
        return df_video_result, df_video_spec
    return df_video_result
