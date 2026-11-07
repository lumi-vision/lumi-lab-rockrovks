"""
LUMI 딥러닝 학습 데이터셋 자동 판별 및 라벨링 빌더 (dsbuilder_v1.py)
=============================================================================
[역할 및 목적]
1. 지정된 데이터셋 폴더 내의 모든 동영상 파일을
   탐색합니다.
2. 핵심 판별 엔진(vdcoreops_v1.py)을 호출하여 영상별 5대 위험 트리거
   (휘도 플래시, 적색 플래시, 지속 플래시, 줄무늬 패턴, 고대비)를 판별합니다.
3. 딥러닝 모델 학습에 즉시 활용할 수 있도록 두 가지 형태의 라벨링 데이터프레임을 생성하고
   폴더에 CSV(UTF-8-SIG, 엑셀 한글 호환)로 저장합니다.

   - [구성 A] pse_video_summary.csv (영상 단위 요약 테이블: 1개 영상 = 1개 Row)
     -> 용도: 영상 전체 위험 여부 이진 분류(Video Classification) 모델 학습 및 전체 데이터셋 Manifest 관리
   - [구성 B] pse_interval_details.csv (구간 단위 상세 테이블: 감지된 위험 구간 1개 = 1개 Row, 방식 B-1 적용)
     -> 용도: 시계열 위험 구간 탐지(Temporal Event Localization) 및 프레임 슬라이싱 학습
     -> 특징: (1) 감지된 위험 구간이 시간/프레임상 겹치더라도 '위험 구분'이 다르면 합치지 않고 별도 행으로 독립 기록
              (2) 위험 구간이 없는 안전 영상도 누락되지 않도록 '위험 여부 (label) = 0, 위험 구분 = 안전'으로 1줄 포함(방식 B-1)
=============================================================================
"""

import os
import re
import sys
import time
import argparse
import cv2
import numpy as np
import pandas as pd
import torch

from vdcoreops_v1 import (
    MAX_SIDE,
    open_video_and_get_info,
    compute_viewing_angle_and_cpd_spec,
    precompute_stripe_scan_lines,
    evaluate_single_frame_all_metrics,
    detect_flashing_intervals,
    detect_sustained_flash_intervals,
    detect_pattern_or_contrast_intervals,
)

# Windows 터미널 한글 출력 깨짐 방지
if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

# =========================================================================
# 1. 기본 경로 및 데이터셋 구축 설정 (추후 영상 폴더나 설정 추가 시 이 부분만 수정)
# =========================================================================
DEFAULT_DATASET_DIR = r"데이터셋 폴더 경로"
DEFAULT_OUTPUT_DIR = r"csv 저장 경로"

SUMMARY_CSV_NAME = "pse_video_summary.csv"       # 구성 A 파일명 (영상 단위 요약)
INTERVAL_CSV_NAME = "pse_interval_details.csv"   # 구성 B 파일명 (구간 단위 상세)

DEFAULT_DEVICE = "laptop"                        # 'laptop' (56.0 cm) 또는 'desktop' (68.0 cm)
DEFAULT_DISPLAY_AREA = 15                        # 디스플레이 대각선 크기 (inch)

# 탐색할 동영상 확장자 및 제외할 하위 폴더명
VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv", ".webm", ".wmv"}
EXCLUDE_DIR_NAMES = {"out"}


def _natural_sort_key(path_str: str) -> list:
    """
    파일명 내 숫자를 인식하여 자연순(1.mp4, 2.mp4, ..., 10.mp4, ..., 19.avi, 22.mp4)으로
    정렬하기 위한 키 함수입니다.
    """
    basename = os.path.basename(path_str)
    return [
        int(tok) if tok.isdigit() else tok.lower()
        for tok in re.split(r"(\d+)", basename)
    ]


def collect_video_files(
    dataset_dir: str = DEFAULT_DATASET_DIR,
    exclude_dirs: set[str] = EXCLUDE_DIR_NAMES,
    recursive: bool = False
) -> list[str]:
    """
    데이터셋 폴더에서 동영상 파일 목록을 수집합니다.
    - 'out' 폴더 등 exclude_dirs에 지정된 폴더 내부 파일은 제외합니다.
    - 추후 하위 카테고리 폴더를 추가해 탐색하고 싶을 경우 recursive=True로 확장 가능합니다.
    """
    if not os.path.exists(dataset_dir):
        raise FileNotFoundError(f"데이터셋 폴더를 찾을 수 없습니다: {dataset_dir}")

    exclude_lower = {d.lower() for d in exclude_dirs}
    video_paths = []

    if recursive:
        for root, dirs, files in os.walk(dataset_dir):
            # 제외 폴더('out' 등)는 아예 하위 탐색에서 배제
            dirs[:] = [d for d in dirs if d.lower() not in exclude_lower]
            for fname in files:
                ext = os.path.splitext(fname)[1].lower()
                if ext in VIDEO_EXTENSIONS:
                    video_paths.append(os.path.join(root, fname))
    else:
        for entry in os.scandir(dataset_dir):
            if entry.is_dir() and entry.name.lower() in exclude_lower:
                continue
            if entry.is_file():
                ext = os.path.splitext(entry.name)[1].lower()
                if ext in VIDEO_EXTENSIONS:
                    video_paths.append(entry.path)

    video_paths.sort(key=_natural_sort_key)
    return video_paths


@torch.no_grad()
def analyze_single_video_for_dataset(
    video_path: str,
    device: str = DEFAULT_DEVICE,
    display_area: int = DEFAULT_DISPLAY_AREA,
    max_side: int = MAX_SIDE,
    scan_cache: dict | None = None,
) -> tuple[dict, list[dict]]:
    """
    단일 동영상을 분석하여:
    1) [구성 A] 영상 단위 요약 딕셔너리 1개 (summary_row)
    2) [구성 B] 구간 단위 상세 딕셔너리 리스트 (interval_rows, 방식 B-1 적용)
    를 반환합니다.
    - 다수 영상 처리 시 동일 해상도에 대한 시야각 및 60개 직선 스캔 좌표를 scan_cache에 캐싱하여 재사용합니다.
    """
    cap, meta = open_video_and_get_info(video_path, max_side=max_side)
    proc_w, proc_h = meta["proc_w"], meta["proc_h"]
    fps = meta["fps"]
    total_frames = meta["total_frames"]
    duration_sec = meta["duration_sec"]

    cache_key = (proc_w, proc_h, device, int(display_area))
    if scan_cache is not None and cache_key in scan_cache:
        cpd_spec, scan_lines = scan_cache[cache_key]
    else:
        cpd_spec = compute_viewing_angle_and_cpd_spec(
            proc_w, proc_h, device=device, display_area=display_area, cpd_min=1.0, cpd_max=10.0
        )
        scan_lines = precompute_stripe_scan_lines(proc_w, proc_h, cpd_spec)
        if scan_cache is not None:
            scan_cache[cache_key] = (cpd_spec, scan_lines)

    device_t = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    history = {
        "lum_flash_danger": [], "lum_flash_prob1": [],
        "red_flash_danger": [], "red_flash_prob1": [],
        "sustained_danger": [], "sustained_prob1": [],
        "stripe_danger": [], "stripe_danger_prob1": [],
        "hc_danger": [], "hc_prob1": [],
    }

    actual_frames = 0
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
        history["lum_flash_danger"].append(res["lum_flash_danger"])
        history["lum_flash_prob1"].append(res["lum_flash_prob1"])
        history["red_flash_danger"].append(res["red_flash_danger"])
        history["red_flash_prob1"].append(res["red_flash_prob1"])
        history["sustained_danger"].append(res["sustained_danger"])
        history["sustained_prob1"].append(res["sustained_prob1"])
        history["stripe_danger"].append(res["stripe_danger"])
        history["stripe_danger_prob1"].append(res["stripe_prob1"])
        history["hc_danger"].append(res["hc_danger"])
        history["hc_prob1"].append(res["hc_prob1"])
        actual_frames += 1

    cap.release()

    if actual_frames > 0 and actual_frames != total_frames:
        total_frames = actual_frames
        duration_sec = total_frames / float(fps)

    # -------------------------------------------------------------------------
    # 5대 위험 트리거별 시계열 구간 독립 탐지
    # (주의: 감지된 구간이 시간/프레임상 겹치더라도 위험 구분이 다르면 절대 병합하지 않고 따로 기록)
    # -------------------------------------------------------------------------
    trigger_specs = [
        (
            "휘도 플래시",
            history["lum_flash_danger"],
            history["lum_flash_prob1"],
            detect_flashing_intervals(history["lum_flash_danger"], fps=fps, min_pairs=4, max_safe_gap_frames=9),
        ),
        (
            "적색 플래시",
            history["red_flash_danger"],
            history["red_flash_prob1"],
            detect_flashing_intervals(history["red_flash_danger"], fps=fps, min_pairs=4, max_safe_gap_frames=9),
        ),
        (
            "지속 플래시",
            history["sustained_danger"],
            history["sustained_prob1"],
            detect_sustained_flash_intervals(history["sustained_danger"], fps=fps, min_duration_sec=1.0),
        ),
        (
            "줄무늬 패턴",
            history["stripe_danger"],
            history["stripe_danger_prob1"],
            detect_pattern_or_contrast_intervals(
                history["stripe_danger"], fps=fps, min_duration_sec=0.5, min_pairs=4, max_safe_gap_frames=9
            ),
        ),
        (
            "고대비",
            history["hc_danger"],
            history["hc_prob1"],
            detect_pattern_or_contrast_intervals(
                history["hc_danger"], fps=fps, min_duration_sec=0.5, min_pairs=4, max_safe_gap_frames=9
            ),
        ),
    ]

    file_name = os.path.basename(video_path)
    raw_intervals = []

    for trigger_name, danger_list, prob1_list, intervals in trigger_specs:
        for s_frame, e_frame in intervals:
            danger_probs = [
                prob1_list[f_idx]
                for f_idx in range(s_frame, e_frame + 1)
                if danger_list[f_idx]
            ]
            mean_prob1 = float(np.mean(danger_probs)) if danger_probs else 0.0
            start_sec = round(s_frame / float(fps), 2)
            end_sec = round(e_frame / float(fps), 2)

            raw_intervals.append({
                "_sort_key": (s_frame, e_frame, trigger_name),
                "파일명": file_name,
                "위험 여부 (label)": 1,
                "위험 구분": trigger_name,
                "시작 프레임": int(s_frame),
                "종료 프레임": int(e_frame),
                "구간 프레임 수": int(e_frame - s_frame + 1),
                "시작 시간 (s)": start_sec,
                "종료 시간 (s)": end_sec,
                "위험 구간 (프레임)": f"{s_frame} ~ {e_frame}",
                "위험 구간 (시간)": f"{start_sec:05.2f}s ~ {end_sec:05.2f}s",
                "판정 근거 - 임계치 달성 정도": f"{mean_prob1:.2f}%",
                "임계치 달성 정도 (수치)": round(mean_prob1, 2),
                "총 프레임 수": int(total_frames),
                "FPS": round(float(fps), 2),
                "영상 경로": video_path,
            })

    # 시작 프레임 순으로 정렬 (동일 시작 프레임이어도 위험 구분이 다르면 별도 Row로 유지)
    raw_intervals.sort(key=lambda r: r["_sort_key"])
    for r in raw_intervals:
        r.pop("_sort_key", None)

    # -------------------------------------------------------------------------
    # [구성 B - 방식 B-1] 위험 구간이 1개도 없는 안전 영상도 1줄(label=0, 구분="안전")로 기록
    # -------------------------------------------------------------------------
    if raw_intervals:
        interval_rows = raw_intervals
        is_video_dangerous = 1
        trigger_order = ["휘도 플래시", "적색 플래시", "지속 플래시", "줄무늬 패턴", "고대비"]
        detected_triggers = [
            t for t in trigger_order
            if any(r["위험 구분"] == t for r in raw_intervals)
        ]
        detected_triggers_str = ", ".join(detected_triggers)
        num_intervals = len(raw_intervals)
        frame_summary_str = " | ".join(
            f"[{r['위험 구분']}] {r['위험 구간 (프레임)']}" for r in raw_intervals
        )
        time_summary_str = " | ".join(
            f"[{r['위험 구분']}] {r['위험 구간 (시간)']}" for r in raw_intervals
        )
        max_prob1 = max(r["임계치 달성 정도 (수치)"] for r in raw_intervals)
    else:
        is_video_dangerous = 0
        detected_triggers_str = "안전"
        num_intervals = 0
        frame_summary_str = "-"
        time_summary_str = "-"
        max_prob1 = 0.0
        interval_rows = [{
            "파일명": file_name,
            "위험 여부 (label)": 0,
            "위험 구분": "안전",
            "시작 프레임": -1,
            "종료 프레임": -1,
            "구간 프레임 수": 0,
            "시작 시간 (s)": 0.0,
            "종료 시간 (s)": 0.0,
            "위험 구간 (프레임)": "-",
            "위험 구간 (시간)": "-",
            "판정 근거 - 임계치 달성 정도": "0.00%",
            "임계치 달성 정도 (수치)": 0.0,
            "총 프레임 수": int(total_frames),
            "FPS": round(float(fps), 2),
            "영상 경로": video_path,
        }]

    # -------------------------------------------------------------------------
    # [구성 A] 영상 단위 요약 Row (1개 영상 = 1개 Row)
    # -------------------------------------------------------------------------
    summary_row = {
        "파일명": file_name,
        "위험 여부 (label)": is_video_dangerous,
        "감지된 위험 구분": detected_triggers_str,
        "위험 구간 수": num_intervals,
        "위험 구간 요약 (프레임)": frame_summary_str,
        "위험 구간 요약 (시간)": time_summary_str,
        "최대 임계치 달성 정도 (%)": f"{max_prob1:.2f}%",
        "최대 임계치 달성 정도 (수치)": round(max_prob1, 2),
        "총 프레임 수": int(total_frames),
        "FPS": round(float(fps), 2),
        "재생 시간 (s)": round(float(duration_sec), 2),
        "분석 해상도": f"{proc_w}x{proc_h}",
        "영상 경로": video_path,
    }

    return summary_row, interval_rows


def build_pse_labeled_dataset(
    dataset_dir: str = DEFAULT_DATASET_DIR,
    output_dir: str = DEFAULT_OUTPUT_DIR,
    device: str = DEFAULT_DEVICE,
    display_area: int = DEFAULT_DISPLAY_AREA,
    max_side: int = MAX_SIDE,
    save_csv: bool = True,
    verbose: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    데이터셋 폴더 내 모든 동영상을 순회하며 판별을 수행하고,
    [구성 A: 영상 단위 요약 DataFrame]과 [구성 B: 구간 단위 상세 DataFrame]을 생성 및 CSV로 저장합니다.
    """
    video_files = collect_video_files(dataset_dir=dataset_dir, exclude_dirs=EXCLUDE_DIR_NAMES)
    if not video_files:
        raise ValueError(f"분석할 동영상 파일이 없습니다: {dataset_dir}")

    os.makedirs(output_dir, exist_ok=True)

    summary_rows = []
    all_interval_rows = []
    scan_cache = {}

    total_videos = len(video_files)
    t_start_all = time.time()

    if verbose:
        print("=" * 95)
        print(f"[LUMI 데이터셋 자동 판별 및 라벨링 시작]")
        print(f" - 입력 폴더 : {dataset_dir} (하위 제외 폴더: {', '.join(EXCLUDE_DIR_NAMES)})")
        print(f" - 대상 영상 : 총 {total_videos}개 파일")
        print(f" - 시청 기준 : {device} ({display_area} inch) / 최대 해상도 변: {max_side}px")
        print("=" * 95)

    for idx, v_path in enumerate(video_files, start=1):
        t0 = time.time()
        fname = os.path.basename(v_path)
        summary_row, interval_rows = analyze_single_video_for_dataset(
            video_path=v_path,
            device=device,
            display_area=display_area,
            max_side=max_side,
            scan_cache=scan_cache,
        )
        elapsed = time.time() - t0

        summary_rows.append(summary_row)
        all_interval_rows.extend(interval_rows)

        if verbose:
            status_str = "위험(1)" if summary_row["위험 여부 (label)"] == 1 else "안전(0)"
            print(
                f"[{idx:02d}/{total_videos:02d}] {fname:<10} | {status_str} | "
                f"구간: {summary_row['위험 구간 수']:2d}개 | "
                f"구분: {summary_row['감지된 위험 구분']:<28} | "
                f"{summary_row['총 프레임 수']:4d}f ({summary_row['재생 시간 (s)']:5.2f}s) | {elapsed:.2f}s 소요"
            )

    summary_columns = [
        "파일명",
        "위험 여부 (label)",
        "감지된 위험 구분",
        "위험 구간 수",
        "위험 구간 요약 (프레임)",
        "위험 구간 요약 (시간)",
        "최대 임계치 달성 정도 (%)",
        "최대 임계치 달성 정도 (수치)",
        "총 프레임 수",
        "FPS",
        "재생 시간 (s)",
        "분석 해상도",
        "영상 경로",
    ]
    interval_columns = [
        "파일명",
        "위험 여부 (label)",
        "위험 구분",
        "시작 프레임",
        "종료 프레임",
        "구간 프레임 수",
        "시작 시간 (s)",
        "종료 시간 (s)",
        "위험 구간 (프레임)",
        "위험 구간 (시간)",
        "판정 근거 - 임계치 달성 정도",
        "임계치 달성 정도 (수치)",
        "총 프레임 수",
        "FPS",
        "영상 경로",
    ]

    df_video_summary = pd.DataFrame(summary_rows, columns=summary_columns)
    df_interval_details = pd.DataFrame(all_interval_rows, columns=interval_columns)

    if save_csv:
        summary_csv_path = os.path.join(output_dir, SUMMARY_CSV_NAME)
        interval_csv_path = os.path.join(output_dir, INTERVAL_CSV_NAME)
        # utf-8-sig 인코딩을 사용하여 Windows Excel에서 바로 열어도 한글이 깨지지 않도록 저장
        df_video_summary.to_csv(summary_csv_path, index=False, encoding="utf-8-sig")
        df_interval_details.to_csv(interval_csv_path, index=False, encoding="utf-8-sig")

        if verbose:
            total_elapsed = time.time() - t_start_all
            num_danger_videos = int((df_video_summary["위험 여부 (label)"] == 1).sum())
            num_safe_videos = int((df_video_summary["위험 여부 (label)"] == 0).sum())
            print("=" * 95)
            print(
                f"[완료] 총 {total_videos}개 영상 판별 완료 (총 소요 시간: {total_elapsed:.2f}s) "
                f"-> 위험 영상: {num_danger_videos}개 / 안전 영상: {num_safe_videos}개"
            )
            print(f" - [구성 A 저장 완료] {summary_csv_path} ({len(df_video_summary)} rows)")
            print(f" - [구성 B 저장 완료] {interval_csv_path} ({len(df_interval_details)} rows)")
            print("=" * 95)

    return df_video_summary, df_interval_details


def _print_df(df: pd.DataFrame, title: str) -> None:
    print(f"\n[{title}]")
    with pd.option_context(
        "display.max_rows", None,
        "display.max_columns", None,
        "display.width", 1200,
        "display.unicode.east_asian_width", True,
    ):
        print(df)


def main():
    parser = argparse.ArgumentParser(
        description="LUMI 딥러닝 데이터셋 구축을 위한 다수 영상 자동 판별 및 라벨링 스크립트"
    )
    parser.add_argument(
        "--dataset_dir",
        type=str,
        default=DEFAULT_DATASET_DIR,
        help="분석할 동영상 데이터셋 폴더 경로",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=DEFAULT_OUTPUT_DIR,
        help="결과 CSV 파일 2종을 저장할 폴더 경로",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=DEFAULT_DEVICE,
        choices=["laptop", "desktop"],
        help="표준 시청거리 기기 종류 ('laptop': 56cm, 'desktop': 68cm)",
    )
    parser.add_argument(
        "--display_area",
        type=int,
        default=DEFAULT_DISPLAY_AREA,
        help="디스플레이 대각선 크기 (inch, 기본값: 15)",
    )
    parser.add_argument(
        "--max_side",
        type=int,
        default=MAX_SIDE,
        help="분석 시 최대 해상도 변 길이 (기본값: 720)",
    )
    args, _ = parser.parse_known_args()

    df_summary, df_intervals = build_pse_labeled_dataset(
        dataset_dir=args.dataset_dir,
        output_dir=args.output_dir,
        device=args.device,
        display_area=args.display_area,
        max_side=args.max_side,
        save_csv=True,
        verbose=True,
    )

    # 화면 출력 시에는 가독성을 위해 긴 절대 경로('영상 경로') 컬럼을 제외하고 핵심 컬럼 위주로 출력
    display_summary_cols = [c for c in df_summary.columns if c not in ("영상 경로", "최대 임계치 달성 정도 (수치)")]
    display_interval_cols = [c for c in df_intervals.columns if c not in ("영상 경로", "임계치 달성 정도 (수치)")]

    _print_df(df_summary[display_summary_cols], "구성 A: 영상 단위 요약 데이터프레임 (pse_video_summary.csv)")
    _print_df(df_intervals[display_interval_cols], "구성 B: 구간 단위 상세 데이터프레임 (pse_interval_details.csv)")


if __name__ == "__main__":
    main()
