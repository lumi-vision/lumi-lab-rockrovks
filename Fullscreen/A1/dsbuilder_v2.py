"""
LUMI 딥러닝 학습 데이터셋 자동 판별 및 라벨링 빌더 (dsbuilder_v2.py)
=============================================================================
[역할 및 목적]
1. 지정된 데이터셋 폴더 내의 모든 동영상 파일을
   탐색합니다. (하위 'out' 폴더는 자동 제외하며, 파일명 숫자 순서대로 자연 정렬)
2. 핵심 판별 엔진(vdcoreops_v2.py - 프레임률 독립적 334ms 점멸 연속성 기준 적용)의
   공통 분석 파이프라인(analyze_video_frames)을 호출하여
   영상별 5대 위험 트리거(휘도 플래시, 적색 플래시, 지속 플래시, 줄무늬 패턴, 고대비)를 판별합니다.
3. 딥러닝 모델 학습에 즉시 활용할 수 있도록 두 가지 형태의 라벨링 데이터프레임을 생성하고
   지정된 출력 폴더(Dataset)에 CSV(UTF-8-SIG, 엑셀 한글 호환)로 저장합니다.

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
import numpy as np
import pandas as pd
import torch

_CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
if _CURRENT_DIR not in sys.path:
    sys.path.insert(0, _CURRENT_DIR)

from vdcoreops_v2 import (
    MAX_SIDE,
    analyze_video_frames,
)

# Windows 터미널 한글 출력 깨짐 방지
if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass
"""경로 입력 필요! 주의!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!"""
# =========================================================================
# 1. 기본 경로 및 데이터셋 구축 설정 (추후 영상 폴더나 설정 추가 시 이 부분만 수정)
# =========================================================================
DEFAULT_DATASET_DIR = r"경로를 입력하세요"
DEFAULT_OUTPUT_DIR = r"경로를 입력하세요"

SUMMARY_CSV_NAME = "pse_video_summary.csv"       # 구성 A 파일명 (영상 단위 요약)
INTERVAL_CSV_NAME = "pse_interval_details.csv"   # 구성 B 파일명 (구간 단위 상세)

DEFAULT_DEVICE = "laptop"                        # 'laptop' (56.0 cm) 또는 'desktop' (68.0 cm)
DEFAULT_DISPLAY_AREA = 15                        # 디스플레이 대각선 크기 (inch)
MAX_SAFE_GAP_MS = 334.0                          # 점멸 간 최대 허용 안전 텀 (334ms 이상이면 비연속으로 제외)

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
    max_safe_gap_ms: float = MAX_SAFE_GAP_MS,
    scan_cache: dict | None = None,
) -> tuple[dict, list[dict]]:
    """
    단일 동영상을 분석하여:
    1) [구성 A] 영상 단위 요약 딕셔너리 1개 (summary_row)
    2) [구성 B] 구간 단위 상세 딕셔너리 리스트 (interval_rows, 방식 B-1 적용)
    를 반환합니다.
    - vdcoreops_v2의 공용 analyze_video_frames를 호출하여 중복 프레임 루프를 제거하고,
      동일 해상도에 대한 시야각 및 60개 직선 스캔 좌표를 scan_cache에 캐싱하여 재사용합니다.
    """
    analysis = analyze_video_frames(
        video_path=video_path,
        device=device,
        display_area=display_area,
        max_side=max_side,
        max_safe_gap_ms=max_safe_gap_ms,
        scan_cache=scan_cache,
        return_physical_stats=False,
    )
    meta = analysis["meta"]
    proc_w, proc_h = meta["proc_w"], meta["proc_h"]
    fps = meta["fps"]
    total_frames = meta["total_frames"]
    duration_sec = meta["duration_sec"]
    trigger_specs = analysis["trigger_specs"]

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
    max_safe_gap_ms: float = MAX_SAFE_GAP_MS,
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
        print(f"[LUMI 데이터셋 자동 판별 및 라벨링 시작 (v2: 334ms 점멸 연속성 기준)]")
        print(f" - 입력 폴더 : {dataset_dir} (하위 제외 폴더: {', '.join(EXCLUDE_DIR_NAMES)})")
        print(f" - 대상 영상 : 총 {total_videos}개 파일")
        print(f" - 시청 기준 : {device} ({display_area} inch) / 최대 해상도 변: {max_side}px / 점멸 안전 텀: < {max_safe_gap_ms:.0f}ms")
        print("=" * 95)

    for idx, v_path in enumerate(video_files, start=1):
        t0 = time.time()
        fname = os.path.basename(v_path)
        summary_row, interval_rows = analyze_single_video_for_dataset(
            video_path=v_path,
            device=device,
            display_area=display_area,
            max_side=max_side,
            max_safe_gap_ms=max_safe_gap_ms,
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
        description="LUMI 딥러닝 데이터셋 구축을 위한 다수 영상 자동 판별 및 라벨링 스크립트 (v2: 334ms 점멸 연속성 기준)"
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

    display_summary_cols = [c for c in df_summary.columns if c not in ("영상 경로", "최대 임계치 달성 정도 (수치)")]
    display_interval_cols = [c for c in df_intervals.columns if c not in ("영상 경로", "임계치 달성 정도 (수치)")]

    _print_df(df_summary[display_summary_cols], "구성 A: 영상 단위 요약 데이터프레임 (pse_video_summary.csv)")
    _print_df(df_intervals[display_interval_cols], "구성 B: 구간 단위 상세 데이터프레임 (pse_interval_details.csv)")


if __name__ == "__main__":
    main()
