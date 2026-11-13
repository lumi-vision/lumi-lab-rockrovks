# A1 전체화면 보정 실행 (기본형: base, 전체 모드: all)
# python Fullscreen\A1\run_fullscreen_1112.py --video <입력영상경로> --output-dir <출력폴더경로> --mode base --no-show

# A1 부분 감광(Partial) 보정 실행 (기본형: base, 전체 모드: all)
# python Fullscreen\A1\run_partial_1112.py --video <입력영상경로> --output-dir <출력폴더경로> --mode base --no-show

# 단일 영상 위험도 진단
# python Fullscreen\A1\runvd.py <입력영상경로> --out-dir <출력폴더경로>

# 데이터셋 일괄 라벨링 CSV 생성
# python Fullscreen\A1\dsbuilder.py --input-dir <영상폴더경로> --output-dir <CSV저장폴더경로>

"""Batch dataset PSE auto-labeling builder (`dsbuilder.py`).

Scans an input video folder in natural filename order,
reuses resolution-keyed `vdcorefs` detector caches across videos, and writes:
  1. `pse_video_summary.csv`    : One row per video with binary label, active categories,
                                  interval summaries, and peak values of the 4 risk metrics.
  2. `pse_interval_details.csv` : One row per detected hazard interval (plus 1 safe row
                                  with `label=0` for safe videos) with all 4 risk metrics.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import re
import sys
import time
from typing import Sequence

if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

_THIS_DIR = Path(__file__).resolve().parent
if str(_THIS_DIR) not in sys.path:
    sys.path.insert(0, str(_THIS_DIR))

import numpy as np
import pandas as pd

from vdcorefs import (
    DetectorCache,
    LABEL_SAFE,
    PSEConfig,
    VideoAnalysisResult,
    ViewingConfig,
    ViewingProfile,
    analyze_video,
)

# ===========================================================================
# Default Input / Output Paths (Replace paths or pass --input-dir / --output-dir)
# ===========================================================================
PLACEHOLDER_PATH = "경로를 입력하세요!"
DEFAULT_INPUT_DIR: str | Path = PLACEHOLDER_PATH
DEFAULT_OUTPUT_DIR: str | Path = _THIS_DIR / "Dataset"
SUMMARY_CSV_NAME = "pse_video_summary.csv"
INTERVAL_CSV_NAME = "pse_interval_details.csv"


def _validate_user_path(raw_path: str | Path | None, arg_label: str) -> Path:
    if raw_path is None:
        raise ValueError(f"{arg_label} 경로가 비어 있습니다. '{PLACEHOLDER_PATH}' 대신 실제 경로를 지정해 주세요.")
    text = str(raw_path).strip()
    if not text or "경로를 입력" in text:
        raise ValueError(
            f"{arg_label}에 '{text or PLACEHOLDER_PATH}'가 설정되어 있습니다. "
            f"스크립트 상단 변수 또는 CLI 옵션(--input-dir / --output-dir)으로 실제 경로를 입력해 주세요."
        )
    return Path(text)

VIDEO_EXTENSIONS: frozenset[str] = frozenset(
    {".mp4", ".avi", ".mov", ".mkv", ".webm", ".m4v", ".wmv"}
)
EXCLUDED_DIR_NAMES: frozenset[str] = frozenset({"out", "output", "__pycache__", ".ipynb_checkpoints"})


def natural_sort_key(path: Path) -> list[int | str]:
    """Sort `1.mp4, 2.mp4, ..., 10.mp4` in human numerical order."""
    parts = re.split(r"(\d+)", path.name.lower())
    return [int(part) if part.isdigit() else part for part in parts]


def collect_video_files(input_dir: Path, recursive: bool = False) -> list[Path]:
    """Collect valid video files sorted in natural filename order."""
    if not input_dir.exists():
        raise FileNotFoundError(f"Input directory not found: {input_dir}")

    candidates = input_dir.rglob("*") if recursive else input_dir.iterdir()
    videos: list[Path] = []
    for p in candidates:
        if not p.is_file():
            continue
        if any(part.lower() in EXCLUDED_DIR_NAMES for part in p.relative_to(input_dir).parts[:-1]):
            continue
        if p.suffix.lower() in VIDEO_EXTENSIONS:
            videos.append(p)
    videos.sort(key=natural_sort_key)
    return videos


def _build_summary_row(res: VideoAnalysisResult, video_path: Path) -> dict[str, object]:
    df = res.frame_metrics_df
    pk_cal = round(float(df["risk_calibrated"].max()), 2) if len(df) else 0.0
    pk_ppr = round(float(df["risk_ppr_pct"].max()), 2) if len(df) else 0.0
    pk_nom = round(float(df["risk_nomura"].max()), 4) if len(df) else 0.0
    pk_vfm = round(float(df["risk_vfm_weibull"].max()), 2) if len(df) else 0.0

    if res.intervals:
        frame_summary = " | ".join(
            f"[{r.label}] {r.start_frame} ~ {r.end_frame}" for r in res.intervals
        )
        time_summary = " | ".join(
            f"[{r.label}] {r.start_sec:05.2f}s ~ {r.end_sec:05.2f}s" for r in res.intervals
        )
        categories_str = ", ".join(res.active_labels)
    else:
        frame_summary = "-"
        time_summary = "-"
        categories_str = LABEL_SAFE

    return {
        "파일명": video_path.name,
        "위험 여부 (label)": 1 if res.is_dangerous else 0,
        "감지된 위험 구분": categories_str,
        "위험 구간 수": len(res.intervals),
        "위험 구간 요약 (프레임)": frame_summary,
        "위험 구간 요약 (시간)": time_summary,
        "최대 risk_calibrated": pk_cal,
        "최대 risk_ppr_pct": pk_ppr,
        "최대 risk_nomura": pk_nom,
        "최대 risk_vfm_weibull": pk_vfm,
        "총 프레임 수": res.total_frames,
        "FPS": round(res.fps, 2),
        "재생 시간 (s)": round(res.total_frames / max(res.fps, 1e-6), 2),
        "분석 해상도": f"{res.analysis_width}x{res.analysis_height}",
        "영상 경로": str(video_path),
    }


def _build_interval_rows(res: VideoAnalysisResult, video_path: Path) -> list[dict[str, object]]:
    df = res.frame_metrics_df
    fps_round = round(res.fps, 2)

    if not res.intervals:
        pk_cal = round(float(df["risk_calibrated"].max()), 2) if len(df) else 0.0
        pk_ppr = round(float(df["risk_ppr_pct"].max()), 2) if len(df) else 0.0
        pk_nom = round(float(df["risk_nomura"].max()), 4) if len(df) else 0.0
        pk_vfm = round(float(df["risk_vfm_weibull"].max()), 2) if len(df) else 0.0
        return [
            {
                "파일명": video_path.name,
                "위험 여부 (label)": 0,
                "위험 구분": LABEL_SAFE,
                "시작 프레임": -1,
                "종료 프레임": -1,
                "구간 프레임 수": 0,
                "시작 시간 (s)": 0.0,
                "종료 시간 (s)": 0.0,
                "지속 시간 (s)": 0.0,
                "위험 구간 (프레임)": "-",
                "위험 구간 (시간)": "-",
                "최대 risk_calibrated": pk_cal,
                "최대 risk_ppr_pct": pk_ppr,
                "최대 risk_nomura": pk_nom,
                "최대 risk_vfm_weibull": pk_vfm,
                "총 프레임 수": res.total_frames,
                "FPS": fps_round,
                "영상 경로": str(video_path),
            }
        ]

    rows: list[dict[str, object]] = []
    for rec in res.intervals:
        rows.append(
            {
                "파일명": video_path.name,
                "위험 여부 (label)": 1,
                "위험 구분": rec.label,
                "시작 프레임": rec.start_frame,
                "종료 프레임": rec.end_frame,
                "구간 프레임 수": rec.end_frame - rec.start_frame + 1,
                "시작 시간 (s)": round(rec.start_sec, 2),
                "종료 시간 (s)": round(rec.end_sec, 2),
                "지속 시간 (s)": round(rec.duration_sec, 2),
                "위험 구간 (프레임)": f"{rec.start_frame} ~ {rec.end_frame}",
                "위험 구간 (시간)": f"{rec.start_sec:05.2f}s ~ {rec.end_sec:05.2f}s",
                "최대 risk_calibrated": rec.peak_risk_calibrated,
                "최대 risk_ppr_pct": rec.peak_risk_ppr_pct,
                "최대 risk_nomura": rec.peak_risk_nomura,
                "최대 risk_vfm_weibull": rec.peak_risk_vfm_weibull,
                "총 프레임 수": res.total_frames,
                "FPS": fps_round,
                "영상 경로": str(video_path),
            }
        )
    return rows


def build_dataset(
    input_dir: str | Path = DEFAULT_INPUT_DIR,
    output_dir: str | Path = DEFAULT_OUTPUT_DIR,
    *,
    viewing: ViewingConfig | None = None,
    pse: PSEConfig | None = None,
    device: str | None = None,
    recursive: bool = False,
    limit: int | None = None,
    frame_csv_dir: Path | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Analyze all videos in `input_dir` and write summary & interval CSV files to `output_dir`."""
    in_folder = _validate_user_path(input_dir, "입력 폴더(DEFAULT_INPUT_DIR / --input-dir)")
    out_folder = _validate_user_path(output_dir, "출력 폴더(DEFAULT_OUTPUT_DIR / --output-dir)")

    videos = collect_video_files(in_folder, recursive=recursive)
    if limit is not None and limit > 0:
        videos = videos[:limit]
    if not videos:
        raise RuntimeError(f"No video files found in {in_folder}")

    out_folder.mkdir(parents=True, exist_ok=True)
    if frame_csv_dir is not None:
        frame_csv_dir.mkdir(parents=True, exist_ok=True)

    view_cfg = viewing or ViewingConfig()
    pse_cfg = pse or PSEConfig()
    cache: DetectorCache = {}

    summary_rows: list[dict[str, object]] = []
    interval_rows: list[dict[str, object]] = []

    t_batch_start = time.perf_counter()
    print(f"[dsbuilder] 입력 폴더: {in_folder} (총 {len(videos)}개 영상)")
    print(f"[dsbuilder] 출력 폴더: {out_folder}")
    print("-" * 88)

    for idx, vpath in enumerate(videos, start=1):
        t0 = time.perf_counter()
        res = analyze_video(
            vpath,
            viewing=view_cfg,
            pse=pse_cfg,
            device=device,
            detector_cache=cache,
        )
        dt = time.perf_counter() - t0

        summary_rows.append(_build_summary_row(res, vpath))
        interval_rows.extend(_build_interval_rows(res, vpath))

        if frame_csv_dir is not None:
            res.frame_metrics_df.to_csv(
                frame_csv_dir / f"{vpath.stem}_frame_metrics.csv",
                index=False,
                encoding="utf-8-sig",
            )

        status = "위험" if res.is_dangerous else "안전"
        cats = ", ".join(res.active_labels) if res.active_labels else LABEL_SAFE
        print(
            f"[{idx:02d}/{len(videos):02d}] {vpath.name:<12} | {status} | "
            f"구간={len(res.intervals):2d} ({cats}) | "
            f"{res.total_frames:4d}f @ {res.fps:5.2f}fps | {dt:5.2f}s"
        )

    summary_df = pd.DataFrame(summary_rows)
    intervals_df = pd.DataFrame(interval_rows)

    summary_path = out_folder / SUMMARY_CSV_NAME
    interval_path = out_folder / INTERVAL_CSV_NAME
    summary_df.to_csv(summary_path, index=False, encoding="utf-8-sig")
    intervals_df.to_csv(interval_path, index=False, encoding="utf-8-sig")

    total_elapsed = time.perf_counter() - t_batch_start
    n_danger = int((summary_df["위험 여부 (label)"] == 1).sum())
    n_safe = int((summary_df["위험 여부 (label)"] == 0).sum())

    print("-" * 88)
    print(
        f"[dsbuilder 완료] 총 {len(videos)}개 영상 (위험 {n_danger}, 안전 {n_safe}) | "
        f"총 행 수={len(intervals_df)} | 소요 시간={total_elapsed:.2f}s"
    )
    print(f"  - 영상 요약 CSV : {summary_path}")
    print(f"  - 구간 상세 CSV : {interval_path}")
    return summary_df, intervals_df


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="LUMI 데이터셋 PSE 자동 라벨링 빌더 (dsbuilder)"
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=DEFAULT_INPUT_DIR,
        help=f"입력 영상 폴더 (기본값: {DEFAULT_INPUT_DIR})",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help=f"CSV 저장 폴더 (기본값: {DEFAULT_OUTPUT_DIR})",
    )
    parser.add_argument(
        "--profile",
        choices=["laptop", "desktop", "tv"],
        default="laptop",
        help="시청 환경 프로파일 (기본값: laptop)",
    )
    parser.add_argument("--diagonal-inch", type=float, default=None, help="화면 대각선 길이 (inch)")
    parser.add_argument("--distance-cm", type=float, default=None, help="시청 거리 (cm)")
    parser.add_argument("--l-max", type=float, default=200.0, help="디스플레이 최대 백색 휘도 (cd/m^2)")
    parser.add_argument("--max-width", type=int, default=640, help="분석 최대 가로 해상도 (기본값: 640)")
    parser.add_argument("--device", type=str, default=None, help="연산 장치 (cuda 또는 cpu)")
    parser.add_argument("--recursive", action="store_true", help="하위 폴더 재귀 탐색")
    parser.add_argument("--limit", type=int, default=None, help="앞에서부터 지정 개수 영상만 분석")
    parser.add_argument(
        "--frame-csv-dir",
        type=Path,
        default=None,
        help="영상별 프레임 단위 상세 지표 CSV 저장 폴더 (선택)",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    viewing = ViewingConfig(
        profile=args.profile,
        diagonal_inch=args.diagonal_inch,
        viewing_distance_cm=args.distance_cm,
        l_max_cd_m2=args.l_max,
        analysis_max_width=args.max_width,
    )
    build_dataset(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        viewing=viewing,
        device=args.device,
        recursive=args.recursive,
        limit=args.limit,
        frame_csv_dir=args.frame_csv_dir,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
