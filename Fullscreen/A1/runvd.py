# 단일 영상 7대 트리거·4대 연속 위험도 진단 실행 예시
# python Fullscreen\A1\runvd.py <입력영상경로> --out-dir <출력폴더경로>

"""Single-video PSE inspection, CSV export, and time-series visualization (`runvd.py`).

Uses the `vdcorefs` package to evaluate a single video against the 7 structured PSE hazard
categories and 4 parallel continuous risk metrics, printing a detailed console summary and
optionally saving per-frame/interval CSVs and a 3-panel `.png` diagnostic plot.
"""

from __future__ import annotations

import argparse
from pathlib import Path
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

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from vdcorefs import (
    ALL_HAZARD_LABELS,
    PSEConfig,
    VideoAnalysisResult,
    ViewingConfig,
    ViewingProfile,
    analyze_video,
)

# Windows Korean glyph & minus-sign configuration
plt.rcParams["font.family"] = ["Malgun Gothic", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

# ===========================================================================
# Default Standalone Configuration (Replace paths or pass CLI arguments)
# ===========================================================================
PLACEHOLDER_PATH = "경로를 입력하세요!"
DEFAULT_VIDEO_PATH: str | Path = PLACEHOLDER_PATH
DEFAULT_OUTPUT_DIR: str | Path = _THIS_DIR / "Output" / "vd"
DEFAULT_PROFILE: ViewingProfile = "laptop"


def _validate_user_path(raw_path: str | Path | None, arg_label: str) -> Path:
    if raw_path is None:
        raise ValueError(f"{arg_label} 경로가 비어 있습니다. '{PLACEHOLDER_PATH}' 대신 실제 경로를 지정해 주세요.")
    text = str(raw_path).strip()
    if not text or "경로를 입력" in text:
        raise ValueError(
            f"{arg_label}에 '{text or PLACEHOLDER_PATH}'가 설정되어 있습니다. "
            f"스크립트 상단 변수 또는 CLI 인자(video / --video-path)로 실제 경로를 입력해 주세요."
        )
    return Path(text)

_LABEL_COLORS: dict[str, str] = {
    "휘도 플래시": "#f59e0b",
    "교집합": "#dc2626",
    "빠른 장면 전환": "#7c3aed",
    "적색 플래시": "#ef4444",
    "휘도 연장": "#d97706",
    "적색 연장": "#b91c1c",
    "정적 공간 패턴": "#2563eb",
}


def build_spec_summary_df(res: VideoAnalysisResult) -> pd.DataFrame:
    """Build a concise key-value DataFrame of viewing geometry & FPS window parameters."""
    g = res.geometry
    f = res.fps_spec
    rows = [
        ("원본 해상도 / 분석 해상도", f"{res.orig_width}x{res.orig_height} -> {res.analysis_width}x{res.analysis_height}"),
        ("총 프레임 수 / FPS / 재생 시간", f"{res.total_frames} frames @ {res.fps:.2f} fps ({res.total_frames / res.fps:.2f}s)"),
        ("물리 화면 크기 / 시청 거리", f"{g.width_cm:.2f} x {g.height_cm:.2f} cm (거리 {g.viewing_distance_cm:.1f} cm)"),
        ("수평·수직 시야각 (FOV)", f"{g.fov_w_deg:.2f} deg x {g.fov_h_deg:.2f} deg ({g.area_deg2:.1f} deg^2)"),
        ("1도당 픽셀 밀도 (px/deg)", f"가로 {g.px_per_deg_x:.2f} / 세로 {g.px_per_deg_y:.2f}"),
        ("시간 창 변환 (전이/동기화/병합)", f"<=66.8ms: {f.k_trans}f | <=20ms: {f.k_sync}f | <=15ms: {f.k_merge}f"),
        ("방송 안전 간격 / 연장·패턴 창", f"안전 간격 {f.safe_gap_ms:.1f} ms | 5s={f.w_5s}f (4s={f.w_4s}f) | 0.5s={f.n_static_0p5s}f"),
        ("연산 장치 (Device)", res.device),
    ]
    return pd.DataFrame(rows, columns=["항목", "설정·산출값"])


def save_analysis_csvs(res: VideoAnalysisResult, out_dir: Path) -> tuple[Path, Path]:
    """Export `<stem>_intervals.csv` and `<stem>_frame_metrics.csv` (`utf-8-sig`)."""
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = Path(res.video_path).stem
    intervals_csv = out_dir / f"{stem}_intervals.csv"
    frames_csv = out_dir / f"{stem}_frame_metrics.csv"
    res.intervals_df.to_csv(intervals_csv, index=False, encoding="utf-8-sig")
    res.frame_metrics_df.to_csv(frames_csv, index=False, encoding="utf-8-sig")
    return intervals_csv, frames_csv


def save_analysis_plot(
    res: VideoAnalysisResult,
    out_png: Path,
    *,
    show_window: bool = False,
) -> Path:
    """Render and save a 3-panel PSE hazard & continuous risk time-series chart (`.png`)."""
    out_png.parent.mkdir(parents=True, exist_ok=True)
    df = res.frame_metrics_df
    t_sec = df["timestamp_ms"].to_numpy(dtype=np.float64) / 1000.0
    dt_sec = 1.0 / max(res.fps, 1.0)
    t_end = float(t_sec[-1] + dt_sec) if len(t_sec) else 1.0

    fig, axes = plt.subplots(
        3,
        1,
        figsize=(14, 9.2),
        sharex=True,
        gridspec_kw={"height_ratios": [1.25, 1.35, 1.15]},
    )
    ax_gantt, ax_risk, ax_cnt = axes

    # 1) Top panel: 7-label Gantt timeline
    labels_rev = list(reversed(ALL_HAZARD_LABELS))
    for y_idx, lbl in enumerate(labels_rev):
        spans = res.intervals_by_label.get(lbl, [])
        color = _LABEL_COLORS.get(lbl, "#475569")
        for s_f, e_f in spans:
            s_t = float(t_sec[s_f])
            w_t = float(t_sec[e_f] + dt_sec - s_t)
            ax_gantt.broken_barh([(s_t, max(w_t, dt_sec))], (y_idx - 0.35, 0.7), facecolors=color, alpha=0.88)

    ax_gantt.set_yticks(range(len(labels_rev)))
    ax_gantt.set_yticklabels(labels_rev, fontsize=9.5)
    ax_gantt.set_ylim(-0.7, len(labels_rev) - 0.3)
    ax_gantt.set_xlim(0.0, t_end)
    ax_gantt.grid(True, axis="x", linestyle=":", alpha=0.45)
    verdict_txt = "위험 (DANGER)" if res.is_dangerous else "안전 (SAFE)"
    ax_gantt.set_title(
        f"[LUMI vdcore] {Path(res.video_path).name} — 판정: {verdict_txt} "
        f"({res.analysis_width}x{res.analysis_height} @ {res.fps:.2f} fps)",
        fontsize=12,
        fontweight="bold",
    )

    # 2) Middle panel: 4 Continuous Risk Metrics [0, 100]
    ax_risk.plot(t_sec, df["risk_calibrated"], label="risk_calibrated (경계=50)", color="#dc2626", linewidth=1.6)
    ax_risk.plot(t_sec, df["risk_ppr_pct"], label="risk_ppr_pct (%)", color="#2563eb", linewidth=1.3)
    ax_risk.plot(t_sec, df["risk_vfm_weibull"], label="risk_vfm_weibull (Apple UFM)", color="#7c3aed", linewidth=1.3)
    ax_risk.plot(
        t_sec,
        df["risk_nomura_scaled"],
        label="risk_nomura_scaled [0..100]",
        color="#059669",
        linewidth=1.2,
        linestyle="--",
    )
    ax_risk.axhline(50.0, color="#dc2626", linestyle=":", linewidth=1.1, alpha=0.75, label="위험 경계 (50.0)")
    ax_risk.set_ylabel("연속 위험도 [0-100]")
    ax_risk.set_ylim(-2.0, 103.0)
    ax_risk.grid(True, linestyle=":", alpha=0.45)
    ax_risk.legend(loc="upper right", ncol=3, fontsize=8.5)

    # 3) Bottom panel: 1s transition counts + raw Nomura e(t) on secondary y-axis
    ax_cnt.plot(t_sec, df["lum_transitions_1s"], label="휘도 전이/1s (>25%)", color="#f59e0b", linewidth=1.4)
    ax_cnt.plot(t_sec, df["red_transitions_1s"], label="적색 전이/1s (>25%)", color="#ef4444", linewidth=1.3)
    ax_cnt.plot(t_sec, df["scene_transitions_1s"], label="장면전환 전이/1s (>=80%)", color="#6d28d9", linewidth=1.2)
    ax_cnt.plot(
        t_sec,
        df["diag_transitions_25pct"],
        label="상위 25% 픽셀 전이/1s (diag)",
        color="#334155",
        linewidth=1.1,
        linestyle=":",
    )
    ax_cnt.axhline(7, color="#ef4444", linestyle="--", linewidth=0.9, alpha=0.55)
    ax_cnt.axhline(4, color="#6d28d9", linestyle="--", linewidth=0.9, alpha=0.55)
    ax_cnt.set_ylabel("1초 반대극성 전이 수")
    ax_cnt.set_xlabel("시간 (초, s)")
    ax_cnt.grid(True, linestyle=":", alpha=0.45)

    ax_nom = ax_cnt.twinx()
    ax_nom.plot(
        t_sec,
        df["risk_nomura"],
        label="risk_nomura 원시값 e(t)",
        color="#0d9488",
        linewidth=1.0,
        alpha=0.75,
    )
    ax_nom.set_ylabel("Nomura e(t)", color="#0d9488")
    ax_nom.tick_params(axis="y", labelcolor="#0d9488")

    lines_l, labels_l = ax_cnt.get_legend_handles_labels()
    lines_r, labels_r = ax_nom.get_legend_handles_labels()
    ax_cnt.legend(lines_l + lines_r, labels_l + labels_r, loc="upper right", ncol=3, fontsize=8.2)

    fig.tight_layout()
    fig.savefig(out_png, dpi=150)
    if show_window:
        plt.show()
    plt.close(fig)
    return out_png


def run_single_video(
    video_path: str | Path,
    *,
    viewing: ViewingConfig | None = None,
    pse: PSEConfig | None = None,
    device: str | None = None,
    out_dir: Path | None = DEFAULT_OUTPUT_DIR,
    save_csv: bool = True,
    save_plot: bool = True,
    show_window: bool = False,
) -> VideoAnalysisResult:
    """Analyze one video, print structured tables, and optionally save CSVs and PNG plot."""
    in_video = _validate_user_path(video_path, "입력 영상(DEFAULT_VIDEO_PATH / video)")
    t0 = time.perf_counter()
    res = analyze_video(in_video, viewing=viewing, pse=pse, device=device)
    elapsed = time.perf_counter() - t0

    spec_df = build_spec_summary_df(res)
    df = res.frame_metrics_df
    pk_cal = float(df["risk_calibrated"].max()) if len(df) else 0.0
    pk_ppr = float(df["risk_ppr_pct"].max()) if len(df) else 0.0
    pk_nom = float(df["risk_nomura"].max()) if len(df) else 0.0
    pk_vfm = float(df["risk_vfm_weibull"].max()) if len(df) else 0.0

    print("=" * 88)
    print(f"[LUMI runvd] 영상 분석 결과: {Path(res.video_path).name} (소요 시간: {elapsed:.2f}s, {res.total_frames / max(elapsed, 1e-6):.1f} fps)")
    print("=" * 88)
    print(spec_df.to_string(index=False))
    print("-" * 88)
    verdict_str = "위험 (label=1)" if res.is_dangerous else "안전 (label=0)"
    labels_str = ", ".join(res.active_labels) if res.active_labels else "안전"
    print(f"최종 판정        : {verdict_str}")
    print(f"감지된 위험 구분 : {labels_str} (총 {len(res.intervals)}개 구간)")
    print(
        f"최대 연속 위험도 : risk_calibrated={pk_cal:.2f} | risk_ppr_pct={pk_ppr:.2f}% | "
        f"risk_nomura={pk_nom:.4f} | risk_vfm_weibull={pk_vfm:.2f}"
    )
    print("-" * 88)
    print("[구간별 상세 내역]")
    print(res.intervals_df.to_string(index=False))

    target_dir = out_dir if out_dir is not None else DEFAULT_OUTPUT_DIR
    if save_csv:
        ic_path, fc_path = save_analysis_csvs(res, target_dir)
        print("-" * 88)
        print(f"구간 요약 CSV 저장   : {ic_path}")
        print(f"프레임 지표 CSV 저장 : {fc_path}")
    if save_plot:
        png_path = target_dir / f"{Path(res.video_path).stem}_pse_graph.png"
        save_analysis_plot(res, png_path, show_window=show_window)
        print(f"시계열 그래프 저장   : {png_path}")
    print("=" * 88)
    return res


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="LUMI 단일 영상 광과민성 뇌전증(PSE) 위험 판별 및 시각화 스크립트 (runvd)"
    )
    parser.add_argument(
        "video",
        nargs="?",
        type=Path,
        default=None,
        help=f"분석할 영상 경로 (생략 시 기본값: {DEFAULT_VIDEO_PATH})",
    )
    parser.add_argument("--video-path", dest="video_opt", type=Path, default=None, help="영상 경로 지정 옵션")
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help=f"CSV 및 그래프 저장 폴더 (기본값: {DEFAULT_OUTPUT_DIR})",
    )
    parser.add_argument(
        "--profile",
        choices=["laptop", "desktop", "tv"],
        default=DEFAULT_PROFILE,
        help="시청 환경 프로파일 (기본값: laptop)",
    )
    parser.add_argument("--diagonal-inch", type=float, default=None, help="화면 대각선 길이 (inch)")
    parser.add_argument("--distance-cm", type=float, default=None, help="시청 거리 (cm)")
    parser.add_argument("--l-max", type=float, default=200.0, help="디스플레이 최대 백색 휘도 (cd/m^2)")
    parser.add_argument("--max-width", type=int, default=640, help="분석 최대 가로 해상도 (기본값: 640)")
    parser.add_argument("--device", type=str, default=None, help="연산 장치 (cuda 또는 cpu)")
    parser.add_argument("--no-csv", action="store_true", help="CSV 파일 저장 생략")
    parser.add_argument("--no-plot", action="store_true", help="시계열 그래프(.png) 저장 생략")
    parser.add_argument("--show", action="store_true", help="그래프 창 화면 표시")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    target_video = args.video or args.video_opt or DEFAULT_VIDEO_PATH
    viewing = ViewingConfig(
        profile=args.profile,
        diagonal_inch=args.diagonal_inch,
        viewing_distance_cm=args.distance_cm,
        l_max_cd_m2=args.l_max,
        analysis_max_width=args.max_width,
    )
    run_single_video(
        target_video,
        viewing=viewing,
        device=args.device,
        out_dir=args.out_dir,
        save_csv=not args.no_csv,
        save_plot=not args.no_plot,
        show_window=args.show,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
