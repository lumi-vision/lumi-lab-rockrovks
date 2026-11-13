# A1 전체화면(Fullscreen) 보정 실행 예시 (모드: base | hold | safe | all)
# python Fullscreen\A1\run_fullscreen_1112.py --video <입력영상경로> --output-dir <출력폴더경로> --mode base --no-show
"""Unified CLI & 4-Panel Comparison Plot Runner for LUMI A1 Full-Screen Dimming (`run_fullscreen_1112.py`).

Supports all three A1 scheduling modes (`base`, `hold`, `safe`, or `all` to run all three)
on top of `vdcore` and `fullscreen_coreops_1112.py`, saving:
  - Original-resolution dimmed `.mp4` video:
      * `base`: `<stem>_fullscreen_dimmed.mp4`
      * `hold`: `<stem>_fullscreen_hold_dimmed.mp4`
      * `safe`: `<stem>_fullscreen_safe_dimmed.mp4`
  - 4-panel Layout A before/after comparison `.png` plot:
      * `base`: `<stem>_fullscreen_graph.png`
      * `hold`: `<stem>_fullscreen_hold_graph.png`
      * `safe`: `<stem>_fullscreen_safe_graph.png`
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

from vdcorefs import (
    DetectorCache,
    PSEConfig,
    ViewingConfig,
    ViewingProfile,
    analyze_video,
    mask_to_spans,
)
from fullscreen_coreops_1112 import (
    A1Config,
    A1DimmingResult,
    A1Mode,
    FADE_DURATION_SEC,
    HALF_HOLD_DURATION_SEC,
    TARGET_SAFE_LUM_CD,
    TARGET_SAFE_RED_EXCESS,
    VALID_A1_MODES,
    dim_video_fullscreen_a1,
)

# Windows Korean font & minus-sign configuration
plt.rcParams["font.family"] = ["Malgun Gothic", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

# ===========================================================================
# Default Standalone Settings (Replace paths or pass --video / --output-dir)
# ===========================================================================
PLACEHOLDER_PATH = "경로를 입력하세요!"
VIDEO_PATH: str = PLACEHOLDER_PATH
OUTPUT_DIR: str = PLACEHOLDER_PATH
DEFAULT_MODE: str = "base"             # 'base' | 'hold' | 'safe' | 'all'
DEFAULT_PROFILE: ViewingProfile = "laptop"

_MODE_TITLES: dict[A1Mode, str] = {
    "base": "기본형 (코사인 Fade-in/out)",
    "hold": "절반 완화 유지형 (Half-Hold)",
    "safe": "Safe형 (확정 구간 ∪ 4종 지표>=50% + Half-Hold)",
}

_MODE_SUFFIXES: dict[A1Mode, tuple[str, str]] = {
    "base": ("_fullscreen_dimmed.mp4", "_fullscreen_graph.png"),
    "hold": ("_fullscreen_hold_dimmed.mp4", "_fullscreen_hold_graph.png"),
    "safe": ("_fullscreen_safe_dimmed.mp4", "_fullscreen_safe_graph.png"),
}


def _validate_user_path(raw_path: str | Path | None, arg_label: str) -> Path:
    if raw_path is None:
        raise ValueError(f"{arg_label} 경로가 비어 있습니다. '{PLACEHOLDER_PATH}' 대신 실제 경로를 지정해 주세요.")
    text = str(raw_path).strip()
    if not text or "경로를 입력" in text:
        raise ValueError(
            f"{arg_label}에 '{text or PLACEHOLDER_PATH}'가 설정되어 있습니다. "
            f"스크립트 상단 변수 또는 CLI 옵션(--video / --output-dir)으로 실제 경로를 입력해 주세요."
        )
    return Path(text)


def _plot_4metrics_panel(
    ax_left: plt.Axes,
    time_sec: np.ndarray,
    res_df,
    panel_title: str,
    y_label_left: str,
) -> None:
    """Render 3 `[0, 100]` risk metrics on left axis + raw `risk_nomura` `[0, 1]` on aligned right axis."""
    ax_left.plot(
        time_sec,
        res_df["risk_calibrated"],
        color="#dc2626",
        linewidth=1.6,
        label="risk_calibrated (경계=50)",
    )
    ax_left.plot(
        time_sec,
        res_df["risk_ppr_pct"],
        color="#2563eb",
        linewidth=1.3,
        label="risk_ppr_pct (%)",
    )
    ax_left.plot(
        time_sec,
        res_df["risk_vfm_weibull"],
        color="#7c3aed",
        linewidth=1.3,
        label="risk_vfm_weibull (Apple UFM)",
    )
    ax_left.axhline(
        50.0,
        color="#dc2626",
        linestyle=":",
        linewidth=1.15,
        alpha=0.85,
        label="위험 경계 (50.0 / Nomura 0.50)",
    )
    ax_left.set_ylabel(y_label_left, fontsize=10.5, fontweight="bold")
    ax_left.set_title(panel_title, fontsize=11, fontweight="bold", loc="left")
    ax_left.set_ylim(-3.0, 105.0)

    ax_right = ax_left.twinx()
    ax_right.plot(
        time_sec,
        res_df["risk_nomura"],
        color="#0d9488",
        linewidth=1.25,
        linestyle="--",
        alpha=0.9,
        label="risk_nomura 원시값 e(t) (경계=0.5)",
    )
    ax_right.set_ylabel("Nomura e(t) [0-1]", color="#0d9488", fontsize=10)
    ax_right.tick_params(axis="y", labelcolor="#0d9488")
    # Exact 1:100 vertical alignment so left y=50.0 coincides with right y=0.50
    ax_right.set_ylim(-0.03, 1.05)

    lines_l, labels_l = ax_left.get_legend_handles_labels()
    lines_r, labels_r = ax_right.get_legend_handles_labels()
    ax_left.legend(lines_l + lines_r, labels_l + labels_r, loc="upper right", ncol=3, fontsize=8.3)


def plot_and_save_a1_layout_a(
    result: A1DimmingResult,
    save_png_path: str | Path,
    *,
    show_window: bool = False,
) -> Path:
    """Save the 4-panel Layout A before/after A1 mitigation comparison chart (`.png`)."""
    out_png = Path(save_png_path)
    out_png.parent.mkdir(parents=True, exist_ok=True)

    df_before = result.before.frame_metrics_df
    df_after = result.after.frame_metrics_df
    phys_before = result.before.physical_stats
    phys_after = result.after.physical_stats
    if phys_before is None or phys_after is None:
        raise ValueError("Physical stats are required on both `before` and `after` results for plotting.")

    fps = max(result.fps, 1e-6)
    dt_sec = 1.0 / fps
    time_sec = df_before["timestamp_ms"].to_numpy(dtype=np.float64) / 1000.0
    t_end = float(time_sec[-1] + dt_sec) if len(time_sec) else 1.0

    orig_max_lum = phys_before["max_lum_cd_m2"]
    orig_mean_lum = phys_before["mean_lum_cd_m2"]
    dim_max_lum = phys_after["max_lum_cd_m2"]
    dim_mean_lum = phys_after["mean_lum_cd_m2"]

    alpha_lin = result.alpha_schedule
    alpha_srgb = np.power(np.clip(alpha_lin, 0.0, 1.0), 1.0 / 2.4)

    confirmed_spans = mask_to_spans(df_before["is_dangerous"].to_numpy(dtype=bool))
    extra_spans = result.extra_safe_runs

    num_iv_before = len(result.before.intervals)
    num_iv_after = len(result.after.intervals)
    video_name = Path(result.input_video_path).name
    mode_title = _MODE_TITLES[result.mode]

    if result.mode == "safe":
        n_u_before = int(np.sum(result.over_4metrics_before))
        n_u_after = int(np.sum(result.over_4metrics_after))
        subtitle = (
            f"(확정 위험 구간: {num_iv_before}개 → {num_iv_after}개  |  "
            f"4종 지표>=50% 합집합 프레임: {n_u_before}f → {n_u_after}f  |  "
            f"저장 해상도: {result.orig_width}x{result.orig_height})"
        )
    else:
        subtitle = (
            f"(보정 전 확정 위험 구간: {num_iv_before}개  →  보정 후 확정 위험 구간: {num_iv_after}개  |  "
            f"저장 해상도: {result.orig_width}x{result.orig_height})"
        )

    fig, axes = plt.subplots(
        4,
        1,
        figsize=(14, 13.0),
        sharex=True,
        gridspec_kw={"height_ratios": [1.15, 1.15, 1.15, 1.0]},
    )
    fig.suptitle(
        f"LUMI A1 전체화면 밝기 보정 [{mode_title}, 1112] 전·후 비교 — {video_name}\n{subtitle}",
        fontsize=13.2,
        fontweight="bold",
        y=0.985,
    )

    # Background interval shading across all 4 panels
    for ax_idx, ax in enumerate(axes):
        for sp_idx, (s_f, e_f) in enumerate(confirmed_spans):
            lbl = "원본 확정 위험 구간" if (ax_idx == 0 and sp_idx == 0) else None
            ax.axvspan(
                float(time_sec[s_f]),
                float(time_sec[e_f] + dt_sec),
                color="#ffcccc",
                alpha=0.36,
                label=lbl,
            )
        for ex_idx, (s_ex, e_ex) in enumerate(extra_spans):
            lbl_ex = "Safe 추가 보정 구간 (4종>=50%)" if (ax_idx == 0 and ex_idx == 0) else None
            ax.axvspan(
                float(time_sec[s_ex]),
                float(time_sec[e_ex] + dt_sec),
                color="#ffe0b2",
                alpha=0.48,
                label=lbl_ex,
            )
        ax.set_xlim(0.0, t_end)
        ax.grid(True, linestyle="--", alpha=0.4)

    # Panel 1: Physical Luminance (Before vs After)
    ax1 = axes[0]
    ax1.plot(time_sec, orig_max_lum, color="#d62728", linewidth=1.4, alpha=0.85, label="보정 전 최대 물리 휘도 (cd/m²)")
    ax1.plot(time_sec, orig_mean_lum, color="#ff7f0e", linewidth=1.1, linestyle=":", alpha=0.85, label="보정 전 평균 물리 휘도 (cd/m²)")
    ax1.plot(time_sec, dim_max_lum, color="#1f77b4", linewidth=1.6, label="보정 후 최대 물리 휘도 (cd/m²)")
    ax1.plot(time_sec, dim_mean_lum, color="#17becf", linewidth=1.2, linestyle="--", label="보정 후 평균 물리 휘도 (cd/m²)")
    ax1.axhline(20.0, color="#2ca02c", linestyle="--", linewidth=1.2, label="국제 표준 안전 역치 (20 cd/m²)")
    ax1.axhline(
        result.config.target_safe_lum_cd,
        color="#2ca02c",
        linestyle=":",
        linewidth=1.0,
        alpha=0.75,
        label=f"보정 목표 상한 ({result.config.target_safe_lum_cd:.0f} cd/m²)",
    )
    ax1.set_ylabel("물리 휘도 (cd/m²)", fontsize=10.5, fontweight="bold")
    ax1.set_title("① 보정 전 vs 보정 후 프레임별 물리 휘도 변화", fontsize=11, fontweight="bold", loc="left")
    ax1.set_ylim(bottom=0.0)
    ax1.legend(loc="upper right", fontsize=8.5, ncol=3)

    # Panel 2: Before-Mitigation 4 Continuous Risk Metrics
    _plot_4metrics_panel(
        axes[1],
        time_sec,
        df_before,
        panel_title="② 보정 전 4종 연속 위험도 시계열",
        y_label_left="보정 전 위험도 [0-100]",
    )

    # Panel 3: After-Mitigation 4 Continuous Risk Metrics
    _plot_4metrics_panel(
        axes[2],
        time_sec,
        df_after,
        panel_title="③ 보정 후 4종 연속 위험도 시계열",
        y_label_left="보정 후 위험도 [0-100]",
    )

    # Panel 4: Applied Alpha Schedule (Linear & Perceptual sRGB)
    ax4 = axes[3]
    ax4.plot(
        time_sec,
        alpha_lin * 100.0,
        color="#2ca02c",
        linewidth=1.8,
        label="선형 물리 휘도 배율 (alpha_t, %)",
    )
    ax4.plot(
        time_sec,
        alpha_srgb * 100.0,
        color="#9467bd",
        linewidth=1.5,
        linestyle="--",
        label="체감 sRGB 픽셀 밝기 배율 (alpha_t^(1/2.4), %)",
    )
    ax4.set_xlabel("시간 (초, sec)", fontsize=11, fontweight="bold")
    ax4.set_ylabel("밝기 유지 배율 (%)", fontsize=10.5, fontweight="bold")
    ax4.set_title(
        f"④ 시간에 따라 적용된 전체 화면 밝기 조절 배율 [{mode_title}]",
        fontsize=11,
        fontweight="bold",
        loc="left",
    )
    ax4.set_ylim(-3.0, 105.0)
    ax4.legend(loc="lower right", fontsize=9)

    fig.tight_layout(rect=[0, 0, 1, 0.955])
    fig.savefig(out_png, dpi=150)

    if show_window:
        try:
            plt.show()
        except Exception:
            pass
    plt.close(fig)
    return out_png


def _print_mode_summary(res: A1DimmingResult, out_png: Path, elapsed_sec: float) -> None:
    min_lin = float(np.min(res.alpha_schedule)) if len(res.alpha_schedule) else 1.0
    min_srgb = min_lin ** (1.0 / 2.4)
    before_cnt = len(res.before.intervals)
    after_cnt = len(res.after.intervals)

    print("-" * 92)
    print(f"[모드: {res.mode.upper()} — {_MODE_TITLES[res.mode]}] (소요 시간: {elapsed_sec:.2f}s)")
    print(
        f" - 원본/저장 해상도 : {res.orig_width}x{res.orig_height} "
        f"(분석 해상도: {res.analysis_width}x{res.analysis_height}, {res.fps:.2f} fps, 총 {res.total_frames}f)"
    )
    print(f" - 확정 위험 구간   : 보정 전 {before_cnt}개  →  보정 후 {after_cnt}개")
    if res.mode == "safe":
        n_u_b = int(np.sum(res.over_4metrics_before))
        n_u_a = int(np.sum(res.over_4metrics_after))
        print(
            f" - 4종>=50% 합집합  : 보정 전 {n_u_b}프레임 (추가 구간 {len(res.extra_safe_runs)}개)  →  보정 후 {n_u_a}프레임"
        )
    print(f" - 최소 밝기 배율   : 선형 {min_lin * 100:.2f}% (체감 sRGB {min_srgb * 100:.2f}%)")
    if after_cnt == 0:
        print(" - 사후 검증 결과   : [검증 완료] 7대 광과민성 위험 구간이 모두 제거되었습니다 (안전).")
    else:
        rem = ", ".join(f"[{r.label}] {r.start_frame}~{r.end_frame}" for r in res.after.intervals)
        print(f" - 사후 검증 결과   : [주의] 잔여 위험 구간 {after_cnt}개 존재 — {rem}")
    print(f" - 보정 영상 저장   : {res.output_video_path}")
    print(f" - 비교 그래프 저장 : {out_png}")


def run_fullscreen_dimming(
    video_path: str | Path = VIDEO_PATH,
    output_dir: str | Path = OUTPUT_DIR,
    *,
    mode: str = DEFAULT_MODE,
    viewing: ViewingConfig | None = None,
    pse: PSEConfig | None = None,
    device: str | None = None,
    target_safe_lum_cd: float = TARGET_SAFE_LUM_CD,
    target_safe_red_excess: float = TARGET_SAFE_RED_EXCESS,
    fade_duration_sec: float = FADE_DURATION_SEC,
    half_hold_duration_sec: float = HALF_HOLD_DURATION_SEC,
    show_window: bool = False,
) -> dict[A1Mode, A1DimmingResult]:
    """Execute A1 dimming for `base`, `hold`, `safe`, or `all` modes and save `.mp4` + `.png`."""
    in_video = _validate_user_path(video_path, "입력 영상(VIDEO_PATH / --video)")
    out_folder = _validate_user_path(output_dir, "출력 폴더(OUTPUT_DIR / --output-dir)")
    if not in_video.exists():
        raise FileNotFoundError(f"입력 영상 파일을 찾을 수 없습니다: {in_video}")

    mode_clean = mode.strip().lower()
    if mode_clean == "all":
        target_modes: list[A1Mode] = list(VALID_A1_MODES)
    elif mode_clean in VALID_A1_MODES:
        target_modes = [mode_clean]  # type: ignore[list-item]
    else:
        raise ValueError(f"지원하지 않는 모드입니다: {mode!r} (선택 가능: base, hold, safe, all)")

    out_folder.mkdir(parents=True, exist_ok=True)
    view_cfg = viewing or ViewingConfig(profile=DEFAULT_PROFILE)
    pse_cfg = pse or PSEConfig()
    cache: DetectorCache = {}

    print("=" * 92)
    print(f"[LUMI A1 전체화면 밝기 보정 실행 (1112, vdcore 기반)]")
    print(f" - 입력 영상 : {in_video}")
    print(f" - 출력 폴더 : {out_folder}")
    print(f" - 실행 모드 : {', '.join(m.upper() for m in target_modes)}")
    print("=" * 92)

    # Run Pass 1 once and share across requested mode(s)
    t_p1 = time.perf_counter()
    pass1_res = analyze_video(
        in_video,
        viewing=view_cfg,
        pse=pse_cfg,
        device=device,
        collect_physical_stats=True,
        detector_cache=cache,
    )
    p1_elapsed = time.perf_counter() - t_p1
    cats_before = ", ".join(pass1_res.active_labels) if pass1_res.active_labels else "안전"
    print(
        f"[Pass 1 원본 진단 완료] {p1_elapsed:.2f}s | 위험 구간 {len(pass1_res.intervals)}개 ({cats_before})"
    )

    stem = in_video.stem
    results: dict[A1Mode, A1DimmingResult] = {}

    for m in target_modes:
        vid_suffix, png_suffix = _MODE_SUFFIXES[m]
        out_mp4 = out_folder / f"{stem}{vid_suffix}"
        out_png = out_folder / f"{stem}{png_suffix}"

        a1_cfg = A1Config(
            mode=m,
            target_safe_lum_cd=target_safe_lum_cd,
            target_safe_red_excess=target_safe_red_excess,
            fade_duration_sec=fade_duration_sec,
            half_hold_duration_sec=half_hold_duration_sec,
        )
        t_m = time.perf_counter()
        res_m = dim_video_fullscreen_a1(
            in_video,
            out_mp4,
            a1_config=a1_cfg,
            viewing=view_cfg,
            pse=pse_cfg,
            device=device,
            detector_cache=cache,
            pass1_result=pass1_res,
        )
        plot_and_save_a1_layout_a(res_m, out_png, show_window=show_window)
        elapsed_m = time.perf_counter() - t_m

        _print_mode_summary(res_m, out_png, elapsed_m)
        results[m] = res_m

    print("=" * 92)
    return results


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="LUMI A1 전체화면 밝기 보정 통합 실행 및 4단 비교 그래프 검증 스크립트 (1112)"
    )
    parser.add_argument(
        "--video",
        type=str,
        default=VIDEO_PATH,
        help=f"보정할 원본 동영상 파일 경로 (기본값: {PLACEHOLDER_PATH})",
    )
    parser.add_argument(
        "--output-dir",
        "--output_dir",
        dest="output_dir",
        type=str,
        default=OUTPUT_DIR,
        help=f"보정 영상(.mp4) 및 그래프(.png) 저장 폴더 경로 (기본값: {PLACEHOLDER_PATH})",
    )
    parser.add_argument(
        "--mode",
        choices=["base", "hold", "safe", "all"],
        default=DEFAULT_MODE,
        help="감광 스케줄링 모드 선택 (기본값: base, 전체 실행: all)",
    )
    parser.add_argument(
        "--profile",
        choices=["laptop", "desktop", "tv"],
        default=DEFAULT_PROFILE,
        help="시청 환경 프로파일 (기본값: laptop)",
    )
    parser.add_argument("--diagonal-inch", type=float, default=None, help="화면 대각선 크기 (inch)")
    parser.add_argument("--distance-cm", type=float, default=None, help="시청 거리 (cm)")
    parser.add_argument("--l-max", type=float, default=200.0, help="디스플레이 최대 휘도 (cd/m^2)")
    parser.add_argument("--max-width", type=int, default=640, help="탐지 분석 최대 가로 해상도 (기본값: 640)")
    parser.add_argument("--device", type=str, default=None, help="연산 장치 (cuda 또는 cpu)")
    parser.add_argument("--fade-sec", type=float, default=FADE_DURATION_SEC, help="코사인 전환 시간 (초, 기본값: 0.20)")
    parser.add_argument(
        "--half-hold-sec",
        type=float,
        default=HALF_HOLD_DURATION_SEC,
        help="절반 완화 유지 시간 (초, 기본값: 0.50)",
    )
    parser.add_argument(
        "--no-show",
        "--no_show",
        dest="no_show",
        action="store_true",
        help="그래프 창을 화면에 띄우지 않고 파일로만 저장",
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
    run_fullscreen_dimming(
        video_path=args.video,
        output_dir=args.output_dir,
        mode=args.mode,
        viewing=viewing,
        device=args.device,
        fade_duration_sec=args.fade_sec,
        half_hold_duration_sec=args.half_hold_sec,
        show_window=not args.no_show,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
