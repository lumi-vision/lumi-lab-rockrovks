import os
import sys
import argparse
import numpy as np
import matplotlib.pyplot as plt

if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

# Windows 한글 폰트 및 마이너스 기호 깨짐 방지 설정
plt.rcParams["font.family"] = "Malgun Gothic"
plt.rcParams["axes.unicode_minus"] = False

_CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
if _CURRENT_DIR not in sys.path:
    sys.path.insert(0, _CURRENT_DIR)

from vdcoreops_v2 import MAX_SIDE
from fullscreen_coreops_hold_1108 import (
    FADE_DURATION_SEC,
    HALF_HOLD_DURATION_SEC,
    TARGET_SAFE_LUM_CD,
    dim_video_fullscreen_hold_method_a,
)

"""경로 입력 필요! 주의!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!"""
# =========================================================================
# 기본 실행 설정 (단독 실행 시 아래 설정값이 사용됩니다)
# =========================================================================
VIDEO_PATH = r"경로를 입력하세요"
OUTPUT_DIR = r"경로를 입력하세요"
DEFAULT_DEVICE = "laptop"     # 'laptop' (56.0 cm) 또는 'desktop' (68.0 cm)
DEFAULT_DISPLAY_AREA = 15     # 디스플레이 대각선 크기 (inch)


def plot_and_save_comparison_graph_c(
    result: dict,
    video_name: str,
    save_png_path: str,
    show_window: bool = True,
) -> None:
    """
    [구성 C: 통합 3단 그래프 생성, 파일 저장 및 창 띄우기 — 절반 완화(Half-Hold) 버전]
    - 1단: 보정 전 vs 보정 후 물리 휘도(cd/m^2, 최대 휘도 및 평균 휘도) 비교 + 위험 구간 배경 음영
    - 2단: 보정 전 vs 보정 후 프레임별 최대 위험도(판정 근거 - 임계치 달성 정도 %) 비교
    - 3단: 적용된 전체 화면 밝기 조절 배율(선형 물리 휘도 배율 alpha_t 및 체감 sRGB 배율 — 절반 완화 Hold 포함) 곡선
    """
    fps = float(result["meta"]["fps"])
    orig_stats = result["orig_frame_stats"]
    dim_stats = result["dimmed_frame_stats"]
    n_frames = len(orig_stats)
    time_sec = np.arange(n_frames, dtype=np.float32) / fps

    orig_max_lum = np.array([s["max_lum"] for s in orig_stats], dtype=np.float32)
    orig_mean_lum = np.array([s["mean_lum"] for s in orig_stats], dtype=np.float32)
    dim_max_lum = np.array([s["max_lum"] for s in dim_stats], dtype=np.float32)
    dim_mean_lum = np.array([s["mean_lum"] for s in dim_stats], dtype=np.float32)

    orig_prob1 = result["orig_max_prob1"]
    dim_prob1 = result["dimmed_max_prob1"]
    alpha_lin = result["alpha_schedule"]
    # sRGB 픽셀 기준 체감 밝기 배율 (gamma 2.4 역변환 근사)
    alpha_srgb = np.power(np.clip(alpha_lin, 0.0, 1.0), 1.0 / 2.4)

    all_orig_intervals = result["all_orig_intervals"]
    num_before = len(result["df_before"])
    num_after = len(result["df_after"])

    fig, axes = plt.subplots(3, 1, figsize=(13, 10), sharex=True)
    fig.suptitle(
        f"LUMI 전체 화면 밝기 보정(방식 A + 선택 1 + 절반 완화 Hold) 전·후 비교 — {video_name}\n"
        f"(보정 전 위험 구간: {num_before}개  →  보정 후 위험 구간: {num_after}개)",
        fontsize=14,
        fontweight="bold",
        y=0.98,
    )

    # 공통: 탐지된 원본 위험 구간 배경 음영 표시
    for ax_idx, ax in enumerate(axes):
        for idx_iv, (s_f, e_f, trig_name) in enumerate(all_orig_intervals):
            lbl = "원본 위험 탐지 구간" if (ax_idx == 0 and idx_iv == 0) else None
            ax.axvspan(s_f / fps, e_f / fps, color="#ffcccc", alpha=0.35, label=lbl)
        ax.grid(True, linestyle="--", alpha=0.4)

    # -------------------------------------------------------------------------
    # [1단] 보정 전 vs 보정 후 물리 휘도 (cd/m^2)
    # -------------------------------------------------------------------------
    ax1 = axes[0]
    ax1.plot(time_sec, orig_max_lum, color="#d62728", linewidth=1.4, alpha=0.85, label="보정 전 최대 물리 휘도 (cd/m²)")
    ax1.plot(time_sec, orig_mean_lum, color="#ff7f0e", linewidth=1.1, linestyle=":", alpha=0.85, label="보정 전 평균 물리 휘도 (cd/m²)")
    ax1.plot(time_sec, dim_max_lum, color="#1f77b4", linewidth=1.6, label="보정 후 최대 물리 휘도 (cd/m²)")
    ax1.plot(time_sec, dim_mean_lum, color="#17becf", linewidth=1.2, linestyle="--", label="보정 후 평균 물리 휘도 (cd/m²)")
    ax1.axhline(20.0, color="#2ca02c", linestyle="--", linewidth=1.2, label="국제 표준 안전 역치 (20 cd/m²)")
    ax1.axhline(TARGET_SAFE_LUM_CD, color="#2ca02c", linestyle=":", linewidth=1.0, alpha=0.7, label=f"보정 목표 상한 ({TARGET_SAFE_LUM_CD:.0f} cd/m²)")
    ax1.set_ylabel("물리 휘도 (cd/m²)", fontsize=11, fontweight="bold")
    ax1.set_title("① 보정 전 vs 보정 후 프레임별 물리 휘도 변화", fontsize=11, fontweight="bold", loc="left")
    ax1.set_ylim(bottom=0.0)
    ax1.legend(loc="upper right", fontsize=9, ncol=2)

    # -------------------------------------------------------------------------
    # [2단] 보정 전 vs 보정 후 프레임별 최대 위험도 (prob1, %)
    # -------------------------------------------------------------------------
    ax2 = axes[1]
    ax2.plot(time_sec, orig_prob1, color="#d62728", linewidth=1.5, alpha=0.85, label="보정 전 프레임 최대 위험도 (prob1 %)")
    ax2.plot(time_sec, dim_prob1, color="#1f77b4", linewidth=1.8, label="보정 후 프레임 최대 위험도 (prob1 %)")
    ax2.axhline(50.0, color="#7f7f7f", linestyle="--", linewidth=1.0, label="단일 프레임 임계점 (50%)")
    ax2.set_ylabel("위험도 달성 정도 (%)", fontsize=11, fontweight="bold")
    ax2.set_title("② 보정 전 vs 보정 후 프레임별 최대 위험도 (판정 근거 - 임계치 달성 정도 %)", fontsize=11, fontweight="bold", loc="left")
    ax2.set_ylim(-3.0, 105.0)
    ax2.legend(loc="upper right", fontsize=9)

    # -------------------------------------------------------------------------
    # [3단] 적용된 화면 밝기 조절 배율 (alpha_t 곡선, 절반 완화 Hold 후 완전 해제 포함)
    # -------------------------------------------------------------------------
    ax3 = axes[2]
    ax3.plot(time_sec, alpha_lin * 100.0, color="#2ca02c", linewidth=1.8, label="선형 물리 휘도 배율 (alpha_t, %)")
    ax3.plot(time_sec, alpha_srgb * 100.0, color="#9467bd", linewidth=1.5, linestyle="--", label="체감 sRGB 픽셀 밝기 배율 (alpha_t^(1/2.4), %)")
    ax3.set_xlabel("시간 (초, sec)", fontsize=11, fontweight="bold")
    ax3.set_ylabel("밝기 유지 배율 (%)", fontsize=11, fontweight="bold")
    ax3.set_title("③ 시간에 따라 적용된 전체 화면 밝기 조절 배율 (절반 완화 Hold 후 완전 해제 포함)", fontsize=11, fontweight="bold", loc="left")
    ax3.set_ylim(-3.0, 105.0)
    ax3.legend(loc="lower right", fontsize=9)

    plt.tight_layout(rect=[0, 0, 1, 0.94])
    os.makedirs(os.path.dirname(os.path.abspath(save_png_path)), exist_ok=True)
    fig.savefig(save_png_path, dpi=150)
    print(f" - [그래프 이미지 저장 완료] {save_png_path}")

    if show_window:
        try:
            plt.show()
        except Exception:
            pass
    plt.close(fig)


def run_fullscreen_hold_dimming(
    video_path: str = VIDEO_PATH,
    output_dir: str = OUTPUT_DIR,
    device: str = DEFAULT_DEVICE,
    display_area: int = DEFAULT_DISPLAY_AREA,
    max_side: int = MAX_SIDE,
    fade_duration_sec: float = FADE_DURATION_SEC,
    half_hold_duration_sec: float = HALF_HOLD_DURATION_SEC,
    show_window: bool = True,
) -> dict:
    """
    단일 동영상을 입력받아 방식 A + 선택 1 + 절반 완화(Half-Hold) 스무딩으로 보정된 동영상(.mp4)과
    구성 C 검증 그래프(.png)를 Output 폴더에 저장하고 창을 띄웁니다.
    """
    if not output_dir or output_dir == r"경로를 입력하세요":
        output_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "Output")

    os.makedirs(output_dir, exist_ok=True)
    stem = os.path.splitext(os.path.basename(video_path))[0]
    out_video_path = os.path.join(output_dir, f"{stem}_fullscreen_hold_dimmed.mp4")
    out_png_path = os.path.join(output_dir, f"{stem}_fullscreen_hold_graph.png")

    print("=" * 90)
    print(f"[LUMI 전체 화면 밝기 보정 실행 (방식 A + 선택 1 + 절반 완화 Hold 스무딩)]")
    print(f" - 입력 영상 : {video_path}")
    print(f" - 출력 폴더 : {output_dir}")
    print("=" * 90)

    result = dim_video_fullscreen_hold_method_a(
        input_video_path=video_path,
        output_video_path=out_video_path,
        device=device,
        display_area=display_area,
        max_side=max_side,
        fade_duration_sec=fade_duration_sec,
        half_hold_duration_sec=half_hold_duration_sec,
    )

    meta = result["meta"]
    alpha_sched = result["alpha_schedule"]
    print(f" - 분석/저장 해상도 : {meta['proc_w']} × {meta['proc_h']} ({meta['fps']:.2f} fps, 총 {meta['total_frames']} 프레임)")
    print(f" - 보정 전 위험 구간 : {len(result['df_before'])}개  →  보정 후 위험 구간 : {len(result['df_after'])}개")
    print(f" - 최소 선형 휘도 배율 : {float(np.min(alpha_sched))*100:.2f}% (체감 sRGB 배율: {float(np.min(alpha_sched)**(1/2.4))*100:.2f}%)")
    print(f" - [보정 영상 저장 완료] {out_video_path}")

    plot_and_save_comparison_graph_c(
        result=result,
        video_name=os.path.basename(video_path),
        save_png_path=out_png_path,
        show_window=show_window,
    )
    print("=" * 90)
    return result


def main():
    parser = argparse.ArgumentParser(
        description="LUMI 위험 구간 전체 화면 밝기 보정(절반 완화 Hold 버전) 및 구성 C 그래프 검증 스크립트 (1108)"
    )
    parser.add_argument(
        "--video",
        type=str,
        default=VIDEO_PATH,
        help="보정할 원본 동영상 파일 경로 (기본값: 스크립트 상단 VIDEO_PATH)",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=OUTPUT_DIR,
        help="보정된 영상(.mp4)과 그래프(.png)를 저장할 폴더 경로",
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
        help="최대 해상도 변 길이 (초과 시 축소, 기본값: 720)",
    )
    parser.add_argument(
        "--half_hold_sec",
        type=float,
        default=HALF_HOLD_DURATION_SEC,
        help="위험 구간 종료 후 절반 완화(alpha_half) 상태로 유지하는 시간 (초, 기본값: 0.50)",
    )
    parser.add_argument(
        "--no_show",
        action="store_true",
        help="그래프 창(plt.show)을 띄우지 않고 파일로만 저장할 때 사용",
    )
    args, _ = parser.parse_known_args()

    run_fullscreen_hold_dimming(
        video_path=args.video,
        output_dir=args.output_dir,
        device=args.device,
        display_area=args.display_area,
        max_side=args.max_side,
        half_hold_duration_sec=args.half_hold_sec,
        show_window=not args.no_show,
    )


if __name__ == "__main__":
    main()
