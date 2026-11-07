import os
import sys
import argparse
import pandas as pd

if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

from vdcoreops_v1 import (
    MAX_SIDE,
    evaluate_video_pse_risk,
)

# =========================================================================
# 기본 실행 설정 (단독 실행 시 아래 설정값이 사용됩니다)
# =========================================================================
VIDEO_PATH = r"영상 파일 위치"
DEFAULT_DEVICE = "laptop"     # 'laptop' (56.0 cm) 또는 'desktop' (68.0 cm)
DEFAULT_DISPLAY_AREA = 15.6     # 디스플레이 대각선 크기 (inch)


def _display_dataframe(df: pd.DataFrame, title: str | None = None) -> None:
    """터미널 및 주피터 환경 모두에서 데이터프레임이 잘리지 않고 깔끔하게 출력되도록 합니다."""
    if title:
        print(f"\n[{title}]")
    with pd.option_context(
        "display.max_rows", None,
        "display.max_columns", None,
        "display.width", 1000,
        "display.unicode.east_asian_width", True,
    ):
        print(df)


def run_single_video(
    video_path: str = VIDEO_PATH,
    device: str = DEFAULT_DEVICE,
    display_area: int = DEFAULT_DISPLAY_AREA,
    max_side: int = MAX_SIDE,
    show_spec: bool = True,
    verbose: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    단일 동영상을 단 1회 열어 시야각/CPD 스펙 데이터프레임과 위험 구간 요약 데이터프레임을 생성 및 출력합니다.
    """
    df_video_result, df_video_spec = evaluate_video_pse_risk(
        video_path=video_path,
        device=device,
        display_area=display_area,
        max_side=max_side,
        return_spec_df=True,
    )

    if verbose:
        if show_spec:
            _display_dataframe(df_video_spec, title=f"입력 영상 및 CPD 설정 정보 ({os.path.basename(video_path)})")
        _display_dataframe(df_video_result, title=f"위험 구간 요약 데이터프레임 ({os.path.basename(video_path)})")

    return df_video_result, df_video_spec


def main():
    parser = argparse.ArgumentParser(
        description="LUMI 동영상 광과민성 발작(PSE) 5대 위험 트리거 구간 판별 실행 스크립트"
    )
    parser.add_argument(
        "--video",
        type=str,
        default=VIDEO_PATH,
        help="분석할 동영상 파일 경로 (기본값: 스크립트 상단 VIDEO_PATH)",
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

    run_single_video(
        video_path=args.video,
        device=args.device,
        display_area=args.display_area,
        max_side=args.max_side,
        show_spec=True,
        verbose=True,
    )


if __name__ == "__main__":
    main()
