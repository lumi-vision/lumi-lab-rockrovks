"""Command-line entry point (`python -m vdcorefs <video_path>`)."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Sequence

from .config import ViewingConfig
from .pipeline import analyze_video


def _build_cli_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m vdcorefs",
        description="LUMI Fullscreen PSE Detection Core Package (vdcorefs)",
    )
    parser.add_argument("video", type=Path, help="Path to input video file")
    parser.add_argument(
        "--profile",
        choices=["laptop", "desktop", "tv"],
        default="laptop",
        help="Viewing geometry preset (default: laptop)",
    )
    parser.add_argument("--diagonal-inch", type=float, default=None, help="Screen diagonal in inches")
    parser.add_argument("--distance-cm", type=float, default=None, help="Viewing distance in cm")
    parser.add_argument("--l-max", type=float, default=200.0, help="Peak display white in cd/m^2")
    parser.add_argument("--device", type=str, default=None, help="Compute device (cuda or cpu)")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _build_cli_parser().parse_args(argv)
    viewing = ViewingConfig(
        profile=args.profile,
        diagonal_inch=args.diagonal_inch,
        viewing_distance_cm=args.distance_cm,
        l_max_cd_m2=args.l_max,
    )
    res = analyze_video(args.video, viewing=viewing, device=args.device)
    print(
        f"[vdcorefs] {Path(res.video_path).name} | {res.orig_width}x{res.orig_height} -> "
        f"{res.analysis_width}x{res.analysis_height} @ {res.fps:.2f} fps | "
        f"frames={res.total_frames} | device={res.device}"
    )
    print(res.intervals_df.to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
