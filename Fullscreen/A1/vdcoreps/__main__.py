"""Command-line entry point (`python -m vdcoreps <video_path>`)."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Sequence

import numpy as np

from .config import PartialMaskConfig, ViewingConfig
from .pipeline import analyze_video


def _build_cli_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m vdcoreps",
        description="LUMI Partial-Screen PSE Detection & Mask Core Package (vdcoreps)",
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
    parser.add_argument(
        "--temporal-window-sec",
        type=float,
        default=0.16,
        help="Sliding temporal max-union half-window in seconds (default: 0.16)",
    )
    parser.add_argument(
        "--temporal-fade-sec",
        type=float,
        default=0.20,
        help="Temporal boundary cosine fade duration in seconds (default: 0.20)",
    )
    parser.add_argument(
        "--feather-deg",
        type=float,
        default=0.40,
        help="Outward spatial feathering width in visual angle degrees (default: 0.40)",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _build_cli_parser().parse_args(argv)
    viewing = ViewingConfig(
        profile=args.profile,
        diagonal_inch=args.diagonal_inch,
        viewing_distance_cm=args.distance_cm,
        l_max_cd_m2=args.l_max,
    )
    mask_cfg = PartialMaskConfig(
        temporal_window_sec=args.temporal_window_sec,
        temporal_fade_sec=args.temporal_fade_sec,
        feather_angle_deg=args.feather_deg,
    )
    res = analyze_video(
        args.video,
        viewing=viewing,
        mask_config=mask_cfg,
        device=args.device,
        collect_physical_stats=True,
        collect_masks=True,
    )
    danger_mask = res.frame_metrics_df["is_dangerous"].to_numpy(dtype=bool)
    if res.active_mask_area is not None and np.any(danger_mask):
        mean_active_pct = 100.0 * float(res.active_mask_area[danger_mask].mean())
        peak_active_pct = 100.0 * float(res.active_mask_area[danger_mask].max())
    else:
        mean_active_pct = 0.0
        peak_active_pct = 0.0

    print(
        f"[vdcoreps] {Path(res.video_path).name} | {res.orig_width}x{res.orig_height} -> "
        f"{res.analysis_width}x{res.analysis_height} @ {res.fps:.2f} fps | "
        f"frames={res.total_frames} | device={res.device} | "
        f"danger_mask_area(mean={mean_active_pct:.1f}%, peak={peak_active_pct:.1f}%)"
    )
    print(res.intervals_df.to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
