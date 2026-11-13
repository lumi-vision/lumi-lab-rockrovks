"""Pixel-level temporal luminance & saturated-red hazard localization for `vdcoreps`.

Implements specification `(1-a)`:
  1. Tracks the exact committed polarity pixel mask (`union_pos` or `union_neg`) for every
     `TransitionEvent` committed by `vdcorefs`'s opposing-transition state machine while
     preserving 100% identical scalar/event state transitions.
  2. Tracks per-pixel `1.0s` sliding opposing reversal counts for both luminance (via
     `PixelDiagnosticTracker`) and saturated red (`RedPixelReversalTracker`), filtering out
     single-step moving edges (`reversals_1s < flash_min_reversals_1s`) while guaranteeing
     residual unmasked transition area stays safely below the `25%` ITU-R threshold.
"""

from __future__ import annotations

from collections import deque
from typing import Sequence

import numpy as np
import torch

from vdcorefs.config import FPSWindowSpec, PSEConfig, TransitionEvent
from vdcorefs.temporal import (
    TemporalTransitionEngine,
    _AlternatingChannelState,
    _SyncMaskItem,
    build_flash_boolean_mask,
    build_iris_extended_mask,
    merge_ultrafast_transitions,
    trailing_1s_event_stats,
)

_POPCOUNT_LUT = np.array([bin(i).count("1") for i in range(256)], dtype=np.int32)


def _pack_tensor_mask_2d(mask_2d: torch.Tensor) -> np.ndarray:
    arr = mask_2d.detach().to(dtype=torch.uint8, device="cpu").numpy()
    return np.packbits(arr, axis=-1)


def _packed_true_fraction(packed_2d: np.ndarray, total_pixels: int) -> float:
    if total_pixels <= 0 or packed_2d.size == 0:
        return 0.0
    return float(_POPCOUNT_LUT[packed_2d].sum()) / float(total_pixels)


class _MaskTrackingChannelState(_AlternatingChannelState):
    """Exact `_AlternatingChannelState` recording bit-packed masks for committed events."""

    def __init__(
        self,
        area_threshold: float,
        inclusive: bool,
        sync_window_ms: float,
    ) -> None:
        super().__init__(
            area_threshold=area_threshold,
            inclusive=inclusive,
            sync_window_ms=sync_window_ms,
        )
        self.event_packed_masks: list[np.ndarray] = []

    def step_masks(
        self,
        frame_idx: int,
        t_ms: float,
        best_pos_ref: tuple[int, float, torch.Tensor, float] | None,
        best_neg_ref: tuple[int, float, torch.Tensor, float] | None,
    ) -> tuple[float, float]:
        while self._sync_buf and (
            (t_ms - self._sync_buf[0].t_ms > self.sync_window_ms + 1e-3)
            or (self._sync_buf[0].frame_idx <= self.last_commit_idx)
        ):
            self._sync_buf.popleft()

        pos_mask = best_pos_ref[2] if best_pos_ref is not None else None
        neg_mask = best_neg_ref[2] if best_neg_ref is not None else None
        if pos_mask is None and neg_mask is None:
            return 0.0, 0.0

        zero_mask = torch.zeros_like(pos_mask if pos_mask is not None else neg_mask)
        cur_pos = pos_mask if pos_mask is not None else zero_mask
        cur_neg = neg_mask if neg_mask is not None else zero_mask
        pos_mag = best_pos_ref[3] if best_pos_ref is not None else 0.0
        neg_mag = best_neg_ref[3] if best_neg_ref is not None else 0.0

        self._sync_buf.append(
            _SyncMaskItem(
                frame_idx=frame_idx,
                t_ms=t_ms,
                pos_mask=cur_pos,
                neg_mask=cur_neg,
                pos_mag=pos_mag,
                neg_mag=neg_mag,
            )
        )

        union_pos = cur_pos
        union_neg = cur_neg
        if len(self._sync_buf) > 1:
            union_pos = cur_pos.clone()
            union_neg = cur_neg.clone()
            for item in self._sync_buf:
                union_pos |= item.pos_mask
                union_neg |= item.neg_mask

        areas = torch.stack([union_pos.float().mean(), union_neg.float().mean()]).tolist()
        area_pos = float(areas[0])
        area_neg = float(areas[1])

        pos_ok = self._passes(area_pos) and (best_pos_ref is not None)
        neg_ok = self._passes(area_neg) and (best_neg_ref is not None)

        chosen_pol = 0
        if self.last_polarity == 0:
            if pos_ok and neg_ok:
                chosen_pol = 1 if area_pos >= area_neg else -1
            elif pos_ok:
                chosen_pol = 1
            elif neg_ok:
                chosen_pol = -1
        elif self.last_polarity == 1 and neg_ok:
            chosen_pol = -1
        elif self.last_polarity == -1 and pos_ok:
            chosen_pol = 1

        if chosen_pol == 1 and best_pos_ref is not None:
            ref_idx, ref_t_ms, _, mag = best_pos_ref
            self.events.append(
                TransitionEvent(
                    ref_frame=ref_idx,
                    frame_idx=frame_idx,
                    t_ref_ms=ref_t_ms,
                    t_ms=t_ms,
                    polarity=1,
                    area=area_pos,
                    mean_magnitude=max(mag, max((x.pos_mag for x in self._sync_buf), default=mag)),
                )
            )
            self.event_packed_masks.append(_pack_tensor_mask_2d(union_pos))
            self.last_polarity = 1
            self.last_commit_idx = frame_idx
            self._sync_buf.clear()
        elif chosen_pol == -1 and best_neg_ref is not None:
            ref_idx, ref_t_ms, _, mag = best_neg_ref
            self.events.append(
                TransitionEvent(
                    ref_frame=ref_idx,
                    frame_idx=frame_idx,
                    t_ref_ms=ref_t_ms,
                    t_ms=t_ms,
                    polarity=-1,
                    area=area_neg,
                    mean_magnitude=max(mag, max((x.neg_mag for x in self._sync_buf), default=mag)),
                )
            )
            self.event_packed_masks.append(_pack_tensor_mask_2d(union_neg))
            self.last_polarity = -1
            self.last_commit_idx = frame_idx
            self._sync_buf.clear()

        return area_pos, area_neg


class RedPixelReversalTracker:
    """Per-pixel `1.0s` opposing saturated-red transition counter (`Delta u'v' > 0.20`)."""

    def __init__(
        self,
        height: int,
        width: int,
        w_1s: int,
        pse: PSEConfig,
        device: torch.device,
    ) -> None:
        self.w_1s = max(1, w_1s)
        self.pse = pse
        self.device = device
        self._u_anchor: torch.Tensor | None = None
        self._v_anchor: torch.Tensor | None = None
        self._red_anchor: torch.Tensor | None = None
        self._pol = torch.zeros((height, width), dtype=torch.int8, device=device)
        self.running_counts = torch.zeros((height, width), dtype=torch.int32, device=device)
        self._ring: deque[torch.Tensor] = deque()

    def update(
        self,
        u_prime: torch.Tensor,
        v_prime: torch.Tensor,
        is_sat_red: torch.Tensor,
    ) -> torch.Tensor:
        if self._u_anchor is None or self._v_anchor is None or self._red_anchor is None:
            self._u_anchor = u_prime.clone()
            self._v_anchor = v_prime.clone()
            self._red_anchor = is_sat_red.clone()
            ev_mask = torch.zeros_like(is_sat_red, dtype=torch.int32)
        else:
            du = u_prime - self._u_anchor
            dv = v_prime - self._v_anchor
            delta_uv = torch.hypot(du, dv)
            either_red = is_sat_red | self._red_anchor
            valid_red = either_red & (delta_uv > self.pse.delta_uv_min)

            pos_hit = valid_red & (du > 0.0) & (self._pol <= 0)
            neg_hit = valid_red & (du < 0.0) & (self._pol >= 0)
            ev_bool = pos_hit | neg_hit

            more_red = (self._pol == 1) & is_sat_red & (u_prime > self._u_anchor)
            less_red = (self._pol == -1) & (~is_sat_red) & (u_prime < self._u_anchor)
            upd = ev_bool | more_red | less_red

            self._u_anchor = torch.where(upd, u_prime, self._u_anchor)
            self._v_anchor = torch.where(upd, v_prime, self._v_anchor)
            self._red_anchor = torch.where(upd, is_sat_red, self._red_anchor)
            self._pol = torch.where(
                pos_hit,
                torch.ones_like(self._pol),
                torch.where(neg_hit, -torch.ones_like(self._pol), self._pol),
            )
            ev_mask = ev_bool.to(torch.int32)

        self.running_counts += ev_mask
        self._ring.append(ev_mask)
        if len(self._ring) > self.w_1s:
            self.running_counts -= self._ring.popleft()
        return self.running_counts


class PartialTemporalTransitionEngine(TemporalTransitionEngine):
    """Extends `TemporalTransitionEngine` with exact committed-event pixel mask tracking."""

    def __init__(self, fps_spec: FPSWindowSpec, pse: PSEConfig = PSEConfig()) -> None:
        super().__init__(fps_spec=fps_spec, pse=pse)
        self.lum_flash_state = _MaskTrackingChannelState(
            area_threshold=pse.flash_area_min,
            inclusive=False,
            sync_window_ms=pse.sync_window_ms,
        )
        self.scene_cut_state = _MaskTrackingChannelState(
            area_threshold=pse.scene_area_min,
            inclusive=True,
            sync_window_ms=pse.sync_window_ms,
        )
        self.red_flash_state = _MaskTrackingChannelState(
            area_threshold=pse.flash_area_min,
            inclusive=False,
            sync_window_ms=pse.sync_window_ms,
        )


def merge_ultrafast_transitions_with_masks(
    events: Sequence[TransitionEvent],
    packed_masks: Sequence[np.ndarray],
    merge_ms: float = 15.0,
) -> tuple[list[TransitionEvent], list[np.ndarray]]:
    """Collapse `>=65 Hz` (`<= 15 ms`) bursts identically to `merge_ultrafast_transitions` with masks."""
    n = len(events)
    if n < 3:
        return list(events), [m.copy() for m in packed_masks]

    merged_ev: list[TransitionEvent] = []
    merged_masks: list[np.ndarray] = []
    i = 0
    limit_ms = merge_ms + 1e-3
    while i < n:
        if (i + 2 < n) and ((events[i + 2].t_ms - events[i].t_ms) <= limit_ms):
            p_start = events[i].polarity
            j = i + 2
            while (j + 1 < n) and ((events[j + 1].t_ms - events[j - 1].t_ms) <= limit_ms):
                j += 1
            burst_union = packed_masks[i].copy()
            for k in range(i + 1, j + 1):
                np.bitwise_or(burst_union, packed_masks[k], out=burst_union)
            if events[j].polarity == -p_start:
                merged_ev.append(events[i])
                merged_masks.append(burst_union)
                merged_ev.append(events[j])
                merged_masks.append(burst_union.copy())
            else:
                merged_ev.append(events[i])
                merged_masks.append(burst_union)
            i = j + 1
        else:
            merged_ev.append(events[i])
            merged_masks.append(packed_masks[i].copy())
            i += 1
    return merged_ev, merged_masks


def forward_window_or_packed(packed_3d: np.ndarray, window_len: int) -> np.ndarray:
    """Compute exact forward sliding bitwise OR over `[t, min(T - 1, t + window_len - 1)]` in O(T)."""
    n_frames = packed_3d.shape[0]
    win = max(1, int(window_len))
    if n_frames <= 1 or win <= 1:
        return packed_3d.copy()

    fwd = np.empty_like(packed_3d)
    bwd = np.empty_like(packed_3d)

    for i in range(n_frames):
        if i % win == 0:
            fwd[i] = packed_3d[i]
        else:
            np.bitwise_or(fwd[i - 1], packed_3d[i], out=fwd[i])

    for i in range(n_frames - 1, -1, -1):
        if (i + 1) % win == 0 or i == n_frames - 1:
            bwd[i] = packed_3d[i]
        else:
            np.bitwise_or(bwd[i + 1], packed_3d[i], out=bwd[i])

    out = np.empty_like(packed_3d)
    for t in range(n_frames):
        lo = t
        hi = (n_frames - 1) if (t + win - 1 >= n_frames) else (t + win - 1)
        if (lo // win) < (hi // win):
            np.bitwise_or(bwd[lo], fwd[hi], out=out[t])
        elif lo % win == 0:
            out[t] = fwd[hi]
        else:
            out[t] = bwd[lo]
    return out


def build_committed_flash_core_packed(
    total_frames: int,
    ana_height: int,
    ana_width: int,
    events: Sequence[TransitionEvent],
    event_masks: Sequence[np.ndarray],
    frame_to_pos: dict[int, int],
    osc_primary_packed: np.ndarray,
    osc_fallback_packed: np.ndarray,
    residual_area_cap: float = 0.18,
) -> np.ndarray:
    """Build `(T, H, ceil(W/8))` flash core mask strictly from committed events & oscillating pixels.

    For each committed `TransitionEvent`:
      1. Intersects `ev_mask` with `osc_primary_packed` (`>= 4` reversals/s) across `[ref_frame, frame_idx]`.
      2. Verifies residual unmasked transition area `area(ev_mask & ~selected) <= residual_area_cap`
         (`< 0.25` ITU-R threshold); relaxes to `osc_fallback_packed` (`>= 2` reversals/s) or `ev_mask`
         only on fast-moving camera events where single-pixel stationarity breaks down.
    """
    packed_w = (ana_width + 7) // 8
    out = np.zeros((total_frames, ana_height, packed_w), dtype=np.uint8)
    if total_frames == 0 or not events:
        return out

    total_px = ana_height * ana_width
    for ev, ev_mask in zip(events, event_masks):
        pos_cur = frame_to_pos.get(ev.frame_idx)
        if pos_cur is None:
            continue
        pos_ref = frame_to_pos.get(ev.ref_frame, pos_cur)
        lo = max(0, min(pos_ref, pos_cur))
        hi = min(total_frames - 1, max(pos_ref, pos_cur))

        osc_p = np.bitwise_or(osc_primary_packed[lo], osc_primary_packed[hi])
        cand_p = np.bitwise_and(ev_mask, osc_p)
        rem_p = np.bitwise_and(ev_mask, np.bitwise_not(cand_p))

        if _packed_true_fraction(rem_p, total_px) <= residual_area_cap and np.any(cand_p):
            chosen = cand_p
        else:
            osc_f = np.bitwise_or(osc_fallback_packed[lo], osc_fallback_packed[hi])
            cand_f = np.bitwise_and(ev_mask, osc_f)
            rem_f = np.bitwise_and(ev_mask, np.bitwise_not(cand_f))
            if _packed_true_fraction(rem_f, total_px) <= residual_area_cap and np.any(cand_f):
                chosen = cand_f
            else:
                chosen = ev_mask

        for p_idx in range(lo, hi + 1):
            np.bitwise_or(out[p_idx], chosen, out=out[p_idx])

    return out


__all__ = [
    "PartialTemporalTransitionEngine",
    "RedPixelReversalTracker",
    "build_committed_flash_core_packed",
    "build_flash_boolean_mask",
    "build_iris_extended_mask",
    "forward_window_or_packed",
    "merge_ultrafast_transitions",
    "merge_ultrafast_transitions_with_masks",
    "trailing_1s_event_stats",
]
