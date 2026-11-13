"""Temporal opposing-transition state machine, HFR provisions, and flash/extended masks."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Sequence

import numpy as np
import torch

from .config import FPSWindowSpec, PSEConfig, TransitionEvent


@dataclass(slots=True)
class _SyncMaskItem:
    frame_idx: int
    t_ms: float
    pos_mask: torch.Tensor
    neg_mask: torch.Tensor
    pos_mag: float
    neg_mag: float


class _AlternatingChannelState:
    """Tracks opposing (+1 <-> -1) transitions for a single area threshold."""

    def __init__(
        self,
        area_threshold: float,
        inclusive: bool,
        sync_window_ms: float,
    ) -> None:
        self.area_threshold = area_threshold
        self.inclusive = inclusive
        self.sync_window_ms = sync_window_ms
        self.last_polarity: int = 0
        self.last_commit_idx: int = -1
        self.events: list[TransitionEvent] = []
        self._sync_buf: deque[_SyncMaskItem] = deque()

    def _passes(self, area: float) -> bool:
        return (area >= self.area_threshold) if self.inclusive else (area > self.area_threshold)

    def step_masks(
        self,
        frame_idx: int,
        t_ms: float,
        best_pos_ref: tuple[int, float, torch.Tensor, float] | None,
        best_neg_ref: tuple[int, float, torch.Tensor, float] | None,
    ) -> tuple[float, float]:
        """Evaluate synchronized +/- areas since `last_commit_idx` and commit opposing events."""
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
            self.last_polarity = -1
            self.last_commit_idx = frame_idx
            self._sync_buf.clear()

        return area_pos, area_neg


@dataclass(slots=True)
class _FrameRefItem:
    frame_idx: int
    t_ms: float
    lum: torch.Tensor
    u_prime: torch.Tensor
    v_prime: torch.Tensor
    is_sat_red: torch.Tensor


class TemporalTransitionEngine:
    """Multi-step (<=66.8 ms), sync-aware (<=20 ms) luminance & red transition tracker."""

    def __init__(self, fps_spec: FPSWindowSpec, pse: PSEConfig = PSEConfig()) -> None:
        self.fps_spec = fps_spec
        self.pse = pse
        self._refs: deque[_FrameRefItem] = deque(maxlen=max(2, fps_spec.k_trans + 2))

        self.lum_flash_state = _AlternatingChannelState(
            area_threshold=pse.flash_area_min,
            inclusive=False,
            sync_window_ms=pse.sync_window_ms,
        )
        self.scene_cut_state = _AlternatingChannelState(
            area_threshold=pse.scene_area_min,
            inclusive=True,
            sync_window_ms=pse.sync_window_ms,
        )
        self.red_flash_state = _AlternatingChannelState(
            area_threshold=pse.flash_area_min,
            inclusive=False,
            sync_window_ms=pse.sync_window_ms,
        )

    def _best_lum_refs_after(
        self,
        min_ref_idx: int,
        t_ms: float,
        lum: torch.Tensor,
    ) -> tuple[
        tuple[int, float, torch.Tensor, float] | None,
        tuple[int, float, torch.Tensor, float] | None,
    ]:
        best_pos: tuple[int, float, torch.Tensor, float] | None = None
        best_neg: tuple[int, float, torch.Tensor, float] | None = None
        best_pos_area = 0.0
        best_neg_area = 0.0

        candidates = [
            r
            for r in self._refs
            if (r.frame_idx >= min_ref_idx)
            and (0.0 < (t_ms - r.t_ms) <= self.pse.max_transition_ms + 1e-3)
        ]
        if not candidates:
            return None, None

        for ref in candidates:
            delta_l = lum - ref.lum
            abs_dl = delta_l.abs()
            l_low = torch.minimum(lum, ref.lum)
            l_sum = torch.clamp(lum + ref.lum, min=1e-6)

            valid_amp = torch.where(
                l_low < self.pse.dark_lum_limit_cd_m2,
                abs_dl >= self.pse.delta_lum_min_cd_m2,
                (abs_dl / l_sum) > self.pse.bright_michelson_min,
            )
            m_pos = valid_amp & (delta_l > 0.0)
            m_neg = valid_amp & (delta_l < 0.0)

            stats = torch.stack(
                [
                    m_pos.float().mean(),
                    m_neg.float().mean(),
                    (abs_dl * m_pos).sum() / torch.clamp(m_pos.float().sum(), min=1.0),
                    (abs_dl * m_neg).sum() / torch.clamp(m_neg.float().sum(), min=1.0),
                ]
            ).tolist()
            a_pos, a_neg, mag_pos, mag_neg = (float(s) for s in stats)

            if a_pos > best_pos_area:
                best_pos_area = a_pos
                best_pos = (ref.frame_idx, ref.t_ms, m_pos, mag_pos)
            if a_neg > best_neg_area:
                best_neg_area = a_neg
                best_neg = (ref.frame_idx, ref.t_ms, m_neg, mag_neg)

        return best_pos, best_neg

    def _best_red_refs_after(
        self,
        min_ref_idx: int,
        t_ms: float,
        u_prime: torch.Tensor,
        v_prime: torch.Tensor,
        is_sat_red: torch.Tensor,
    ) -> tuple[
        tuple[int, float, torch.Tensor, float] | None,
        tuple[int, float, torch.Tensor, float] | None,
    ]:
        best_pos: tuple[int, float, torch.Tensor, float] | None = None
        best_neg: tuple[int, float, torch.Tensor, float] | None = None
        best_pos_area = 0.0
        best_neg_area = 0.0

        candidates = [
            r
            for r in self._refs
            if (r.frame_idx >= min_ref_idx)
            and (0.0 < (t_ms - r.t_ms) <= self.pse.max_transition_ms + 1e-3)
        ]
        if not candidates:
            return None, None

        for ref in candidates:
            du = u_prime - ref.u_prime
            dv = v_prime - ref.v_prime
            delta_uv = torch.hypot(du, dv)
            either_red = is_sat_red | ref.is_sat_red
            valid_red = either_red & (delta_uv > self.pse.delta_uv_min)

            m_to_red = valid_red & (du > 0.0)
            m_from_red = valid_red & (du < 0.0)

            stats = torch.stack(
                [
                    m_to_red.float().mean(),
                    m_from_red.float().mean(),
                    (delta_uv * m_to_red).sum() / torch.clamp(m_to_red.float().sum(), min=1.0),
                    (delta_uv * m_from_red).sum() / torch.clamp(m_from_red.float().sum(), min=1.0),
                ]
            ).tolist()
            a_pos, a_neg, mag_pos, mag_neg = (float(s) for s in stats)

            if a_pos > best_pos_area:
                best_pos_area = a_pos
                best_pos = (ref.frame_idx, ref.t_ms, m_to_red, mag_pos)
            if a_neg > best_neg_area:
                best_neg_area = a_neg
                best_neg = (ref.frame_idx, ref.t_ms, m_from_red, mag_neg)

        return best_pos, best_neg

    def update(
        self,
        frame_idx: int,
        t_ms: float,
        lum: torch.Tensor,
        u_prime: torch.Tensor,
        v_prime: torch.Tensor,
        is_sat_red: torch.Tensor,
    ) -> tuple[float, float]:
        """Process one frame and return `(max_lum_trans_area, max_red_trans_area)`."""
        while self._refs and (t_ms - self._refs[0].t_ms > self.pse.max_transition_ms + 1e-3):
            self._refs.popleft()

        lum_area = 0.0
        red_area = 0.0
        if self._refs:
            lum_pos_25, lum_neg_25 = self._best_lum_refs_after(
                max(0, self.lum_flash_state.last_commit_idx), t_ms, lum
            )
            a_lp, a_ln = self.lum_flash_state.step_masks(frame_idx, t_ms, lum_pos_25, lum_neg_25)
            lum_area = max(a_lp, a_ln)

            if self.scene_cut_state.last_commit_idx == self.lum_flash_state.last_commit_idx:
                lum_pos_80, lum_neg_80 = lum_pos_25, lum_neg_25
            else:
                lum_pos_80, lum_neg_80 = self._best_lum_refs_after(
                    max(0, self.scene_cut_state.last_commit_idx), t_ms, lum
                )
            a_sp, a_sn = self.scene_cut_state.step_masks(frame_idx, t_ms, lum_pos_80, lum_neg_80)
            lum_area = max(lum_area, a_sp, a_sn)

            red_pos, red_neg = self._best_red_refs_after(
                max(0, self.red_flash_state.last_commit_idx), t_ms, u_prime, v_prime, is_sat_red
            )
            a_rp, a_rn = self.red_flash_state.step_masks(frame_idx, t_ms, red_pos, red_neg)
            red_area = max(a_rp, a_rn)

        self._refs.append(
            _FrameRefItem(
                frame_idx=frame_idx,
                t_ms=t_ms,
                lum=lum,
                u_prime=u_prime,
                v_prime=v_prime,
                is_sat_red=is_sat_red,
            )
        )
        return lum_area, red_area


def merge_ultrafast_transitions(
    events: Sequence[TransitionEvent],
    merge_ms: float = 15.0,
) -> list[TransitionEvent]:
    """Collapse >=65 Hz (same-polarity spacing <= 15 ms) oscillation bursts into 1 flash pair.

    Preserves strict opposing polarity alternation across the entire event sequence.
    """
    n = len(events)
    if n < 3:
        return list(events)

    merged: list[TransitionEvent] = []
    i = 0
    limit_ms = merge_ms + 1e-3
    while i < n:
        if (i + 2 < n) and ((events[i + 2].t_ms - events[i].t_ms) <= limit_ms):
            p_start = events[i].polarity
            j = i + 2
            while (j + 1 < n) and ((events[j + 1].t_ms - events[j - 1].t_ms) <= limit_ms):
                j += 1
            if events[j].polarity == -p_start:
                merged.append(events[i])
                merged.append(events[j])
            else:
                merged.append(events[i])
            i = j + 1
        else:
            merged.append(events[i])
            i += 1
    return merged


def build_flash_boolean_mask(
    total_frames: int,
    timestamps_ms: np.ndarray,
    events: Sequence[TransitionEvent],
    min_transitions: int,
    spacing_stride: int,
    safe_gap_ms: float,
    window_1s_ms: float,
) -> np.ndarray:
    """Convert opposing transition events into a merged boolean frame mask.

    Args:
        spacing_stride: `2` for flash leading-edge spacing (`t_{m+2} - t_m < safe_gap_ms`),
                        `1` for rapid scene transition spacing (`t_{m+1} - t_m < safe_gap_ms`).
    """
    mask = np.zeros(total_frames, dtype=bool)
    if len(events) < min_transitions or total_frames == 0:
        return mask

    raw_spans: list[tuple[int, int]] = []
    max_start = len(events) - min_transitions + 1
    for k in range(max_start):
        win = events[k : k + min_transitions]
        if (win[-1].t_ms - win[0].t_ms) > window_1s_ms:
            continue
        gaps_ok = all(
            (win[m + spacing_stride].t_ms - win[m].t_ms) < safe_gap_ms
            for m in range(min_transitions - spacing_stride)
        )
        if not gaps_ok:
            continue
        s_idx = max(0, min(total_frames - 1, win[0].ref_frame))
        e_idx = max(s_idx, min(total_frames - 1, win[-1].frame_idx))
        raw_spans.append((s_idx, e_idx))

    if not raw_spans:
        return mask

    raw_spans.sort(key=lambda sp: (sp[0], sp[1]))
    merged_spans: list[list[int]] = [[raw_spans[0][0], raw_spans[0][1]]]
    for s_idx, e_idx in raw_spans[1:]:
        prev_s, prev_e = merged_spans[-1]
        if s_idx <= prev_e or (timestamps_ms[s_idx] - timestamps_ms[prev_e]) < safe_gap_ms:
            merged_spans[-1][1] = max(prev_e, e_idx)
        else:
            merged_spans.append([s_idx, e_idx])

    for s_idx, e_idx in merged_spans:
        mask[s_idx : e_idx + 1] = True
    return mask


def trailing_1s_event_stats(
    timestamps_ms: np.ndarray,
    events: Sequence[TransitionEvent],
    window_1s_ms: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute per-frame `(count_1s, mean_area_1s, mean_mag_1s)` in O(N + |E|) two-pointer time."""
    n_frames = len(timestamps_ms)
    counts = np.zeros(n_frames, dtype=np.int32)
    mean_areas = np.zeros(n_frames, dtype=np.float32)
    mean_mags = np.zeros(n_frames, dtype=np.float32)
    if not events or n_frames == 0:
        return counts, mean_areas, mean_mags

    ev_t = np.array([e.t_ms for e in events], dtype=np.float64)
    ev_a = np.array([e.area for e in events], dtype=np.float64)
    ev_m = np.array([e.mean_magnitude for e in events], dtype=np.float64)

    left = 0
    right = 0
    area_sum = 0.0
    mag_sum = 0.0
    n_ev = len(events)

    for idx in range(n_frames):
        t_now = timestamps_ms[idx]
        while right < n_ev and ev_t[right] <= t_now + 1e-6:
            area_sum += ev_a[right]
            mag_sum += ev_m[right]
            right += 1
        while left < right and (t_now - ev_t[left]) > window_1s_ms:
            area_sum -= ev_a[left]
            mag_sum -= ev_m[left]
            left += 1
        cnt = right - left
        counts[idx] = cnt
        if cnt > 0:
            mean_areas[idx] = float(area_sum / cnt)
            mean_mags[idx] = float(mag_sum / cnt)

    return counts, mean_areas, mean_mags


def build_iris_extended_mask(
    total_frames: int,
    timestamps_ms: np.ndarray,
    events: Sequence[TransitionEvent],
    counts_1s: np.ndarray,
    fps_spec: FPSWindowSpec,
    pse: PSEConfig,
) -> np.ndarray:
    """EA IRIS 5-second extended flash detector (4..6 transitions/s for >=4.0s within 5.0s)."""
    mask = np.zeros(total_frames, dtype=bool)
    if total_frames < fps_spec.w_5s or not events:
        return mask

    in_band = (
        (counts_1s >= pse.extended_band_min_transitions)
        & (counts_1s <= pse.extended_band_max_transitions)
    ).astype(np.int32)
    prefix = np.concatenate(([0], np.cumsum(in_band)))

    ev_times = np.array([e.t_ms for e in events], dtype=np.float64)
    ev_refs = np.array([e.ref_frame for e in events], dtype=np.int32)
    win_ms = pse.extended_window_sec * 1000.0 + 0.05

    w5 = fps_spec.w_5s
    w4 = fps_spec.w_4s
    for end_idx in range(w5 - 1, total_frames):
        start_win_idx = end_idx - w5 + 1
        active_cnt = int(prefix[end_idx + 1] - prefix[start_win_idx])
        if active_cnt >= w4:
            t_start = timestamps_ms[end_idx] - win_ms
            pos = int(np.searchsorted(ev_times, t_start, side="left"))
            if pos < len(events) and ev_times[pos] <= timestamps_ms[end_idx] + 1e-6:
                span_start = max(0, min(start_win_idx, int(ev_refs[pos])))
            else:
                span_start = start_win_idx
            mask[span_start : end_idx + 1] = True
    return mask
