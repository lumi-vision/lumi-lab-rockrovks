"""Pixel-level static spatial pattern localization for `vdcoreps` (`spatial_mask.py`).

Implements specification `(2-a)`:
  - Preserves 100% identical scalar verdicts `(is_pat, effective_area, best_pairs, best_cpd)`
    with `vdcorefs.spatial.StaticPatternDetector`.
  - Eliminates non-periodic whole-scene `contrast_mask` inclusion, full `360 deg` circular shell
    bleed, and coarse `::4` closing bloat.
  - Back-projects ONLY the exact 1D segments (`[s_lo, s_hi]` on linear scan lines,
    `[alpha_0, alpha_1] x [r_lo, r_hi]` on ring arcs, `[r_lo, r_hi] x [alpha_sp +- d_alpha]`
    on radial spokes, and `active_wave` when 2D FFT fallback triggers) where `>= 6` alternating
    stripe/ring/ray pairs within `[1.0, 10.0] cpd` actually occur, intersected with local
    periodic contrast support and verified via a 2-step self-consistency check.
"""

from __future__ import annotations

import math
from typing import Sequence

import numpy as np
import torch
import torch.nn.functional as F

from vdcorefs.config import PSEConfig, ViewingGeometry
from vdcorefs.spatial import (
    StaticPatternDetector,
    extract_hysteresis_extrema_1d,
    filter_static_pattern_runs,
    separable_max_min_2d,
)


def morphological_close_2d(mask_2d: torch.Tensor, radius: int) -> torch.Tensor:
    """Exact separable 2D binary morphological closing on `2*radius` replicate-padded canvas."""
    if radius <= 0 or not bool(mask_2d.any().item()):
        return mask_2d

    r = int(radius)
    k = 2 * r + 1
    pad2 = 2 * r
    x = mask_2d.to(torch.float32).unsqueeze(0).unsqueeze(0)
    x_pad = F.pad(x, (pad2, pad2, pad2, pad2), mode="replicate")

    d_h = F.max_pool2d(x_pad, kernel_size=(1, k), stride=1)
    dilated = F.max_pool2d(d_h, kernel_size=(k, 1), stride=1)

    e_h = -F.max_pool2d(-dilated, kernel_size=(1, k), stride=1)
    eroded = -F.max_pool2d(-e_h, kernel_size=(k, 1), stride=1)
    return mask_2d | (eroded[0, 0] > 0.5)


def _find_valid_open_runs_1d(
    values: np.ndarray,
    theta_deg: np.ndarray,
    pse: PSEConfig,
    cpd_lo: float | None = None,
) -> list[tuple[float, float, int, float]]:
    """Return `[(t_start_deg, t_end_deg, pairs, median_cpd), ...]` for all runs `>= min_pairs`."""
    ext_t, ext_v = extract_hysteresis_extrema_1d(
        values, theta_deg, pse.pattern_hysteresis_cd_m2
    )
    if len(ext_v) < 2:
        return []

    lo_f = pse.pattern_cpd_min if cpd_lo is None else cpd_lo
    v0 = ext_v[:-1]
    v1 = ext_v[1:]
    l_low = np.minimum(v0, v1)
    l_high = np.maximum(v0, v1)
    dl = l_high - l_low
    contrast_ok = np.where(
        l_low < pse.dark_lum_limit_cd_m2,
        dl >= pse.delta_lum_min_cd_m2,
        (dl / np.maximum(l_high + l_low, 1e-6)) > pse.bright_michelson_min,
    )
    d_theta = ext_t[1:] - ext_t[:-1]
    cpds = np.where(d_theta > 1e-6, 0.5 / np.maximum(d_theta, 1e-6), 0.0)
    valid = contrast_ok & (cpds >= lo_f) & (cpds <= pse.pattern_cpd_max)
    if not np.any(valid):
        return []

    padded = np.concatenate(([False], valid, [False]))
    edges = np.flatnonzero(padded[1:] != padded[:-1])
    out: list[tuple[float, float, int, float]] = []
    for k in range(0, len(edges), 2):
        rs = int(edges[k])
        re = int(edges[k + 1])
        pairs = (re - rs + 1) // 2
        if pairs >= pse.pattern_min_pairs:
            med_cpd = float(np.median(cpds[rs:re]))
            out.append((float(ext_t[rs]), float(ext_t[re]), pairs, med_cpd))
    return out


def _find_valid_closed_runs_1d(
    values: np.ndarray,
    cum_theta: np.ndarray,
    ring_perimeter_deg: float,
    pse: PSEConfig,
    cpd_lo: float | None = None,
) -> list[tuple[float, float, int, float]]:
    """Return `[(t_start_deg, t_end_deg, pairs, median_cpd), ...]` on a closed `360 deg` ring."""
    n = len(values)
    if n < 12:
        return []
    half_n = max(4, n // 2)
    wrapped_v = np.concatenate([values, values[:half_n]])
    wrapped_t = np.concatenate([cum_theta, cum_theta[:half_n] + ring_perimeter_deg])

    ext_t, ext_v = extract_hysteresis_extrema_1d(
        wrapped_v, wrapped_t, pse.pattern_hysteresis_cd_m2
    )
    if len(ext_v) < 4:
        return []

    lo_f = pse.pattern_cpd_min if cpd_lo is None else cpd_lo
    v0 = ext_v[:-1]
    v1 = ext_v[1:]
    l_low = np.minimum(v0, v1)
    l_high = np.maximum(v0, v1)
    dl = l_high - l_low
    contrast_ok = np.where(
        l_low < pse.dark_lum_limit_cd_m2,
        dl >= pse.delta_lum_min_cd_m2,
        (dl / np.maximum(l_high + l_low, 1e-6)) > pse.bright_michelson_min,
    )
    d_theta = ext_t[1:] - ext_t[:-1]
    cpds = np.where(d_theta > 1e-6, 0.5 / np.maximum(d_theta, 1e-6), 0.0)
    valid = contrast_ok & (cpds >= lo_f) & (cpds <= pse.pattern_cpd_max)
    if not np.any(valid):
        return []

    padded = np.concatenate(([False], valid, [False]))
    edges = np.flatnonzero(padded[1:] != padded[:-1])
    limit_deg = ring_perimeter_deg + 1e-4
    out: list[tuple[float, float, int, float]] = []

    for k in range(0, len(edges), 2):
        rs0 = int(edges[k])
        re0 = int(edges[k + 1])
        # Evaluate sub-window bounded by 1 full turn identically to `_best_closed_run_from_extrema`
        best_p = 0
        best_sl = (rs0, re0)
        run_s = rs0
        for j in range(rs0, re0):
            while run_s <= j and (ext_t[j + 1] - ext_t[run_s]) > limit_deg:
                run_s += 1
            run_len = (j + 1) - run_s
            span_deg = ext_t[j + 1] - ext_t[run_s]
            if span_deg >= 0.92 * ring_perimeter_deg:
                in_one_turn = int(
                    np.searchsorted(ext_t, ext_t[run_s] + ring_perimeter_deg - 1e-4, side="left")
                    - np.searchsorted(ext_t, ext_t[run_s] - 1e-5, side="left")
                )
                pairs = min((run_len + 1) // 2, in_one_turn // 2)
            else:
                pairs = (run_len + 1) // 2
            if pairs > best_p and run_len > 0:
                best_p = pairs
                best_sl = (run_s, j + 1)

        if best_p >= pse.pattern_min_pairs and best_sl[1] > best_sl[0]:
            med_cpd = float(np.median(cpds[best_sl[0] : best_sl[1]]))
            out.append((float(ext_t[best_sl[0]]), float(ext_t[best_sl[1]]), best_p, med_cpd))

    return out


def _in_azimuth_arc(ang_2d: torch.Tensor, a0: float, a1: float) -> torch.Tensor:
    """Boolean mask on `ang_2d in [0, 2*pi)` for counter-clockwise arc `a0 -> a1`."""
    two_pi = 2.0 * math.pi
    span = a1 - a0
    if span >= two_pi - 1e-4:
        return torch.ones_like(ang_2d, dtype=torch.bool)
    lo = a0 % two_pi
    hi = a1 % two_pi
    if lo <= hi:
        return (ang_2d >= lo) & (ang_2d <= hi)
    return (ang_2d >= lo) | (ang_2d <= hi)


class PartialStaticPatternDetector(StaticPatternDetector):
    """Extends `StaticPatternDetector` with `(2-a)` exact 1D-segment pattern mask localization."""

    def __init__(
        self,
        geometry: ViewingGeometry,
        pse: PSEConfig = PSEConfig(),
        device: torch.device | None = None,
    ) -> None:
        super().__init__(geometry=geometry, pse=pse, device=device)
        self._prev_mask: torch.Tensor | None = None
        self._prev_close_scale: float | None = None

        # Precompute linear scan line coordinate metadata matching `_build_linear_scan_cache`
        w, h = self.geom.width, self.geom.height
        self._lin_cx = 0.5 * (w - 1)
        self._lin_cy = 0.5 * (h - 1)
        self._lin_s0 = -float(math.ceil(0.5 * math.hypot(w, h)))
        self._lin_angle_meta: list[tuple[float, float, float, np.ndarray, float]] = []
        for m in range(12):
            phi = math.radians(15.0 * m)
            c_p, s_p = math.cos(phi), math.sin(phi)
            p_phi = float(self.geom.pixels_per_degree(phi))
            o_max = 0.5 * (w - 1) * abs(s_p) + 0.5 * (h - 1) * abs(c_p)
            offsets = np.linspace(-0.75 * o_max, 0.75 * o_max, 9)
            self._lin_angle_meta.append((c_p, s_p, p_phi, offsets, o_max))

        # Precompute ring azimuth angle lookup tables matching `_build_ring_templates`
        self._ring_alpha_luts: list[tuple[np.ndarray, np.ndarray]] = []
        for (s_idx, e_idx), (cum_theta, perim_deg) in zip(self._ring_slices, self._ring_theta):
            n_alpha = e_idx - s_idx
            alphas = np.linspace(0.0, 2.0 * math.pi, n_alpha, endpoint=False, dtype=np.float64)
            self._ring_alpha_luts.append(
                (np.append(cum_theta, perim_deg), np.append(alphas, 2.0 * math.pi))
            )

        self._spoke_alphas = np.linspace(0.0, 2.0 * math.pi, 24, endpoint=False, dtype=np.float64)

    def reset(self) -> None:
        super().reset()
        self._prev_mask = None
        self._prev_close_scale = None

    def _compute_contrast_mask(self, lum_4d: torch.Tensor) -> torch.Tensor:
        l_max_loc, l_min_loc = separable_max_min_2d(lum_4d, self.pool_radius)
        dl_loc = l_max_loc - l_min_loc
        return torch.where(
            l_min_loc < self.pse.dark_lum_limit_cd_m2,
            dl_loc >= self.pse.delta_lum_min_cd_m2,
            (dl_loc / torch.clamp(l_max_loc + l_min_loc, min=1e-6)) > self.pse.bright_michelson_min,
        )

    def _evaluate_2d_fft_wave_mask(self, lum: torch.Tensor) -> tuple[int, float, torch.Tensor]:
        """Step 4 2D FFT fallback returning `(est_pairs, f_dom_cpd, active_wave_mask)`."""
        empty = torch.zeros_like(lum, dtype=torch.bool)
        centered = (lum - lum.mean()) * self._hann_2d
        spec = torch.fft.rfft2(centered)
        power = spec.real * spec.real + spec.imag * spec.imag
        power[0, 0] = 0.0
        total_energy = float(power.sum().item())
        if total_energy <= 1e-6:
            return 0, 0.0, empty

        bp_power = power * self._fft_bp_mask
        bp_ratio = float(bp_power.sum().item()) / total_energy
        if bp_ratio < self.pse.pattern_fft_energy_ratio_min:
            return 0, 0.0, empty

        wave = torch.fft.irfft2(spec * self._fft_bp_mask, s=lum.shape)
        w4 = wave.unsqueeze(0).unsqueeze(0)
        w_max, w_min = separable_max_min_2d(w4, self.pool_radius)
        active_wave = (w_max - w_min) >= self.pse.delta_lum_min_cd_m2
        wave_area = float(active_wave.float().mean().item())
        if wave_area <= self.pse.pattern_area_min:
            return 0, 0.0, empty

        flat_idx = int(torch.argmax(bp_power.reshape(-1)).item())
        rw = bp_power.shape[1]
        iy, ix = flat_idx // rw, flat_idx % rw
        f_dom_cpd = float(self._fft_f_cpd[iy, ix].item())
        phi_dom = float(self._fft_phi[iy, ix].item())

        ys, xs = torch.where(active_wave)
        if xs.numel() < 32:
            return 0, 0.0, empty
        proj_deg = (
            (xs.float() * math.cos(phi_dom)) / self.geom.px_per_deg_x
            + (ys.float() * math.sin(phi_dom)) / self.geom.px_per_deg_y
        )
        q_lo, q_hi = torch.quantile(
            proj_deg, torch.tensor([0.02, 0.98], device=self.device)
        ).tolist()
        span_deg = max(0.0, float(q_hi - q_lo))
        est_pairs = int(math.floor(f_dom_cpd * span_deg))
        return est_pairs, f_dom_cpd, active_wave

    def _extract_active_segments_mask(
        self,
        lum: torch.Tensor,
        close_scale: float,
    ) -> torch.Tensor:
        """Extract 2D periodic pattern mask strictly from `[1..10] cpd`, `>= min_pairs` segments."""
        w, h = self.geom.width, self.geom.height
        seg_union = torch.zeros((h, w), dtype=torch.bool, device=self.device)
        lum_4d = lum.unsqueeze(0).unsqueeze(0)
        contrast_mask = self._compute_contrast_mask(lum_4d)
        if not bool(contrast_mask.any().item()):
            return seg_union

        r_close = max(3, int(round(max(0.55, close_scale) * self.pool_radius)))
        closed_contrast = morphological_close_2d(contrast_mask, r_close)
        fine_area = float(contrast_mask.float().mean().item())
        found_1d = False

        # 1) Linear scan line segment back-projection (when `fine_area > pattern_area_min`)
        if fine_area > self.pse.pattern_area_min:
            lin_profiles = (
                F.grid_sample(
                    lum_4d,
                    self._linear_grid,
                    mode="bilinear",
                    padding_mode="border",
                    align_corners=True,
                )
                .squeeze(0)
                .squeeze(0)
                .cpu()
                .numpy()
            )
            ptp = np.ptp(lin_profiles, axis=1)
            active_by_angle: dict[int, list[tuple[int, float, float, float]]] = {}
            for idx in np.flatnonzero(ptp >= self.pse.pattern_hysteresis_cd_m2):
                v_mask = self._linear_valid[idx]
                if int(np.sum(v_mask)) < 12:
                    continue
                runs = _find_valid_open_runs_1d(
                    lin_profiles[idx, v_mask], self._linear_theta[idx, v_mask], self.pse
                )
                if not runs:
                    continue
                m_ang = int(idx // 9)
                k_off = int(idx % 9)
                lst = active_by_angle.setdefault(m_ang, [])
                for t0, t1, _, med_cpd in runs:
                    lst.append((k_off, t0, t1, med_cpd))

            if active_by_angle:
                dx = self._xx_2d - self._lin_cx
                dy = self._yy_2d - self._lin_cy
                for m_ang, seg_list in active_by_angle.items():
                    c_p, s_p, p_phi, offsets, o_max = self._lin_angle_meta[m_ang]
                    u_2d = dx * c_p + dy * s_p
                    n_2d = -dx * s_p + dy * c_p
                    d_off = (1.5 * o_max) / 8.0 if o_max > 0.0 else 1.0
                    ang_corridor = torch.zeros_like(seg_union)

                    for k_off, t0, t1, med_cpd in seg_list:
                        pad_s = 0.80 * (p_phi / max(med_cpd, 1.0))
                        s_lo = self._lin_s0 + t0 * p_phi - pad_s
                        s_hi = self._lin_s0 + t1 * p_phi + pad_s
                        n_lo = (-1.05 * o_max) if k_off == 0 else (float(offsets[k_off]) - 0.65 * d_off)
                        n_hi = (1.05 * o_max) if k_off == 8 else (float(offsets[k_off]) + 0.65 * d_off)
                        ang_corridor |= (u_2d >= s_lo) & (u_2d <= s_hi) & (n_2d >= n_lo) & (n_2d <= n_hi)

                    seg_union |= ang_corridor & closed_contrast
                    found_1d = True

        # 2) Concentric ring azimuthal arc & radial spoke segment back-projection
        centers = self._estimate_candidate_centers(lum_4d, contrast_mask)
        if centers:
            c_np = np.asarray(centers, dtype=np.float32)
            c_gpu = torch.from_numpy(c_np).to(self.device)

            ring_xy = self._ring_unit_xy[None, :, :] + c_gpu[:, None, :]
            gx = (2.0 * ring_xy[..., 0] / max(w - 1, 1)) - 1.0
            gy = (2.0 * ring_xy[..., 1] / max(h - 1, 1)) - 1.0
            ring_grid = torch.stack([gx, gy], dim=-1).unsqueeze(0)
            ring_vals_batch = (
                F.grid_sample(lum_4d, ring_grid, mode="bilinear", padding_mode="border", align_corners=True)
                .squeeze(0)
                .squeeze(0)
                .cpu()
                .numpy()
            )
            ring_xy_np = self._ring_unit_xy_np[None, :, :] + c_np[:, None, :]
            in_bounds_batch = (
                (ring_xy_np[..., 0] >= 0.0)
                & (ring_xy_np[..., 0] <= (w - 1))
                & (ring_xy_np[..., 1] >= 0.0)
                & (ring_xy_np[..., 1] <= (h - 1))
            )

            radii = self._ring_radii
            n_r = len(radii)

            for c_idx, (cx, cy) in enumerate(centers):
                ring_vals_all = ring_vals_batch[c_idx]
                in_bounds_all = in_bounds_batch[c_idx]
                ring_arc_specs: list[tuple[int, float, float, bool]] = []
                wedge_shell_indices: list[int] = []
                c_best_ring_pairs = 0

                for k_idx, ((s_idx, e_idx), (cum_theta, perim_deg)) in enumerate(
                    zip(self._ring_slices, self._ring_theta)
                ):
                    vals = ring_vals_all[s_idx:e_idx]
                    valid = in_bounds_all[s_idx:e_idx]
                    if not np.any(valid):
                        continue
                    theta_lut, alpha_lut = self._ring_alpha_luts[k_idx]

                    if np.all(valid):
                        if float(np.ptp(vals)) < self.pse.pattern_hysteresis_cd_m2:
                            continue
                        band_runs = _find_valid_closed_runs_1d(vals, cum_theta, perim_deg, self.pse)
                        wedge_runs = _find_valid_closed_runs_1d(
                            vals, cum_theta, perim_deg, self.pse, cpd_lo=0.05
                        )
                    else:
                        padded = np.concatenate(([False], valid, [False]))
                        edges = np.flatnonzero(padded[1:] != padded[:-1])
                        runs_ib = [(int(edges[k]), int(edges[k + 1])) for k in range(0, len(edges), 2)]
                        arc_list: list[tuple[np.ndarray, np.ndarray]] = []
                        if len(runs_ib) >= 2 and runs_ib[0][0] == 0 and runs_ib[-1][1] == len(valid):
                            fs, fe = runs_ib[0]
                            ls, le = runs_ib[-1]
                            mv = np.concatenate([vals[ls:le], vals[fs:fe]])
                            mt = np.concatenate([cum_theta[ls:le], cum_theta[fs:fe] + perim_deg])
                            runs_ib = runs_ib[1:-1]
                            if len(mv) >= 12:
                                arc_list.append((mv, mt))
                        for rs, re in runs_ib:
                            if (re - rs) >= 12:
                                arc_list.append((vals[rs:re], cum_theta[rs:re]))

                        band_runs = []
                        wedge_runs = []
                        for av, at in arc_list:
                            if float(np.ptp(av)) < self.pse.pattern_hysteresis_cd_m2:
                                continue
                            band_runs.extend(_find_valid_open_runs_1d(av, at, self.pse))
                            wedge_runs.extend(
                                _find_valid_open_runs_1d(av, at, self.pse, cpd_lo=0.05)
                            )

                    if band_runs or wedge_runs:
                        wedge_shell_indices.append(k_idx)

                    for t0, t1, pairs, med_cpd in band_runs:
                        if pairs > c_best_ring_pairs:
                            c_best_ring_pairs = pairs
                        pad_deg = 0.80 / max(med_cpd, 1.0)
                        t0_p = t0 - pad_deg
                        t1_p = t1 + pad_deg
                        if (t1_p - t0_p) >= 0.88 * perim_deg:
                            ring_arc_specs.append((k_idx, 0.0, 2.0 * math.pi, True))
                        else:
                            a0 = float(np.interp(t0_p % perim_deg, theta_lut, alpha_lut))
                            a1 = float(np.interp(t1_p % perim_deg, theta_lut, alpha_lut))
                            if a1 < a0:
                                a1 += 2.0 * math.pi
                            ring_arc_specs.append((k_idx, a0, a1, False))

                if c_best_ring_pairs >= self.pse.pattern_min_pairs and wedge_shell_indices:
                    dist_2d = torch.hypot(self._xx_2d - cx, self._yy_2d - cy)
                    # Check exact `vdcorefs` shell area gate before accepting this center's rings
                    full_shell_check = torch.zeros_like(dist_2d, dtype=torch.bool)
                    for k_idx in wedge_shell_indices:
                        r_lo = 0.0 if k_idx == 0 else 0.5 * (radii[k_idx - 1] + radii[k_idx])
                        r_hi = (
                            (1.25 * radii[-1])
                            if (k_idx == n_r - 1)
                            else 0.5 * (radii[k_idx] + radii[k_idx + 1])
                        )
                        full_shell_check |= (dist_2d >= r_lo) & (dist_2d < r_hi)

                    if float(full_shell_check.float().mean().item()) > self.pse.pattern_area_min:
                        ang_2d = torch.remainder(
                            torch.atan2(self._yy_2d - cy, self._xx_2d - cx), 2.0 * math.pi
                        )
                        center_ring_mask = torch.zeros_like(seg_union)
                        has_inner_full = any(is_f and k_i <= 2 for k_i, _, _, is_f in ring_arc_specs)

                        for k_idx, a0, a1, is_full in ring_arc_specs:
                            r_lo = 0.0 if k_idx == 0 else 0.5 * (radii[k_idx - 1] + radii[k_idx])
                            r_hi = (
                                (1.25 * radii[-1])
                                if (k_idx == n_r - 1)
                                else 0.5 * (radii[k_idx] + radii[k_idx + 1])
                            )
                            annulus = (dist_2d >= r_lo) & (dist_2d <= r_hi)
                            if is_full:
                                if k_idx <= 2:
                                    center_ring_mask |= annulus
                                else:
                                    center_ring_mask |= annulus & closed_contrast
                            else:
                                center_ring_mask |= (
                                    annulus & _in_azimuth_arc(ang_2d, a0, a1) & closed_contrast
                                )

                        if has_inner_full:
                            center_ring_mask |= dist_2d <= (1.2 * radii[0])

                        seg_union |= center_ring_mask
                        found_1d = True

            # Radial spokes (`check_spokes=True` when `fine_area > pattern_area_min`)
            if fine_area > self.pse.pattern_area_min:
                c_count = len(centers)
                spoke_xy = self._spoke_unit_xy[None, :, :, :] + c_gpu[:, None, None, :]
                spoke_xy_flat = spoke_xy.view(c_count * 24, -1, 2)
                sgx = (2.0 * spoke_xy_flat[..., 0] / max(w - 1, 1)) - 1.0
                sgy = (2.0 * spoke_xy_flat[..., 1] / max(h - 1, 1)) - 1.0
                spoke_grid = torch.stack([sgx, sgy], dim=-1).unsqueeze(0)
                spoke_vals_all = (
                    F.grid_sample(
                        lum_4d, spoke_grid, mode="bilinear", padding_mode="border", align_corners=True
                    )
                    .squeeze(0)
                    .squeeze(0)
                    .cpu()
                    .numpy()
                )
                spoke_xy_np = (self._spoke_unit_xy_np[None, :, :, :] + c_np[:, None, None, :]).reshape(
                    c_count * 24, -1, 2
                )
                spoke_valid_all = (
                    (spoke_xy_np[..., 0] >= 0.0)
                    & (spoke_xy_np[..., 0] <= (w - 1))
                    & (spoke_xy_np[..., 1] >= 0.0)
                    & (spoke_xy_np[..., 1] <= (h - 1))
                )
                sp_ptp = np.ptp(spoke_vals_all, axis=1)
                d_sp = 0.68 * (2.0 * math.pi / 24.0)

                spokes_by_center: dict[int, list[tuple[int, float, float]]] = {}
                for row_idx in np.flatnonzero(sp_ptp >= self.pse.pattern_hysteresis_cd_m2):
                    v_mask = spoke_valid_all[row_idx]
                    if int(np.sum(v_mask)) < 12:
                        continue
                    c_idx = int(row_idx // 24)
                    sp_idx = int(row_idx % 24)
                    vals = spoke_vals_all[row_idx, v_mask]
                    p_phi_sp = float(self._spoke_p_phi[sp_idx])
                    theta_sp = self._spoke_s_px[v_mask] / p_phi_sp
                    runs_sp = _find_valid_open_runs_1d(vals, theta_sp, self.pse)
                    if not runs_sp:
                        continue
                    lst_sp = spokes_by_center.setdefault(c_idx, [])
                    for t0, t1, _, med_cpd in runs_sp:
                        pad_r = 0.80 * (p_phi_sp / max(med_cpd, 1.0))
                        r_lo = max(0.0, t0 * p_phi_sp - pad_r)
                        r_hi = t1 * p_phi_sp + pad_r
                        lst_sp.append((sp_idx, r_lo, r_hi))

                for c_idx, sp_list in spokes_by_center.items():
                    cx, cy = centers[c_idx]
                    dist_2d = torch.hypot(self._xx_2d - cx, self._yy_2d - cy)
                    ang_2d = torch.remainder(
                        torch.atan2(self._yy_2d - cy, self._xx_2d - cx), 2.0 * math.pi
                    )
                    sp_corridor = torch.zeros_like(seg_union)
                    for sp_idx, r_lo, r_hi in sp_list:
                        a_mid = float(self._spoke_alphas[sp_idx])
                        sp_corridor |= (
                            (dist_2d >= r_lo)
                            & (dist_2d <= r_hi)
                            & _in_azimuth_arc(ang_2d, a_mid - d_sp, a_mid + d_sp)
                        )
                    seg_union |= sp_corridor & closed_contrast
                    found_1d = True

        # 3) 2D FFT fallback (ONLY when 1D linear/ring/spoke did not explain the pattern)
        if not found_1d and fine_area > self.pse.pattern_area_min:
            fft_pairs, _, active_wave = self._evaluate_2d_fft_wave_mask(lum)
            if fft_pairs >= self.pse.pattern_min_pairs:
                seg_union |= morphological_close_2d(active_wave & contrast_mask, r_close)

        return seg_union

    def _probe_is_pattern_stateless(self, lum: torch.Tensor) -> bool:
        """Evaluate `StaticPatternDetector.inspect_frame` without mutating `(self._prev_lum, self._prev_result)`."""
        saved_lum = self._prev_lum
        saved_res = self._prev_result
        try:
            self._prev_lum = None
            self._prev_result = None
            is_p, _, _, _ = super().inspect_frame(lum)
            return is_p
        finally:
            self._prev_lum = saved_lum
            self._prev_result = saved_res

    def inspect_frame_with_mask(
        self,
        lum: torch.Tensor,
        close_scale: float = 0.50,
    ) -> tuple[bool, float, int, float, torch.Tensor]:
        """Inspect `lum` `(H, W)` -> `(is_pattern, area, pairs, cpd, pattern_core_mask)`."""
        if (
            self._prev_lum is not None
            and self._prev_result is not None
            and self._prev_mask is not None
            and self._prev_close_scale == close_scale
            and float((lum - self._prev_lum).abs().max().item()) < 0.5
        ):
            is_p, ar, pr, cp = self._prev_result
            return is_p, ar, pr, cp, self._prev_mask

        # Compute exact scalar verdict via upstream `vdcorefs`
        self._prev_lum = None
        self._prev_result = None
        is_pat, effective_area, best_pairs, best_cpd = super().inspect_frame(lum)
        res = (is_pat, effective_area, best_pairs, best_cpd)

        if not is_pat:
            empty_mask = torch.zeros_like(lum, dtype=torch.bool)
            self._prev_mask = empty_mask
            self._prev_close_scale = close_scale
            return (*res, empty_mask)

        pat_mask = self._extract_active_segments_mask(lum, close_scale=close_scale)

        # Up to 2 self-consistency refinement steps if a shifted center or curved residual remains
        for _ in range(2):
            if not bool(pat_mask.any().item()):
                break
            max_masked = float(lum[pat_mask].max().item())
            alpha_sim = min(1.0, 18.0 / max(max_masked, 1e-6))
            w_sim = F.avg_pool2d(
                F.pad(pat_mask.to(torch.float32).unsqueeze(0).unsqueeze(0), (2, 2, 2, 2), mode="replicate"),
                kernel_size=5,
                stride=1,
            )[0, 0]
            w_sim = torch.maximum(pat_mask.to(torch.float32), w_sim)
            lum_sim = lum * (1.0 - w_sim * (1.0 - alpha_sim))
            if not self._probe_is_pattern_stateless(lum_sim):
                break
            extra_mask = self._extract_active_segments_mask(lum_sim, close_scale=close_scale)
            if not bool((extra_mask & (~pat_mask)).any().item()):
                # Widen local closing slightly along already-localized periodic edges
                lum_4d = lum.unsqueeze(0).unsqueeze(0)
                c_mask = self._compute_contrast_mask(lum_4d)
                pat_mask = morphological_close_2d(pat_mask, max(4, self.pool_radius)) & (
                    morphological_close_2d(c_mask, max(4, self.pool_radius)) | pat_mask
                )
                if self._probe_is_pattern_stateless(
                    lum * (1.0 - pat_mask.to(torch.float32) * (1.0 - alpha_sim))
                ):
                    _, _, active_w = self._evaluate_2d_fft_wave_mask(lum_sim)
                    pat_mask |= active_w
                break
            pat_mask |= extra_mask

        self._prev_lum = lum.clone()
        self._prev_result = res
        self._prev_mask = pat_mask
        self._prev_close_scale = close_scale
        return (*res, pat_mask)


__all__ = [
    "PartialStaticPatternDetector",
    "morphological_close_2d",
    "extract_hysteresis_extrema_1d",
    "separable_max_min_2d",
    "filter_static_pattern_runs",
]
