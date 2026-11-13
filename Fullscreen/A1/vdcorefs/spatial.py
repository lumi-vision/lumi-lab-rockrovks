"""Static spatial pattern detection (linear, radial, concentric, and 2D FFT fallback)."""

from __future__ import annotations

import math
from typing import Sequence

import numpy as np
import torch
import torch.nn.functional as F

from .config import PSEConfig, ViewingGeometry
from .photometry import resolve_device


def extract_hysteresis_extrema_1d(
    values: np.ndarray,
    theta_deg: np.ndarray,
    hysteresis: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Exact 1D alternating peaks & troughs with hysteresis, accelerated via local-extrema compression."""
    n = len(values)
    if n < 3 or (float(values.max()) - float(values.min())) < hysteresis:
        return np.empty(0, dtype=np.float64), np.empty(0, dtype=np.float64)

    # Deduplicate flat runs while keeping the leading index of each plateau
    keep_nonflat = np.empty(n, dtype=bool)
    keep_nonflat[0] = True
    keep_nonflat[1:] = values[1:] != values[:-1]
    v_dedup = values[keep_nonflat]
    t_dedup = theta_deg[keep_nonflat]
    m = len(v_dedup)
    if m < 3:
        return np.empty(0, dtype=np.float64), np.empty(0, dtype=np.float64)

    # Strict local extrema + endpoints preserve 100% of hysteresis peaks/troughs
    is_ext = np.zeros(m, dtype=bool)
    is_ext[0] = True
    is_ext[-1] = True
    is_ext[1:-1] = ((v_dedup[1:-1] > v_dedup[:-2]) & (v_dedup[1:-1] > v_dedup[2:])) | (
        (v_dedup[1:-1] < v_dedup[:-2]) & (v_dedup[1:-1] < v_dedup[2:])
    )
    v_cand = v_dedup[is_ext]
    t_cand = t_dedup[is_ext]

    ext_t: list[float] = []
    ext_v: list[float] = []

    cur_max = float(v_cand[0])
    cur_min = float(v_cand[0])
    max_t = float(t_cand[0])
    min_t = float(t_cand[0])
    state = 0  # 0: undecided, 1: rising (seeking peak), -1: falling (seeking trough)

    for k in range(1, len(v_cand)):
        val = float(v_cand[k])
        pos = float(t_cand[k])
        if state == 0:
            if val > cur_max:
                cur_max = val
                max_t = pos
            if val < cur_min:
                cur_min = val
                min_t = pos
            if (cur_max - val) >= hysteresis and max_t < pos:
                ext_t.append(max_t)
                ext_v.append(cur_max)
                cur_min = val
                min_t = pos
                state = -1
            elif (val - cur_min) >= hysteresis and min_t < pos:
                ext_t.append(min_t)
                ext_v.append(cur_min)
                cur_max = val
                max_t = pos
                state = 1
        elif state == 1:
            if val > cur_max:
                cur_max = val
                max_t = pos
            elif (cur_max - val) >= hysteresis:
                ext_t.append(max_t)
                ext_v.append(cur_max)
                cur_min = val
                min_t = pos
                state = -1
        else:
            if val < cur_min:
                cur_min = val
                min_t = pos
            elif (val - cur_min) >= hysteresis:
                ext_t.append(min_t)
                ext_v.append(cur_min)
                cur_max = val
                max_t = pos
                state = 1

    # Append trailing extremum if it forms a valid >= hysteresis swing from the last confirmed one
    if ext_v:
        if state == 1 and (cur_max - ext_v[-1]) >= hysteresis:
            ext_t.append(max_t)
            ext_v.append(cur_max)
        elif state == -1 and (ext_v[-1] - cur_min) >= hysteresis:
            ext_t.append(min_t)
            ext_v.append(cur_min)

    return np.asarray(ext_t, dtype=np.float64), np.asarray(ext_v, dtype=np.float64)


def separable_max_min_2d(x_4d: torch.Tensor, radius: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Exact separable O(2*k) rectangular 2D max and min pooling for `(1, 1, H, W)` tensors."""
    k = 2 * radius + 1
    mx_h = F.max_pool2d(x_4d, kernel_size=(1, k), stride=1, padding=(0, radius))
    mx = F.max_pool2d(mx_h, kernel_size=(k, 1), stride=1, padding=(radius, 0))[0, 0]
    mn_h = -F.max_pool2d(-x_4d, kernel_size=(1, k), stride=1, padding=(0, radius))
    mn = -F.max_pool2d(-mn_h, kernel_size=(k, 1), stride=1, padding=(radius, 0))[0, 0]
    return mx, mn


def _best_open_run_from_extrema(
    ext_t: np.ndarray,
    ext_v: np.ndarray,
    pse: PSEConfig,
    cpd_lo: float,
) -> tuple[int, float]:
    if len(ext_v) < 2:
        return 0, 0.0
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
    valid = contrast_ok & (cpds >= cpd_lo) & (cpds <= pse.pattern_cpd_max)
    if not np.any(valid):
        return 0, 0.0

    padded = np.concatenate(([False], valid, [False]))
    edges = np.flatnonzero(padded[1:] != padded[:-1])
    lengths = edges[1::2] - edges[0::2]
    best_i = int(np.argmax(lengths))
    best_len = int(lengths[best_i])
    best_pairs = (best_len + 1) // 2
    if best_pairs <= 0:
        return 0, 0.0
    rs, re = int(edges[2 * best_i]), int(edges[2 * best_i + 1])
    return best_pairs, float(np.median(cpds[rs:re]))


def evaluate_1d_open_profile(
    values: np.ndarray,
    theta_deg: np.ndarray,
    pse: PSEConfig,
    cpd_min_override: float | None = None,
) -> tuple[int, float]:
    """Return `(max_pairs, dominant_cpd)` along a contiguous open 1D line/spoke/arc."""
    ext_t, ext_v = extract_hysteresis_extrema_1d(
        values, theta_deg, pse.pattern_hysteresis_cd_m2
    )
    cpd_lo = pse.pattern_cpd_min if cpd_min_override is None else cpd_min_override
    return _best_open_run_from_extrema(ext_t, ext_v, pse, cpd_lo)


def evaluate_1d_open_dual(
    values: np.ndarray,
    theta_deg: np.ndarray,
    pse: PSEConfig,
    wedge_cpd_min: float = 0.05,
) -> tuple[int, float, int]:
    """Extract extrema once and return `(band_pairs, band_cpd, wedge_pairs)`."""
    ext_t, ext_v = extract_hysteresis_extrema_1d(
        values, theta_deg, pse.pattern_hysteresis_cd_m2
    )
    if len(ext_v) < 2:
        return 0, 0.0, 0
    band_pairs, band_cpd = _best_open_run_from_extrema(ext_t, ext_v, pse, pse.pattern_cpd_min)
    wedge_pairs, _ = _best_open_run_from_extrema(ext_t, ext_v, pse, wedge_cpd_min)
    return band_pairs, band_cpd, wedge_pairs


def _best_closed_run_from_extrema(
    ext_t: np.ndarray,
    ext_v: np.ndarray,
    ring_perimeter_deg: float,
    pse: PSEConfig,
    cpd_lo: float,
) -> tuple[int, float]:
    if len(ext_v) < 4:
        return 0, 0.0

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
    valid = contrast_ok & (cpds >= cpd_lo) & (cpds <= pse.pattern_cpd_max)
    if not np.any(valid):
        return 0, 0.0

    best_pairs = 0
    best_slice = (0, 0)
    run_start_idx = 0
    limit_deg = ring_perimeter_deg + 1e-4

    for j in range(len(valid)):
        if valid[j]:
            while run_start_idx <= j and (ext_t[j + 1] - ext_t[run_start_idx]) > limit_deg:
                run_start_idx += 1
            run_len = (j + 1) - run_start_idx
            span_deg = ext_t[j + 1] - ext_t[run_start_idx]
            if span_deg >= 0.92 * ring_perimeter_deg:
                in_one_turn = int(
                    np.searchsorted(ext_t, ext_t[run_start_idx] + ring_perimeter_deg - 1e-4, side="left")
                    - np.searchsorted(ext_t, ext_t[run_start_idx] - 1e-5, side="left")
                )
                pairs = min((run_len + 1) // 2, in_one_turn // 2)
            else:
                pairs = (run_len + 1) // 2
            if pairs > best_pairs and run_len > 0:
                best_pairs = pairs
                best_slice = (run_start_idx, j + 1)
        else:
            run_start_idx = j + 1

    if best_pairs <= 0 or best_slice[1] <= best_slice[0]:
        return 0, 0.0
    return best_pairs, float(np.median(cpds[best_slice[0] : best_slice[1]]))


def evaluate_1d_closed_dual(
    values: np.ndarray,
    theta_deg: np.ndarray,
    ring_perimeter_deg: float,
    pse: PSEConfig,
    wedge_cpd_min: float = 0.05,
) -> tuple[int, float, int]:
    """Evaluate a full 360-degree closed ring once for both `[1..10] cpd` and outer wedge continuation."""
    n = len(values)
    if n < 12:
        return 0, 0.0, 0
    half_n = max(4, n // 2)
    wrapped_v = np.concatenate([values, values[:half_n]])
    wrapped_t = np.concatenate([theta_deg, theta_deg[:half_n] + ring_perimeter_deg])

    ext_t, ext_v = extract_hysteresis_extrema_1d(
        wrapped_v, wrapped_t, pse.pattern_hysteresis_cd_m2
    )
    if len(ext_v) < 4:
        return 0, 0.0, 0
    bp, bc = _best_closed_run_from_extrema(ext_t, ext_v, ring_perimeter_deg, pse, pse.pattern_cpd_min)
    wp, _ = _best_closed_run_from_extrema(ext_t, ext_v, ring_perimeter_deg, pse, wedge_cpd_min)
    return bp, bc, wp


class StaticPatternDetector:
    """Detects static linear, concentric, radial, and curved spatial patterns (Stage 3 Section 2.5)."""

    def __init__(
        self,
        geometry: ViewingGeometry,
        pse: PSEConfig = PSEConfig(),
        device: torch.device | None = None,
    ) -> None:
        self.geom = geometry
        self.pse = pse
        self.device = resolve_device(device)

        w, h = geometry.width, geometry.height
        self.pool_radius = max(3, int(round(0.55 * max(geometry.px_per_deg_x, geometry.px_per_deg_y))))
        self._coarse_radius = max(2, int(round(0.07 * min(w, h))))

        yy_1d = torch.arange(h, dtype=torch.float32, device=self.device)
        xx_1d = torch.arange(w, dtype=torch.float32, device=self.device)
        self._yy_2d, self._xx_2d = torch.meshgrid(yy_1d, xx_1d, indexing="ij")

        # Sobel kernels for radial-apex / concentric-center voting
        sobel_x = torch.tensor(
            [[-1.0, 0.0, 1.0], [-2.0, 0.0, 2.0], [-1.0, 0.0, 1.0]],
            dtype=torch.float32,
            device=self.device,
        ).view(1, 1, 3, 3)
        sobel_y = sobel_x.transpose(-1, -2).contiguous()
        self._sobel = torch.cat([sobel_x, sobel_y], dim=0)

        # Precompute cached batched linear scan grid (12 angles x 9 offsets, step = 0.5 px)
        self._linear_step_px = 0.5
        self._linear_grid, self._linear_valid, self._linear_theta = self._build_linear_scan_cache()

        # Precompute unit-circle ring templates & spoke templates (reused for any center (cx, cy))
        (
            self._ring_radii,
            self._ring_unit_xy,
            self._ring_unit_xy_np,
            self._ring_slices,
            self._ring_theta,
        ) = self._build_ring_templates()
        (
            self._spoke_unit_xy,
            self._spoke_unit_xy_np,
            self._spoke_s_px,
            self._spoke_p_phi,
        ) = self._build_spoke_templates()

        # Precompute 2D rfft2 annular bandpass mask & Hann window
        self._hann_2d, self._fft_bp_mask, self._fft_f_cpd, self._fft_phi = (
            self._build_fft_cache(w, h)
        )
        self._prev_lum: torch.Tensor | None = None
        self._prev_result: tuple[bool, float, int, float] | None = None

    def _to_norm_grid(self, x_px: np.ndarray, y_px: np.ndarray) -> torch.Tensor:
        w, h = self.geom.width, self.geom.height
        gx = (2.0 * x_px / max(w - 1, 1)) - 1.0
        gy = (2.0 * y_px / max(h - 1, 1)) - 1.0
        grid = np.stack([gx, gy], axis=-1).astype(np.float32)
        return torch.from_numpy(grid).unsqueeze(0).to(self.device)

    def _build_linear_scan_cache(self) -> tuple[torch.Tensor, np.ndarray, np.ndarray]:
        w, h = self.geom.width, self.geom.height
        cx = 0.5 * (w - 1)
        cy = 0.5 * (h - 1)
        s_max = math.ceil(0.5 * math.hypot(w, h))
        s_vals = np.arange(-s_max, s_max + 1e-6, self._linear_step_px, dtype=np.float64)
        l_len = len(s_vals)

        n_angles = 12
        n_offsets = 9
        total_lines = n_angles * n_offsets
        x_all = np.zeros((total_lines, l_len), dtype=np.float64)
        y_all = np.zeros((total_lines, l_len), dtype=np.float64)
        theta_all = np.zeros((total_lines, l_len), dtype=np.float64)

        line_idx = 0
        for m in range(n_angles):
            phi = math.radians(15.0 * m)
            c_p, s_p = math.cos(phi), math.sin(phi)
            nx, ny = -s_p, c_p
            o_max = 0.5 * (w - 1) * abs(s_p) + 0.5 * (h - 1) * abs(c_p)
            offsets = np.linspace(-0.75 * o_max, 0.75 * o_max, n_offsets)
            p_phi = float(self.geom.pixels_per_degree(phi))
            theta_line = (s_vals - s_vals[0]) / p_phi

            for off in offsets:
                x_all[line_idx] = cx + off * nx + s_vals * c_p
                y_all[line_idx] = cy + off * ny + s_vals * s_p
                theta_all[line_idx] = theta_line
                line_idx += 1

        valid = (x_all >= 0.0) & (x_all <= (w - 1)) & (y_all >= 0.0) & (y_all <= (h - 1))
        return self._to_norm_grid(x_all, y_all), valid, theta_all

    def _build_ring_templates(
        self,
    ) -> tuple[
        list[float],
        torch.Tensor,
        np.ndarray,
        list[tuple[int, int]],
        list[tuple[np.ndarray, float]],
    ]:
        w, h = self.geom.width, self.geom.height
        r_min = max(8.0, round(0.4 * self.geom.px_per_deg_ring))
        r_max = 1.15 * math.hypot(w, h)

        radii: list[float] = []
        r = r_min
        while r <= r_max + 1e-6:
            radii.append(r)
            r *= 1.28

        unit_xy_parts: list[np.ndarray] = []
        slices: list[tuple[int, int]] = []
        theta_info: list[tuple[np.ndarray, float]] = []
        cursor = 0

        for rad in radii:
            n_alpha = int(np.clip(math.ceil(2.0 * math.pi * rad / 0.75), 72, 1440))
            alphas = np.linspace(0.0, 2.0 * math.pi, n_alpha, endpoint=False, dtype=np.float64)
            ux = rad * np.cos(alphas)
            uy = rad * np.sin(alphas)
            unit_xy_parts.append(np.stack([ux, uy], axis=-1).astype(np.float32))
            slices.append((cursor, cursor + n_alpha))
            cursor += n_alpha

            phi_tan = np.mod(alphas + 0.5 * math.pi, math.pi)
            d_arc_px = (2.0 * math.pi * rad) / n_alpha
            d_theta = d_arc_px / self.geom.pixels_per_degree(phi_tan)
            cum_theta = np.concatenate(([0.0], np.cumsum(d_theta[:-1])))
            perimeter_deg = float(np.sum(d_theta))
            theta_info.append((cum_theta, perimeter_deg))

        unit_xy_np = np.concatenate(unit_xy_parts, axis=0)
        unit_xy = torch.from_numpy(unit_xy_np).to(self.device)
        return radii, unit_xy, unit_xy_np, slices, theta_info

    def _build_spoke_templates(
        self,
    ) -> tuple[torch.Tensor, np.ndarray, np.ndarray, np.ndarray]:
        w, h = self.geom.width, self.geom.height
        s_max = math.ceil(math.hypot(w, h))
        s_vals = np.arange(0.0, s_max + 1e-6, 0.5, dtype=np.float64)
        n_spokes = 24
        alphas = np.linspace(0.0, 2.0 * math.pi, n_spokes, endpoint=False, dtype=np.float64)
        ux = np.cos(alphas)[:, None] * s_vals[None, :]
        uy = np.sin(alphas)[:, None] * s_vals[None, :]
        unit_xy_np = np.stack([ux, uy], axis=-1).astype(np.float32)
        unit_xy = torch.from_numpy(unit_xy_np).to(self.device)
        p_phi = self.geom.pixels_per_degree(np.mod(alphas, math.pi))
        return unit_xy, unit_xy_np, s_vals, np.asarray(p_phi, dtype=np.float64)

    def _build_fft_cache(
        self, w: int, h: int
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        wy = torch.hann_window(h, device=self.device, dtype=torch.float32)
        wx = torch.hann_window(w, device=self.device, dtype=torch.float32)
        hann_2d = wy[:, None] * wx[None, :]

        fy_cpd = torch.fft.fftfreq(h, d=1.0, device=self.device) * self.geom.px_per_deg_y
        fx_cpd = torch.fft.rfftfreq(w, d=1.0, device=self.device) * self.geom.px_per_deg_x
        f_rad = torch.hypot(fy_cpd[:, None], fx_cpd[None, :])
        phi_grid = torch.atan2(fy_cpd[:, None], fx_cpd[None, :])
        bp_mask = (f_rad >= self.pse.pattern_cpd_min) & (f_rad <= self.pse.pattern_cpd_max)
        return hann_2d, bp_mask, f_rad, phi_grid

    def _estimate_candidate_centers(
        self, lum_4d: torch.Tensor, contrast_mask: torch.Tensor
    ) -> list[tuple[float, float]]:
        """Estimate up to 4 centers: frame center, mask centroid, radial apex, concentric center."""
        w, h = self.geom.width, self.geom.height
        cx0 = 0.5 * (w - 1)
        cy0 = 0.5 * (h - 1)
        centers: list[tuple[float, float]] = [(cx0, cy0)]
        min_sep = 0.08 * min(w, h)

        def _add_if_distinct(cand_x: float, cand_y: float) -> None:
            if not (math.isfinite(cand_x) and math.isfinite(cand_y)):
                return
            cand_x = float(np.clip(cand_x, -0.15 * w, 1.15 * w))
            cand_y = float(np.clip(cand_y, -0.15 * h, 1.15 * h))
            if all(math.hypot(cand_x - x0, cand_y - y0) > min_sep for x0, y0 in centers):
                centers.append((cand_x, cand_y))

        ys_m, xs_m = torch.where(contrast_mask)
        if len(xs_m) > 32:
            cent = torch.stack([xs_m.float().mean(), ys_m.float().mean()]).tolist()
            _add_if_distinct(float(cent[0]), float(cent[1]))

        grads = F.conv2d(lum_4d, self._sobel, padding=1)[0]
        gx_sub = grads[0, 2:-2:4, 2:-2:4]
        gy_sub = grads[1, 2:-2:4, 2:-2:4]
        mag = torch.hypot(gx_sub, gy_sub)
        flat_mag = mag.reshape(-1)
        if flat_mag.numel() == 0:
            return centers

        k_top = min(512, flat_mag.numel())
        top_vals, top_idx = torch.topk(flat_mag, k=k_top)
        valid_g = top_vals > 15.0
        if int(valid_g.sum().item()) < 24:
            return centers

        top_idx = top_idx[valid_g]
        sub_w = gx_sub.shape[1]
        py = ((top_idx // sub_w).float() * 4.0) + 2.0
        px = ((top_idx % sub_w).float() * 4.0) + 2.0
        gxv = gx_sub.reshape(-1)[top_idx]
        gyv = gy_sub.reshape(-1)[top_idx]
        gnorm = torch.clamp(torch.hypot(gxv, gyv), min=1e-6)
        nx = gxv / gnorm
        ny = gyv / gnorm
        tx = -ny
        ty = nx

        bins_y, bins_x = 24, 32
        x_min, x_max = -0.15 * w, 1.15 * w
        y_min, y_max = -0.15 * h, 1.15 * h
        diag = math.hypot(w, h)
        t_steps = torch.linspace(-diag, diag, 41, device=self.device)[None, :]

        for vx, vy in ((tx, ty), (nx, ny)):
            cand_x = px[:, None] + vx[:, None] * t_steps
            cand_y = py[:, None] + vy[:, None] * t_steps
            bx = ((cand_x - x_min) / max(x_max - x_min, 1.0) * bins_x).long()
            by = ((cand_y - y_min) / max(y_max - y_min, 1.0) * bins_y).long()
            in_b = (bx >= 0) & (bx < bins_x) & (by >= 0) & (by < bins_y)
            flat_bins = (by * bins_x + bx)[in_b]
            if flat_bins.numel() > 0:
                hist = torch.bincount(flat_bins, minlength=bins_y * bins_x)
                best_bin = int(torch.argmax(hist).item())
                if int(hist[best_bin].item()) >= 12:
                    vy_idx = best_bin // bins_x
                    vx_idx = best_bin % bins_x
                    est_x = x_min + (vx_idx + 0.5) * (x_max - x_min) / bins_x
                    est_y = y_min + (vy_idx + 0.5) * (y_max - y_min) / bins_y
                    _add_if_distinct(est_x, est_y)

        return centers[:4]

    def _scan_centers_batched(
        self,
        lum_4d: torch.Tensor,
        centers: Sequence[tuple[float, float]],
        check_spokes: bool = True,
    ) -> tuple[int, float, float]:
        """Batch all candidate centers into a single `grid_sample` for rings (and spokes)."""
        if not centers:
            return 0, 0.0, 0.0
        w, h = self.geom.width, self.geom.height
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

        best_pairs = 0
        best_cpd = 0.0
        best_radial_area = 0.0
        radii = self._ring_radii
        n_r = len(radii)

        for c_idx, (cx, cy) in enumerate(centers):
            ring_vals_all = ring_vals_batch[c_idx]
            in_bounds_all = in_bounds_batch[c_idx]
            c_best_ring_pairs = 0
            c_best_ring_cpd = 0.0
            active_shell_indices: list[int] = []

            for k_idx, ((s_idx, e_idx), (cum_theta, perim_deg)) in enumerate(
                zip(self._ring_slices, self._ring_theta)
            ):
                vals = ring_vals_all[s_idx:e_idx]
                valid = in_bounds_all[s_idx:e_idx]
                if not np.any(valid):
                    continue

                ring_band_pairs = 0
                ring_band_cpd = 0.0
                ring_wedge_pairs = 0

                if np.all(valid):
                    if float(np.ptp(vals)) < self.pse.pattern_hysteresis_cd_m2:
                        continue
                    ring_band_pairs, ring_band_cpd, ring_wedge_pairs = evaluate_1d_closed_dual(
                        vals, cum_theta, perim_deg, self.pse
                    )
                else:
                    padded = np.concatenate(([False], valid, [False]))
                    edges = np.flatnonzero(padded[1:] != padded[:-1])
                    runs = [(int(edges[k]), int(edges[k + 1])) for k in range(0, len(edges), 2)]
                    arc_list: list[tuple[np.ndarray, np.ndarray]] = []
                    if len(runs) >= 2 and runs[0][0] == 0 and runs[-1][1] == len(valid):
                        first_s, first_e = runs[0]
                        last_s, last_e = runs[-1]
                        merged_v = np.concatenate([vals[last_s:last_e], vals[first_s:first_e]])
                        merged_t = np.concatenate(
                            [cum_theta[last_s:last_e], cum_theta[first_s:first_e] + perim_deg]
                        )
                        runs = runs[1:-1]
                        if len(merged_v) >= 12:
                            arc_list.append((merged_v, merged_t))

                    for rs, re in runs:
                        if (re - rs) >= 12:
                            arc_list.append((vals[rs:re], cum_theta[rs:re]))

                    for av, at in arc_list:
                        if float(np.ptp(av)) < self.pse.pattern_hysteresis_cd_m2:
                            continue
                        pb, cb, pw = evaluate_1d_open_dual(av, at, self.pse)
                        if pb > ring_band_pairs:
                            ring_band_pairs, ring_band_cpd = pb, cb
                        if pw > ring_wedge_pairs:
                            ring_wedge_pairs = pw

                if ring_band_pairs > c_best_ring_pairs:
                    c_best_ring_pairs, c_best_ring_cpd = ring_band_pairs, ring_band_cpd
                if max(ring_band_pairs, ring_wedge_pairs) >= self.pse.pattern_min_pairs:
                    active_shell_indices.append(k_idx)

            if c_best_ring_pairs >= self.pse.pattern_min_pairs and active_shell_indices:
                dist_2d = torch.hypot(self._xx_2d - cx, self._yy_2d - cy)
                shell_mask = torch.zeros_like(dist_2d, dtype=torch.bool)
                for k_idx in active_shell_indices:
                    r_lo = 0.0 if k_idx == 0 else 0.5 * (radii[k_idx - 1] + radii[k_idx])
                    r_hi = (
                        (1.25 * radii[-1])
                        if (k_idx == n_r - 1)
                        else 0.5 * (radii[k_idx] + radii[k_idx + 1])
                    )
                    shell_mask |= (dist_2d >= r_lo) & (dist_2d < r_hi)
                radial_area = float(shell_mask.float().mean().item())
                if radial_area > self.pse.pattern_area_min and c_best_ring_pairs > best_pairs:
                    best_pairs = c_best_ring_pairs
                    best_cpd = c_best_ring_cpd
                    best_radial_area = max(best_radial_area, radial_area)

        if not check_spokes:
            return best_pairs, best_cpd, best_radial_area

        c_count = len(centers)
        spoke_xy = self._spoke_unit_xy[None, :, :, :] + c_gpu[:, None, None, :]
        spoke_xy_flat = spoke_xy.view(c_count * 24, -1, 2)
        sgx = (2.0 * spoke_xy_flat[..., 0] / max(w - 1, 1)) - 1.0
        sgy = (2.0 * spoke_xy_flat[..., 1] / max(h - 1, 1)) - 1.0
        spoke_grid = torch.stack([sgx, sgy], dim=-1).unsqueeze(0)
        spoke_vals_all = (
            F.grid_sample(lum_4d, spoke_grid, mode="bilinear", padding_mode="border", align_corners=True)
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
        for row_idx in np.flatnonzero(sp_ptp >= self.pse.pattern_hysteresis_cd_m2):
            v_mask = spoke_valid_all[row_idx]
            if int(np.sum(v_mask)) < 12:
                continue
            sp_idx = int(row_idx % 24)
            vals = spoke_vals_all[row_idx, v_mask]
            theta_sp = self._spoke_s_px[v_mask] / self._spoke_p_phi[sp_idx]
            pairs, cpd = evaluate_1d_open_profile(vals, theta_sp, self.pse)
            if pairs > best_pairs:
                best_pairs, best_cpd = pairs, cpd

        return best_pairs, best_cpd, best_radial_area

    def _evaluate_2d_fft_fallback(self, lum: torch.Tensor) -> tuple[int, float]:
        """Step 4: 2D rfft2 1..10 cpd annular bandpass fallback for curved/wavy gratings."""
        centered = (lum - lum.mean()) * self._hann_2d
        spec = torch.fft.rfft2(centered)
        power = spec.real * spec.real + spec.imag * spec.imag
        power[0, 0] = 0.0
        total_energy = float(power.sum().item())
        if total_energy <= 1e-6:
            return 0, 0.0

        bp_power = power * self._fft_bp_mask
        bp_ratio = float(bp_power.sum().item()) / total_energy
        if bp_ratio < self.pse.pattern_fft_energy_ratio_min:
            return 0, 0.0

        wave = torch.fft.irfft2(spec * self._fft_bp_mask, s=lum.shape)
        w4 = wave.unsqueeze(0).unsqueeze(0)
        w_max, w_min = separable_max_min_2d(w4, self.pool_radius)
        wave_amp = w_max - w_min
        active_wave = wave_amp >= self.pse.delta_lum_min_cd_m2
        wave_area = float(active_wave.float().mean().item())
        if wave_area <= self.pse.pattern_area_min:
            return 0, 0.0

        flat_idx = int(torch.argmax(bp_power.reshape(-1)).item())
        rw = bp_power.shape[1]
        iy, ix = flat_idx // rw, flat_idx % rw
        f_dom_cpd = float(self._fft_f_cpd[iy, ix].item())
        phi_dom = float(self._fft_phi[iy, ix].item())

        ys, xs = torch.where(active_wave)
        if xs.numel() < 32:
            return 0, 0.0
        proj_deg = (
            (xs.float() * math.cos(phi_dom)) / self.geom.px_per_deg_x
            + (ys.float() * math.sin(phi_dom)) / self.geom.px_per_deg_y
        )
        q_lo, q_hi = torch.quantile(proj_deg, torch.tensor([0.02, 0.98], device=self.device)).tolist()
        span_deg = max(0.0, float(q_hi - q_lo))
        est_pairs = int(math.floor(f_dom_cpd * span_deg))
        return est_pairs, f_dom_cpd

    def inspect_frame(self, lum: torch.Tensor) -> tuple[bool, float, int, float]:
        """Inspect a single luminance frame `(H, W)` -> `(is_pattern, area, pairs, cpd)`."""
        if (
            self._prev_lum is not None
            and self._prev_result is not None
            and float((lum - self._prev_lum).abs().max().item()) < 0.5
        ):
            return self._prev_result

        lum_4d = lum.unsqueeze(0).unsqueeze(0)
        l_max_loc, l_min_loc = separable_max_min_2d(lum_4d, self.pool_radius)
        dl_loc = l_max_loc - l_min_loc

        contrast_mask = torch.where(
            l_min_loc < self.pse.dark_lum_limit_cd_m2,
            dl_loc >= self.pse.delta_lum_min_cd_m2,
            (dl_loc / torch.clamp(l_max_loc + l_min_loc, min=1e-6)) > self.pse.bright_michelson_min,
        )
        fine_area = float(contrast_mask.float().mean().item())

        if fine_area <= self.pse.pattern_area_min:
            if fine_area < 0.12:
                res = (False, fine_area, 0, 0.0)
                self._prev_lum = lum.clone()
                self._prev_result = res
                return res
            sub = lum_4d[..., ::4, ::4]
            c_max, c_min = separable_max_min_2d(sub, self._coarse_radius)
            c_dl = c_max - c_min
            coarse_ok = torch.where(
                c_min < self.pse.dark_lum_limit_cd_m2,
                c_dl >= self.pse.delta_lum_min_cd_m2,
                (c_dl / torch.clamp(c_max + c_min, min=1e-6)) > self.pse.bright_michelson_min,
            )
            if float(coarse_ok.float().mean().item()) <= self.pse.pattern_area_min:
                res = (False, fine_area, 0, 0.0)
                self._prev_lum = lum.clone()
                self._prev_result = res
                return res

            centers = self._estimate_candidate_centers(lum_4d, contrast_mask)
            rad_pairs, rad_cpd, rad_area = self._scan_centers_batched(
                lum_4d, centers, check_spokes=False
            )
            if rad_pairs >= self.pse.pattern_min_pairs and rad_area > self.pse.pattern_area_min:
                res = (True, rad_area, rad_pairs, rad_cpd)
            else:
                res = (False, fine_area, 0, 0.0)
            self._prev_lum = lum.clone()
            self._prev_result = res
            return res

        best_pairs = 0
        best_cpd = 0.0
        effective_area = fine_area

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
        for idx in np.flatnonzero(ptp >= self.pse.pattern_hysteresis_cd_m2):
            v_mask = self._linear_valid[idx]
            if int(np.sum(v_mask)) < 12:
                continue
            pairs, cpd = evaluate_1d_open_profile(
                lin_profiles[idx, v_mask], self._linear_theta[idx, v_mask], self.pse
            )
            if pairs > best_pairs:
                best_pairs, best_cpd = pairs, cpd

        centers = self._estimate_candidate_centers(lum_4d, contrast_mask)
        rc_pairs, rc_cpd, rad_area = self._scan_centers_batched(
            lum_4d, centers, check_spokes=True
        )
        if rc_pairs > best_pairs:
            best_pairs, best_cpd = rc_pairs, rc_cpd
            effective_area = max(effective_area, rad_area)

        if best_pairs < self.pse.pattern_min_pairs:
            fft_pairs, fft_cpd = self._evaluate_2d_fft_fallback(lum)
            if fft_pairs > best_pairs:
                best_pairs, best_cpd = fft_pairs, fft_cpd

        is_pat = best_pairs >= self.pse.pattern_min_pairs
        res = (is_pat, effective_area, best_pairs, best_cpd)
        self._prev_lum = lum.clone()
        self._prev_result = res
        return res

    def reset(self) -> None:
        """Clear single-frame temporal cache before starting a new video stream."""
        self._prev_lum = None
        self._prev_result = None


def filter_static_pattern_runs(
    raw_pattern_mask: np.ndarray,
    min_frames: int,
) -> np.ndarray:
    """Method A duration filter: retain only contiguous `True` runs of length >= `min_frames`."""
    out = np.zeros_like(raw_pattern_mask, dtype=bool)
    if len(raw_pattern_mask) == 0:
        return out
    padded = np.concatenate(([False], raw_pattern_mask, [False]))
    edges = np.flatnonzero(padded[1:] != padded[:-1])
    for i in range(0, len(edges), 2):
        s_idx, e_idx = int(edges[i]), int(edges[i + 1])
        if (e_idx - s_idx) >= min_frames:
            out[s_idx:e_idx] = True
    return out
