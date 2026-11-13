"""Four parallel continuous PSE risk metrics and pixel-level diagnostic tracker."""

from __future__ import annotations

from collections import deque
import math

import numpy as np
import torch

from .config import FPSWindowSpec, PSEConfig, ViewingGeometry

# Harding & Jeavons (1994) clinical PPR sensitivity curve control points (Hz -> fraction)
_PPR_FREQ_HZ = np.array(
    [0.0, 1.0, 3.0, 5.0, 10.0, 16.0, 20.0, 25.0, 30.0, 40.0, 50.0, 60.0, 65.0, 80.0],
    dtype=np.float64,
)
_PPR_FREQ_SENS = np.array(
    [0.0, 0.03, 0.09, 0.28, 0.65, 0.89, 0.89, 0.80, 0.72, 0.58, 0.49, 0.15, 0.04, 0.0],
    dtype=np.float64,
)

# Wilkins et al. (1979, 2005) spatial-frequency pattern sensitivity curve (cpd -> fraction)
_PPR_CPD_AXIS = np.array([0.0, 0.5, 1.0, 2.0, 3.0, 4.0, 6.0, 10.0, 12.0, 20.0], dtype=np.float64)
_PPR_CPD_SENS = np.array([0.0, 0.05, 0.35, 0.80, 0.89, 0.85, 0.65, 0.25, 0.05, 0.0], dtype=np.float64)


class PixelDiagnosticTracker:
    """HardingFPA-style top-25% active pixel 1-second opposing transition counter (`diag_transitions_25pct`)."""

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
        self._lum_anchor: torch.Tensor | None = None
        self._lum_pol = torch.zeros((height, width), dtype=torch.int8, device=device)
        self._running_counts = torch.zeros((height, width), dtype=torch.int32, device=device)
        self._ring: deque[torch.Tensor] = deque()
        self._total_pixels = height * width
        self._top25_rank = int(math.ceil(0.75 * self._total_pixels))

    def update(self, lum: torch.Tensor) -> int:
        if self._lum_anchor is None:
            self._lum_anchor = lum.clone()
            ev_mask = torch.zeros_like(lum, dtype=torch.int32)
        else:
            dl = lum - self._lum_anchor
            l_low = torch.minimum(lum, self._lum_anchor)
            valid = torch.where(
                l_low < self.pse.dark_lum_limit_cd_m2,
                dl.abs() >= self.pse.delta_lum_min_cd_m2,
                (dl.abs() / torch.clamp(lum + self._lum_anchor, min=1e-6))
                > self.pse.bright_michelson_min,
            )
            pos_hit = valid & (dl > 0.0) & (self._lum_pol <= 0)
            neg_hit = valid & (dl < 0.0) & (self._lum_pol >= 0)
            ev_bool = pos_hit | neg_hit

            # Advance monotonic extrema even without polarity flip so reversals are accurate
            higher_same = (self._lum_pol == 1) & (lum > self._lum_anchor)
            lower_same = (self._lum_pol == -1) & (lum < self._lum_anchor)
            update_anchor = ev_bool | higher_same | lower_same
            self._lum_anchor = torch.where(update_anchor, lum, self._lum_anchor)
            self._lum_pol = torch.where(
                pos_hit,
                torch.ones_like(self._lum_pol),
                torch.where(neg_hit, -torch.ones_like(self._lum_pol), self._lum_pol),
            )
            ev_mask = ev_bool.to(torch.int32)

        self._running_counts += ev_mask
        self._ring.append(ev_mask)
        if len(self._ring) > self.w_1s:
            self._running_counts -= self._ring.popleft()

        # Exact O(N) 75th-percentile integer transition count via bincount
        hist = torch.bincount(self._running_counts.reshape(-1), minlength=self.w_1s + 2)
        cdf = torch.cumsum(hist, dim=0)
        idx = int(torch.searchsorted(cdf, self._top25_rank).item())
        return idx


class NomuraTemporalFilter:
    """Nomura (1999) / Nomura et al. (2000) 4-stage temporal cortical filter risk e(t)."""

    def __init__(self, fps: float, device: torch.device, tau_sec: float = 0.015) -> None:
        dt = 1.0 / max(fps, 1.0)
        self.alpha = float(1.0 - math.exp(-dt / tau_sec))
        self.weights = torch.tensor([0.5, 0.3, 0.2], dtype=torch.float32, device=device).view(
            1, 1, 3
        )
        self._stages: list[torch.Tensor] | None = None

    def update(self, rgb_lin: torch.Tensor) -> tuple[float, float]:
        if self._stages is None:
            self._stages = [rgb_lin.clone() for _ in range(4)]
            return 0.0, 0.0

        a = self.alpha
        prev = rgb_lin
        for k in range(4):
            self._stages[k].mul_(1.0 - a).add_(prev, alpha=a)
            prev = self._stages[k]

        diff = (self._stages[3] - rgb_lin).abs()
        weighted = (diff * self.weights).sum(dim=-1)
        e_raw = float(weighted.mean().item())
        e_scaled = float(np.clip(100.0 * (1.0 - math.exp(-4.5 * e_raw)), 0.0, 100.0))
        return e_raw, e_scaled


def _build_ufm_biphasic_kernels(
    fps_w: float,
    size_deg: float,
) -> tuple[list[np.ndarray], np.ndarray]:
    """Construct the 5 standard-luminance UFM biphasic FIR kernels (Apple VFR Appendix pp. 15-18)."""
    std_nits = np.array([0.2, 1.0, 10.0, 150.0, 500.0], dtype=np.float64)
    if size_deg <= 12.0:
        peak_gains = [80.0, 175.0, 290.0, 350.0, 390.0]
        peak_freqs = [6.0, 8.0, 10.0, 14.0, 16.0]
    elif size_deg <= 32.0:
        peak_gains = [125.0, 220.0, 400.0, 520.0, 500.0]
        peak_freqs = [7.0, 9.0, 11.0, 15.0, 17.0]
    else:
        peak_gains = [195.0, 250.0, 440.0, 540.0, 600.0]
        peak_freqs = [7.0, 9.0, 12.0, 15.0, 18.0]

    kernels: list[np.ndarray] = []
    for gain, f_pk in zip(peak_gains, peak_freqs):
        tau1 = 1.0 / (2.0 * math.pi * f_pk * 1.15)
        tau2 = 1.85 * tau1
        duration = min(0.28, max(5.0 / fps_w, 8.5 * tau2))
        n_taps = max(4, int(math.ceil(duration * fps_w)))
        t = np.arange(n_taps, dtype=np.float64) / fps_w
        # Biphasic difference-of-gammas impulse response with exact zero DC sum
        h = (t / (tau1 * tau1)) * np.exp(-t / tau1) - (t / (tau2 * tau2)) * np.exp(-t / tau2)
        h -= np.mean(h)
        freqs = np.linspace(0.5, min(50.0, 0.49 * fps_w), 256)
        mtf = np.abs(h[None, :] @ np.exp(-2j * math.pi * freqs[:, None] * t[None, :]).T).ravel()
        cur_pk = float(np.max(mtf)) if len(mtf) > 0 else 1.0
        if cur_pk > 1e-9:
            h = h * (gain / cur_pk)
        kernels.append(h.astype(np.float64))

    return kernels, np.log10(std_nits)


def compute_vfm_weibull_series(
    mean_y_rel: np.ndarray,
    fps: float,
    geometry: ViewingGeometry,
    l_max_cd_m2: float = 200.0,
) -> np.ndarray:
    """Apple VFR / UFM Video Flashing Metric (`risk_vfm_weibull` in [0, 100], Eqs. 1-14)."""
    n_frames = len(mean_y_rel)
    if n_frames == 0:
        return np.zeros(0, dtype=np.float32)

    w_fps = float(max(fps, 1.0))
    area_deg2 = max(1.0, geometry.area_deg2)
    s0 = math.sqrt(1.6 * area_deg2)
    std_sizes = (6.0, 20.0, 45.0)
    s_std = min(std_sizes, key=lambda sv: abs(sv - s0))

    kernels, log_std_nits = _build_ufm_biphasic_kernels(w_fps, s_std)
    kernel_norms = [max(float(np.linalg.norm(hk)), 1e-9) for hk in kernels]

    # Eq. (3)-(5): Adapting normalized luminance (tau_L = 1.5 s) & Weber contrast
    tau_l = 1.5
    mu_l = 1.0 - math.exp(-1.0 / (tau_l * w_fps))
    l_sig = np.maximum(mean_y_rel.astype(np.float64), 0.5 / max(l_max_cd_m2, 1.0))
    adapting = np.empty(n_frames, dtype=np.float64)
    adapting[0] = l_sig[0]
    for i in range(1, n_frames):
        adapting[i] = mu_l * l_sig[i] + (1.0 - mu_l) * adapting[i - 1]
    contrast = (l_sig / np.maximum(adapting, 1e-6)) - 1.0

    # Eq. (8): Gamma energy pooling kernel (kappa=2, lambda=0.15 s, alpha=2)
    kappa = 2.0
    lam = 0.15
    alpha = 2.0
    n_e = max(4, int(math.ceil(1.05 * w_fps)))
    t_e = np.arange(n_e, dtype=np.float64) / w_fps
    h_e = np.exp(-t_e / lam) * (lam ** (-kappa)) * (t_e ** (kappa - 1.0)) / math.gamma(kappa)

    # Eq. (6), (7), (11)-(14): 5-filter causal convolution, energy pooling, & step-mask correlation
    e_bank = np.zeros((5, n_frames), dtype=np.float64)
    f_bank = np.zeros((5, n_frames), dtype=np.float64)
    c_sq = contrast * contrast

    for k, (hk, hk_norm) in enumerate(zip(kernels, kernel_norms)):
        nk = len(hk)
        rk = np.convolve(contrast, hk, mode="full")[:n_frames]
        ek_sq = np.convolve(np.abs(rk) ** alpha, h_e, mode="full")[:n_frames]
        e_bank[k] = np.power(np.maximum(ek_sq, 0.0), 1.0 / alpha)

        # Normalized cross-correlation with non-reversed kernel hk[::-1] in conv form (Eq. 13)
        raw_corr = np.convolve(contrast, hk[::-1], mode="full")[:n_frames]
        c_win_sq = np.convolve(c_sq, np.ones(nk, dtype=np.float64), mode="full")[:n_frames]
        c_norm = np.sqrt(np.maximum(c_win_sq, 0.0))
        denom = hk_norm * c_norm
        n_corr = np.where(denom > 1e-6, raw_corr / np.maximum(denom, 1e-6), 0.0)
        fk_sq = np.convolve(np.abs(n_corr) ** alpha, h_e, mode="full")[:n_frames]
        f_bank[k] = np.power(np.maximum(fk_sq, 0.0), 1.0 / alpha) * math.sqrt(24.0 / w_fps)

    # Eq. (9): Linear interpolation over log10(adapting_nits) + size & frame-rate adjustment g
    log_adapt_nits = np.log10(np.clip(adapting * l_max_cd_m2, 0.2, 500.0))
    c_area = 0.263
    g_adj = ((s0 / s_std) ** (2.0 * c_area)) * (w_fps ** (-1.0 / alpha))

    e_interp = np.empty(n_frames, dtype=np.float64)
    f_interp = np.empty(n_frames, dtype=np.float64)
    for i in range(n_frames):
        la = log_adapt_nits[i]
        e_interp[i] = np.interp(la, log_std_nits, e_bank[:, i]) * g_adj
        f_interp[i] = np.interp(la, log_std_nits, f_bank[:, i])

    # Eq. (10) & Step Masking (mu=33, xi=200, nu=3, delta=1.8)
    mu_w, xi_w, nu_w, delta_step = 33.0, 200.0, 3.0, 1.8
    active = (e_interp > mu_w) & (f_interp >= delta_step)
    z = np.where(active, (e_interp - mu_w) / xi_w, 0.0)
    risk = np.where(active, 100.0 * (1.0 - np.exp(-(z**nu_w))), 0.0)
    return np.clip(risk, 0.0, 100.0).astype(np.float32)


def compute_ppr_and_calibrated_series(
    total_frames: int,
    geometry: ViewingGeometry,
    pse: PSEConfig,
    fps_spec: FPSWindowSpec,
    lum_counts_1s: np.ndarray,
    lum_areas_1s: np.ndarray,
    lum_mags_1s: np.ndarray,
    scene_counts_1s: np.ndarray,
    scene_areas_1s: np.ndarray,
    scene_mags_1s: np.ndarray,
    red_counts_1s: np.ndarray,
    red_areas_1s: np.ndarray,
    red_mags_1s: np.ndarray,
    pat_areas: np.ndarray,
    pat_pairs: np.ndarray,
    pat_cpds: np.ndarray,
    any_danger_mask: np.ndarray,
    b_le: np.ndarray,
    b_re: np.ndarray,
    b_sp: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute Candidate A (`risk_ppr_pct`) and Candidate B (`risk_calibrated`) per frame."""
    area_fov = geometry.area_deg2

    def _p_area(active_frac: np.ndarray) -> np.ndarray:
        theta_stim = 0.5 * np.sqrt(np.maximum(active_frac, 0.0) * area_fov)
        q_cort = 100.0 * (1.0 - np.exp(-0.0574 * theta_stim))
        return np.clip(0.021 * q_cort - 0.1827, 0.0, 1.0)

    # Candidate A: Epidemiological PPR (%)
    f_lum_hz = 0.5 * lum_counts_1s.astype(np.float64)
    p_freq_lum = np.interp(f_lum_hz, _PPR_FREQ_HZ, _PPR_FREQ_SENS)
    p_mod_lum = np.clip(
        np.log10(np.maximum(lum_mags_1s.astype(np.float64), 1.0)) / math.log10(200.0), 0.0, 1.0
    )
    ppr_lum = _p_area(lum_areas_1s) * p_freq_lum * p_mod_lum

    f_red_hz = 0.5 * red_counts_1s.astype(np.float64)
    p_freq_red = np.interp(f_red_hz, _PPR_FREQ_HZ, _PPR_FREQ_SENS)
    p_mod_red = np.clip(red_mags_1s.astype(np.float64) / 0.40, 0.0, 1.0)
    ppr_red = _p_area(red_areas_1s) * p_freq_red * p_mod_red

    p_freq_pat = np.interp(pat_cpds.astype(np.float64), _PPR_CPD_AXIS, _PPR_CPD_SENS)
    p_mod_pat = np.clip(pat_pairs.astype(np.float64) / 12.0, 0.0, 1.0)
    ppr_pat = np.where(pat_pairs > 0, _p_area(pat_areas) * p_freq_pat * p_mod_pat, 0.0)

    risk_ppr = np.clip(
        100.0 * np.maximum(np.maximum(ppr_lum, ppr_red), ppr_pat), 0.0, 100.0
    ).astype(np.float32)

    # Candidate B: Calibrated continuous index (50.0 exact boundary aligned with binary hazard)
    def _chan_ratios(
        mag: np.ndarray,
        mag_thr: float,
        area: np.ndarray,
        area_thr: float,
        freq: np.ndarray,
        freq_thr: float,
    ) -> tuple[np.ndarray, np.ndarray]:
        r_m = np.maximum(mag / max(mag_thr, 1e-6), 0.0)
        r_a = np.maximum(area / max(area_thr, 1e-6), 0.0)
        r_f = np.maximum(freq / max(freq_thr, 1e-6), 0.0)
        eta = np.minimum(np.minimum(r_m, r_a), r_f)
        geom = np.cbrt(r_m * r_a * r_f)
        return eta, geom

    eta_lf, g_lf = _chan_ratios(
        lum_mags_1s,
        pse.delta_lum_min_cd_m2,
        lum_areas_1s,
        pse.flash_area_min,
        lum_counts_1s,
        pse.flash_min_transitions,
    )
    eta_st, g_st = _chan_ratios(
        scene_mags_1s,
        pse.delta_lum_min_cd_m2,
        scene_areas_1s,
        pse.scene_area_min,
        scene_counts_1s,
        pse.scene_min_transitions,
    )
    eta_rf, g_rf = _chan_ratios(
        red_mags_1s,
        pse.delta_uv_min,
        red_areas_1s,
        pse.flash_area_min,
        red_counts_1s,
        pse.flash_min_transitions,
    )

    in_lum_ext = (
        (lum_counts_1s >= pse.extended_band_min_transitions)
        & (lum_counts_1s <= pse.extended_band_max_transitions)
    ).astype(np.int32)
    in_red_ext = (
        (red_counts_1s >= pse.extended_band_min_transitions)
        & (red_counts_1s <= pse.extended_band_max_transitions)
    ).astype(np.int32)
    lum_ext_prog = np.convolve(in_lum_ext, np.ones(fps_spec.w_5s, dtype=np.int32), mode="full")[
        :total_frames
    ] / max(fps_spec.w_4s, 1)
    red_ext_prog = np.convolve(in_red_ext, np.ones(fps_spec.w_5s, dtype=np.int32), mode="full")[
        :total_frames
    ] / max(fps_spec.w_4s, 1)

    eta_le = np.where(b_le, np.maximum(lum_ext_prog, 1.0), np.minimum(lum_ext_prog, 0.98))
    eta_re = np.where(b_re, np.maximum(red_ext_prog, 1.0), np.minimum(red_ext_prog, 0.98))

    pat_consec = np.zeros(total_frames, dtype=np.float64)
    run = 0
    for i in range(total_frames):
        run = (
            (run + 1)
            if (pat_pairs[i] >= pse.pattern_min_pairs and pat_areas[i] > pse.pattern_area_min)
            else 0
        )
        pat_consec[i] = run / max(fps_spec.n_static_0p5s, 1)
    eta_sp, g_sp = _chan_ratios(
        pat_consec, 1.0, pat_areas, pse.pattern_area_min, pat_pairs, pse.pattern_min_pairs
    )
    eta_sp = np.where(b_sp, np.maximum(eta_sp, 1.0), np.minimum(eta_sp, 0.99))

    eta_raw = np.maximum.reduce([eta_lf, eta_st, eta_rf, eta_le, eta_re, eta_sp])
    geom_raw = np.maximum.reduce([g_lf, g_st, g_rf, eta_le, eta_re, g_sp])

    eta_aligned = np.where(any_danger_mask, np.maximum(eta_raw, 1.0), np.minimum(eta_raw, 0.998))
    geom_aligned = np.where(any_danger_mask, np.maximum(geom_raw, 1.0), geom_raw)

    risk_cal = np.where(
        eta_aligned < 1.0,
        50.0 * eta_aligned,
        50.0 + 50.0 * (1.0 - np.exp(-0.8 * np.maximum(geom_aligned - 1.0, 0.0))),
    )
    return risk_ppr, np.clip(risk_cal, 0.0, 100.0).astype(np.float32)
