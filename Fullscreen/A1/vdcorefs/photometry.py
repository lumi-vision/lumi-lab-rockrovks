"""Device selection, IEC 61966-2-1 sRGB linearization, luminance, and CIE 1976 u'v' conversion."""

from __future__ import annotations

import numpy as np
import torch

from .config import PSEConfig


def resolve_device(device: str | torch.device | None = None) -> torch.device:
    """Select CUDA automatically when available, falling back cleanly to CPU."""
    if device is not None:
        resolved = torch.device(device)
        if resolved.type == "cuda" and not torch.cuda.is_available():
            return torch.device("cpu")
        return resolved
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _build_srgb_lut(device: torch.device) -> torch.Tensor:
    """Exact 256-entry IEC 61966-2-1 8-bit sRGB -> linear light lookup table."""
    c_s = np.arange(256, dtype=np.float64) / 255.0
    c_lin = np.where(c_s <= 0.04045, c_s / 12.92, ((c_s + 0.055) / 1.055) ** 2.4)
    return torch.tensor(c_lin, dtype=torch.float32, device=device)


class PhotometricConverter:
    """Converts BGR frames into linear RGB, physical luminance L, CIE 1976 u'v', and red masks."""

    def __init__(
        self,
        l_max_cd_m2: float = 200.0,
        pse: PSEConfig = PSEConfig(),
        device: torch.device | None = None,
    ) -> None:
        self.l_max_cd_m2 = float(l_max_cd_m2)
        self.pse = pse
        self.device = resolve_device(device)
        self._srgb_lut = _build_srgb_lut(self.device)

    def convert_bgr(
        self, bgr: np.ndarray | torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return `(rgb_lin, lum, u_prime, v_prime, is_sat_red)` on `self.device`.

        Shapes:
            rgb_lin    : (H, W, 3) float32 in [0, 1]
            lum        : (H, W) float32 in [0, l_max_cd_m2]
            u_prime    : (H, W) float32
            v_prime    : (H, W) float32
            is_sat_red : (H, W) bool
        """
        if isinstance(bgr, np.ndarray):
            t_bgr = torch.from_numpy(np.ascontiguousarray(bgr)).to(self.device)
        else:
            t_bgr = bgr.to(self.device)

        if t_bgr.dtype == torch.uint8:
            # BGR -> RGB while indexing the exact 256-entry IEC 61966-2-1 LUT
            t_rgb_u8 = t_bgr.flip(-1).long()
            rgb_lin = self._srgb_lut[t_rgb_u8]
        else:
            c_s = torch.clamp(t_bgr.flip(-1).to(torch.float32), 0.0, 1.0)
            rgb_lin = torch.where(
                c_s <= 0.04045,
                c_s / 12.92,
                torch.pow((c_s + 0.055) / 1.055, 2.4),
            )

        r_lin = rgb_lin[..., 0]
        g_lin = rgb_lin[..., 1]
        b_lin = rgb_lin[..., 2]

        y_rel = 0.2126 * r_lin + 0.7152 * g_lin + 0.0722 * b_lin
        lum = self.l_max_cd_m2 * y_rel

        # CIE 1931 XYZ (D65) -> CIE 1976 UCS (u', v') with D65 floor for D <= 1e-4
        x_val = 0.4124564 * r_lin + 0.3575761 * g_lin + 0.1804375 * b_lin
        y_val = 0.2126729 * r_lin + 0.7151522 * g_lin + 0.0721750 * b_lin
        z_val = 0.0193339 * r_lin + 0.1191920 * g_lin + 0.9503041 * b_lin

        denom = x_val + 15.0 * y_val + 3.0 * z_val
        valid_uv = denom > self.pse.uv_denom_floor
        safe_denom = torch.where(valid_uv, denom, torch.ones_like(denom))
        u_prime = torch.where(valid_uv, (4.0 * x_val) / safe_denom, self.pse.d65_u_prime)
        v_prime = torch.where(valid_uv, (9.0 * y_val) / safe_denom, self.pse.d65_v_prime)

        # Method A saturated red: ratio >= 0.8 AND excess R_lin - G_lin - B_lin >= 0.0625
        rgb_sum = torch.clamp(r_lin + g_lin + b_lin, min=1e-6)
        is_sat_red = ((r_lin / rgb_sum) >= self.pse.sat_red_ratio_min) & (
            (r_lin - g_lin - b_lin) >= self.pse.sat_red_excess_min
        )
        return rgb_lin, lum, u_prime, v_prime, is_sat_red
