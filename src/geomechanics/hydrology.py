"""
Yağış → boşluk suyu basıncı → kritik ivme (sonsuz şev, yamaca paralel sızma).

Su tablası kayma yüzeyinin m·z kadar üzerinde ve yamaca paralel kabul edilir
(Skempton & DeLory 1957; Iverson 2000'in uzun süreli sızma limiti):

    u = m · γ_w · z · cos²θ          (0 ≤ m ≤ 1, m = 1 → yüzeye kadar doygun)

Bu u, StaticFS'e girer: FS = [c' + (γ z cos²θ − u) tanφ'] / (γ z sinθ cosθ).
Kᵧ = (FS − 1) sinθ; Dₙ (Newmark) buna göre artar.

m'nin olay için tahmini: ERA5-Land kök bölgesi (0–100 cm) hacimsel toprak nemi
θ'nın doygunluk nemine oranı (S = θ/θ_s). Bu, su tablası oranı için basit ve
şeffaf bir VEKİLDİR; sonuçlar m = 0…1 duyarlılık taramasıyla birlikte verilir.

Referanslar:
  Skempton A.W., DeLory F.A. (1957). Stability of natural slopes in London Clay.
  Iverson R.M. (2000). Landslide triggering by rain infiltration. Water Resour. Res. 36(7).
  Muñoz-Sabater J. ve ark. (2021). ERA5-Land. Earth Syst. Sci. Data 13: 4349–4383.
"""
from __future__ import annotations

import numpy as np

GAMMA_W = 9.81          # kN/m³


def pore_pressure(m, z, theta_deg):
    """Yamaca paralel sızmada kayma yüzeyindeki boşluk suyu basıncı (kPa)."""
    m = np.clip(np.asarray(m, dtype=float), 0.0, 1.0)
    return m * GAMMA_W * np.asarray(z, dtype=float) * np.cos(np.deg2rad(theta_deg)) ** 2


def saturation_from_moisture(theta_vol, theta_sat: float = 0.45):
    """Hacimsel nemden doygunluk derecesi S = θ/θ_s (0–1)."""
    return np.clip(np.asarray(theta_vol, dtype=float) / theta_sat, 0.0, 1.0)
