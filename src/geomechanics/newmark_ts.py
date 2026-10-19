"""
Newmark (1965) rijit-blok analizi — ZAMAN-TANIM alanında (gerçek ivme kaydı ile).

Jibson (1993) algoritması: blok, yamaç aşağı yönde yer ivmesi a(t) kritik ivme
Kᵧ'yi aştığında kaymaya başlar; göreli hız

    v̇ = (a(t) − Kᵧ)·g    (kayma sürerken)

ile integre edilir, v ≤ 0 olduğunda blok durur. Kalıcı yer değiştirme
Dₙ = ∫ v dt. Kayma tek yönlüdür (yamaç aşağı); kaydın her iki polaritesi ve
her iki yatay bileşeni ayrı ayrı uygulanır, en büyüğü raporlanır.

Bu çözüm, Jibson (2007) regresyon denklemlerinin aksine kaydın süresini,
frekans içeriğini ve darbe dizilimini doğrudan hesaba katar.

Referanslar:
  Newmark N.M. (1965). Géotechnique 15(2), 139–160.
  Jibson R.W. (1993). Transportation Research Record 1411, 9–17.
  Jibson R.W. & Jibson M.W. (2003). Java programs for using Newmark's method. USGS OFR 03-005.
"""
from __future__ import annotations

import numpy as np

G = 9.80665


def rigid_block(acc_g: np.ndarray, dt: float, ky_g: np.ndarray, scale: np.ndarray | float = 1.0,
                history_step_s: float | None = None):
    """
    Çok sayıda blok için (vektörel) rijit-blok integrasyonu.

    acc_g   : (T,) ivme kaydı (g), pozitif = yamaç aşağı
    ky_g    : (N,) kritik ivmeler (g)
    scale   : (N,) ya da skaler — kayıt ölçek çarpanı (ör. yerel PGA / kayıt PGA'sı)
    history_step_s : verilirse her bu kadar saniyede bir yer değiştirme anlık görüntüsü

    Döndürür: (Dn_m (N,), times (K,), history_m (K, N) | None)
    """
    ky = np.asarray(ky_g, dtype=np.float64).ravel()
    sc = np.broadcast_to(np.asarray(scale, dtype=np.float64), ky.shape)
    v = np.zeros_like(ky)
    d = np.zeros_like(ky)
    every = int(round(history_step_s / dt)) if history_step_s else 0
    hist, times = [], []
    for i, a in enumerate(acc_g):
        ar = (a * sc - ky) * G                     # göreli ivme (m/s²)
        sliding = (v > 0) | (ar > 0)
        v = np.where(sliding, np.maximum(v + ar * dt, 0.0), 0.0)
        d += v * dt
        if every and i % every == 0:
            hist.append(d.copy()); times.append(i * dt)
    if every:
        hist.append(d.copy()); times.append(len(acc_g) * dt)
        return d, np.asarray(times), np.vstack(hist)
    return d, None, None


def rigid_block_both_directions(components: list[np.ndarray], dt: float, ky_g, scale=1.0,
                                history_step_s: float | None = None):
    """
    Her yatay bileşenin iki polaritesi için çözer; blok başına en büyük Dₙ'yi
    (ve istenirse o çözümün zaman geçmişini) döndürür.
    """
    best_d, best_h, times = None, None, None
    for comp in components:
        for sgn in (1.0, -1.0):
            d, t, h = rigid_block(sgn * comp, dt, ky_g, scale, history_step_s)
            if best_d is None:
                best_d, best_h, times = d, h, t
                continue
            better = d > best_d
            best_d = np.where(better, d, best_d)
            if h is not None:
                n = min(len(h), len(best_h))
                best_h = np.where(better[None, :], h[:n], best_h[:n])
                times = times[:n]
    return best_d, times, best_h
