"""
Değerlendirme metrikleri: sınıflama + fiziksel tutarlılık.

Fiziksel tutarlılık (physics consistency)
-----------------------------------------
Bir heyelan modeli, diğer her şey sabitken PGA arttığında P(heyelan)'ı
düşürmemelidir (Newmark: Dn, PGA ile monoton artar). Klasik ML modelleri bu
kısıtı garanti etmez. `pga_monotonicity` her örnek için PGA'yı bir ızgarada
tarar (Ky sabit, Dn Jibson 2007 ile yeniden hesaplanır) ve P'nin düştüğü
örneklerin oranını ölçer.
"""
from __future__ import annotations

from typing import Callable

import numpy as np
from sklearn.metrics import (
    roc_auc_score, average_precision_score, f1_score, precision_score,
    recall_score, accuracy_score, brier_score_loss, cohen_kappa_score,
)

from ..data_engineering.feature_matrix import DN_LOG_OFFSET_CM
from ..geomechanics.newmark import NewmarkDisplacement

PGA_SWEEP = np.round(np.arange(0.05, 1.201, 0.05), 3)


def best_f1_threshold(y: np.ndarray, p: np.ndarray, lo=0.05, hi=0.95, step=0.01) -> float:
    grid = np.arange(lo, hi + 1e-9, step)
    f1s = [f1_score(y, (p >= t).astype(int), zero_division=0) for t in grid]
    return float(grid[int(np.argmax(f1s))])


def classification_metrics(
    y: np.ndarray,
    p: np.ndarray,
    threshold: float,
    slope: np.ndarray | None = None,
    steep_deg: float = 15.0,
) -> dict:
    """AUC, PR-AUC, F1/precision/recall @eşik, Brier, Kappa + dik-arazi AUC."""
    y = np.asarray(y).astype(int)
    p = np.asarray(p, dtype=float)
    yhat = (p >= threshold).astype(int)
    out = {
        "roc_auc":   float(roc_auc_score(y, p)),
        "pr_auc":    float(average_precision_score(y, p)),
        "f1":        float(f1_score(y, yhat, zero_division=0)),
        "precision": float(precision_score(y, yhat, zero_division=0)),
        "recall":    float(recall_score(y, yhat, zero_division=0)),
        "accuracy":  float(accuracy_score(y, yhat)),
        "kappa":     float(cohen_kappa_score(y, yhat)),
        "brier":     float(brier_score_loss(y, np.clip(p, 0, 1))),
        "threshold": float(threshold),
    }
    # "Zor" alt küme: yalnız dik araziler. Rastgele arka plan negatiflerinin çoğu
    # düzlüktür; eğim tek başına yüksek AUC verir. Dik arazide ayırma gücü,
    # modelin eğim dışında ne öğrendiğini gösterir.
    if slope is not None:
        m = np.asarray(slope) >= steep_deg
        if m.sum() > 50 and 0 < y[m].mean() < 1:
            out["roc_auc_steep"] = float(roc_auc_score(y[m], p[m]))
            out["n_steep"] = int(m.sum())
    return out


def set_pga(X: np.ndarray, feat_cols: list[str], pga: float | np.ndarray) -> np.ndarray:
    """Ham öznitelik matrisinde PGA'yı değiştirir ve log Dn'yi tutarlı günceller."""
    Xs = X.copy()
    ip = feat_cols.index("pga_g")
    Xs[:, ip] = pga
    if "log_dn_cm" in feat_cols or "dn_cm" in feat_cols:
        ky = X[:, feat_cols.index("ky_g")].astype(float)
        dn_cm = NewmarkDisplacement(ky, np.broadcast_to(pga, ky.shape)) * 100.0
        if "log_dn_cm" in feat_cols:
            Xs[:, feat_cols.index("log_dn_cm")] = np.log10(dn_cm + DN_LOG_OFFSET_CM)
        if "dn_cm" in feat_cols:
            Xs[:, feat_cols.index("dn_cm")] = dn_cm
    return Xs


def pga_response(
    predict: Callable[[np.ndarray], np.ndarray],
    X: np.ndarray,
    feat_cols: list[str],
    grid: np.ndarray = PGA_SWEEP,
) -> np.ndarray:
    """(n_grid, n_samples) P matrisi: her PGA değeri için tahmin."""
    return np.stack([predict(set_pga(X, feat_cols, pg)) for pg in grid])


def pga_monotonicity(P: np.ndarray, tol: float = 0.01) -> dict:
    """
    PGA taramasındaki monotonluk ihlalleri.

    violation_rate : P'nin herhangi bir adımda > tol düştüğü örnek oranı
    mean_drop      : örnek başına toplam düşüş miktarının ortalaması
    """
    d = np.diff(P, axis=0)
    drops = np.clip(-d, 0, None)
    return {
        "mono_violation_rate": float((drops.max(axis=0) > tol).mean()),
        "mono_mean_drop": float(drops.sum(axis=0).mean()),
    }


def critical_pga_from_curve(grid: np.ndarray, curve: np.ndarray, level: float) -> float | None:
    """Eğrinin `level`'i ilk geçtiği PGA (doğrusal enterpolasyonla)."""
    over = np.where(curve >= level)[0]
    if len(over) == 0:
        return None
    i = int(over[0])
    if i == 0:
        return float(grid[0])
    x0, x1, y0, y1 = grid[i - 1], grid[i], curve[i - 1], curve[i]
    if y1 == y0:
        return float(x1)
    return float(x0 + (level - y0) * (x1 - x0) / (y1 - y0))
