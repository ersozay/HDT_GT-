"""
Fizik yolları üzerinde kolokasyon çiftleri (physics collocation pairs).

PINN'lerde (Raissi ve ark. 2019) fizik kısıtı, gözlem noktalarına ek olarak
alanın tamamına yayılmış "kolokasyon noktalarında" uygulanır. Burada fizik,
Mohr-Coulomb sonsuz şev + Newmark (Jibson 2007) zinciridir; ondan üç kesin
işaret kısıtı çıkar:

  (1) PGA yolu   : diğer her şey sabit, PGA ↑  ⇒ Dn ↑            ⇒ P ↓ olamaz
  (2) Eğim yolu  : litoloji sabit,  θ ↑        ⇒ FS ↓, Ky ↓, Dn ↑ ⇒ P ↓ olamaz
  (3) Doygunluk yolu: yağışla su tablası oranı m ↑ ⇒ u ↑ ⇒ FS ↓, Ky ↓, Dn ↑
                      ⇒ P ↓ olamaz  (u = m·γ_w·z·cos²θ; bkz. geomechanics/hydrology.py)

Her çift (x_a, x_b) bir yol üzerinde küçük bir adımdır; Ky ve Dn fizik
denklemleriyle YENİDEN hesaplanır (öznitelikler birbirinden bağımsız
oynatılmaz). Kayıp:  L_path = mean( ReLU(P(x_a) − P(x_b)) ).

Kolokasyon PGA'ları gözlenen aralıktan geniş (0.02–1.30 g) örneklenir; böylece
Kritik PGA taramasının gözlem dışı (yüksek PGA) bölgesi de fizikle kısıtlanır.
"""
from __future__ import annotations

import numpy as np

from ..data_engineering.feature_matrix import DN_LOG_OFFSET_CM
from ..geomechanics.mohr_coulomb import StaticFS, CriticalAcceleration
from ..geomechanics.newmark import NewmarkDisplacement


class PhysicsPathSampler:
    """
    Ham öznitelik matrisinden normalize kolokasyon çiftleri üretir.

    Parametreler
    -----------
    X_raw     : (n, d) eğitim öznitelikleri (ham)
    feat_cols : sütun adları (pga_g, slope_deg, ky_g, log_dn_cm, cohesion_kpa,
                friction_deg, unit_weight gerekir)
    mean, std : eğitim ölçekleyicisi
    z_ref     : sonsuz şev hesap derinliği (m)
    """

    def __init__(self, X_raw, feat_cols, mean, std, z_ref: float = 5.0,
                 pga_range=(0.02, 1.30), slope_range=(0.0, 50.0), seed: int = 0):
        self.X = np.asarray(X_raw, dtype=np.float64)
        self.c = {k: feat_cols.index(k) for k in
                  ("pga_g", "slope_deg", "ky_g", "log_dn_cm",
                   "cohesion_kpa", "friction_deg", "unit_weight")}
        self.mean = np.asarray(mean, dtype=np.float64)
        self.std = np.asarray(std, dtype=np.float64)
        self.z_ref = z_ref
        self.pga_range = pga_range
        self.slope_range = slope_range
        self.rng = np.random.RandomState(seed)

    # -- fizik zinciri --------------------------------------------------- #
    def _set_slope(self, X, slope, m=0.0):
        from ..geomechanics.hydrology import pore_pressure
        c = self.c
        u = pore_pressure(m, self.z_ref, slope)
        fs = StaticFS(X[:, c["cohesion_kpa"]], X[:, c["friction_deg"]],
                      X[:, c["unit_weight"]], self.z_ref, slope, u)
        X[:, c["slope_deg"]] = slope
        X[:, c["ky_g"]] = CriticalAcceleration(fs, slope)

    def _update_dn(self, X):
        c = self.c
        dn_cm = NewmarkDisplacement(X[:, c["ky_g"]], X[:, c["pga_g"]]) * 100.0
        X[:, c["log_dn_cm"]] = np.log10(dn_cm + DN_LOG_OFFSET_CM)

    def _norm(self, X):
        return ((X - self.mean) / self.std).astype(np.float32)

    # -- çift üretimi ---------------------------------------------------- #
    def sample(self, n: int) -> tuple[np.ndarray, np.ndarray]:
        """n çift: ⅓ PGA yolu, ⅓ eğim yolu, ⅓ doygunluk yolu → (Xa_n, Xb_n)."""
        idx = self.rng.randint(0, len(self.X), size=n)
        Xa = self.X[idx].copy()
        Xb = Xa.copy()
        c = self.c
        n1, n2 = n // 3, 2 * n // 3

        # (1) PGA yolu
        lo, hi = self.pga_range
        pa = self.rng.uniform(lo, hi, size=n1)
        Xa[:n1, c["pga_g"]] = pa
        Xb[:n1, c["pga_g"]] = pa + self.rng.uniform(0.02, 0.15, size=n1)

        # (2) Eğim yolu (PGA gözlenen değerinde, kuru)
        slo, shi = self.slope_range
        sa = self.rng.uniform(slo, shi, size=n2 - n1)
        sb = np.minimum(sa + self.rng.uniform(1.0, 6.0, size=n2 - n1), 89.0)
        A, B = Xa[n1:n2], Xb[n1:n2]
        self._set_slope(A, sa); self._set_slope(B, sb)
        Xa[n1:n2], Xb[n1:n2] = A, B

        # (3) Doygunluk yolu (eğim gözlenen değerinde)
        k = n - n2
        ma = self.rng.uniform(0.0, 0.85, size=k)
        mb = np.minimum(ma + self.rng.uniform(0.05, 0.3, size=k), 1.0)
        A, B = Xa[n2:], Xb[n2:]
        th = A[:, c["slope_deg"]].copy()
        self._set_slope(A, th, ma); self._set_slope(B, th, mb)
        Xa[n2:], Xb[n2:] = A, B

        self._update_dn(Xa)
        self._update_dn(Xb)
        return self._norm(Xa), self._norm(Xb)
