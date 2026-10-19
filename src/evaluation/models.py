"""
Karşılaştırmalı modeller — ortak arayüz: fit(...) → self, predict(X_raw) → P.

Tüm modeller HAM (normalize edilmemiş) öznitelik matrisi alır; ölçekleme
gereken modeller ölçekleyiciyi yalnız eğitim katmanından öğrenir.

Modeller
--------
  newmark    : Saf fizik — skor = log Dn (öğrenme yok), eşik doğrulamada seçilir
  logreg     : Lojistik regresyon (doğrusal istatistiksel referans)
  rf         : Random Forest
  gbm        : Gradyan artırmalı karar ağaçları (scikit-learn HistGradientBoosting,
               LightGBM/XGBoost eşdeğeri; tablo verisinde en güçlü referans)
  gbm_mono   : Aynı GBM + fiziksel monoton kısıtlar (PGA↑, Dn↑, eğim↑, Ky↓)
  mlp        : Aynı mimari, yalnız Focal Loss (fizik yok)
  pinn_v1    : Focal + stabilite cezası (tezdeki ilk sürüm)
  pinn_v2    : Focal + stabilite + fizik-yolu (PGA, eğim) kolokasyon kısıtı (önerilen)
"""
from __future__ import annotations

import tempfile
from dataclasses import dataclass, field

import numpy as np

from ..data_engineering.feature_matrix import MONOTONE_FEATURES


def _standardize_fit(X):
    mean = X.mean(axis=0)
    std = X.std(axis=0)
    std = np.where(std < 1e-8, 1.0, std)
    return mean.astype(np.float32), std.astype(np.float32)


@dataclass
class NewmarkScore:
    feat_cols: list[str]
    name: str = "newmark"

    def fit(self, X, y, X_va=None, y_va=None):
        self._i = self.feat_cols.index("log_dn_cm")
        lo, hi = X[:, self._i].min(), X[:, self._i].max()
        self._lo, self._hi = float(lo), float(hi)
        return self

    def predict(self, X):
        # Monoton dönüşüm: AUC'yi değiştirmez; [0,1] aralığına taşır.
        return np.clip((X[:, self._i] - self._lo) / (self._hi - self._lo + 1e-9), 0, 1)


@dataclass
class SklearnModel:
    feat_cols: list[str]
    name: str
    params: dict = field(default_factory=dict)

    def fit(self, X, y, X_va=None, y_va=None):
        if self.name == "logreg":
            from sklearn.linear_model import LogisticRegression
            self._mean, self._std = _standardize_fit(X)
            self._m = LogisticRegression(max_iter=2000, C=1.0)
            self._m.fit((X - self._mean) / self._std, y)
        elif self.name == "rf":
            from sklearn.ensemble import RandomForestClassifier
            self._m = RandomForestClassifier(
                n_estimators=400, min_samples_leaf=5, max_features="sqrt",
                n_jobs=-1, random_state=42)
            self._m.fit(X, y)
        elif self.name in ("gbm", "gbm_mono"):
            from sklearn.ensemble import HistGradientBoostingClassifier
            kw = dict(max_iter=1500, learning_rate=0.05, max_leaf_nodes=63,
                      min_samples_leaf=40, l2_regularization=1.0,
                      early_stopping=False, random_state=42)
            if self.name == "gbm_mono":
                kw["monotonic_cst"] = [MONOTONE_FEATURES.get(c, 0) for c in self.feat_cols]
            kw.update(self.params)
            # Erken durdurma: iterasyon sayısı blok-bazlı doğrulama kümesinde seçilir
            # (sklearn'ün dahili rastgele ayrımı mekânsal sızıntı yaratırdı).
            m = HistGradientBoostingClassifier(**kw).fit(X, y)
            if X_va is not None:
                from sklearn.metrics import log_loss
                losses = [log_loss(y_va, p[:, 1]) for p in m.staged_predict_proba(X_va)]
                best = int(np.argmin(losses)) + 1
                kw["max_iter"] = best
                m = HistGradientBoostingClassifier(**kw).fit(X, y)
            self._m = m
        else:
            raise ValueError(self.name)
        return self

    def predict(self, X):
        if self.name == "logreg":
            return self._m.predict_proba((X - self._mean) / self._std)[:, 1]
        return self._m.predict_proba(X)[:, 1]


@dataclass
class NeuralModel:
    """LandslidePINN sarmalayıcısı (mlp / pinn_v1 / pinn_v2)."""
    feat_cols: list[str]
    name: str = "pinn_v2"
    epochs: int = 120
    lambda_mono: float = 12.0
    device: str = "cpu"
    seed: int = 42
    verbose: bool = False
    checkpoint_dir: str | None = None

    def _loss(self):
        from config import (FOCAL_ALPHA, FOCAL_GAMMA, PHYS_SLOPE_THRESHOLD,
                            PHYS_PGA_THRESHOLD, TRAIN_CONFIG)
        from ..pinn.loss import PhysicsInformedLoss
        use_phys = self.name in ("pinn_v1", "pinn_v2")
        return PhysicsInformedLoss(
            lambda_phys=TRAIN_CONFIG["lambda_phys"] if use_phys else 0.0,
            alpha=FOCAL_ALPHA, gamma=FOCAL_GAMMA,
            slope_threshold_deg=PHYS_SLOPE_THRESHOLD,
            pga_threshold_g=PHYS_PGA_THRESHOLD,
            lambda_mono=self.lambda_mono if self.name == "pinn_v2" else 0.0,
        ), use_phys

    def fit(self, X, y, X_va, y_va):
        from config import TRAIN_CONFIG
        from ..pinn.model import LandslidePINN
        from ..pinn.trainer import PINNTrainer, TrainConfig

        self._mean, self._std = _standardize_fit(X)
        Xn = (X - self._mean) / self._std
        Xvn = (X_va - self._mean) / self._std
        isl, ipg = self.feat_cols.index("slope_deg"), self.feat_cols.index("pga_g")

        loss_fn, use_phys = self._loss()
        ckdir = self.checkpoint_dir or tempfile.mkdtemp(prefix=f"{self.name}_")
        cfg = TrainConfig(
            epochs=self.epochs, batch_size=TRAIN_CONFIG["batch_size"],
            lr=TRAIN_CONFIG["lr"], weight_decay=TRAIN_CONFIG["weight_decay"],
            patience=TRAIN_CONFIG["patience"], checkpoint_dir=ckdir,
            lambda_phys=loss_fn.lambda_phys, use_lambda_annealing=use_phys,
            max_lambda=TRAIN_CONFIG["max_lambda"] if use_phys else 0.0,
            device=self.device, seed=self.seed, verbose=self.verbose,
        )
        model = LandslidePINN(in_features=X.shape[1])
        self.trainer = PINNTrainer(model, loss_fn, cfg)
        self.trainer.metadata = {
            "feat_cols": list(self.feat_cols),
            "feature_medians": {c: float(np.median(X[:, i])) for i, c in enumerate(self.feat_cols)},
            "model_kind": self.name,
            "scaler": {"mean": self._mean, "std": self._std},
        }
        sampler = None
        if self.name == "pinn_v2":
            from config import Z_REF_M
            from ..pinn.physics_paths import PhysicsPathSampler
            sampler = PhysicsPathSampler(X, self.feat_cols, self._mean, self._std,
                                         z_ref=Z_REF_M, seed=self.seed)
        self.history = self.trainer.fit(
            Xn.astype(np.float32), y.astype(np.float32),
            X[:, isl].astype(np.float32), X[:, ipg].astype(np.float32),
            Xvn.astype(np.float32), y_va.astype(np.float32),
            X_va[:, isl].astype(np.float32), X_va[:, ipg].astype(np.float32),
            path_sampler=sampler,
        )
        self.model = self.trainer.load_best().to("cpu").eval()
        return self

    def predict(self, X, batch: int = 65536):
        import torch
        Xn = ((X - self._mean) / self._std).astype(np.float32)
        out = []
        with torch.no_grad():
            for i in range(0, len(Xn), batch):
                out.append(self.model(torch.from_numpy(Xn[i:i + batch])).numpy().reshape(-1))
        return np.concatenate(out) if out else np.zeros(0)


def make_model(name: str, feat_cols: list[str], **kw):
    if name == "newmark":
        return NewmarkScore(feat_cols)
    if name in ("logreg", "rf", "gbm", "gbm_mono"):
        return SklearnModel(feat_cols, name)
    if name in ("mlp", "pinn_v1", "pinn_v2"):
        return NeuralModel(feat_cols, name, **kw)
    raise ValueError(f"Bilinmeyen model: {name}")
