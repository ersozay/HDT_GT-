"""
PINN Eğitim Pipeline'ı.

Özellikler:
  - AdamW optimizer + CosineAnnealingLR
  - Early stopping (patience tabanlı)
  - Val F1, ROC-AUC metrikleri
  - Model checkpoint kaydı
  - SHAP değer hesaplama desteği
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from pathlib import Path
from dataclasses import dataclass, field


@dataclass
class TrainConfig:
    """Eğitim hiper-parametreleri."""
    epochs: int = 100
    batch_size: int = 512
    lr: float = 1e-3
    weight_decay: float = 1e-4
    patience: int = 15
    min_delta: float = 1e-4
    checkpoint_dir: str = "outputs/models"
    device: str = "auto"
    lambda_phys: float = 1.0
    use_lambda_annealing: bool = True
    max_lambda: float = 5.0


@dataclass
class TrainHistory:
    """Eğitim tarihçesi."""
    train_loss: list[float] = field(default_factory=list)
    val_loss: list[float]   = field(default_factory=list)
    val_f1: list[float]     = field(default_factory=list)
    val_auc: list[float]    = field(default_factory=list)
    focal_loss: list[float] = field(default_factory=list)
    phys_loss: list[float]  = field(default_factory=list)


class PINNTrainer:
    """
    LandslidePINN eğitim sınıfı.

    Kullanım
    --------
    trainer = PINNTrainer(model, loss_fn, config)
    history = trainer.fit(X_train, y_train, slope_train, pga_train,
                          X_val, y_val, slope_val, pga_val)
    """

    def __init__(
        self,
        model: nn.Module,
        loss_fn: nn.Module,
        config: TrainConfig | None = None,
    ):
        self.config = config or TrainConfig()

        if self.config.device == "auto":
            self.device = torch.device(
                "cuda" if torch.cuda.is_available()
                else "mps" if torch.backends.mps.is_available()
                else "cpu"
            )
        else:
            self.device = torch.device(self.config.device)

        self.model   = model.to(self.device)
        self.loss_fn = loss_fn

        self.optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=self.config.lr,
            weight_decay=self.config.weight_decay,
        )
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=self.config.epochs,
            eta_min=1e-6,
        )

        self.checkpoint_dir = Path(self.config.checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

    def _to_tensor(self, arr: np.ndarray, dtype=torch.float32) -> torch.Tensor:
        return torch.tensor(arr, dtype=dtype).to(self.device)

    def _make_dataloader(
        self,
        X: np.ndarray,
        y: np.ndarray,
        slope: np.ndarray,
        pga: np.ndarray,
        shuffle: bool = True,
    ) -> DataLoader:
        dataset = TensorDataset(
            self._to_tensor(X),
            self._to_tensor(y),
            self._to_tensor(slope),
            self._to_tensor(pga),
        )
        return DataLoader(dataset, batch_size=self.config.batch_size, shuffle=shuffle)

    def fit(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        slope_train: np.ndarray,
        pga_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
        slope_val: np.ndarray,
        pga_val: np.ndarray,
    ) -> TrainHistory:
        """
        Modeli eğitir.

        Parametreler
        -----------
        X_train, X_val   : Öznitelik matrisleri
        y_train, y_val   : Binary hedef (0/1)
        slope_train/val  : Eğim değerleri (fizik penaltı için)
        pga_train/val    : PGA değerleri (fizik penaltı için)

        Döndürür
        --------
        TrainHistory
        """
        train_loader = self._make_dataloader(X_train, y_train, slope_train, pga_train)
        history = TrainHistory()

        best_val_loss = float("inf")
        patience_counter = 0
        best_epoch = 0

        print(f"Cihaz: {self.device}")
        print(f"Model parametreleri: {self.model.count_parameters():,}")
        print(f"Eğitim örnekleri: {len(X_train):,} | Val örnekleri: {len(X_val):,}")
        print("-" * 60)

        for epoch in range(1, self.config.epochs + 1):
            # Lambda annealing
            if self.config.use_lambda_annealing:
                self.loss_fn.anneal_lambda(epoch, self.config.epochs, self.config.max_lambda)

            # --- Train ---
            self.model.train()
            train_losses, focal_losses, phys_losses = [], [], []

            for X_b, y_b, slope_b, pga_b in train_loader:
                self.optimizer.zero_grad()
                p_hat = self.model(X_b)
                loss, components = self.loss_fn(p_hat, y_b, slope_b, pga_b)
                loss.backward()
                nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                self.optimizer.step()

                train_losses.append(components["total"])
                focal_losses.append(components["focal"])
                phys_losses.append(components["physics"])

            # --- Validation ---
            val_loss, val_f1, val_auc = self._evaluate(X_val, y_val, slope_val, pga_val)
            self.scheduler.step()

            history.train_loss.append(np.mean(train_losses))
            history.val_loss.append(val_loss)
            history.val_f1.append(val_f1)
            history.val_auc.append(val_auc)
            history.focal_loss.append(np.mean(focal_losses))
            history.phys_loss.append(np.mean(phys_losses))

            # Early stopping
            if val_loss < best_val_loss - self.config.min_delta:
                best_val_loss    = val_loss
                patience_counter = 0
                best_epoch       = epoch
                self._save_checkpoint(epoch, val_loss, val_f1, val_auc)
            else:
                patience_counter += 1

            if epoch % 10 == 0 or epoch == 1:
                print(
                    f"Epoch {epoch:4d}/{self.config.epochs} | "
                    f"Train={np.mean(train_losses):.4f} | "
                    f"Val={val_loss:.4f} | "
                    f"F1={val_f1:.4f} | "
                    f"AUC={val_auc:.4f} | "
                    f"λ={self.loss_fn.lambda_phys:.2f}"
                )

            if patience_counter >= self.config.patience:
                print(f"\nEarly stopping: epoch {best_epoch} en iyi (val_loss={best_val_loss:.4f})")
                break

        return history

    def _evaluate(
        self,
        X: np.ndarray,
        y: np.ndarray,
        slope: np.ndarray,
        pga: np.ndarray,
    ) -> tuple[float, float, float]:
        """Val loss, F1, ROC-AUC hesapla."""
        from sklearn.metrics import f1_score, roc_auc_score

        self.model.eval()
        with torch.no_grad():
            X_t     = self._to_tensor(X)
            y_t     = self._to_tensor(y)
            slope_t = self._to_tensor(slope)
            pga_t   = self._to_tensor(pga)

            p_hat = self.model(X_t)
            loss, _ = self.loss_fn(p_hat, y_t, slope_t, pga_t)
            proba   = p_hat.squeeze().cpu().numpy()

        y_pred = (proba >= 0.5).astype(int)
        f1  = f1_score(y, y_pred, zero_division=0)
        try:
            auc = roc_auc_score(y, proba)
        except ValueError:
            auc = 0.5

        return loss.item(), f1, auc

    def _save_checkpoint(self, epoch: int, val_loss: float, f1: float, auc: float):
        path = self.checkpoint_dir / "best_model.pt"
        torch.save({
            "epoch":     epoch,
            "model_state": self.model.state_dict(),
            "optimizer":   self.optimizer.state_dict(),
            "val_loss":    val_loss,
            "val_f1":      f1,
            "val_auc":     auc,
        }, path)

    def load_best(self) -> nn.Module:
        """En iyi checkpoint'i yükle."""
        path = self.checkpoint_dir / "best_model.pt"
        ckpt = torch.load(path, map_location=self.device)
        self.model.load_state_dict(ckpt["model_state"])
        print(f"En iyi model yüklendi — epoch={ckpt['epoch']}, "
              f"F1={ckpt['val_f1']:.4f}, AUC={ckpt['val_auc']:.4f}")
        return self.model

    def compute_shap(
        self,
        X_background: np.ndarray,
        X_explain: np.ndarray,
        feature_names: list[str],
        n_background: int = 100,
    ):
        """
        SHAP (SHapley Additive exPlanations) değerlerini hesaplar.

        Gereksinim: pip install shap

        Parametreler
        -----------
        X_background : Referans veri seti (train setinden örnek)
        X_explain    : Açıklanacak örnekler
        feature_names: Öznitelik isimleri
        n_background : Kullanılacak arka plan örnek sayısı

        Döndürür
        --------
        shap_values : (n_explain, n_features) ndarray
        """
        try:
            import shap

            self.model.eval()

            def predict_fn(X_np):
                X_t = torch.tensor(np.atleast_2d(X_np), dtype=torch.float32).to(self.device)
                with torch.no_grad():
                    out = self.model(X_t).detach().cpu().numpy()
                return out.reshape(-1)   # her zaman 1-B (shap 0-B array'i kabul etmez)

            idx = np.random.choice(len(X_background), min(n_background, len(X_background)), replace=False)
            background = X_background[idx]

            explainer   = shap.KernelExplainer(predict_fn, background)
            shap_values = explainer.shap_values(X_explain, nsamples=100)
            # shap bazı sürümlerde liste döndürür (sınıf başına) — ilkini al
            if isinstance(shap_values, list):
                shap_values = shap_values[0]
            return np.asarray(shap_values)
        except Exception as e:
            print(f"SHAP KernelExplainer kullanılamadı ({e}). "
                  f"Gradyan-tabanlı öznitelik önemine geçiliyor...")
            return self._gradient_importance(X_explain)

    def _gradient_importance(self, X_explain: np.ndarray) -> np.ndarray:
        """
        SHAP alternatifi: girdi gradyanı × girdi (saliency) tabanlı yerel önem.

        Her örnek ve öznitelik için ∂P(heyelan)/∂x · x değerini döndürür.
        Fiziksel yorum: modelin o özniteliğe yerel duyarlılığı. Uydurma/rastgele
        değer ÜRETMEZ; modelin gerçek türevlerini kullanır (tez için savunulabilir).
        """
        self.model.eval()
        X_t = torch.tensor(X_explain, dtype=torch.float32,
                           requires_grad=True).to(self.device)
        out = self.model(X_t)
        out.sum().backward()
        grads = X_t.grad.detach().cpu().numpy()
        return grads * X_explain

