"""
Script 3: PINN Model Eğitimi

Adımlar:
  1. Öznitelik matrisini yükle
  2. Polygon-based spatial split uygula
  3. Özellikleri normalize et
  4. LandslidePINN modelini başlat
  5. PhysicsInformedLoss ile eğit
  6. Eğitim grafiklerini kaydet
  7. SHAP önem analizi yap

Kullanım:
  python scripts/03_train_pinn.py [--synthetic] [--epochs 50]
"""
import sys
import argparse
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from config import (
    DATA_PROCESSED, FIGURES_DIR, MODELS_DIR,
    TRAIN_CONFIG, FOCAL_ALPHA, FOCAL_GAMMA,
    PHYS_SLOPE_THRESHOLD, PHYS_PGA_THRESHOLD,
    N_LITHOLOGY_CLASSES, THRESHOLD_SEARCH, ensure_dirs,
)
from src.pinn.model import LandslidePINN
from src.pinn.loss import PhysicsInformedLoss
from src.pinn.trainer import PINNTrainer, TrainConfig
from src.data_engineering.spatial_split import polygon_based_split
from src.data_engineering.feature_matrix import normalize_features, ALL_FEATURE_COLUMNS
from src.visualization.training_plots import (
    plot_training_history, plot_confusion_matrix
)


def load_or_generate_data(synthetic: bool = False) -> tuple:
    """
    Öznitelik matrisini yükler veya sentetik üretir.

    Döndürür: X, y, slope_col, pga_col, polygon_ids, feat_cols
    """
    parquet_path = DATA_PROCESSED / "feature_matrix.parquet"

    if not synthetic and parquet_path.exists():
        import pandas as pd
        print(f"Öznitelik matrisi yükleniyor: {parquet_path}")
        df = pd.read_parquet(parquet_path)
    else:
        from src.data_engineering.feature_matrix import prepare_synthetic_data, build_feature_matrix
        from src.geomechanics.newmark import LandslideLabel
        import pandas as pd

        raw = prepare_synthetic_data(n_pixels=50_000)
        n   = raw["n_pixels"]

        df, _ = build_feature_matrix(
            elevation_m = raw["elevation_m"].reshape(n),
            slope_deg   = raw["slope_deg"].reshape(n),
            aspect_deg  = raw["aspect_deg"].reshape(n),
            litho_array = raw["litho_array"].reshape(n),
            pga_array   = raw["pga_array"].reshape(n),
            u_array     = raw["u_array"].reshape(n),
        )
        df["label"]      = LandslideLabel(df["dn_cm"].values / 100.0)
        df["polygon_id"] = raw["polygon_ids"]

    feat_cols = [c for c in ALL_FEATURE_COLUMNS if c in df.columns]
    X           = df[feat_cols].values.astype(np.float32)
    y           = df["label"].values.astype(np.float32)
    slope_col   = df["slope_deg"].values.astype(np.float32)
    pga_col     = df["pga_g"].values.astype(np.float32)
    polygon_ids = df.get("polygon_id", df.index).values

    print(f"  Örnekler: {len(X):,} | Öznitelikler: {X.shape[1]}")
    print(f"  Heyelan oranı: {y.mean():.1%}")

    return X, y, slope_col, pga_col, polygon_ids, feat_cols


def run(epochs: int | None = None, synthetic: bool = False):
    ensure_dirs()

    # Veri yükle
    X, y, slope, pga, polygon_ids, feat_cols = load_or_generate_data(synthetic)

    # Spatial split — 3'lü polygon-based (train/val/test).
    # Doğrulama seti erken-durdurma VE karar eşiği seçimi için kullanılır; test
    # seti yalnızca nihai raporlamada dokunulur (eşik teste göre ayarlanmaz →
    # veri sızıntısı önlenir).
    unique_poly = np.unique(polygon_ids)
    rng_local   = np.random.RandomState(TRAIN_CONFIG["random_state"])
    rng_local.shuffle(unique_poly)
    n_test_poly = max(1, int(len(unique_poly) * TRAIN_CONFIG["test_ratio"]))
    n_val_poly  = max(1, int(len(unique_poly) * 0.15))
    test_polys  = set(unique_poly[:n_test_poly])
    val_polys   = set(unique_poly[n_test_poly:n_test_poly + n_val_poly])

    test_mask = np.isin(polygon_ids, list(test_polys))
    val_mask  = np.isin(polygon_ids, list(val_polys))
    tr_mask   = ~(test_mask | val_mask)

    X_tr, X_va, X_te    = X[tr_mask],     X[val_mask],     X[test_mask]
    y_tr, y_va, y_te    = y[tr_mask],     y[val_mask],     y[test_mask]
    sl_tr, sl_va, sl_te = slope[tr_mask], slope[val_mask], slope[test_mask]
    pg_tr, pg_va, pg_te = pga[tr_mask],   pga[val_mask],   pga[test_mask]

    print(f"\nTrain: {len(X_tr):,} | Val: {len(X_va):,} | Test: {len(X_te):,}")

    # Normalize — scaler yalnızca train'den; val ve test aynı ölçekle dönüştürülür
    X_tr_n, X_te_n, scaler = normalize_features(X_tr, X_te)
    X_va_n = (X_va - scaler["mean"]) / scaler["std"]

    # Model
    n_features = X_tr_n.shape[1]
    model = LandslidePINN(in_features=n_features)
    print(f"\n{model}")

    # Loss
    loss_fn = PhysicsInformedLoss(
        lambda_phys=TRAIN_CONFIG["lambda_phys"],
        alpha=FOCAL_ALPHA,
        gamma=FOCAL_GAMMA,
        slope_threshold_deg=PHYS_SLOPE_THRESHOLD,
        pga_threshold_g=PHYS_PGA_THRESHOLD,
    )

    # Trainer
    n_epochs = epochs or TRAIN_CONFIG["epochs"]
    cfg = TrainConfig(
        epochs=n_epochs,
        batch_size=TRAIN_CONFIG["batch_size"],
        lr=TRAIN_CONFIG["lr"],
        weight_decay=TRAIN_CONFIG["weight_decay"],
        patience=TRAIN_CONFIG["patience"],
        checkpoint_dir=str(MODELS_DIR),
        lambda_phys=TRAIN_CONFIG["lambda_phys"],
        use_lambda_annealing=True,
        max_lambda=TRAIN_CONFIG["max_lambda"],
    )
    trainer = PINNTrainer(model, loss_fn, cfg)

    # Eğit — erken durdurma VAL setinde (test'e dokunulmaz)
    print("\nEğitim başlıyor...")
    history = trainer.fit(
        X_tr_n, y_tr, sl_tr, pg_tr,
        X_va_n, y_va, sl_va, pg_va,
    )

    # En iyi modeli yükle
    model = trainer.load_best()

    # Eğitim grafikleri
    plot_training_history(
        history,
        save_path=FIGURES_DIR / "training_history.png",
    )

    import torch
    from sklearn.metrics import classification_report, roc_auc_score, f1_score

    # Karar eşiğini VAL setinde F1-maks ile seç (test'e bakılmaz)
    X_va_t   = torch.tensor(X_va_n, dtype=torch.float32).to(trainer.device)
    proba_va = model.predict_proba(X_va_t).cpu().numpy().squeeze()
    lo, hi, step = THRESHOLD_SEARCH
    thr_grid = np.arange(lo, hi + 1e-9, step)
    f1s = [f1_score(y_va, (proba_va >= t).astype(int), zero_division=0) for t in thr_grid]
    best_thr = float(thr_grid[int(np.argmax(f1s))])
    print(f"\nSeçilen karar eşiği (VAL F1-maks): {best_thr:.2f} "
          f"(VAL F1={max(f1s):.3f})")

    # Test değerlendirmesi
    X_te_t  = torch.tensor(X_te_n, dtype=torch.float32).to(trainer.device)
    proba   = model.predict_proba(X_te_t).cpu().numpy().squeeze()
    y_pred_05  = (proba >= 0.5).astype(int)
    y_pred_thr = (proba >= best_thr).astype(int)

    print("\n" + "="*50)
    print("Test Sonuçları — varsayılan eşik 0.50")
    print("="*50)
    print(classification_report(y_te, y_pred_05,
                                  target_names=["Kararlı", "Heyelan"]))
    print("="*50)
    print(f"Test Sonuçları — ayarlı eşik {best_thr:.2f} (recall odaklı)")
    print("="*50)
    print(classification_report(y_te, y_pred_thr,
                                  target_names=["Kararlı", "Heyelan"]))
    print(f"ROC-AUC: {roc_auc_score(y_te, proba):.4f}")

    # Nihai rapor + confusion matrix ayarlı eşikte
    plot_confusion_matrix(
        y_te, y_pred_thr,
        save_path=FIGURES_DIR / "confusion_matrix.png",
    )

    # Eşiği kaydet (harita/çıkarım tutarlılığı için)
    import json
    (MODELS_DIR / "threshold.json").write_text(
        json.dumps({"decision_threshold": best_thr, "selected_on": "validation_f1max"})
    )

    # SHAP Önem Analizi ve Grafikleştirme
    print("\nSHAP önem analizi yapılıyor...")
    try:
        from src.visualization.training_plots import plot_shap_summary
        # Train setinden referans arka plan (background) ve test setinden açıklanacak verileri (explain) seçelim
        shap_values = trainer.compute_shap(
            X_background=X_tr_n,
            X_explain=X_te_n[:200],  # Hızlı olması için ilk 200 testi açıklayalım
            feature_names=feat_cols,
            n_background=100,
        )
        plot_shap_summary(
            shap_values=shap_values,
            X_explain=X_te_n[:200],
            feature_names=feat_cols,
            save_path=FIGURES_DIR / "shap_summary.png",
        )
        print(f"  SHAP önem analizi grafiği kaydedildi: {FIGURES_DIR / 'shap_summary.png'}")
    except Exception as e:
        print(f"  SHAP analizi başarısız oldu veya atlandı: {e}")

    # Scaler'ı kaydet
    import pickle
    (MODELS_DIR / "scaler.pkl").write_bytes(pickle.dumps(scaler))
    print(f"\nScaler kaydedildi: {MODELS_DIR / 'scaler.pkl'}")
    print("Eğitim tamamlandı.")



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="PINN model eğitimi")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--synthetic", action="store_true")
    args = parser.parse_args()

    run(epochs=args.epochs, synthetic=args.synthetic)
