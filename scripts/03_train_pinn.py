"""
Script 3: PINN v2 Nihai Model Eğitimi

Adımlar:
  1. Öznitelik matrisini yükle (v2 öznitelikleri: log Dn, litho_id/u yok)
  2. Mekânsal blok ayrımı (0.25° bloklar): train / val / test — test blokları
     coğrafi olarak eğitimden ayrıktır (komşu-poligon sızıntısı yok)
  3. PINN v2: Focal + stabilite cezası + fizik-yolu (PGA, eğim) kolokasyon kısıtı
  4. Karar eşiği VAL setinde F1-maks ile seçilir (teste bakılmaz)
  5. Test raporu + fiziksel tutarlılık (PGA-monotonluk) ölçümü
  6. SHAP (GradientExplainer; litoloji one-hot'ları tek grup)
  7. Checkpoint (+ feat_cols, scaler, eşik), scaler.pkl, threshold.json,
     backend/metrics.json (uygulama metrikleri elle değil buradan yazılır)

Kullanım:
  python3 scripts/03_train_pinn.py [--epochs 150] [--lambda-mono 12]
"""
import sys
import json
import pickle
import argparse
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from config import (
    DATA_PROCESSED, FIGURES_DIR, MODELS_DIR, OUTPUTS_DIR, PROJECT_ROOT, ensure_dirs,
)
from src.data_engineering.feature_matrix import load_feature_table, MODEL_FEATURES
from src.evaluation.spatial_cv import assign_spatial_blocks, spatial_folds, inner_validation_split
from src.evaluation.models import NeuralModel
from src.evaluation.metrics import (
    classification_metrics, best_f1_threshold, pga_response, pga_monotonicity,
)
from src.visualization.training_plots import plot_training_history, plot_confusion_matrix

BLOCK_DEG = 0.25


def run(epochs: int = 150, lambda_mono: float = 12.0, seed: int = 42):
    ensure_dirs()
    df = load_feature_table(DATA_PROCESSED / "feature_matrix.parquet")
    y = df["label"].to_numpy(int)
    slope = df["slope_deg"].to_numpy(float)
    feat_cols = list(MODEL_FEATURES)
    X = df[feat_cols].to_numpy(np.float32)

    # Mekânsal ayrım: 5 bloğun 1'i test (%20), kalan blokların %15'i doğrulama
    blocks = assign_spatial_blocks(df, BLOCK_DEG)
    tr_all, te = spatial_folds(y, blocks, n_splits=5, seed=seed)[0]
    tr, va = inner_validation_split(tr_all, y, blocks, val_frac=0.15, seed=seed)
    print(f"Train: {len(tr):,} | Val: {len(va):,} | Test: {len(te):,}  "
          f"(blok: {len(np.unique(blocks[tr]))}/{len(np.unique(blocks[va]))}/"
          f"{len(np.unique(blocks[te]))})")

    m = NeuralModel(feat_cols, "pinn_v2", epochs=epochs, lambda_mono=lambda_mono,
                    seed=seed, verbose=True, checkpoint_dir=str(MODELS_DIR))
    m.fit(X[tr], y[tr], X[va], y[va])
    plot_training_history(m.history, save_path=FIGURES_DIR / "training_history.png")

    thr = best_f1_threshold(y[va], m.predict(X[va]))
    p_te = m.predict(X[te])
    met = classification_metrics(y[te], p_te, thr, slope=slope[te])
    rng = np.random.RandomState(0)
    s = rng.choice(te, min(3000, len(te)), replace=False)
    met.update(pga_monotonicity(pga_response(m.predict, X[s], feat_cols)))
    print(f"\nKarar eşiği (VAL F1-maks): {thr:.2f}")
    print(json.dumps({k: round(v, 4) if isinstance(v, float) else v for k, v in met.items()},
                     indent=2, ensure_ascii=False))
    plot_confusion_matrix(y[te], (p_te >= thr).astype(int),
                          save_path=FIGURES_DIR / "confusion_matrix.png")

    # Checkpoint'i zenginleştir (servis ve haritalar feat_cols'u buradan okur)
    import torch
    ck_path = MODELS_DIR / "best_model.pt"
    ck = torch.load(ck_path, map_location="cpu", weights_only=False)
    ck.update({"threshold": thr, "test_metrics": met, "model_version": "pinn_v2",
               "split": f"mekânsal blok ({BLOCK_DEG}°) train/val/test"})
    torch.save(ck, ck_path)
    scaler = {"mean": m._mean, "std": m._std}
    (MODELS_DIR / "scaler.pkl").write_bytes(pickle.dumps(scaler))
    (MODELS_DIR / "threshold.json").write_text(json.dumps(
        {"decision_threshold": thr, "selected_on": "validation_f1max (spatial blocks)"}))

    shap_order = _shap(m, X[tr], X[te], feat_cols)
    _write_app_metrics(met, thr, len(df), float(y.mean()), feat_cols, shap_order)
    print("Eğitim tamamlandı.")


def _shap(m, X_bg, X_ex, feat_cols, n_bg=300, n_ex=600):
    """GradientExplainer (beklenen gradyanlar) — litoloji one-hot'ları tek grup."""
    import torch
    import shap
    from src.visualization.training_plots import plot_shap_summary

    rng = np.random.RandomState(1)
    bg = X_bg[rng.choice(len(X_bg), n_bg, replace=False)]
    ex = X_ex[rng.choice(len(X_ex), min(n_ex, len(X_ex)), replace=False)]
    to_n = lambda a: torch.from_numpy(((a - m._mean) / m._std).astype(np.float32))
    explainer = shap.GradientExplainer(m.model, to_n(bg))
    sv = explainer.shap_values(to_n(ex))
    sv = np.asarray(sv[0] if isinstance(sv, list) else sv).reshape(len(ex), -1)

    litho = [i for i, c in enumerate(feat_cols) if c.startswith("litho_")]
    keep = [i for i in range(len(feat_cols)) if i not in litho]
    names = [feat_cols[i] for i in keep] + ["litoloji (one-hot toplam)"]
    sv_g = np.column_stack([sv[:, keep], sv[:, litho].sum(axis=1)])
    x_g = np.column_stack([ex[:, keep], ex[:, litho].argmax(axis=1)])
    plot_shap_summary(sv_g, x_g, names, top_n=len(names),
                      save_path=FIGURES_DIR / "shap_summary.png")
    imp = np.abs(sv_g).mean(axis=0)
    order = [names[i] for i in np.argsort(imp)[::-1]]
    (OUTPUTS_DIR / "results").mkdir(parents=True, exist_ok=True)
    (OUTPUTS_DIR / "results" / "shap_importance.json").write_text(json.dumps(
        {n: round(float(v), 5) for n, v in sorted(zip(names, imp), key=lambda t: -t[1])},
        indent=2, ensure_ascii=False))
    print("SHAP sırası:", order)
    return order


def _write_app_metrics(met, thr, n, pos_rate, feat_cols, shap_order):
    path = PROJECT_ROOT / "backend" / "metrics.json"
    old = json.loads(path.read_text()) if path.exists() else {}
    bench = OUTPUTS_DIR / "results" / "benchmark_summary.json"
    cv = None
    if bench.exists():
        cv = json.loads(bench.read_text()).get("models", {}).get("pinn_v2")
    out = {
        "model_version": "pinn_v2",
        "roc_auc": round(met["roc_auc"], 3),
        "roc_auc_steep": round(met.get("roc_auc_steep", float("nan")), 3),
        "pr_auc": round(met["pr_auc"], 3),
        "accuracy": round(met["accuracy"], 3),
        "recall": round(met["recall"], 3),
        "precision": round(met["precision"], 3),
        "f1": round(met["f1"], 3),
        "kappa": round(met["kappa"], 3),
        "brier": round(met["brier"], 3),
        "mono_violation_rate": round(met["mono_violation_rate"], 4),
        "critical_pga_ml": old.get("critical_pga_ml"),
        "n_samples": int(n),
        "landslide_ratio": round(pos_rate, 3),
        "n_features": len(feat_cols),
        "threshold": round(thr, 2),
        "shap_order": shap_order[:6],
        "split": f"mekânsal blok ({BLOCK_DEG}°) 3'lü (train/val/test)",
        "earthquake": "Kahramanmaraş 6 Şubat 2023 (Mw7.8 + Mw7.5)",
        "notes": old.get("notes", {}),
    }
    if cv:
        out["cv"] = {k: v for k, v in cv.items() if k != "label"}
    for k in ("critical_pga", "v1_reference"):
        if k in old:
            out[k] = old[k]
    out.setdefault("v1_reference", {
        "roc_auc": 0.963, "split": "rastgele poligon ayrımı",
        "note": "v1 (Temmuz 2026) sonucu; komşu poligon sızıntısı nedeniyle iyimser.",
    })
    path.write_text(json.dumps(out, indent=2, ensure_ascii=False))
    print(f"Uygulama metrikleri güncellendi: {path}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="PINN v2 nihai model eğitimi")
    ap.add_argument("--epochs", type=int, default=150)
    ap.add_argument("--lambda-mono", type=float, default=12.0)
    ap.add_argument("--seed", type=int, default=42)
    a = ap.parse_args()
    run(a.epochs, a.lambda_mono, a.seed)
