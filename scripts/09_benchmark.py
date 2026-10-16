"""
Script 9: Mekânsal Blok Çapraz Doğrulama ile Karşılaştırmalı Değerlendirme

Tezin ana savını sınar: "fizik + veri hibrit modeli, saf fizikten ve saf
makine öğrenmesinden daha iyi ya da eşit doğrulukta VE fiziksel olarak
tutarlıdır."

Protokol
--------
  · 5 katlı mekânsal blok CV (0.25° ≈ 25 km bloklar, StratifiedGroupKFold).
    Her katın eğitim bloklarından blok-bazlı iç doğrulama kümesi ayrılır
    (erken durdurma + karar eşiği). Test bloğuna yalnız raporlamada dokunulur.
  · Modeller: newmark, logreg, rf, gbm, gbm_mono, mlp, pinn_v1, pinn_v2
    (--models ile alt küme koşulursa diğer modellerin önceki sonuçları korunur)
    + ablasyon: pinn_v2 fizik öznitelikleri olmadan (pinn_v2_nophysfeat)
  · Metrikler: ROC-AUC, PR-AUC, F1, Kappa, Brier, dik-arazi AUC (eğim ≥ 15°),
    PGA-monotonluk ihlal oranı.
  · Referans: v1 değerlendirmesi (rastgele poligon ayrımı) pinn_v2 için de
    tekrarlanır → mekânsal CV'nin iyimserliği ne kadar azalttığı ölçülür.

Çıktılar
--------
  outputs/results/benchmark_folds.csv      — kat × model metrikleri
  outputs/results/benchmark_summary.json   — ortalama ± std
  outputs/results/oof_pinn_v2.parquet      — katlar-dışı tahminler
  outputs/models/cv_ensemble/fold{k}.pt     — pinn_v2 kat modelleri (belirsizlik)
  outputs/figures/benchmark_auc.png, benchmark_consistency.png,
  outputs/figures/benchmark_pga_response.png

Kullanım:
  python3 scripts/09_benchmark.py [--models pinn_v2,gbm] [--folds 5] [--epochs 120]
"""
import sys
import json
import time
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))

from config import DATA_PROCESSED, FIGURES_DIR, MODELS_DIR, OUTPUTS_DIR, ensure_dirs
from src.data_engineering.feature_matrix import load_feature_table, MODEL_FEATURES, LITHO_ONEHOT_COLS
from src.evaluation.spatial_cv import (
    assign_spatial_blocks, spatial_folds, inner_validation_split, fold_summary,
)
from src.evaluation.models import make_model
from src.evaluation.metrics import (
    classification_metrics, best_f1_threshold, pga_response, pga_monotonicity, PGA_SWEEP,
)

RESULTS_DIR = OUTPUTS_DIR / "results"
ALL_MODELS = ["newmark", "logreg", "rf", "gbm", "gbm_mono",
              "mlp", "pinn_v1", "pinn_v2", "pinn_v2_nophysfeat"]

# Fizik öznitelikleri çıkarılmış (yalnız veri) öznitelik kümesi — ablasyon.
# PGA ve eğim kalır (fizik-yolu kısıtı bunlar üzerinden kurulur); Ky, log Dn ve
# c', φ', γ (litoloji tablosundan türetilen jeomekanik) çıkarılır.
DATA_ONLY_FEATURES = ["elevation_m", "slope_deg", "aspect_sin", "aspect_cos",
                      "pga_g"] + LITHO_ONEHOT_COLS

LABELS = {
    "newmark": "Newmark (saf fizik)", "logreg": "Lojistik regresyon",
    "rf": "Random Forest", "gbm": "Gradient Boosting",
    "gbm_mono": "GBM + monoton kısıt", "mlp": "MLP (fiziksiz)",
    "pinn_v1": "PINN v1", "pinn_v2": "PINN v2 (önerilen)",
    "pinn_v2_nophysfeat": "PINN v2 – fizik girdisiz",
}


def _model_and_features(name, epochs):
    if name == "pinn_v2_nophysfeat":
        # Fizik-yolu örnekleyicisi Ky/Dn sütunlarına ihtiyaç duyar; bu ablasyonda
        # yalnız stabilite cezası + MLP kullanılır (pinn_v1 eşdeğeri, fizik girdisiz).
        return make_model("pinn_v1", DATA_ONLY_FEATURES, epochs=epochs), DATA_ONLY_FEATURES
    kw = {"epochs": epochs} if name in ("mlp", "pinn_v1", "pinn_v2") else {}
    return make_model(name, MODEL_FEATURES, **kw), MODEL_FEATURES


def run(models, n_folds=5, epochs=120, block_deg=0.25, n_mono=2000):
    ensure_dirs()
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    ens_dir = MODELS_DIR / "cv_ensemble"
    ens_dir.mkdir(parents=True, exist_ok=True)

    df = load_feature_table(DATA_PROCESSED / "feature_matrix.parquet")
    y = df["label"].to_numpy(int)
    slope = df["slope_deg"].to_numpy(float)
    blocks = assign_spatial_blocks(df, block_deg)
    folds = spatial_folds(y, blocks, n_folds)
    print(f"Örnek: {len(df):,} | blok: {len(np.unique(blocks))} ({block_deg}°)")
    print(fold_summary(y, blocks, folds).to_string(index=False))

    rows, oof = [], np.full(len(df), np.nan)
    curves = {}   # model → (n_folds, n_grid) dik test hücrelerinde ort. P
    rng = np.random.RandomState(0)

    for k, (tr, te) in enumerate(folds, 1):
        tr_in, va = inner_validation_split(tr, y, blocks, seed=k)
        steep_te = te[slope[te] >= np.percentile(slope, 90)]
        mono_idx = rng.choice(te, min(n_mono, len(te)), replace=False)
        for name in models:
            t0 = time.time()
            m, cols = _model_and_features(name, epochs)
            X = df[cols].to_numpy(np.float32)
            m.fit(X[tr_in], y[tr_in], X[va], y[va])
            thr = best_f1_threshold(y[va], m.predict(X[va]))
            p_te = m.predict(X[te])
            met = classification_metrics(y[te], p_te, thr, slope=slope[te])
            if "pga_g" in cols:
                P = pga_response(m.predict, X[mono_idx], cols)
                met.update(pga_monotonicity(P))
                curves.setdefault(name, []).append(
                    pga_response(m.predict, X[steep_te], cols).mean(axis=1))
            met.update({"fold": k, "model": name, "fit_s": round(time.time() - t0, 1)})
            rows.append(met)
            print(f"[kat {k}] {name:20s} AUC={met['roc_auc']:.3f} "
                  f"steep={met.get('roc_auc_steep', float('nan')):.3f} "
                  f"F1={met['f1']:.3f} mono={met.get('mono_violation_rate', float('nan')):.3f} "
                  f"({met['fit_s']}s)", flush=True)
            if name == "pinn_v2":
                oof[te] = p_te
                import torch
                ck = m.trainer.metadata | {"model_state": m.model.state_dict(),
                                           "threshold": thr, "fold": k}
                torch.save(ck, ens_dir / f"fold{k}.pt")

    res = pd.DataFrame(rows)
    # Yalnız seçilen modeller yeniden koşulduysa önceki diğer sonuçlar korunur
    prev_csv = RESULTS_DIR / "benchmark_folds.csv"
    if prev_csv.exists():
        prev = pd.read_csv(prev_csv)
        res = pd.concat([prev[~prev.model.isin(models)], res], ignore_index=True)
    res.to_csv(prev_csv, index=False)
    prev_curves = RESULTS_DIR / "pga_curves.npy"
    if prev_curves.exists():
        old = np.load(prev_curves, allow_pickle=True).item()
        curves = {**{k: list(v) for k, v in old.items() if k not in models}, **curves}
    models_all = [m for m in ALL_MODELS if m in set(res.model)]

    # v1 protokolü (rastgele poligon ayrımı) — iyimserlik karşılaştırması
    optimistic = None
    prev_sum = RESULTS_DIR / "benchmark_summary.json"
    if "pinn_v2" in models:
        optimistic = _polygon_split_reference(df, epochs)
    elif prev_sum.exists():
        optimistic = json.loads(prev_sum.read_text()).get("reference_polygon_split_pinn_v2")

    summary = _summarize(res, models_all, optimistic, block_deg, n_folds)
    (RESULTS_DIR / "benchmark_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False))
    if "pinn_v2" in models:
        pd.DataFrame({"oof_p": oof, "label": y, "block": blocks,
                      "lon": df["lon"], "lat": df["lat"]}).to_parquet(
            RESULTS_DIR / "oof_pinn_v2.parquet", index=False)
    np.save(RESULTS_DIR / "pga_curves.npy",
            {k: np.array(v) for k, v in curves.items()}, allow_pickle=True)

    _plot_auc(summary)
    _plot_consistency(summary)
    _plot_pga_response(curves)
    print(f"\nÖzet: {RESULTS_DIR / 'benchmark_summary.json'}")


def _polygon_split_reference(df, epochs):
    """v1 değerlendirme protokolü: poligon kimliğine göre rastgele %65/15/20."""
    y = df["label"].to_numpy(int)
    pid = df["polygon_id"].to_numpy()
    u = np.unique(pid)
    rng = np.random.RandomState(42)
    rng.shuffle(u)
    nt, nv = int(len(u) * 0.2), int(len(u) * 0.15)
    te = np.isin(pid, u[:nt]); va = np.isin(pid, u[nt:nt + nv]); tr = ~(te | va)
    X = df[MODEL_FEATURES].to_numpy(np.float32)
    m = make_model("pinn_v2", MODEL_FEATURES, epochs=epochs)
    m.fit(X[tr], y[tr], X[va], y[va])
    thr = best_f1_threshold(y[va], m.predict(X[va]))
    met = classification_metrics(y[te], m.predict(X[te]), thr,
                                 slope=df["slope_deg"].to_numpy()[te])
    print(f"[referans] pinn_v2 rastgele-poligon ayrımı AUC={met['roc_auc']:.3f}")
    return met


def _summarize(res, models, optimistic, block_deg, n_folds):
    keys = ["roc_auc", "pr_auc", "f1", "precision", "recall", "kappa", "brier",
            "roc_auc_steep", "mono_violation_rate", "mono_mean_drop"]
    out = {"protocol": {
        "cv": f"{n_folds}-katlı mekânsal blok CV ({block_deg}° bloklar, StratifiedGroupKFold)",
        "inner_validation": "eğitim bloklarından blok-bazlı %15 (erken durdurma + eşik)",
        "steep_subset": "eğim ≥ 15° test örnekleri",
        "monotonicity": "test örneklerinde PGA 0.05–1.20 g taraması; P'nin >0.01 düştüğü örnek oranı",
    }, "models": {}}
    for name in models:
        r = res[res.model == name]
        out["models"][name] = {"label": LABELS.get(name, name)}
        for k in keys:
            if k in r and r[k].notna().any():
                out["models"][name][k] = {"mean": round(float(r[k].mean()), 4),
                                          "std": round(float(r[k].std(ddof=1)), 4)}
    if optimistic:
        out["reference_polygon_split_pinn_v2"] = {k: round(v, 4) for k, v in optimistic.items()
                                                 if isinstance(v, float)}
    return out


# ------------------------------------------------------------------ grafikler
BLUE, GRAY, INK, MUTED = "#2a78d6", "#a3a29c", "#0b0b0b", "#52514e"
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]


def _style(ax):
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.spines["left"].set_color("#c9c8c2"); ax.spines["bottom"].set_color("#c9c8c2")
    ax.tick_params(colors=MUTED)
    ax.grid(axis="x", color="#ecebe7", lw=0.8); ax.set_axisbelow(True)


def _plot_auc(summary):
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    items = [(k, v) for k, v in summary["models"].items() if "roc_auc" in v]
    items.sort(key=lambda kv: kv[1]["roc_auc"]["mean"])
    fig, axes = plt.subplots(1, 2, figsize=(12, 0.5 * len(items) + 1.6), sharey=True)
    for ax, key, title in [(axes[0], "roc_auc", "ROC-AUC (tüm test)"),
                           (axes[1], "roc_auc_steep", "ROC-AUC (yalnız dik arazi, eğim ≥ 15°)")]:
        names = [LABELS.get(k, k) for k, _ in items]
        m = [v.get(key, {}).get("mean", np.nan) for _, v in items]
        s = [v.get(key, {}).get("std", 0) for _, v in items]
        col = [BLUE if k == "pinn_v2" else GRAY for k, _ in items]
        ax.barh(names, m, xerr=s, color=col, height=0.6, edgecolor="white", linewidth=2,
                error_kw=dict(ecolor=MUTED, lw=1, capsize=3))
        for i, (mi, si) in enumerate(zip(m, s)):
            ax.text(mi + si + 0.004, i, f"{mi:.3f}", va="center", fontsize=9, color=INK)
        lo = np.nanmin(m) - 0.08
        ax.set_xlim(max(0.5, lo), 1.0)
        ax.set_title(title, fontsize=11, color=INK, loc="left")
        _style(ax)
    fig.suptitle("Mekânsal blok CV (5 kat, ort. ± std) — PINN v2 mavi", x=0.01, ha="left",
                 fontsize=12, color=INK)
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "benchmark_auc.png", dpi=170); plt.close(fig)


def _plot_consistency(summary):
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(8.5, 5.6))
    for k, v in summary["models"].items():
        if "mono_violation_rate" not in v:
            continue
        x = 100 * v["mono_violation_rate"]["mean"]; yv = v["roc_auc"]["mean"]
        c = BLUE if k == "pinn_v2" else GRAY
        ax.scatter(x, yv, s=110 if k == "pinn_v2" else 70, color=c, edgecolor="white",
                   linewidth=2, zorder=3)
        ax.annotate(LABELS.get(k, k), (x, yv), xytext=(7, 4), textcoords="offset points",
                    fontsize=9, color=INK)
    ax.set_xlabel("PGA-monotonluk ihlali (% örnek; PGA artınca P düşüyor)", color=MUTED)
    ax.set_ylabel("ROC-AUC (mekânsal CV)", color=MUTED)
    ax.set_title("Doğruluk – fiziksel tutarlılık düzlemi (sol üst = en iyi)",
                 loc="left", fontsize=12, color=INK)
    _style(ax); ax.grid(axis="y", color="#ecebe7", lw=0.8)
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "benchmark_consistency.png", dpi=170); plt.close(fig)


def _plot_pga_response(curves):
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    show = [m for m in ("gbm", "mlp", "pinn_v1", "pinn_v2") if m in curves]
    if not show:
        return
    fig, ax = plt.subplots(figsize=(9, 5.6))
    ends = {}
    for i, m in enumerate(show):
        C = np.array(curves[m])
        mu = C.mean(axis=0)
        ax.plot(PGA_SWEEP, mu, color=SERIES[i], lw=2.4 if m == "pinn_v2" else 2)
        ax.fill_between(PGA_SWEEP, C.min(axis=0), C.max(axis=0), color=SERIES[i], alpha=0.12, lw=0)
        ends[m] = mu[-1]
    # Doğrudan etiketler — üst üste binmeyi önlemek için en az 0.04 aralık
    ys, prev = {}, None
    for m in sorted(ends, key=ends.get):
        y = ends[m] if prev is None else max(ends[m], prev + 0.04)
        ys[m] = prev = y
    for m, y in ys.items():
        ax.text(PGA_SWEEP[-1] + 0.02, y, LABELS[m], color=INK, fontsize=9, va="center")
    ax.axhline(0.5, color=MUTED, ls=":", lw=1)
    ax.set_xlim(0, 1.55); ax.set_ylim(0, 1)
    ax.set_xlabel("PGA (g)", color=MUTED); ax.set_ylabel("Ortalama P(heyelan)", color=MUTED)
    ax.set_title("PGA tepkisi — en dik %10 test hücresi (bant: 5 kat min–maks)",
                 loc="left", fontsize=12, color=INK)
    _style(ax); ax.grid(axis="y", color="#ecebe7", lw=0.8)
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "benchmark_pga_response.png", dpi=170); plt.close(fig)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Mekânsal CV karşılaştırmalı değerlendirme")
    ap.add_argument("--models", default=",".join(ALL_MODELS))
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--epochs", type=int, default=120)
    ap.add_argument("--block-deg", type=float, default=0.25)
    ap.add_argument("--plots-only", action="store_true", help="Kayıtlı sonuçlardan yalnız figürleri yeniden çiz")
    a = ap.parse_args()
    if a.plots_only:
        summ = json.loads((RESULTS_DIR / "benchmark_summary.json").read_text())
        _plot_auc(summ); _plot_consistency(summ)
        _plot_pga_response(np.load(RESULTS_DIR / "pga_curves.npy", allow_pickle=True).item())
    else:
        run(a.models.split(","), a.folds, a.epochs, a.block_deg)
