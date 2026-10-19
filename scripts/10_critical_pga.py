"""
Script 10: Kritik PGA — Belirsizlik, Tasarım Tablosu ve Ampirik Doğrulama

Tezin ana sorusu: "Yamaç hangi PGA'da kaymaya başlar?" Bu betik v1'deki tek
sayılık yanıtı (≈0.55 g) üç yönden güçlendirir:

  (A) Belirsizlik: 5 mekânsal-CV kat modeli (epistemik) + hücre bootstrap'ı
      (örnekleme) → Kritik PGA için medyan ve %95 aralık.
  (B) Tasarım tablosu: eğim sınıfı × litoloji için Kritik PGA (mühendislik
      kullanımına uygun "hangi yamaç hangi PGA'da" çizelgesi).
  (C) Ampirik doğrulama: gözlenen heyelan alan yoğunluğu (envanter poligonları
      ızgaraya dökülür) PGA ve eğim sınıflarına göre, model tahminiyle aynı
      ızgarada karşılaştırılır (Spearman ρ).
  (D) Duyarlılık: negatif/pozitif örnekleme oranının Kritik PGA'ya etkisi.

İki tanım raporlanır:
  · BAŞLANGIÇ Kritik PGA (önerilen, örneklemeden bağımsız): olabilirlik oranı
      LR(x) = p(x | heyelan) / p(x | arka plan) = [P/(1−P)] · (N_neg/N_pos)
    değerinin 1'i geçtiği PGA. LR = 1 ⇔ P = π (eğitimdeki pozitif oranı).
    Bu, Frekans Oranı (FR > 1) yönteminin olasılıksal karşılığıdır ve gözlenen
    envanterden doğrudan hesaplanan ampirik FR eğrisiyle kıyaslanabilir.
  · Geleneksel P ≥ 0.5 (v1 tanımı): LR = N_neg/N_pos'a karşılık gelir; bu
    nedenle negatif örnekleme oranına bağlıdır (bkz. D).

PGA taramasında Dn her PGA için Jibson (2007) ile yeniden hesaplanır (v1'deki
betikte Dn sabit kalıyordu; PGA–Dn tutarsızlığı giderildi).

Kullanım:
  python3 scripts/10_critical_pga.py [--res 0.008] [--skip-sensitivity]
"""
import sys
import json
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))

from config import (DATA_PROCESSED, FIGURES_DIR, MODELS_DIR, OUTPUTS_DIR, PROJECT_ROOT,
                    Z_REF_M, DN_THRESHOLD_M, ensure_dirs)
from src.data_engineering.regional_grid import (
    load_regional_stack, stack_features, build_grid, rasterize_inventory, STUDY_BOUNDS,
)
from src.evaluation.metrics import pga_response, critical_pga_from_curve, PGA_SWEEP
from src.geomechanics.lithology import LITHOLOGY_TABLE
from src.geomechanics.newmark import NewmarkDisplacement
from src.pinn.io import load_checkpoint

RESULTS = OUTPUTS_DIR / "results"
GRID = np.round(np.arange(0.02, 1.501, 0.02), 3)
LEVEL = 0.5          # geleneksel (v1) tanım
PI_TRAIN = None      # eğitim pozitif oranı → LR = 1 seviyesi (run() içinde atanır)
SLOPE_BINS = [(15, 25), (25, 35), (35, 45), (45, 90)]

INK, MUTED, BLUE, ORANGE, AQUA, GRAYL = "#0b0b0b", "#52514e", "#2a78d6", "#eb6834", "#1baf7a", "#c9c8c2"


def _style(ax):
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(GRAYL)
    ax.tick_params(colors=MUTED)
    ax.grid(color="#ecebe7", lw=0.8); ax.set_axisbelow(True)


def load_models():
    final = load_checkpoint(MODELS_DIR / "best_model.pt", MODELS_DIR / "scaler.pkl",
                            MODELS_DIR / "threshold.json")
    ver = final.meta.get("model_version")
    if ver not in ("pinn_v2", "pinn_v2_L"):
        raise SystemExit("best_model.pt v2 değil — önce scripts/03_train_pinn.py çalıştırın.")
    folds = [load_checkpoint(p) for p in sorted((MODELS_DIR / "cv_ensemble" / ver).glob("fold*.pt"))]
    return final, folds


def run(res=0.008, sensitivity=True):
    ensure_dirs(); RESULTS.mkdir(parents=True, exist_ok=True)
    final, folds = load_models()
    cols = final.feat_cols
    print(f"Model: final + {len(folds)} CV katı | ızgara {res}°")

    stack = load_regional_stack(res)
    df = stack_features(stack, Z_REF_M)
    X = df[cols].to_numpy(np.float32)
    v = stack.valid.ravel()
    slope = df["slope_deg"].to_numpy()
    litho = df["litho_id"].to_numpy()
    ky = df["ky_g"].to_numpy()

    # ---------------- (A) en dik %5 hücre — belirsizlik --------------------
    thr95 = float(np.percentile(slope[v], 95))
    steep = np.where(v & (slope >= thr95))[0]
    P_final = pga_response(final.predict, X[steep], cols, GRID)          # (G, n)
    P_folds = [pga_response(m.predict, X[steep], cols, GRID) for m in folds]
    global PI_TRAIN
    from src.data_engineering.feature_matrix import load_feature_table
    PI_TRAIN = float(load_feature_table(DATA_PROCESSED / "feature_matrix.parquet")["label"].mean())
    levels = {"onset_lr1": PI_TRAIN, "p05": LEVEL}
    curves_all = [P_final.mean(1)] + [P.mean(1) for P in P_folds]
    per_level = {}
    rng = np.random.RandomState(0)
    for key, lev in levels.items():
        c_models = [critical_pga_from_curve(GRID, cv, lev) for cv in curves_all]
        boots = []
        for _ in range(500):
            b = rng.randint(0, len(steep), len(steep))
            boots.append(critical_pga_from_curve(GRID, P_final[:, b].mean(1), lev))
        boots = np.array([x for x in boots if x is not None])
        vals = np.array([x for x in c_models if x is not None])
        ens_curve = np.mean(curves_all, axis=0)
        P_ens = np.mean([P_final] + P_folds, axis=0)
        boots_e = []
        for _ in range(300):
            b = rng.randint(0, len(steep), len(steep))
            boots_e.append(critical_pga_from_curve(GRID, P_ens[:, b].mean(1), lev))
        boots_e = np.array([x for x in boots_e if x is not None])
        per_level[key] = {
            "level_p": round(lev, 4),
            "critical_pga_g": _r(critical_pga_from_curve(GRID, ens_curve, lev)),
            "ensemble_bootstrap_95ci_g": [round(float(np.percentile(boots_e, 2.5)), 3),
                                          round(float(np.percentile(boots_e, 97.5)), 3)] if len(boots_e) else None,
            "median_of_models_g": round(float(np.median(vals)), 3) if len(vals) else None,
            "final_model_g": _r(c_models[0]),
            "cv_folds_g": [_r(x) for x in c_models[1:]],
            "model_range_g": [round(float(vals.min()), 3), round(float(vals.max()), 3)] if len(vals) else None,
            "final_bootstrap_95ci_g": [round(float(np.percentile(boots, 2.5)), 3),
                                       round(float(np.percentile(boots, 97.5)), 3)] if len(boots) else None,
        }

    # Newmark referansı: statik-stabil (Ky > 0.02) dik hücrelerde Dn ≥ 10 cm oranı
    ky_s = ky[steep]; stable = ky_s > 0.02
    frac_dn = np.array([(NewmarkDisplacement(ky_s[stable], pg) >= DN_THRESHOLD_M).mean()
                        if stable.any() else 0.0 for pg in GRID])
    c_newmark = critical_pga_from_curve(GRID, frac_dn, LEVEL)

    op = per_level["p05"]
    crit = {
        "definition": f"En dik %5 hücrede (eğim ≥ {thr95:.1f}°, n={len(steep)}) 6 modelin (nihai + 5 "
                      f"mekânsal-CV katı) ortalama P(heyelan) eğrisinin {LEVEL} düzeyini geçtiği PGA",
        "critical_pga_g": op["critical_pga_g"],
        "ensemble_range_g": op["model_range_g"],
        "bootstrap_95ci_g": op["ensemble_bootstrap_95ci_g"],
        "onset_lr1_g": per_level["onset_lr1"]["critical_pga_g"],
        "levels": per_level,
        "newmark_critical_pga_g": _r(c_newmark),
        "newmark_max_fraction": round(float(frac_dn.max()), 3),
        "static_marginal_fraction": round(float((~stable).mean()), 3),
        "steep_slope_threshold_deg": round(thr95, 1),
        "n_cells": int(len(steep)),
    }
    print(json.dumps(crit, indent=2, ensure_ascii=False))
    _plot_uncertainty(P_final, P_folds, frac_dn, crit)

    # ---------------- (B) eğim × litoloji tasarım tablosu ------------------
    table = _design_table(final, folds, X, cols, v, slope, litho)
    table.to_csv(RESULTS / "critical_pga_table.csv", index=False)
    _plot_design_table(table)

    # ---------------- (C) ampirik doğrulama -------------------------------
    emp = _empirical(stack, final, X, cols, v, slope, df["pga_g"].to_numpy())

    # ---------------- (D) örnekleme oranı duyarlılığı ---------------------
    sens = _neg_ratio_sensitivity(stack, df, steep, cols) if sensitivity else None

    out = {"critical_pga": crit, "empirical_validation": emp, "neg_ratio_sensitivity": sens,
           "design_table_csv": "outputs/results/critical_pga_table.csv"}
    (RESULTS / "critical_pga.json").write_text(json.dumps(out, indent=2, ensure_ascii=False))
    _update_app_metrics(crit, emp, sens)
    print(f"\nKaydedildi: {RESULTS / 'critical_pga.json'}")


def _ensemble_predict(models, X):
    return np.mean([m.predict(X) for m in models], axis=0)


def _design_table(final, folds, X, cols, v, slope, litho, min_cells=40):
    models = [final] + folds
    rows = []
    for lo, hi in SLOPE_BINS:
        for lid in sorted(np.unique(litho[v & (slope >= 15)])):
            idx = np.where(v & (slope >= lo) & (slope < hi) & (litho == lid))[0]
            if len(idx) < min_cells:
                continue
            sub = idx if len(idx) <= 3000 else np.random.RandomState(lid).choice(idx, 3000, replace=False)
            curves = [pga_response(m.predict, X[sub], cols, GRID).mean(1) for m in models]
            per_model = [critical_pga_from_curve(GRID, cv, LEVEL) for cv in curves]
            vals = np.array([x for x in per_model if x is not None])
            ens = critical_pga_from_curve(GRID, np.mean(curves, axis=0), LEVEL)
            onset = critical_pga_from_curve(GRID, np.mean(curves, axis=0), PI_TRAIN)
            rows.append({
                "slope_class": f"{lo}–{hi if hi < 90 else '90'}°", "slope_lo": lo,
                "litho_id": int(lid),
                "litho": LITHOLOGY_TABLE[int(lid)]["description"],
                "n_cells": int(len(idx)),
                "critical_pga_g": _r(ens),
                "range_lo_g": round(float(vals.min()), 3) if len(vals) else None,
                "range_hi_g": round(float(vals.max()), 3) if len(vals) else None,
                "n_models_reached": int(len(vals)), "n_models": len(models),
                "onset_lr1_g": _r(onset),
            })
    t = pd.DataFrame(rows)
    print("\nTasarım tablosu (medyan Kritik PGA, g):")
    if len(t):
        print(t.pivot(index="litho", columns="slope_class", values="critical_pga_g").to_string())
    return t


def _empirical(stack, final, X, cols, v, slope, pga_obs):
    """Gözlenen heyelan alan yoğunluğu vs model, aynı ızgarada (eğim × PGA sınıfı)."""
    from rasterio.warp import reproject, Resampling
    from scipy.stats import spearmanr

    nat = 1.0 / 1200.0
    t_nat, w_nat, h_nat = build_grid(STUDY_BOUNDS, nat)
    mask = rasterize_inventory(t_nat, w_nat, h_nat).astype(np.float32)
    h, w = stack.shape
    dens = np.zeros((h, w), np.float32)
    reproject(mask, dens, src_transform=t_nat, src_crs="EPSG:4326",
              dst_transform=stack.transform, dst_crs="EPSG:4326", resampling=Resampling.average)
    dens = dens.ravel()
    p_obs = final.predict(X)
    prevalence = float(dens[v].mean())

    pga_bins = np.arange(0.0, 1.01, 0.1)
    rows = []
    for lo, hi in [(0, 15)] + SLOPE_BINS[:2] + [(35, 90)]:
        for a, b in zip(pga_bins[:-1], pga_bins[1:]):
            m = v & (slope >= lo) & (slope < hi) & (pga_obs >= a) & (pga_obs < b)
            if m.sum() < 30:
                continue
            rows.append({"slope_class": f"{lo}–{hi}°", "pga_mid": round(a + 0.05, 2),
                         "n_cells": int(m.sum()),
                         "obs_density": float(dens[m].mean()),
                         "model_p": float(p_obs[m].mean())})
    e = pd.DataFrame(rows)
    e["frequency_ratio"] = e["obs_density"] / prevalence
    pi = PI_TRAIN
    e["model_lr"] = (e["model_p"] / (1 - e["model_p"])) * ((1 - pi) / pi)
    e.to_csv(RESULTS / "empirical_density.csv", index=False)

    # Ampirik Kritik PGA: en dik %5 hücrede gözlenen FR'nin 1'i geçtiği PGA
    thr95 = float(np.percentile(slope[v], 95))
    pb = np.round(np.arange(0.0, 1.001, 0.05), 3)
    fr_rows = []
    for a, b in zip(pb[:-1], pb[1:]):
        m = v & (slope >= thr95) & (pga_obs >= a) & (pga_obs < b)
        if m.sum() >= 30:
            fr_rows.append((a + 0.025, float(dens[m].mean()) / prevalence, int(m.sum())))
    fr = np.array(fr_rows)
    emp_crit = critical_pga_from_curve(fr[:, 0], fr[:, 1], 1.0) if len(fr) else None
    pd.DataFrame(fr, columns=["pga_mid", "frequency_ratio", "n_cells"]).to_csv(
        RESULTS / "empirical_fr_steep.csv", index=False)
    rho, pval = spearmanr(e["obs_density"], e["model_p"])
    rho_cell, _ = spearmanr(dens[v], p_obs[v])
    _plot_empirical(e)
    res = {"empirical_critical_pga_fr1_g": _r(emp_crit),
           "empirical_fr_steep": [{"pga_mid": round(a, 3), "fr": round(f, 2), "n": int(n)}
                                  for a, f, n in fr_rows],
           "spearman_binned": round(float(rho), 3), "spearman_binned_p": float(pval),
           "spearman_cells": round(float(rho_cell), 3),
           "landslide_area_prevalence": round(prevalence, 5), "n_bins": int(len(e)),
           "note": "Gözlenen = envanter poligonlarının ızgara hücresindeki alan oranı; "
                   "model = gözlenen PGA'da P(heyelan). Aynı 0.008° ızgara."}
    print("\nAmpirik doğrulama:", res)
    return res


def _neg_ratio_sensitivity(stack, df_grid, steep, cols, ratios=(1.0, 1.5, 2.5), epochs=150,
                           seeds=(1, 2, 3)):
    """Negatif/pozitif oranı değişince Kritik PGA (P=0.5 ve P=eşik) nasıl değişir?"""
    from src.data_engineering.feature_matrix import load_feature_table
    from src.evaluation.spatial_cv import assign_spatial_blocks, spatial_folds, inner_validation_split
    from src.evaluation.models import NeuralModel
    from src.evaluation.metrics import best_f1_threshold, classification_metrics

    t = load_feature_table(DATA_PROCESSED / "feature_matrix.parquet")
    y = t["label"].to_numpy(int)
    blocks = assign_spatial_blocks(t, 0.25)
    tr_all, te = spatial_folds(y, blocks, 5, seed=42)[0]
    Xt = t[cols].to_numpy(np.float32)
    Xg = df_grid[cols].to_numpy(np.float32)[steep]
    base_ratio = (y == 0).sum() / (y == 1).sum()
    out = []
    for r in ratios:
        res = []
        for sd in seeds:
            rng = np.random.RandomState(sd)
            keep_neg = rng.rand(len(t)) < min(1.0, r / base_ratio)
            sel = (y == 1) | keep_neg
            tr_r = tr_all[sel[tr_all]]
            tr, va = inner_validation_split(tr_r, y, blocks, seed=42)
            m = NeuralModel(cols, "pinn_v2", epochs=epochs, seed=sd).fit(Xt[tr], y[tr], Xt[va], y[va])
            curve = pga_response(m.predict, Xg, cols, GRID).mean(1)
            pi_r = float(y[tr].mean())
            res.append((critical_pga_from_curve(GRID, curve, 0.5),
                        critical_pga_from_curve(GRID, curve, pi_r),
                        classification_metrics(y[te], m.predict(Xt[te]), 0.5)["roc_auc"], pi_r))
        c05 = np.array([x[0] for x in res if x[0] is not None])
        con = np.array([x[1] for x in res if x[1] is not None])
        out.append({"neg_pos_ratio": r, "train_pos_rate": round(float(np.mean([x[3] for x in res])), 3),
                    "critical_pga_at_0.5_mean": _r(float(c05.mean())) if len(c05) else None,
                    "critical_pga_at_0.5_sd": _r(float(c05.std(ddof=1))) if len(c05) > 1 else None,
                    "critical_pga_onset_lr1_mean": _r(float(con.mean())) if len(con) else None,
                    "test_roc_auc_mean": round(float(np.mean([x[2] for x in res])), 3),
                    "n_seeds": len(res)})
        print("  duyarlılık:", out[-1])
    return out


def _r(x):
    return round(x, 3) if x is not None else None


# ------------------------------------------------------------------ grafikler
def _plot_uncertainty(P_final, P_folds, frac_dn, crit):
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(9.5, 6))
    F = np.array([P.mean(1) for P in P_folds]) if P_folds else None
    if F is not None and len(F):
        ax.fill_between(GRID, F.min(0), F.max(0), color=BLUE, alpha=0.15, lw=0)
        ax.text(GRID[-1] + 0.02, F.max(0)[-1] + 0.06, "bant: 5 CV katı (min–maks)", color=MUTED, fontsize=8.5, va="center")
    mu = np.mean([P_final.mean(1)] + ([*F] if F is not None else []), axis=0)
    ax.plot(GRID, mu, color=BLUE, lw=2.4)
    ax.text(GRID[-1] + 0.02, mu[-1], "PINN v2 — 6 model ort. P(heyelan)", color=INK, fontsize=9, va="center")
    ax.plot(GRID, frac_dn, color=ORANGE, lw=2, ls="--")
    ax.text(GRID[-1] + 0.02, frac_dn[-1] - 0.05, "Newmark — Dn ≥ 10 cm oranı", color=INK, fontsize=9, va="center")
    lv = crit["levels"]
    ax.axhline(lv["onset_lr1"]["level_p"], color=INK, ls=":", lw=1)
    ax.text(0.6, lv["onset_lr1"]["level_p"] + 0.012, "LR = 1 (heyelan lehine başlangıç)", color=INK, fontsize=8.5)
    ax.axhline(LEVEL, color=MUTED, ls=":", lw=1)
    ax.text(0.6, LEVEL + 0.012, "P = 0.5 (örnekleme oranına bağlı)", color=MUTED, fontsize=8.5)
    c = crit["critical_pga_g"]; rg = crit.get("ensemble_range_g")
    if c:
        if rg:
            ax.axvspan(rg[0], rg[1], color=BLUE, alpha=0.08, lw=0)
        ax.axvline(c, color=BLUE, lw=1, alpha=0.7)
        txt = f"Kritik PGA (P = 0.5) ≈ {c:.2f} g"
        if rg:
            txt += f"\n6 model aralığı {rg[0]:.2f}–{rg[1]:.2f} g"
        on = lv["onset_lr1"]["critical_pga_g"]
        if on:
            txt += f"\nBaşlangıç (LR = 1) ≈ {on:.2f} g"
        ax.annotate(txt, xy=(c, LEVEL), xytext=(c + 0.3, 0.12), fontsize=10,
                    color=INK, arrowprops=dict(arrowstyle="->", color=MUTED, lw=1))
    ax.set_xlim(0, 2.15); ax.set_ylim(0, 1)
    ax.set_xlabel("PGA (g)", color=MUTED); ax.set_ylabel("Olasılık / oran", color=MUTED)
    ax.set_title(f"Kritik PGA — en dik %5 hücre (eğim ≥ {crit['steep_slope_threshold_deg']}°, "
                 f"n = {crit['n_cells']:,})", loc="left", fontsize=12, color=INK)
    _style(ax); fig.tight_layout()
    fig.savefig(FIGURES_DIR / "critical_pga_uncertainty.png", dpi=170)
    fig.savefig(FIGURES_DIR / "critical_pga_curve_real.png", dpi=170)   # uygulamanın kullandığı ad
    plt.close(fig)


def _plot_design_table(t):
    if not len(t):
        return
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import LinearSegmentedColormap
    piv = t.pivot(index="litho", columns="slope_class", values="critical_pga_g")
    piv = piv[[c for c in [f"{lo}–{hi if hi < 90 else '90'}°" for lo, hi in SLOPE_BINS] if c in piv.columns]]
    # Tek ton (mavi) sıralı skala: düşük Kritik PGA = koyu = daha tehlikeli
    cmap = LinearSegmentedColormap.from_list("blue_seq", ["#0d3a75", "#2a78d6", "#a9cdf4", "#eef5fd"])
    fig, ax = plt.subplots(figsize=(8.5, 0.55 * len(piv) + 1.8))
    im = ax.imshow(piv.values.astype(float), cmap=cmap, vmin=0.05, vmax=1.5, aspect="auto")
    for i in range(piv.shape[0]):
        for j in range(piv.shape[1]):
            val = piv.values[i, j]
            if val is None or not np.isfinite(val):
                ax.text(j, i, "—", ha="center", va="center", color=MUTED, fontsize=9)
            else:
                ax.text(j, i, f"{val:.2f}", ha="center", va="center", fontsize=9,
                        color="white" if val < 0.6 else INK)
    ax.set_xticks(range(piv.shape[1]), piv.columns); ax.set_yticks(range(piv.shape[0]), piv.index)
    ax.tick_params(length=0, colors=MUTED)
    for s in ax.spines.values():
        s.set_visible(False)
    cb = fig.colorbar(im, ax=ax, shrink=0.8); cb.set_label("Kritik PGA (g)", color=MUTED)
    ax.set_title("Kritik PGA (P = 0.5, 6 model ortalama eğrisi) — eğim sınıfı × litoloji (— : 1.5 g'ye kadar ulaşılmaz)",
                 loc="left", fontsize=10.5, color=INK)
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "critical_pga_design_table.png", dpi=170); plt.close(fig)


def _plot_empirical(e):
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    classes = list(dict.fromkeys(e["slope_class"]))
    colors = ["#a3a29c", BLUE, ORANGE, AQUA]
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.2), sharey=True)
    for ax, col, title in [(axes[0], "frequency_ratio", "Gözlenen Frekans Oranı (envanter alanı / bölge ortalaması)"),
                           (axes[1], "model_lr", "Model olabilirlik oranı LR (gözlenen PGA'da)")]:
        for c, colr in zip(classes, colors):
            sub = e[(e.slope_class == c) & (e[col] > 0)]
            if not len(sub):
                continue
            ax.plot(sub["pga_mid"], sub[col], "o-", color=colr, lw=2, ms=6, mec="white", mew=1.5)
            ax.text(sub["pga_mid"].iloc[-1] + 0.02, sub[col].iloc[-1], f"eğim {c}", fontsize=8.5,
                    color=INK, va="center")
        ax.axhline(1.0, color=INK, ls=":", lw=1)
        ax.set_yscale("log")
        ax.set_xlabel("Gözlenen PGA sınıfı (g)", color=MUTED)
        ax.set_title(title, loc="left", fontsize=10.5, color=INK)
        ax.set_xlim(0, 1.15); _style(ax)
    axes[0].set_ylabel("Oran (log; 1 = nötr)", color=MUTED)
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "empirical_validation.png", dpi=170); plt.close(fig)


def _update_app_metrics(crit, emp, sens):
    p = PROJECT_ROOT / "backend" / "metrics.json"
    m = json.loads(p.read_text()) if p.exists() else {}
    m["critical_pga_ml"] = crit["critical_pga_g"]
    m["critical_pga"] = {k: crit[k] for k in ("critical_pga_g", "ensemble_range_g", "bootstrap_95ci_g",
                                               "newmark_critical_pga_g", "steep_slope_threshold_deg",
                                               "n_cells", "definition", "levels")}
    m["critical_pga_p05"] = crit["levels"]["p05"]["critical_pga_g"]
    m["empirical_validation"] = emp
    if sens:
        m["neg_ratio_sensitivity"] = sens
    p.write_text(json.dumps(m, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--res", type=float, default=0.008)
    ap.add_argument("--skip-sensitivity", action="store_true")
    a = ap.parse_args()
    run(a.res, not a.skip_sensitivity)
