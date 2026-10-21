"""
Script 16: "Heyelan yok" Etiketinin Kirliliğine Duyarlılık (PU öğrenme bakışı)

Envanter eksik olabilir (kar, gölge; Kocaman ve ark. 2025). Bu durumda y = 0
"kesin yokluk" değil, "etiketsiz"dir (positive–unlabeled, PU). Dört eğitim
varyantı AYNI mekânsal test bloklarında (tüm negatiflerle) karşılaştırılır:

  V0  temel        : negatifler heyelanlara ≥ 200 m (mevcut)
  V1  tampon 1 km  : eğitim negatifleri heyelanlara ≥ 1 km
  V2  tampon 2 km  : eğitim negatifleri heyelanlara ≥ 2 km
  V3  PU-casus     : V0'ın katlar-dışı tahmininde en heyelan-benzeri %5 negatif
                     (olası kaçmış heyelan) eğitimden çıkarılır

Raporlanan: test AUC / dik arazi AUC, Kritik PGA (P = 0,5 ve LR = 1, en dik %5
ızgara hücresi) ve Elkan & Noto (2008) etiketleme olasılığı c = E[P(s=1|x) | y=1].

Çıktı: outputs/results/label_noise.json
"""
import sys
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))

from config import DATA_PROCESSED, OUTPUTS_DIR, Z_REF_M, INVENTORY_KML
from src.data_engineering.feature_matrix import load_feature_table, MODEL_FEATURES_V3
from src.data_engineering.regional_grid import load_regional_stack, stack_features
from src.evaluation.spatial_cv import assign_spatial_blocks, spatial_folds, inner_validation_split
from src.evaluation.models import NeuralModel
from src.evaluation.metrics import classification_metrics, best_f1_threshold, pga_response, critical_pga_from_curve

GRID = np.round(np.arange(0.02, 1.501, 0.02), 3)


def dist_to_inventory_m(df):
    import geopandas as gpd
    inv = gpd.read_file(INVENTORY_KML)
    inv = inv[inv.geom_type.isin(["Polygon", "MultiPolygon"])].to_crs(32637)
    pts = gpd.GeoDataFrame(geometry=gpd.points_from_xy(df["lon"], df["lat"]), crs=4326).to_crs(32637)
    j = gpd.sjoin_nearest(pts, inv[["geometry"]], distance_col="d")
    return j.groupby(level=0)["d"].min().reindex(range(len(df))).to_numpy()


CACHE = OUTPUTS_DIR / "cache" / "label_noise"


def run(epochs=150, budget_s=480):
    t0 = time.time()
    df = load_feature_table(DATA_PROCESSED / "feature_matrix.parquet").reset_index(drop=True)
    cols = list(MODEL_FEATURES_V3)
    X = df[cols].to_numpy(np.float32); y = df["label"].to_numpy(int)
    slope = df["slope_deg"].to_numpy()
    d_inv = dist_to_inventory_m(df)
    blocks = assign_spatial_blocks(df, 0.25)
    folds = spatial_folds(y, blocks, 5)
    print(f"Negatiflerin heyelana uzaklığı: medyan {np.median(d_inv[y==0])/1000:.1f} km; "
          f"<1 km: %{100*(d_inv[y==0] < 1000).mean():.1f}, <2 km: %{100*(d_inv[y==0] < 2000).mean():.1f}")

    CACHE.mkdir(parents=True, exist_ok=True)
    gfile = CACHE / "steep_grid_X.npy"
    if gfile.exists():
        Xg = np.load(gfile)
    else:
        stack = load_regional_stack(0.008)
        G = stack_features(stack, Z_REF_M)
        v = stack.valid.ravel(); gs = G["slope_deg"].to_numpy()
        steep = np.where(v & (gs >= np.percentile(gs[v], 95)))[0]
        Xg = G[cols].to_numpy(np.float32)[steep]
        np.save(gfile, Xg)
    print(f"Izgara hazır ({time.time()-t0:.0f} s)", flush=True)

    # V0 katlar-dışı tahminleri (PU-casus için) + kat sonuçları önbellekten
    ofile, pfile = CACHE / "oof_v0.npy", CACHE / "partial.json"
    oof = np.load(ofile) if ofile.exists() else np.full(len(df), np.nan)
    partial = json.loads(pfile.read_text()) if pfile.exists() else {}
    variants = {"V0_temel": None, "V1_tampon_1km": 1000, "V2_tampon_2km": 2000, "V3_PU_casus": "spy"}
    res = {k: {"auc": [], "auc_steep": [], "crit05": [], "crit_lr1": [], "n_neg_train": [], "c_hat": []}
           for k in variants}
    for name, rule in variants.items():
        for k, (tr, te) in enumerate(folds, 1):
            tag = f"{name}|{k}"
            if tag in partial:
                for kk, vv in partial[tag].items():
                    res[name][kk].append(vv)
                continue
            if time.time() - t0 > budget_s:
                print("Süre bütçesi doldu — yeniden çalıştırınca kaldığı yerden devam eder.")
                return
            tr_in, va = inner_validation_split(tr, y, blocks, seed=k)
            keep = np.ones(len(df), bool)
            if isinstance(rule, (int, float)):
                keep = (y == 1) | (d_inv >= rule)
            elif rule == "spy":
                thr = np.nanpercentile(oof[tr_in][y[tr_in] == 0], 95)
                keep = (y == 1) | ~(oof >= thr)
            tr2 = tr_in[keep[tr_in]]
            m = NeuralModel(cols, "pinn_v2", epochs=epochs, seed=k).fit(X[tr2], y[tr2], X[va], y[va])
            p_te = m.predict(X[te])
            if name == "V0_temel":
                oof[te] = p_te          # 5 kat test blokları tüm örnekleri kapsar → tam OOF
                np.save(ofile, oof)
            met = classification_metrics(y[te], p_te, best_f1_threshold(y[va], m.predict(X[va])), slope=slope[te])
            curve = pga_response(m.predict, Xg, cols, GRID).mean(1)
            pi = float(y[tr2].mean())
            r = res[name]
            r["auc"].append(met["roc_auc"]); r["auc_steep"].append(met.get("roc_auc_steep", np.nan))
            r["crit05"].append(critical_pga_from_curve(GRID, curve, 0.5))
            r["crit_lr1"].append(critical_pga_from_curve(GRID, curve, pi))
            r["n_neg_train"].append(int((y[tr2] == 0).sum()))
            r["c_hat"].append(float(np.mean(m.predict(X[va][y[va] == 1]))))
            partial[tag] = {kk: r[kk][-1] for kk in r}
            pfile.write_text(json.dumps(partial))
            print(f"  {name} kat {k}: AUC {met['roc_auc']:.3f} | Kritik PGA 0.5: {r['crit05'][-1]} | "
                  f"LR=1: {r['crit_lr1'][-1]} ({time.time()-t0:.0f} s)", flush=True)
    out = {"description": "Aynı mekânsal test blokları (tüm negatiflerle); yalnız eğitim negatifleri değişir.",
           "neg_distance_km": {"median": round(float(np.median(d_inv[y == 0]) / 1000), 2),
                               "share_lt_1km": round(float((d_inv[y == 0] < 1000).mean()), 4),
                               "share_lt_2km": round(float((d_inv[y == 0] < 2000).mean()), 4)},
           "variants": {}}
    f = lambda a: [x for x in a if x is not None and np.isfinite(x)]
    for name, r in res.items():
        out["variants"][name] = {
            "roc_auc": [round(float(np.mean(r["auc"])), 4), round(float(np.std(r["auc"], ddof=1)), 4)],
            "roc_auc_steep": [round(float(np.nanmean(r["auc_steep"])), 4), round(float(np.nanstd(r["auc_steep"], ddof=1)), 4)],
            "critical_pga_p05_g": [round(float(np.mean(f(r["crit05"]))), 3), round(float(np.std(f(r["crit05"]), ddof=1)), 3)] if len(f(r["crit05"])) > 1 else None,
            "critical_pga_lr1_g": [round(float(np.mean(f(r["crit_lr1"]))), 3), round(float(np.std(f(r["crit_lr1"]), ddof=1)), 3)] if len(f(r["crit_lr1"])) > 1 else None,
            "n_neg_train_mean": int(np.mean(r["n_neg_train"])),
            "elkan_noto_c": round(float(np.mean(r["c_hat"])), 3),
        }
    (OUTPUTS_DIR / "results" / "label_noise.json").write_text(json.dumps(out, indent=2, ensure_ascii=False))
    print(json.dumps(out, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--budget", type=float, default=480, help="çalıştırma başına saniye")
    run(budget_s=ap.parse_args().budget)
