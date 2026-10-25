"""
Script 17: Yağış Senaryosu — Deprem + Zemin Doygunluğu (hidro-mekanik)

Yağış, boşluk suyu basıncını (u = m·γ_w·z·cos²θ) artırır → FS ve Kᵧ düşer → aynı
PGA'da Dₙ ve heyelan olasılığı artar. PINN v2'nin fizik-yolu kısıtı doygunluk
yolunu da içerir (m ↑ ⇒ P ↓ olamaz).

Adımlar
  1. Olay nemi: ERA5-Land kök bölgesi (0–100 cm) toprak nemi, 6 Şubat 2023 01:00 UTC
     (Pazarcık depreminden hemen önce) ve önceki 30 günlük yağış (Open-Meteo arşivi,
     0.25° ızgara) → doygunluk derecesi S = θ/θ_s (θ_s = 0,45) → m ≈ S (vekil)
  2. Senaryo taraması: m = 0 … 1 için en dik %5 hücrede Kritik PGA (P = 0,5 ve
     LR = 1; 6 model) ve Newmark Dₙ ≥ 10 cm oranı
  3. Olay doğrulaması: nihai modelin mekânsal test bloklarında, gözlenen PGA ile
     kuru (m = 0) ve olay nemi (m = S) tahminlerinin envanteri ayırma gücü
  4. Bölgesel kuru / ıslak / fark haritaları

Çıktılar: outputs/results/rainfall_scenario.json, figures rainfall_*.png,
          data/processed/layers/saturation_20230206.tif, precip_30d_mm.tif

Kullanım: caffeinate -i -s python3 scripts/17_rainfall_scenario.py
"""
import sys
import json
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from config import DATA_PROCESSED, OUTPUTS_DIR, FIGURES_DIR, MODELS_DIR, Z_REF_M, DN_THRESHOLD_M
from src.data_engineering.regional_grid import (load_regional_stack, stack_features, STUDY_BOUNDS,
                                                reproject_to_grid)
from src.data_engineering.extra_layers import write_layer, fine_grid, LAYER_DIR
from src.data_engineering.feature_matrix import load_feature_table
from src.data_engineering.real_dataset import sample_raster
from src.evaluation.metrics import pga_response, critical_pga_from_curve, classification_metrics
from src.evaluation.spatial_cv import assign_spatial_blocks, spatial_folds
from src.geomechanics.hydrology import saturation_from_moisture, pore_pressure
from src.geomechanics.newmark import NewmarkDisplacement
from src.pinn.io import load_checkpoint

ARCHIVE = "https://archive-api.open-meteo.com/v1/archive"
EVENT_HOUR = "2023-02-06T01:00"
GRID = np.round(np.arange(0.02, 1.501, 0.02), 3)
M_LEVELS = [0.0, 0.25, 0.5, 0.75, 1.0]
INK, MUTED, BLUE, ORANGE = "#0b0b0b", "#52514e", "#2a78d6", "#eb6834"


def fetch_event_moisture(step=0.25):
    """ERA5-Land (Open-Meteo) 0.25° ızgara: kök bölgesi nemi ve 30 günlük yağış."""
    from urllib.request import urlopen
    from urllib.parse import urlencode
    from src.data_engineering.strong_motion import _ssl_context
    lon0, lat0, lon1, lat1 = STUDY_BOUNDS
    lats = np.round(np.arange(lat0, lat1 + 1e-9, step), 3)
    lons = np.round(np.arange(lon0, lon1 + 1e-9, step), 3)
    LA, LO = np.meshgrid(lats, lons, indexing="ij")
    pts = list(zip(LA.ravel(), LO.ravel()))
    theta = np.full(len(pts), np.nan); p30 = np.full(len(pts), np.nan)
    for i in range(0, len(pts), 50):
        chunk = pts[i:i + 50]
        q = urlencode({"latitude": ",".join(str(a) for a, _ in chunk),
                       "longitude": ",".join(str(b) for _, b in chunk),
                       "start_date": "2023-01-07", "end_date": "2023-02-06",
                       "hourly": "soil_moisture_0_to_7cm,soil_moisture_7_to_28cm,soil_moisture_28_to_100cm",
                       "daily": "precipitation_sum", "timezone": "GMT"})
        data = json.load(urlopen(f"{ARCHIVE}?{q}", context=_ssl_context(), timeout=120))
        data = data if isinstance(data, list) else [data]
        for j, d in enumerate(data):
            h = d["hourly"]; t = h["time"].index(EVENT_HOUR)
            s1, s2, s3 = (h[k][t] for k in ("soil_moisture_0_to_7cm", "soil_moisture_7_to_28cm",
                                             "soil_moisture_28_to_100cm"))
            if None not in (s1, s2, s3):
                theta[i + j] = (7 * s1 + 21 * s2 + 72 * s3) / 100.0       # 0–100 cm ağırlıklı
            pr = [x for x in d["daily"]["precipitation_sum"][:-1] if x is not None]
            p30[i + j] = float(np.sum(pr)) if pr else np.nan
        print(f"  ERA5-Land: {min(i + 50, len(pts))}/{len(pts)} nokta", flush=True)
    from rasterio.transform import from_origin
    tr = from_origin(lons[0] - step / 2, lats[-1] + step / 2, step, step)
    th = theta.reshape(len(lats), len(lons))[::-1]
    pr = p30.reshape(len(lats), len(lons))[::-1]
    # Kıyı/deniz hücrelerini en yakın kara değeriyle doldur (ERA5-Land yalnız kara)
    from scipy.ndimage import distance_transform_edt
    for a in (th, pr):
        bad = ~np.isfinite(a)
        if bad.any():
            idx = distance_transform_edt(bad, return_distances=False, return_indices=True)
            a[bad] = a[tuple(i[bad] for i in idx)]
    return th, pr, tr


def run():
    t0 = time.time()
    import rasterio
    th, pr, tr = fetch_event_moisture()
    S = saturation_from_moisture(th)
    print(f"Olay anı kök bölgesi nemi: medyan θ={np.median(th):.3f} m³/m³ → S={np.median(S):.2f}; "
          f"30 günlük yağış medyan {np.median(pr):.0f} mm")
    ftr, w, h = fine_grid(STUDY_BOUNDS)
    tmp = LAYER_DIR / "_sat_coarse.tif"
    prof = {"driver": "GTiff", "height": S.shape[0], "width": S.shape[1], "count": 1,
            "dtype": "float32", "crs": "EPSG:4326", "transform": tr}
    for name, arr in (("saturation_20230206", S), ("precip_30d_mm", pr)):
        with rasterio.open(tmp, "w", **prof) as d:
            d.write(arr.astype(np.float32), 1)
        write_layer(name, reproject_to_grid(tmp, ftr, w, h, "bilinear"), ftr)
    tmp.unlink(missing_ok=True)

    final = load_checkpoint(MODELS_DIR / "best_model.pt", MODELS_DIR / "scaler.pkl", MODELS_DIR / "threshold.json")
    ver = final.meta.get("model_version", "pinn_v2")
    folds = [load_checkpoint(p) for p in sorted((MODELS_DIR / "cv_ensemble" / ver).glob("fold*.pt"))]
    models = [final] + folds
    cols = final.feat_cols
    pi = float(load_feature_table(DATA_PROCESSED / "feature_matrix.parquet")["label"].mean())

    stack = load_regional_stack(0.008)
    S_grid = reproject_to_grid(LAYER_DIR / "saturation_20230206.tif", stack.transform,
                               stack.shape[1], stack.shape[0], "bilinear")
    S_grid = np.clip(np.nan_to_num(S_grid, nan=float(np.nanmedian(S_grid))), 0, 1)
    v = stack.valid.ravel()
    sl = stack.slope.ravel()
    steep = np.where(v & (sl >= np.percentile(sl[v], 95)))[0]

    # ---- 2) senaryo taraması -------------------------------------------------
    scen = []
    for m in M_LEVELS:
        G = stack_features(stack, Z_REF_M, saturation=m)
        X = G[cols].to_numpy(np.float32)[steep]
        curves = [pga_response(mm.predict, X, cols, GRID).mean(1) for mm in models]
        ens = np.mean(curves, axis=0)
        ky = G["ky_g"].to_numpy()[steep]
        frac = [float((NewmarkDisplacement(np.maximum(ky, 0.0), pg) >= DN_THRESHOLD_M).mean()) for pg in GRID]
        c05 = [critical_pga_from_curve(GRID, c, 0.5) for c in curves]
        c05 = [x for x in c05 if x is not None]
        scen.append({"m": m,
                     "critical_pga_p05_g": _r(critical_pga_from_curve(GRID, ens, 0.5)),
                     "p05_model_range_g": [_r(min(c05)), _r(max(c05))] if c05 else None,
                     "onset_lr1_g": _r(critical_pga_from_curve(GRID, ens, pi)),
                     "newmark_p50_g": _r(critical_pga_from_curve(GRID, np.array(frac), 0.5)),
                     "median_ky_g": round(float(np.median(ky)), 3),
                     "share_static_unstable": round(float((ky <= 0.0).mean()), 3),
                     "curve": [round(float(x), 4) for x in ens],
                     "newmark_curve": [round(x, 4) for x in frac]})
        print(f"  m={m:.2f}: Kritik PGA P05={scen[-1]['critical_pga_p05_g']} g, LR1={scen[-1]['onset_lr1_g']} g, "
              f"Newmark %50={scen[-1]['newmark_p50_g']} g, medyan Kᵧ={scen[-1]['median_ky_g']} g "
              f"({time.time()-t0:.0f} s)", flush=True)

    # ---- 3) olay doğrulaması (nihai modelin test blokları) -------------------
    df = load_feature_table(DATA_PROCESSED / "feature_matrix.parquet").reset_index(drop=True)
    y = df["label"].to_numpy(int)
    _, te = spatial_folds(y, assign_spatial_blocks(df, 0.25), 5, seed=42)[0]
    s_pts = np.clip(np.nan_to_num(sample_raster(LAYER_DIR / "saturation_20230206.tif",
                                                df["lon"].to_numpy(), df["lat"].to_numpy()),
                                  nan=float(np.median(S))), 0, 1)
    dry = df.iloc[te]
    wet = _apply_saturation(df.iloc[te].copy(), s_pts[te])
    p_dry = final.predict(dry[cols].to_numpy(np.float32))
    p_wet = final.predict(wet[cols].to_numpy(np.float32))
    m_dry = classification_metrics(y[te], p_dry, final.threshold, slope=dry["slope_deg"].to_numpy())
    m_wet = classification_metrics(y[te], p_wet, final.threshold, slope=dry["slope_deg"].to_numpy())
    dn_dry = dry["dn_cm"].to_numpy(); dn_wet = wet["dn_cm"].to_numpy()
    from sklearn.metrics import roc_auc_score
    event = {
        "saturation_median": round(float(np.median(S)), 3),
        "saturation_range": [round(float(np.min(S)), 3), round(float(np.max(S)), 3)],
        "precip_30d_median_mm": round(float(np.nanmedian(pr)), 1),
        "model_auc_dry": round(m_dry["roc_auc"], 4), "model_auc_event_wet": round(m_wet["roc_auc"], 4),
        "model_auc_steep_dry": round(m_dry.get("roc_auc_steep", np.nan), 4),
        "model_auc_steep_wet": round(m_wet.get("roc_auc_steep", np.nan), 4),
        "newmark_auc_dry": round(float(roc_auc_score(y[te], np.log10(dn_dry + 0.1))), 4),
        "newmark_auc_wet": round(float(roc_auc_score(y[te], np.log10(dn_wet + 0.1))), 4),
        "newmark_hit_rate_dry": round(float((dn_dry[y[te] == 1] >= 10).mean()), 4),
        "newmark_hit_rate_wet": round(float((dn_wet[y[te] == 1] >= 10).mean()), 4),
        "newmark_false_rate_dry": round(float((dn_dry[y[te] == 0] >= 10).mean()), 4),
        "newmark_false_rate_wet": round(float((dn_wet[y[te] == 0] >= 10).mean()), 4),
        "mean_p_landslide_dry": round(float(p_dry[y[te] == 1].mean()), 4),
        "mean_p_landslide_wet": round(float(p_wet[y[te] == 1].mean()), 4),
    }
    print(json.dumps(event, indent=2, ensure_ascii=False))

    # ---- 4) bölgesel haritalar ----------------------------------------------
    hgt, wid = stack.shape
    P = {}
    for key, sat in (("dry", None), ("event", S_grid)):
        G = stack_features(stack, Z_REF_M, saturation=sat)
        P[key] = np.where(stack.valid, final.predict(G[cols].to_numpy(np.float32)).reshape(hgt, wid), np.nan)
    with rasterio.open(DATA_PROCESSED / "susceptibility_pinn.tif") as ds:
        prof = ds.profile
    with rasterio.open(DATA_PROCESSED / "susceptibility_wet_20230206.tif", "w", **prof) as d:
        d.write(P["event"].astype(np.float32), 1)

    out = {"assumptions": {
               "pore_pressure": "u = m·γw·z·cos²θ, yamaca paralel sızma, z = 5 m",
               "m_event": "m ≈ S = θ(0–100 cm, ERA5-Land, 2023-02-06 01 UTC)/θs, θs = 0,45",
               "model": "PINN v2 + katmanlar; doygunluk yolu fizik-yolu kısıtında; eğitim verisi kuru (u = 0) — "
                        "ıslak senaryo, fiziksel girdiler (Kᵧ, Dₙ) üzerinden kısıtlı bir dışkestirimdir"},
           "pga_grid_g": GRID.tolist(), "scenarios": scen, "event_2023_02_06": event,
           "regional_mean_p": {"dry": round(float(np.nanmean(P['dry'])), 4),
                               "event_wet": round(float(np.nanmean(P['event'])), 4)}}
    (OUTPUTS_DIR / "results" / "rainfall_scenario.json").write_text(json.dumps(out, indent=2, ensure_ascii=False))
    _plots(scen, P, S, pr, stack)
    print(f"Süre: {time.time()-t0:.0f} s")


def _apply_saturation(df, m):
    """Eğitim tablosu satırlarında Kᵧ ve Dₙ'yi doygunluk m ile yeniden hesaplar."""
    from src.geomechanics.mohr_coulomb import StaticFS, CriticalAcceleration
    from src.data_engineering.feature_matrix import DN_LOG_OFFSET_CM
    th = df["slope_deg"].to_numpy()
    u = pore_pressure(m, Z_REF_M, th)
    fs = StaticFS(df["cohesion_kpa"].to_numpy(), df["friction_deg"].to_numpy(),
                  df["unit_weight"].to_numpy(), Z_REF_M, th, u)
    ky = CriticalAcceleration(fs, th)
    dn = NewmarkDisplacement(ky, df["pga_g"].to_numpy()) * 100
    df["ky_g"] = ky; df["dn_cm"] = dn; df["log_dn_cm"] = np.log10(dn + DN_LOG_OFFSET_CM)
    return df


def _r(x):
    return None if x is None else round(float(x), 3)


def _plots(scen, P, S, pr, stack):
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    # Senaryo eğrileri
    cols = ["#9ec5f0", "#5f9fe3", "#2a78d6", "#1a56a3", "#0d3a75"]
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    for s, c in zip(scen, cols):
        axes[0].plot(GRID, s["curve"], color=c, lw=2)
        if s["m"] in (0.0, 1.0):   # eğriler birbirine yakın — yalnız uçları etiketle
            axes[0].text(GRID[-1] + 0.02, s["curve"][-1] + (0.015 if s["m"] == 1.0 else -0.015),
                         f"m = {s['m']:.0f}" + (" (doygun)" if s["m"] == 1.0 else " (kuru)"),
                         fontsize=8.5, va="center")
        axes[1].plot(GRID, s["newmark_curve"], color=c, lw=2)
        if s["m"] in (0.0, 0.25, 1.0):
            axes[1].text(GRID[-1] + 0.02, s["newmark_curve"][-1], f"m = {s['m']:.2f}", fontsize=8.5, va="center")
    axes[0].axhline(0.5, color=MUTED, ls=":", lw=1)
    axes[0].text(0.02, 0.96, "koyu = daha ıslak (m = 0; 0,25; 0,5; 0,75; 1)", transform=axes[0].transAxes,
                 fontsize=8.5, color=MUTED, va="top")
    axes[0].set_title("PINN — en dik %5 hücre, ort. P(heyelan)", loc="left", fontsize=11)
    axes[1].axhline(0.5, color=MUTED, ls=":", lw=1)
    axes[1].set_title("Newmark — Dₙ ≥ 10 cm hücre oranı", loc="left", fontsize=11)
    for ax in axes:
        ax.set_xlabel("PGA (g)", color=MUTED); ax.set_xlim(0, 1.75); ax.set_ylim(0, 1)
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
    fig.suptitle("Yağış senaryosu: su tablası oranı m arttıkça (u = m·γw·z·cos²θ) tetiklenme daha düşük PGA'da",
                 x=0.01, ha="left", fontsize=11.5)
    fig.tight_layout(); fig.savefig(FIGURES_DIR / "rainfall_scenario_curves.png", dpi=160); plt.close(fig)

    lon0, lat0, lon1, lat1 = stack.bounds
    ext = [lon0, lon1, lat0, lat1]
    fig, axes = plt.subplots(1, 3, figsize=(17, 5))
    im0 = axes[0].imshow(P["dry"], extent=ext, cmap="RdYlGn_r", vmin=0, vmax=1)
    axes[0].set_title("Kuru (m = 0)", loc="left")
    axes[1].imshow(P["event"], extent=ext, cmap="RdYlGn_r", vmin=0, vmax=1)
    axes[1].set_title("6 Şubat 2023 nemi (ERA5-Land)", loc="left")
    d = P["event"] - P["dry"]
    im2 = axes[2].imshow(d, extent=ext, cmap="Blues", vmin=0, vmax=max(0.05, float(np.nanpercentile(d, 99))))
    axes[2].set_title("Fark (ıslak − kuru)", loc="left")
    fig.colorbar(im0, ax=axes[:2], shrink=0.8, label="P(heyelan)")
    fig.colorbar(im2, ax=axes[2], shrink=0.8, label="ΔP")
    fig.savefig(FIGURES_DIR / "rainfall_maps.png", dpi=140, bbox_inches="tight"); plt.close(fig)


if __name__ == "__main__":
    run()
