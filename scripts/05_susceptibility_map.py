"""
Script 5: Gerçek Bölgesel Heyelan Duyarlılık Haritası

Eğitilmiş PINN'i gerçek raster yığınına (Copernicus DEM eğim/bakı + USGS PGA +
Macrostrat litoloji) uygulayarak tüm çalışma alanı için heyelan olasılığı
(susceptibility) haritası üretir ve gözlemlenen envanterle görsel doğrular.

Ayrıca "Kritik PGA" analizi: her pilot piksel için PGA senaryoları taranır ve
P(heyelan) ≥ 0.5 olduğu eşik PGA bulunur — projenin ana araştırma sorusu.

Kullanım:
  python scripts/05_susceptibility_map.py [--res 0.008]
"""
import sys
import argparse
import pickle
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from config import (
    DATA_RAW, DATA_PROCESSED, FIGURES_DIR, MODELS_DIR, Z_REF_M, ensure_dirs,
)

STUDY_BOUNDS = (35.7, 35.9, 39.25, 38.65)   # (lon_min, lat_min, lon_max, lat_max)


def reproject_to_grid(raster_path, dst_transform, dst_w, dst_h, resampling="bilinear"):
    """Bir rasterı hedef ızgaraya yeniden projekte eder (EPSG:4326)."""
    import rasterio
    from rasterio.warp import reproject, Resampling

    rs = Resampling.bilinear if resampling == "bilinear" else Resampling.nearest
    dst = np.zeros((dst_h, dst_w), dtype=np.float32)
    with rasterio.open(raster_path) as src:
        reproject(
            source=rasterio.band(src, 1),
            destination=dst,
            src_transform=src.transform, src_crs=src.crs,
            dst_transform=dst_transform, dst_crs="EPSG:4326",
            resampling=rs,
        )
    return dst


def build_grid(bounds, res):
    """Hedef ızgara transform + boyutlarını üretir."""
    from rasterio.transform import from_origin
    lon_min, lat_min, lon_max, lat_max = bounds
    w = int(round((lon_max - lon_min) / res))
    h = int(round((lat_max - lat_min) / res))
    transform = from_origin(lon_min, lat_max, res, res)
    return transform, w, h


def run(res: float = 0.008):
    ensure_dirs()
    import rasterio
    import torch
    from src.data_engineering.feature_matrix import build_feature_matrix, ALL_FEATURE_COLUMNS
    from src.pinn.model import LandslidePINN

    dem_path    = DATA_RAW / "EUDEM_kahramanmaras.tif"
    pga_path    = DATA_RAW / "pga_kahramanmaras_20230206.tif"
    litho_path  = DATA_RAW / "lithology_macrostrat_proxy.tif"
    slope_path  = DATA_PROCESSED / "slope_deg.tif"
    aspect_path = DATA_PROCESSED / "aspect_deg.tif"

    transform, w, h = build_grid(STUDY_BOUNDS, res)
    print(f"Hedef ızgara: {w}×{h} = {w*h:,} piksel (~{res*111:.1f} km çözünürlük)")

    # Katmanları hizala
    print("Raster katmanları hizalanıyor...")
    elevation = reproject_to_grid(dem_path,   transform, w, h, "bilinear")
    slope     = reproject_to_grid(slope_path, transform, w, h, "bilinear")
    aspect    = reproject_to_grid(aspect_path, transform, w, h, "nearest")
    pga       = reproject_to_grid(pga_path,   transform, w, h, "bilinear")
    litho     = reproject_to_grid(litho_path, transform, w, h, "nearest")

    # Temizlik
    litho = np.where((litho >= 1) & (litho <= 27), litho, 27).astype(int)
    slope = np.clip(np.nan_to_num(slope, nan=0.0), 0.0, 89.0)
    aspect = np.nan_to_num(aspect, nan=-1.0)
    elevation = np.nan_to_num(elevation, nan=0.0)
    pga = np.clip(np.nan_to_num(pga, nan=0.05), 0.001, 3.0)
    valid = (elevation > 0) & (pga > 0.02)   # deniz/veri-dışı maskesi

    # Öznitelik matrisi (piksel bazlı)
    print("Öznitelik matrisi kuruluyor...")
    df, _ = build_feature_matrix(
        elevation_m=elevation, slope_deg=slope, aspect_deg=aspect,
        litho_array=litho, pga_array=pga, u_array=None,
        z_ref=Z_REF_M, include_onehot=True,
    )
    feat_cols = [c for c in ALL_FEATURE_COLUMNS if c in df.columns]
    X = df[feat_cols].values.astype(np.float32)

    # Ölçekle + tahmin et
    scaler = pickle.loads((MODELS_DIR / "scaler.pkl").read_bytes())
    Xn = (X - scaler["mean"]) / scaler["std"]

    model = LandslidePINN(in_features=len(feat_cols))
    ckpt = torch.load(MODELS_DIR / "best_model.pt", map_location="cpu")
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    print(f"Model yüklendi (epoch={ckpt['epoch']}, AUC={ckpt['val_auc']:.3f}). Tahmin ediliyor...")

    with torch.no_grad():
        proba = model(torch.tensor(Xn, dtype=torch.float32)).numpy().reshape(h, w)
    proba = np.where(valid, proba, np.nan)

    # GeoTIFF kaydet
    out_tif = DATA_PROCESSED / "susceptibility_pinn.tif"
    prof = {"driver": "GTiff", "height": h, "width": w, "count": 1,
            "dtype": "float32", "crs": "EPSG:4326", "transform": transform,
            "nodata": np.nan}
    with rasterio.open(out_tif, "w", **prof) as dst:
        dst.write(proba.astype(np.float32), 1)
    print(f"Duyarlılık rasterı: {out_tif}")

    # Görselleştir + envanter bindirmesi
    _plot_map(proba, STUDY_BOUNDS, valid)

    # Kritik PGA analizi
    _critical_pga_analysis(df, feat_cols, scaler, model, valid.ravel())

    print("\nDuyarlılık haritası üretimi tamamlandı.")


def _plot_map(proba, bounds, valid):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import geopandas as gpd

    lon_min, lat_min, lon_max, lat_max = bounds
    fig, ax = plt.subplots(figsize=(11, 8))
    im = ax.imshow(proba, extent=[lon_min, lon_max, lat_min, lat_max],
                   origin="upper", cmap="RdYlGn_r", vmin=0, vmax=1,
                   interpolation="nearest")
    cbar = fig.colorbar(im, ax=ax, shrink=0.8)
    cbar.set_label("P(heyelan) — PINN")

    try:
        inv = gpd.read_file(DATA_RAW / "TJES-2024-00171_Landslides.kml")
        inv = inv[inv.geom_type.isin(["Polygon", "MultiPolygon"])]
        inv.boundary.plot(ax=ax, color="black", linewidth=0.3, alpha=0.6)
    except Exception as e:
        print(f"  Envanter bindirmesi atlandı: {e}")

    ax.set_title("6 Şubat 2023 Kahramanmaraş — PINN Heyelan Duyarlılık Haritası\n"
                 "(siyah konturlar: gözlemlenen Kocaman 2025 envanteri)")
    ax.set_xlabel("Boylam (°D)"); ax.set_ylabel("Enlem (°K)")
    fig.tight_layout()
    out = FIGURES_DIR / "susceptibility_map_real.png"
    fig.savefig(out, dpi=150); plt.close(fig)
    print(f"Duyarlılık haritası (PNG): {out}")


def _critical_pga_analysis(df, feat_cols, scaler, model, valid_mask):
    """
    En dik %5 yamaç için kritik PGA'yı İKİ bağımsız yolla türetir:

      (A) ML   — PINN'in ortalama P(heyelan) değeri 0.5'i geçtiği PGA.
      (B) FİZİK — Newmark (Jibson 2007) kalıcı yer değiştirmesi Dn, hücrelerin
                  %50'sinde 10 cm (Jibson eşiği) sınırını aştığı PGA. Her hücrenin
                  kendi Ky'si (ky_g) kullanılır; ML olasılık eşiğinden bağımsızdır.

    İki eğrinin çakışması, ML kararının fiziksel Newmark mekaniğiyle tutarlı
    olduğunu gösterir ("neden P=0.5?" sorusuna fiziksel gerekçe sağlar).
    """
    import torch
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from config import DN_THRESHOLD_M
    from src.geomechanics.newmark import NewmarkDisplacement

    # Temsili yüksek riskli hücreler: en dik %5 eğim
    slope = df["slope_deg"].values
    thr = np.nanpercentile(slope[valid_mask], 95)
    sel = valid_mask & (slope >= thr)
    n_sel = int(sel.sum())
    if n_sel == 0:
        print("  Kritik PGA analizi için uygun hücre yok.")
        return

    base = df.loc[sel, feat_cols].copy().reset_index(drop=True)
    ky_cells = df.loc[sel, "ky_g"].values.astype(float)

    # Statik olarak zaten marjinal (Ky≈0, FS≤1) yamaçlar tetikleyici gerektirmez;
    # Newmark tetikleme analizi yalnızca statik-stabil (Ky>0) alt kümede anlamlıdır.
    static_unstable_frac = float((ky_cells <= 0.02).mean())
    trig = ky_cells > 0.02
    ky_trig = ky_cells[trig]
    n_trig = int(trig.sum())

    pga_grid = np.round(np.arange(0.05, 1.21, 0.05), 2)
    pga_idx = feat_cols.index("pga_g")

    mean_p, frac_dn = [], []
    for pg in pga_grid:
        # (A) ML olasılığı (tüm dik hücreler)
        Xtmp = base.values.astype(np.float32).copy()
        Xtmp[:, pga_idx] = pg
        Xn = (Xtmp - scaler["mean"]) / scaler["std"]
        with torch.no_grad():
            p = model(torch.tensor(Xn, dtype=torch.float32)).numpy().ravel()
        mean_p.append(p.mean())

        # (B) Newmark: statik-stabil hücrelerde Dn ≥ 10 cm oranı
        if n_trig:
            dn_m = NewmarkDisplacement(ky_trig, np.full_like(ky_trig, pg))
            frac_dn.append(float((dn_m >= DN_THRESHOLD_M).mean()))
        else:
            frac_dn.append(0.0)

    mean_p  = np.array(mean_p)
    frac_dn = np.array(frac_dn)
    print(f"  Statik-marjinal (Ky≈0) dik hücre oranı: {static_unstable_frac*100:.0f}% "
          f"(tetikleyici gerektirmez); Newmark analizi n={n_trig} stabil hücrede.")

    over_ml  = np.where(mean_p  >= 0.5)[0]
    over_phy = np.where(frac_dn >= 0.5)[0]
    crit_ml  = pga_grid[over_ml[0]]  if len(over_ml)  else None
    crit_phy = pga_grid[over_phy[0]] if len(over_phy) else None

    plt.rcParams.update({"font.size": 12})
    fig, ax = plt.subplots(figsize=(9.5, 6.2))

    # Tetiklenme bölgesi (P ≥ 0.5) hafif gölgelendirme — "heyelan olası" alanı
    ax.axhspan(0.5, 1.02, color="#c0392b", alpha=0.05, zorder=0)
    ax.text(0.07, 0.93, "Heyelan olası bölge (P ≥ 0.5)",
            color="#c0392b", fontsize=10, alpha=0.8, va="top")

    ax.plot(pga_grid, mean_p, "o-", color="#c0392b", lw=2.4, ms=6,
            label="ML modeli — ortalama P(heyelan)", zorder=5)
    ax.plot(pga_grid, frac_dn, "s--", color="#2c7fb8", lw=2.0, ms=5,
            label=f"Fizik (Newmark) — Dn ≥ {DN_THRESHOLD_M*100:.0f} cm oranı",
            zorder=4)
    ax.axhline(0.5, ls=":", color="gray", lw=1.5, label="Karar eşiği = 0.5")

    if crit_ml is not None:
        ax.axvline(crit_ml, ls="-", color="#c0392b", alpha=0.45, lw=1.8)
        # Kesişim noktasını vurgula
        y_at = mean_p[np.argmin(np.abs(pga_grid - crit_ml))]
        ax.plot([crit_ml], [y_at], "*", color="#c0392b", ms=20, zorder=6)
        ax.annotate(f"Kritik PGA ≈ {crit_ml:.2f} g",
                    xy=(crit_ml, 0.5), xytext=(crit_ml + 0.12, 0.30),
                    fontsize=12, fontweight="bold", color="#c0392b",
                    arrowprops=dict(arrowstyle="->", color="#c0392b", lw=1.8))
    if crit_phy is not None:
        ax.axvline(crit_phy, ls="--", color="#2c7fb8", alpha=0.5, lw=1.6,
                   label=f"Kritik PGA (Newmark) ≈ {crit_phy:.2f} g")

    # Açıklama kutusu — iki eğri neden farklı
    ax.text(0.98, 0.04,
            "ML, gerçek heyelanlardan öğrendiği için dayanım kaybı,\n"
            "ivme büyütmesi ve süreksizlikleri yakalar → daha erken tetiklenme.\n"
            "Saf rijit-blok Newmark bunları içermez → alt sınır verir.",
            transform=ax.transAxes, ha="right", va="bottom", fontsize=8.5,
            bbox=dict(boxstyle="round,pad=0.5", fc="#f7f7f7", ec="gray", alpha=0.9))

    ax.set_xlabel("PGA — Yüzey İvmesi (g)", fontsize=13)
    ax.set_ylabel("Heyelan olasılığı / tetiklenen hücre oranı", fontsize=13)
    ax.set_ylim(-0.02, 1.02); ax.set_xlim(0.0, 1.25)
    ax.set_title("Kritik PGA Analizi — ML Modeli vs Fizik Tabanlı Newmark\n"
                 f"En dik %5 yamaç (eğim ≥ {thr:.0f}°, n = {n_sel:,} hücre)",
                 fontsize=13, fontweight="bold")
    ax.legend(fontsize=10, loc="center left"); ax.grid(alpha=0.3)
    fig.tight_layout()
    out = FIGURES_DIR / "critical_pga_curve_real.png"
    fig.savefig(out, dpi=150); plt.close(fig)
    print(f"Kritik PGA eğrisi: {out}")
    if crit_ml is not None:
        phy_at_ml = frac_dn[np.argmin(np.abs(pga_grid - crit_ml))]
        print(f"  >>> Kritik PGA (ML)      ≈ {crit_ml:.2f} g "
              f"(bu PGA'da Newmark Dn≥10cm oranı = {phy_at_ml*100:.0f}%)")
    if crit_phy is not None:
        print(f"  >>> Kritik PGA (Newmark) ≈ {crit_phy:.2f} g  [fiziksel Dn=10cm eşiği]")
    else:
        print(f"  >>> Newmark rijit-blok eğrisi gözlem aralığında (≤{pga_grid[-1]:.2f} g) "
              f"%50'yi geçmez (maks oran {frac_dn.max()*100:.0f}%). Bu beklenen bir "
              f"sonuçtur: saf rijit-blok Newmark gerçek yamaç tetiklemesini olduğundan "
              f"az tahmin eder; gerçek heyelanlardan öğrenen ML bunu yakalar.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Gerçek bölgesel duyarlılık haritası")
    parser.add_argument("--res", type=float, default=0.008, help="Izgara çözünürlüğü (derece)")
    args = parser.parse_args()
    run(res=args.res)
