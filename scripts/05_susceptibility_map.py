"""
Script 5: Gerçek Bölgesel Heyelan Duyarlılık Haritası

Eğitilmiş PINN'i gerçek raster yığınına (Copernicus DEM eğim/bakı + USGS PGA +
Macrostrat litoloji) uygulayarak tüm çalışma alanı için heyelan olasılığı
(susceptibility) haritası üretir ve gözlemlenen envanterle görsel doğrular.

Kritik PGA analizi (belirsizlik + tasarım tablosu + ampirik doğrulama)
scripts/10_critical_pga.py'ye taşındı.

Kullanım:
  python3 scripts/05_susceptibility_map.py [--res 0.008]
"""
import sys
import argparse
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from config import DATA_RAW, DATA_PROCESSED, FIGURES_DIR, MODELS_DIR, Z_REF_M, ensure_dirs
from src.data_engineering.regional_grid import load_regional_stack, stack_features, STUDY_BOUNDS


def run(res: float = 0.008):
    ensure_dirs()
    import rasterio
    from src.pinn.io import load_checkpoint

    stack = load_regional_stack(res)
    h, w = stack.shape
    print(f"Hedef ızgara: {w}×{h} = {w*h:,} piksel (~{res*111:.1f} km çözünürlük)")
    lm = load_checkpoint(MODELS_DIR / "best_model.pt", MODELS_DIR / "scaler.pkl",
                         MODELS_DIR / "threshold.json")
    print(f"Model: {lm.meta.get('model_version', 'pinn_v1')} "
          f"(epoch={lm.meta.get('epoch')}, {len(lm.feat_cols)} öznitelik)")

    df = stack_features(stack, Z_REF_M)
    proba = lm.predict(df[lm.feat_cols].to_numpy(np.float32)).reshape(h, w)
    proba = np.where(stack.valid, proba, np.nan)

    out_tif = DATA_PROCESSED / "susceptibility_pinn.tif"
    prof = {"driver": "GTiff", "height": h, "width": w, "count": 1,
            "dtype": "float32", "crs": "EPSG:4326", "transform": stack.transform,
            "nodata": np.nan}
    with rasterio.open(out_tif, "w", **prof) as dst:
        dst.write(proba.astype(np.float32), 1)
    print(f"Duyarlılık rasterı: {out_tif}")
    _plot_map(proba, STUDY_BOUNDS, stack.valid)
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


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Gerçek bölgesel duyarlılık haritası")
    parser.add_argument("--res", type=float, default=0.008, help="Izgara çözünürlüğü (derece)")
    args = parser.parse_args()
    run(res=args.res)
