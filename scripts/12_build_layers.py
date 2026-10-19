"""
Script 12: Ek koşullandırıcı katmanları üret (tamamı açık kaynak)

PGV/MMI/Vs30 (USGS ShakeMap), kırılmaya ve faya uzaklık (USGS rupture + GEM),
DEM türevleri (pürüzlülük, TRI, eğrilik, rölyef), dereye uzaklık + TWI (pysheds),
arazi örtüsü (ESA WorldCover), kil/kum oranı (SoilGrids). Ayrıntı:
src/data_engineering/extra_layers.py

Çıktı: data/processed/layers/*.tif (DEM ızgarası, çalışma alanı)

Kullanım:
  python3 scripts/12_build_layers.py [--skip-remote]
"""
import sys
import time
import argparse
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data_engineering import extra_layers as L
from src.data_engineering.regional_grid import STUDY_BOUNDS, DEM_PATH, SLOPE_PATH, reproject_to_grid


def run(skip_remote=False):
    t0 = time.time()
    transform, w, h = L.fine_grid(STUDY_BOUNDS)
    print(f"Izgara: {w}×{h} ({w*h/1e6:.1f} M piksel, 0.000833°)")
    dem = reproject_to_grid(DEM_PATH, transform, w, h, "bilinear")
    slope = reproject_to_grid(SLOPE_PATH, transform, w, h, "bilinear")
    dem_f = np.where(np.isfinite(dem), dem, np.nanmedian(dem))
    dem_clip = L.write_layer("dem_clip", dem_f, transform)

    print("ShakeMap katmanları (PGV, MMI, Vs30)…")
    for k, a in L.shakemap_layers(transform, w, h).items():
        L.write_layer(k, a, transform)
    print("Fay / kırılma uzaklıkları…")
    faults, traces = L.fault_layers(transform, w, h, STUDY_BOUNDS)
    for k, a in faults.items():
        L.write_layer(k, a, transform)
    import geopandas as gpd
    gpd.GeoDataFrame(geometry=traces, crs=4326).to_file(L.LAYER_DIR / "rupture_traces.geojson", driver="GeoJSON")
    print("DEM türevleri…")
    for k, a in L.terrain_layers(dem_f, transform).items():
        L.write_layer(k, a, transform)
    print("Hidroloji (pysheds)…")
    for k, a in L.hydrology_layers(dem_clip, np.nan_to_num(slope), transform).items():
        L.write_layer(k, a, transform)
    if not skip_remote:
        print("ESA WorldCover…")
        L.write_layer("landcover", L.worldcover_layer(transform, w, h, STUDY_BOUNDS), transform,
                      nodata=0, dtype="uint8")
        for prop in ("clay", "sand"):
            print(f"SoilGrids {prop}…")
            try:
                L.write_layer(f"{prop}_pct", L.soilgrids_layer(prop, STUDY_BOUNDS, transform, w, h), transform)
            except Exception as e:
                print(f"  SoilGrids {prop} atlandı: {e}")

    import rasterio
    print("\nKatman özeti:")
    for p in sorted(L.LAYER_DIR.glob("*.tif")):
        with rasterio.open(p) as d:
            a = d.read(1, out_shape=(d.height // 8, d.width // 8)).astype(float)
            if d.nodata is not None and not np.isnan(d.nodata):
                a[a == d.nodata] = np.nan
            print(f"  {p.stem:16s} min {np.nanmin(a):9.2f}  medyan {np.nanmedian(a):9.2f}  "
                  f"maks {np.nanmax(a):9.2f}  geçerli {np.isfinite(a).mean()*100:5.1f}%")
    enrich_feature_table()
    print(f"Süre: {time.time()-t0:.0f} s")


def enrich_feature_table():
    """Eğitim noktalarında ham katman değerlerini örnekleyip parquet'e ekler."""
    import pandas as pd
    from config import DATA_PROCESSED
    from src.data_engineering.real_dataset import sample_raster
    from src.data_engineering.feature_matrix import LAYER_RASTERS
    path = DATA_PROCESSED / "feature_matrix.parquet"
    df = pd.read_parquet(path)
    for name in LAYER_RASTERS:
        p = L.LAYER_DIR / f"{name}.tif"
        if not p.exists():
            continue
        v = sample_raster(p, df["lon"].to_numpy(), df["lat"].to_numpy())
        if name == "landcover":
            v = np.where(np.isfinite(v) & (v > 0), v, 30)
        else:
            v = np.where(np.isfinite(v), v, np.nanmedian(v))
        df[name] = v.astype(np.float32)
    df.to_parquet(path, index=False)
    print(f"Eğitim tablosu zenginleştirildi: {path} ({df.shape[1]} sütun)")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--skip-remote", action="store_true")
    run(ap.parse_args().skip_remote)
