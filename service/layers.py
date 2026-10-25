"""
Harita katmanları servisi — açık kaynak ek katmanlar, envanter, faylar, istasyonlar.

Rasterlar (data/processed/layers/*.tif) renklendirilmiş şeffaf PNG olarak,
vektörler GeoJSON olarak sunulur. Tümü süreç içinde önbelleklenir.
"""
from __future__ import annotations

import io
import json
from functools import lru_cache

import numpy as np

from config import DATA_PROCESSED, DATA_RAW, OUTPUTS_DIR, INVENTORY_KML

LAYER_DIR = DATA_PROCESSED / "layers"

# ad → (başlık, birim, colormap, ters?, kaynak)
RASTER_LAYERS = {
    "pgv_cms":         ("PGV — en büyük yer hızı (2 olay zarfı)", "cm/s", "Oranges", False, "USGS ShakeMap"),
    "mmi":             ("MMI — değiştirilmiş Mercalli şiddeti", "", "Oranges", False, "USGS ShakeMap"),
    "vs30":            ("Vs30 — zemin kayma dalgası hızı", "m/s", "Purples", False, "USGS ShakeMap"),
    "dist_rupture_km": ("2023 kırılmasına uzaklık", "km", "Blues", True, "USGS rupture.json"),
    "dist_fault_km":   ("Aktif faya uzaklık", "km", "Blues", True, "GEM Global Active Faults"),
    "tri_m":           ("Arazi pürüzlülük indeksi (TRI)", "m", "Greys", False, "Copernicus DEM"),
    "relief_1km_m":    ("Yerel rölyef (≈1 km)", "m", "Greys", False, "Copernicus DEM"),
    "twi":             ("Topoğrafik ıslaklık indeksi (TWI)", "", "Blues", False, "DEM + pysheds"),
    "dist_stream_km":  ("Dereye uzaklık", "km", "Blues", True, "DEM + pysheds"),
    "clay_pct":        ("Kil oranı (15–30 cm)", "%", "YlOrBr", False, "ISRIC SoilGrids 2.0"),
    "landcover":       ("Arazi örtüsü (ESA WorldCover 2021)", "", None, False, "ESA WorldCover"),
    "saturation_20230206": ("Zemin doygunluğu — 6 Şubat 2023 sabahı", "S", "Blues", False, "ERA5-Land (Open-Meteo)"),
    "precip_30d_mm":   ("Depremden önceki 30 gün yağış", "mm", "Blues", False, "ERA5-Land (Open-Meteo)"),
}

LANDCOVER_COLORS = {10: ("Ağaç", "#006400"), 20: ("Çalılık", "#ffbb22"), 30: ("Çayır", "#ffff4c"),
                    40: ("Tarım", "#f096ff"), 50: ("Yapılaşma", "#fa0000"),
                    60: ("Çıplak/seyrek", "#b4b4b4"), 70: ("Kar-buz", "#f0f0f0"),
                    80: ("Su", "#0064c8"), 90: ("Sulak alan", "#0096a0")}

DISPLAY_STRIDE = 3          # 0.000833° → ~0.0025° (PNG boyutu için)


def _read_display(name):
    import rasterio
    with rasterio.open(LAYER_DIR / f"{name}.tif") as ds:
        h, w = ds.height // DISPLAY_STRIDE, ds.width // DISPLAY_STRIDE
        from rasterio.enums import Resampling
        rs = Resampling.mode if name == "landcover" else Resampling.average
        a = ds.read(1, out_shape=(h, w), resampling=rs).astype(float)
        b = ds.bounds
        nd = ds.nodata
    if nd is not None and not np.isnan(nd):
        a[a == nd] = np.nan
    return a, [[b.bottom, b.left], [b.top, b.right]]


def available_layers() -> list[dict]:
    out = []
    for k, (title, unit, _, rev, src) in RASTER_LAYERS.items():
        if (LAYER_DIR / f"{k}.tif").exists():
            out.append({"id": k, "title": title, "unit": unit, "source": src,
                        "categorical": k == "landcover"})
    return out


@lru_cache(maxsize=32)
def layer_png(name: str) -> bytes:
    import matplotlib
    from matplotlib.colors import Normalize
    from PIL import Image
    a, _ = _read_display(name)
    h, w = a.shape
    if name == "landcover":
        rgba = np.zeros((h, w, 4), np.uint8)
        for code, (_, hexv) in LANDCOVER_COLORS.items():
            m = a == code
            rgba[m, :3] = [int(hexv[i:i + 2], 16) for i in (1, 3, 5)]
            rgba[m, 3] = 190
    else:
        title, unit, cmap, rev, _ = RASTER_LAYERS[name]
        lo, hi = np.nanpercentile(a, [2, 98])
        norm = Normalize(lo, hi)
        cm = matplotlib.colormaps[cmap + ("_r" if rev else "")]
        rgba = (cm(norm(np.nan_to_num(a, nan=lo))) * 255).astype(np.uint8)
        rgba[..., 3] = np.where(np.isfinite(a), 185, 0)
    buf = io.BytesIO()
    Image.fromarray(rgba, mode="RGBA").save(buf, format="PNG", optimize=True)
    return buf.getvalue()


@lru_cache(maxsize=32)
def layer_meta(name: str) -> dict:
    import matplotlib
    a, bounds = _read_display(name)
    title, unit, cmap, rev, src = RASTER_LAYERS[name]
    meta = {"id": name, "title": title, "unit": unit, "source": src, "bounds": bounds}
    if name == "landcover":
        present = set(np.unique(a[np.isfinite(a)]).astype(int))
        meta["legend"] = [{"code": c, "label": l, "color": col}
                          for c, (l, col) in LANDCOVER_COLORS.items() if c in present]
    else:
        lo, hi = np.nanpercentile(a, [2, 98])
        cm = matplotlib.colormaps[cmap + ("_r" if rev else "")]
        stops = [matplotlib.colors.to_hex(cm(t)) for t in np.linspace(0, 1, 5)]
        meta.update({"vmin": float(lo), "vmax": float(hi), "gradient": stops})
    return meta


@lru_cache(maxsize=1)
def inventory_geojson() -> dict:
    """Kocaman (2025) envanter poligonları — sadeleştirilmiş, alan bilgisiyle."""
    import geopandas as gpd
    g = gpd.read_file(INVENTORY_KML)
    g = g[g.geom_type.isin(["Polygon", "MultiPolygon"])].reset_index(drop=True)
    area = g.to_crs(32637).area
    g = g[["Name", "geometry"]].rename(columns={"Name": "id"})
    g["area_m2"] = area.round(0)
    g["geometry"] = g.geometry.simplify(0.00015, preserve_topology=True)
    sim = OUTPUTS_DIR / "results" / "newmark_th_polygons.json"
    if sim.exists():
        s = json.loads(sim.read_text())
        dn = {str(k): v for k, v in s.get("final_dn_cm", {}).items()}
        g["dn_th_cm"] = [dn.get(str(i)) for i in range(len(g))]
    return json.loads(g.to_json(drop_id=True))


@lru_cache(maxsize=1)
def faults_geojson() -> dict:
    import geopandas as gpd
    from src.data_engineering.regional_grid import STUDY_BOUNDS
    lon0, lat0, lon1, lat1 = STUDY_BOUNDS
    feats = []
    gem = gpd.read_file(DATA_RAW / "faults" / "gem_active_faults.geojson").cx[lon0 - .5:lon1 + .5, lat0 - .5:lat1 + .5]
    for _, r in gem.iterrows():
        feats.append({"type": "Feature", "geometry": r.geometry.__geo_interface__,
                      "properties": {"kind": "active_fault", "slip_type": r.get("slip_type"),
                                     "source": "GEM Global Active Faults"}})
    rt = LAYER_DIR / "rupture_traces.geojson"
    if rt.exists():
        for f in json.loads(rt.read_text())["features"]:
            f["properties"] = {"kind": "rupture_2023", "source": "USGS ShakeMap rupture"}
            feats.append(f)
    return {"type": "FeatureCollection", "features": feats}


@lru_cache(maxsize=1)
def stations_geojson() -> dict:
    import pandas as pd
    p = OUTPUTS_DIR / "results" / "strong_motion_stations.csv"
    if not p.exists():
        return {"type": "FeatureCollection", "features": []}
    df = pd.read_csv(p)
    feats = []
    for (st, lat, lon), g in df.groupby(["station", "lat", "lon"]):
        props = {"station": str(st)}
        for _, r in g.iterrows():
            props[f"pga_{r.event}"] = round(float(r.pga_g), 3)
            props[f"arias_{r.event}"] = round(float(r.arias_ms), 3)
        feats.append({"type": "Feature", "geometry": {"type": "Point", "coordinates": [lon, lat]},
                      "properties": props})
    return {"type": "FeatureCollection", "features": feats}
