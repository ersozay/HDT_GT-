"""
Bölgesel raster yığını — tüm katmanları ortak bir EPSG:4326 ızgarasına hizalar.

05_susceptibility_map.py ve 10_critical_pga.py aynı ızgarayı kullanır.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from config import DATA_RAW, DATA_PROCESSED

# (lon_min, lat_min, lon_max, lat_max) — Kocaman (2025) envanteri + tampon
STUDY_BOUNDS = (35.7, 35.9, 39.25, 38.65)

DEM_PATH    = DATA_RAW / "EUDEM_kahramanmaras.tif"
PGA_PATH    = DATA_RAW / "pga_kahramanmaras_20230206.tif"
LITHO_PATH  = DATA_RAW / "lithology_macrostrat_proxy.tif"
INV_PATH    = DATA_RAW / "TJES-2024-00171_Landslides.kml"
SLOPE_PATH  = DATA_PROCESSED / "slope_deg.tif"
ASPECT_PATH = DATA_PROCESSED / "aspect_deg.tif"


@dataclass
class RegionalStack:
    elevation: np.ndarray
    slope: np.ndarray
    aspect: np.ndarray
    pga: np.ndarray
    litho: np.ndarray
    valid: np.ndarray
    layers: dict
    transform: object
    bounds: tuple
    res: float

    @property
    def shape(self):
        return self.elevation.shape


def build_grid(bounds, res):
    from rasterio.transform import from_origin
    lon_min, lat_min, lon_max, lat_max = bounds
    w = int(round((lon_max - lon_min) / res))
    h = int(round((lat_max - lat_min) / res))
    return from_origin(lon_min, lat_max, res, res), w, h


def reproject_to_grid(raster_path, dst_transform, dst_w, dst_h, resampling="bilinear"):
    import rasterio
    from rasterio.warp import reproject, Resampling
    rs = {"bilinear": Resampling.bilinear, "nearest": Resampling.nearest,
          "average": Resampling.average, "max": Resampling.max,
          "mode": Resampling.mode}[resampling]
    dst = np.full((dst_h, dst_w), np.nan, dtype=np.float32)
    with rasterio.open(raster_path) as src:
        reproject(source=rasterio.band(src, 1), destination=dst,
                  src_transform=src.transform, src_crs=src.crs,
                  dst_transform=dst_transform, dst_crs="EPSG:4326",
                  resampling=rs, src_nodata=src.nodata, dst_nodata=np.nan)
    return dst


def load_regional_stack(res: float = 0.008, bounds=STUDY_BOUNDS) -> RegionalStack:
    """Tüm girdi katmanlarını `res` derecelik ızgarada döndürür (temizlenmiş)."""
    transform, w, h = build_grid(bounds, res)
    elevation = reproject_to_grid(DEM_PATH, transform, w, h, "bilinear")
    slope     = reproject_to_grid(SLOPE_PATH, transform, w, h, "bilinear")
    aspect    = reproject_to_grid(ASPECT_PATH, transform, w, h, "nearest")
    pga       = reproject_to_grid(PGA_PATH, transform, w, h, "bilinear")
    litho     = reproject_to_grid(LITHO_PATH, transform, w, h, "nearest")

    # Deniz / veri-dışı maskesi — NaN doldurmadan ÖNCE hesaplanır
    valid = (np.isfinite(elevation) & (elevation > 0) &
             np.isfinite(pga) & (pga > 0.02) & np.isfinite(slope))

    litho = np.where(np.isfinite(litho) & (litho >= 1) & (litho <= 27), litho, 27).astype(int)
    slope = np.clip(np.nan_to_num(slope, nan=0.0), 0.0, 89.0)
    aspect = np.nan_to_num(aspect, nan=-1.0)
    elevation = np.nan_to_num(elevation, nan=0.0)
    pga = np.clip(np.nan_to_num(pga, nan=0.05), 0.001, 3.0)
    return RegionalStack(elevation, slope, aspect, pga, litho, valid,
                         load_layers(transform, w, h), transform, bounds, res)


def load_layers(transform, w, h) -> dict:
    """data/processed/layers/*.tif katmanlarını ızgaraya alır (yoksa boş sözlük)."""
    from .feature_matrix import LAYER_RASTERS
    layer_dir = DATA_PROCESSED / "layers"
    out = {}
    for name in LAYER_RASTERS:
        p = layer_dir / f"{name}.tif"
        if not p.exists():
            continue
        a = reproject_to_grid(p, transform, w, h, "mode" if name == "landcover" else "average")
        if name == "landcover":
            a = np.where(np.isfinite(a) & (a > 0), a, 30)
        else:
            a = np.where(np.isfinite(a), a, np.nanmedian(a))
        out[name] = a.astype(np.float32)
    return out


def stack_features(stack: RegionalStack, z_ref: float, pga_override: float | None = None,
                   saturation=None):
    """
    Izgara için öznitelik DataFrame'i. `saturation` (skaler ya da ızgara, 0–1):
    yağış senaryosu — su tablası oranı m; u = m·γ_w·z·cos²θ (geomechanics/hydrology).
    """
    from .feature_matrix import build_feature_matrix
    from ..geomechanics.hydrology import pore_pressure
    pga = stack.pga if pga_override is None else np.full(stack.shape, pga_override, np.float32)
    u = None if saturation is None else pore_pressure(np.broadcast_to(saturation, stack.shape), z_ref, stack.slope)
    df, _ = build_feature_matrix(
        elevation_m=stack.elevation, slope_deg=stack.slope, aspect_deg=stack.aspect,
        litho_array=stack.litho, pga_array=pga, u_array=u,
        z_ref=z_ref, include_onehot=True,
    )
    if stack.layers:
        from .feature_matrix import derive_layer_features
        for name, a in stack.layers.items():
            df[name] = a.ravel()
        df = derive_layer_features(df)
    return df


def rasterize_inventory(transform, w, h, all_touched: bool = False) -> np.ndarray:
    """Envanter poligonlarını ızgaraya ikili maske olarak döker (1 = heyelan)."""
    import geopandas as gpd
    from rasterio.features import rasterize
    gdf = gpd.read_file(INV_PATH)
    gdf = gdf[gdf.geom_type.isin(["Polygon", "MultiPolygon"])]
    return rasterize(((g, 1) for g in gdf.geometry), out_shape=(h, w), transform=transform,
                     fill=0, dtype="uint8", all_touched=all_touched)
