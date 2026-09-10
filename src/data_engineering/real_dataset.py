"""
Gerçek Veri Seti Oluşturucu.

USGS ShakeMap (PGA) + Copernicus DEM (yükseklik/eğim/bakı) + Macrostrat
litoloji proxy + Kocaman (2025) heyelan envanteri (etiketler) katmanlarını
birleştirerek PINN eğitimi için nokta-bazlı öznitelik matrisi üretir.

Pozitif örnekler  : heyelan poligonları içi (gözlem etiketi y=1)
Negatif örnekler  : poligonlardan uzak rastgele arka plan noktaları (y=0)
Mekânsal grup     : pozitiflerde poligon ID'si, negatiflerde ızgara hücresi
                    (Polygon-Based GroupKFold ile veri sızıntısını önler)
"""
from __future__ import annotations

from pathlib import Path

import numpy as np


# ---------------------------------------------------------------------------
# DEM'den eğim (slope) ve bakı (aspect) türetme — Horn (1981) 3×3 yöntemi
# ---------------------------------------------------------------------------
def compute_slope_aspect(
    dem_path: str | Path,
    out_slope: str | Path,
    out_aspect: str | Path,
) -> tuple[Path, Path]:
    """
    DEM rasterından eğim (derece) ve bakı (derece) rasterları üretir.

    Coğrafi (EPSG:4326) DEM'de piksel boyu metreye çevrilir:
      dx = 111320 · cos(lat) · dlon,  dy = 110540 · dlat
    Eğim, Horn (1981) 3×3 gradyan operatörü ile hesaplanır.
    """
    import rasterio

    dem_path = Path(dem_path)
    out_slope = Path(out_slope)
    out_aspect = Path(out_aspect)
    out_slope.parent.mkdir(parents=True, exist_ok=True)

    with rasterio.open(dem_path) as src:
        z = src.read(1).astype(np.float64)
        transform = src.transform
        profile = src.profile.copy()
        nodata = src.nodata
        # Piksel merkezlerinin ortalama enlemi
        h, w = z.shape
        lat_top = transform.f
        lat_bot = transform.f + transform.e * h
        mean_lat = (lat_top + lat_bot) / 2.0

    # Geçersiz/deniz değerlerini komşu ortalamasına yaklaştırmak yerine maskele
    if nodata is not None:
        z = np.where(z == nodata, np.nan, z)
    z = np.where(z < -1000, np.nan, z)

    dlon = abs(transform.a)
    dlat = abs(transform.e)
    dx = 111_320.0 * np.cos(np.deg2rad(mean_lat)) * dlon   # m
    dy = 110_540.0 * dlat                                   # m

    # NaN'ları geçici olarak civar değeriyle doldur (gradyan için)
    zf = z.copy()
    mask_nan = np.isnan(zf)
    if mask_nan.any():
        zf[mask_nan] = np.nanmedian(zf)

    # Horn 3×3 penceresi (kenarlar 'edge' ile genişletilir)
    zp = np.pad(zf, 1, mode="edge")
    a = zp[:-2, :-2]; b = zp[:-2, 1:-1]; c = zp[:-2, 2:]
    d = zp[1:-1, :-2];                    f = zp[1:-1, 2:]
    g = zp[2:, :-2]; hh = zp[2:, 1:-1]; i = zp[2:, 2:]

    dzdx = ((c + 2 * f + i) - (a + 2 * d + g)) / (8.0 * dx)
    dzdy = ((g + 2 * hh + i) - (a + 2 * b + c)) / (8.0 * dy)

    slope_rad = np.arctan(np.sqrt(dzdx ** 2 + dzdy ** 2))
    slope_deg = np.rad2deg(slope_rad).astype(np.float32)

    # Bakı: 0=Kuzey, saat yönünde. atan2 tabanlı standart CBS formülü
    aspect = np.rad2deg(np.arctan2(dzdy, -dzdx))
    aspect = np.where(aspect < 0, 90.0 - aspect,
                      np.where(aspect > 90.0, 360.0 - aspect + 90.0, 90.0 - aspect))
    aspect_deg = aspect.astype(np.float32)

    slope_deg[mask_nan] = 0.0
    aspect_deg[mask_nan] = -1.0

    profile.update(dtype="float32", count=1, nodata=None)
    with rasterio.open(out_slope, "w", **profile) as dst:
        dst.write(slope_deg, 1)
    with rasterio.open(out_aspect, "w", **profile) as dst:
        dst.write(aspect_deg, 1)

    print(f"Eğim rasterı  : {out_slope}  (maks {np.nanmax(slope_deg):.1f}°)")
    print(f"Bakı rasterı  : {out_aspect}")
    return out_slope, out_aspect


# ---------------------------------------------------------------------------
# Raster nokta örnekleme
# ---------------------------------------------------------------------------
def sample_raster(raster_path: str | Path, lons, lats, band: int = 1) -> np.ndarray:
    """Verilen (lon, lat) noktalarında raster değerlerini döndürür (nearest)."""
    import rasterio

    coords = list(zip(np.asarray(lons), np.asarray(lats)))
    with rasterio.open(raster_path) as src:
        vals = np.array([v[band - 1] for v in src.sample(coords)], dtype=np.float64)
        nodata = src.nodata
    if nodata is not None:
        vals = np.where(vals == nodata, np.nan, vals)
    return vals


# ---------------------------------------------------------------------------
# Örnekleme noktaları üretme (pozitif + negatif)
# ---------------------------------------------------------------------------
def _points_in_polygon(geom, n_points: int, rng) -> list[tuple[float, float]]:
    """Bir poligon içinde n adet rastgele nokta üretir (reddetme örneklemesi)."""
    minx, miny, maxx, maxy = geom.bounds
    from shapely.geometry import Point

    pts: list[tuple[float, float]] = []
    # Merkez temsil noktası her zaman dahil
    rp = geom.representative_point()
    pts.append((rp.x, rp.y))
    tries = 0
    while len(pts) < n_points and tries < n_points * 50:
        x = rng.uniform(minx, maxx)
        y = rng.uniform(miny, maxy)
        if geom.contains(Point(x, y)):
            pts.append((x, y))
        tries += 1
    return pts


def generate_samples(
    inventory_path: str | Path,
    bounds: tuple[float, float, float, float],
    pts_per_polygon_scale: float = 3.0,
    max_pts_per_polygon: int = 12,
    neg_ratio: float = 2.5,
    neg_buffer_m: float = 200.0,
    seed: int = 42,
):
    """
    Heyelan envanterinden pozitif ve arka plandan negatif örnek noktaları üretir.

    Parametreler
    -----------
    inventory_path        : Heyelan poligonları (KML/GeoJSON/SHP)
    bounds                : (lon_min, lat_min, lon_max, lat_max) çalışma alanı
    pts_per_polygon_scale : Poligon başına nokta ~ scale·log(alan) ile ölçeklenir
    max_pts_per_polygon   : Poligon başına üst sınır nokta sayısı
    neg_ratio             : Negatif/pozitif oranı
    neg_buffer_m          : Negatiflerin heyelanlara min mesafesi (m)
    seed                  : Rastgelelik tohumu

    Döndürür
    --------
    dict : lon, lat, y (0/1), group_id (mekânsal grup)
    """
    import geopandas as gpd
    from shapely.geometry import Point

    rng = np.random.RandomState(seed)

    gdf = gpd.read_file(inventory_path)
    gdf = gdf[gdf.geom_type.isin(["Polygon", "MultiPolygon"])].reset_index(drop=True)
    gdf = gdf.set_crs(4326, allow_override=True)
    gdf_m = gdf.to_crs(32637)                       # metrik (UTM 37N)

    # --- Pozitif örnekler ---
    pos_lon: list[float] = []
    pos_lat: list[float] = []
    pos_grp: list[int] = []
    for pid, (_, row) in enumerate(gdf.iterrows()):
        geom = row.geometry
        if geom is None or geom.is_empty:
            continue
        area_m2 = gdf_m.geometry.iloc[pid].area
        n_pts = int(np.clip(pts_per_polygon_scale * np.log10(max(area_m2, 10.0)),
                            1, max_pts_per_polygon))
        for (x, y) in _points_in_polygon(geom, n_pts, rng):
            pos_lon.append(x); pos_lat.append(y); pos_grp.append(pid)

    n_pos = len(pos_lon)

    # --- Negatif örnekler ---
    # Heyelan birleşik tamponu (yakın çevreyi negatif olarak dışla)
    union_buf = gdf_m.geometry.buffer(neg_buffer_m).unary_union
    lon_min, lat_min, lon_max, lat_max = bounds
    n_neg_target = int(n_pos * neg_ratio)

    from shapely.ops import transform as shp_transform
    import pyproj

    to_m = pyproj.Transformer.from_crs(4326, 32637, always_xy=True).transform

    neg_lon: list[float] = []
    neg_lat: list[float] = []
    tries = 0
    max_tries = n_neg_target * 60
    while len(neg_lon) < n_neg_target and tries < max_tries:
        x = rng.uniform(lon_min, lon_max)
        y = rng.uniform(lat_min, lat_max)
        pm = shp_transform(to_m, Point(x, y))
        if not union_buf.contains(pm):
            neg_lon.append(x); neg_lat.append(y)
        tries += 1

    n_neg = len(neg_lon)

    # Negatif mekânsal gruplar: ~0.1° ızgara hücresi (pozitif ID'lerinden sonra)
    grid = 0.1
    base = max(pos_grp) + 1 if pos_grp else 0
    neg_grp = [
        base + int(((lat - lat_min) // grid)) * 10000 + int(((lon - lon_min) // grid))
        for lon, lat in zip(neg_lon, neg_lat)
    ]

    lon = np.array(pos_lon + neg_lon)
    lat = np.array(pos_lat + neg_lat)
    y = np.array([1] * n_pos + [0] * n_neg, dtype=np.int8)
    group = np.array(pos_grp + neg_grp)

    print(f"Örnek üretimi: {n_pos} pozitif + {n_neg} negatif = {len(y)} nokta")
    print(f"  Heyelan oranı: {100 * n_pos / len(y):.1f}%  |  "
          f"benzersiz grup: {len(np.unique(group))}")
    return {"lon": lon, "lat": lat, "y": y, "group": group}


# ---------------------------------------------------------------------------
# Tam öznitelik matrisi kurulumu
# ---------------------------------------------------------------------------
def build_real_feature_matrix(
    samples: dict,
    dem_path: str | Path,
    slope_path: str | Path,
    aspect_path: str | Path,
    pga_path: str | Path,
    litho_path: str | Path,
    z_ref: float = 5.0,
):
    """
    Örnek noktalarında tüm katmanları örnekleyip PINN öznitelik matrisini kurar.

    Döndürür
    --------
    df    : pd.DataFrame (öznitelikler + y + group + lon/lat)
    feats : öznitelik sütun isimleri
    """
    from .feature_matrix import build_feature_matrix

    lon, lat = samples["lon"], samples["lat"]

    elevation = sample_raster(dem_path, lon, lat)
    slope = sample_raster(slope_path, lon, lat)
    aspect = sample_raster(aspect_path, lon, lat)
    pga = sample_raster(pga_path, lon, lat)
    litho = sample_raster(litho_path, lon, lat)

    # Temizlik: geçersiz litoloji → 27, negatif eğim → 0, NaN PGA → 0.05
    litho = np.where(np.isfinite(litho) & (litho >= 1) & (litho <= 27),
                     litho, 27).astype(int)
    slope = np.clip(np.nan_to_num(slope, nan=0.0), 0.0, 89.0)
    aspect = np.nan_to_num(aspect, nan=-1.0)
    elevation = np.nan_to_num(elevation, nan=0.0)
    pga = np.clip(np.nan_to_num(pga, nan=0.05), 0.001, 3.0)

    df, feats = build_feature_matrix(
        elevation_m=elevation,
        slope_deg=slope,
        aspect_deg=aspect,
        litho_array=litho,
        pga_array=pga,
        u_array=None,
        z_ref=z_ref,
        include_onehot=True,
    )

    df["y"] = samples["y"]
    df["group"] = samples["group"]
    df["lon"] = lon
    df["lat"] = lat
    return df, feats
