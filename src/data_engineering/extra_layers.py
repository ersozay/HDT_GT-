"""
Ek koşullandırıcı katmanlar — tamamı açık kaynak.

Kocaman ve ark. (2025) ile DepremBİS'in kullandığı statik faktörlerin açık
veriden üretilmiş karşılıkları + 2023 depremlerine özgü yer hareketi katmanları:

  Katman              Kaynak (lisans)                                   Çözünürlük
  ------------------  ------------------------------------------------  ----------
  PGV, MMI, Vs30      USGS ShakeMap grid.xml (kamu malı)                 ~1 km
  kırılmaya uzaklık   USGS ShakeMap rupture.json, 2 olay (kamu malı)     vektör
  faya uzaklık        GEM Global Active Faults (CC BY-SA 4.0)            vektör
  pürüzlülük, TRI,    Copernicus DEM türevleri (Horn 1981;               ~90 m
  eğrilik, rölyef     Zevenbergen & Thorne 1987)
  dereye uzaklık, TWI pysheds D8 akış birikimi (DEM'den)                 ~90 m
  arazi örtüsü        ESA WorldCover 2021 v200 (CC BY 4.0)               10 m → 90 m
  kil / kum oranı     ISRIC SoilGrids 2.0, 15–30 cm (CC BY 4.0)          250 m

Tüm rasterlar çalışma alanında DEM'in kendi ızgarasına (0.000833°) hizalanır
ve data/processed/layers/ altına yazılır.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from config import DATA_RAW, DATA_PROCESSED

LAYER_DIR = DATA_PROCESSED / "layers"
FAULT_DIR = DATA_RAW / "faults"
EVENTS_USGS = ("us6000jllz", "us6000jlqa")

WORLDCOVER_URL = ("https://esa-worldcover.s3.eu-central-1.amazonaws.com/v200/2021/map/"
                  "ESA_WorldCover_10m_2021_v200_{tile}_Map.tif")
SOILGRIDS_WCS = "https://maps.isric.org/mapserv?map=/map/{prop}.map"

# ESA WorldCover sınıfları
WORLDCOVER_CLASSES = {10: "ağaç", 20: "çalılık", 30: "çayır", 40: "tarım", 50: "yapılaşma",
                      60: "çıplak/seyrek", 70: "kar-buz", 80: "su", 90: "sulak alan",
                      95: "mangrov", 100: "yosun-liken"}


# --------------------------------------------------------------------------- #
# Ortak ızgara
# --------------------------------------------------------------------------- #
def fine_grid(bounds, res=1.0 / 1200.0):
    from rasterio.transform import from_origin
    lon0, lat0, lon1, lat1 = bounds
    w = int(round((lon1 - lon0) / res)); h = int(round((lat1 - lat0) / res))
    return from_origin(lon0, lat1, res, res), w, h


def write_layer(name, arr, transform, nodata=np.nan, dtype="float32"):
    import rasterio
    LAYER_DIR.mkdir(parents=True, exist_ok=True)
    path = LAYER_DIR / f"{name}.tif"
    prof = {"driver": "GTiff", "height": arr.shape[0], "width": arr.shape[1], "count": 1,
            "dtype": dtype, "crs": "EPSG:4326", "transform": transform, "nodata": nodata,
            "compress": "deflate", "tiled": True}
    with rasterio.open(path, "w", **prof) as dst:
        dst.write(arr.astype(dtype), 1)
    return path


def _pixel_size_km(transform, h):
    lat_mid = transform.f + transform.e * h / 2
    dx = abs(transform.a) * 111.32 * np.cos(np.deg2rad(lat_mid))
    dy = abs(transform.e) * 110.57
    return dx, dy


def _distance_to_lines_km(geoms, transform, w, h):
    """Çizgileri rasterlaştırıp öklid uzaklık dönüşümü (km)."""
    from rasterio.features import rasterize
    from scipy.ndimage import distance_transform_edt
    mask = rasterize(((g, 1) for g in geoms), out_shape=(h, w), transform=transform,
                     fill=0, all_touched=True, dtype="uint8")
    dx, dy = _pixel_size_km(transform, h)
    if mask.sum() == 0:
        return np.full((h, w), np.nan, np.float32)
    return distance_transform_edt(mask == 0, sampling=(dy, dx)).astype(np.float32)


# --------------------------------------------------------------------------- #
# 1) ShakeMap: PGV, MMI, Vs30 (iki olayın zarfı)
# --------------------------------------------------------------------------- #
def shakemap_layers(transform, w, h):
    from rasterio.warp import reproject, Resampling
    out = {}
    stacks = {"pgv_cms": [], "mmi": [], "vs30": []}
    for ev in EVENTS_USGS:
        g = _parse_fields(DATA_RAW / f"shakemap_{ev}_grid.xml")
        for key, field in (("pgv_cms", "PGV"), ("mmi", "MMI"), ("vs30", "SVEL")):
            dst = np.full((h, w), np.nan, np.float32)
            reproject(g[field], dst, src_transform=g["transform"], src_crs="EPSG:4326",
                      dst_transform=transform, dst_crs="EPSG:4326", resampling=Resampling.bilinear,
                      dst_nodata=np.nan)
            stacks[key].append(dst)
    out["pgv_cms"] = np.fmax(*stacks["pgv_cms"])
    out["mmi"] = np.fmax(*stacks["mmi"])
    out["vs30"] = stacks["vs30"][0]      # Vs30 iki olayda aynı haritadır
    return out


def _parse_fields(path):
    """ShakeMap grid.xml → {alan: 2B dizi, transform}."""
    import re
    from rasterio.transform import from_origin
    txt = Path(path).read_text()
    spec = re.search(r"<grid_specification([^>]*)>", txt).group(1)
    attr = dict(re.findall(r'(\w+)="([^"]*)"', spec))
    nlon, nlat = int(attr["nlon"]), int(attr["nlat"])
    dlon, dlat = float(attr["nominal_lon_spacing"]), float(attr["nominal_lat_spacing"])
    fields = {m.group(2): int(m.group(1)) - 1 for m in re.finditer(
        r'<grid_field index="(\d+)" name="(\w+)"', txt)}
    body = txt[txt.index("<grid_data>") + 11: txt.index("</grid_data>")]
    data = np.fromstring(body, sep=" ").reshape(-1, len(fields))
    lon, lat = data[:, fields["LON"]], data[:, fields["LAT"]]
    # "nominal" aralık yuvarlatılmıştır; gerçek aralık uç noktalardan hesaplanır
    dlon = (lon.max() - lon.min()) / (nlon - 1)
    dlat = (lat.max() - lat.min()) / (nlat - 1)
    col = np.rint((lon - lon.min()) / dlon).astype(int)
    row = np.rint((lat.max() - lat) / dlat).astype(int)
    out = {"transform": from_origin(lon.min() - dlon / 2, lat.max() + dlat / 2, dlon, dlat)}
    for name, i in fields.items():
        a = np.full((nlat, nlon), np.nan, np.float32)
        a[row, col] = data[:, i]
        out[name] = a
    return out


# --------------------------------------------------------------------------- #
# 2) Faya ve kırılmaya uzaklık
# --------------------------------------------------------------------------- #
def fault_layers(transform, w, h, bounds):
    import geopandas as gpd
    from shapely.geometry import shape, LineString
    lon0, lat0, lon1, lat1 = bounds
    gem = gpd.read_file(FAULT_DIR / "gem_active_faults.geojson").cx[lon0 - 1:lon1 + 1, lat0 - 1:lat1 + 1]
    out = {"dist_fault_km": _distance_to_lines_km(list(gem.geometry), transform, w, h)}

    # USGS ShakeMap rupture.json: sonlu fay düzlemleri (poligon) → üst kenar = yüzey izi
    traces = []
    for ev in EVENTS_USGS:
        rj = json.loads((FAULT_DIR / f"{ev}_rupture.json").read_text())
        for feat in rj["features"]:
            geom = shape(feat["geometry"])
            for poly in getattr(geom, "geoms", [geom]):
                xyz = np.asarray(poly.exterior.coords if hasattr(poly, "exterior") else poly.coords)
                if len(xyz) > 2 and np.allclose(xyz[0], xyz[-1]):
                    xyz = xyz[:-1]          # kapanış noktası → üst kenar kendine geri dönmesin
                if xyz.shape[1] >= 3:
                    top = xyz[xyz[:, 2] <= xyz[:, 2].min() + 0.5][:, :2]
                else:
                    top = xyz[:, :2]
                if len(top) >= 2:
                    traces.append(LineString(top))
                else:
                    traces.append(poly.boundary if hasattr(poly, "boundary") else poly)
    out["dist_rupture_km"] = _distance_to_lines_km(traces, transform, w, h)
    return out, traces


# --------------------------------------------------------------------------- #
# 3) DEM türevleri + 4) hidroloji
# --------------------------------------------------------------------------- #
def terrain_layers(dem, transform):
    """Pürüzlülük (3×3 std), TRI, profil/plan eğriliği, yerel rölyef (≈1 km)."""
    from scipy.ndimage import generic_filter, uniform_filter, maximum_filter, minimum_filter
    z = dem.astype(np.float64)
    h = z.shape[0]
    dx_km, dy_km = _pixel_size_km(transform, h)
    dx, dy = dx_km * 1000, dy_km * 1000
    m = uniform_filter(z, 3); m2 = uniform_filter(z * z, 3)
    rough = np.sqrt(np.clip(m2 - m * m, 0, None))
    # TRI (Riley 1999): komşularla mutlak fark ortalaması ≈ |z − ortalama| ile yaklaşık
    zp = np.pad(z, 1, mode="edge")
    tri = np.zeros_like(z)
    for di in (-1, 0, 1):
        for dj in (-1, 0, 1):
            if di or dj:
                tri += np.abs(zp[1 + di:1 + di + z.shape[0], 1 + dj:1 + dj + z.shape[1]] - z)
    tri /= 8.0
    # Zevenbergen & Thorne (1987) eğrilikleri
    Z = {k: zp[1 + a:1 + a + z.shape[0], 1 + b:1 + b + z.shape[1]]
         for k, (a, b) in {"z1": (-1, -1), "z2": (-1, 0), "z3": (-1, 1), "z4": (0, -1),
                           "z6": (0, 1), "z7": (1, -1), "z8": (1, 0), "z9": (1, 1)}.items()}
    D = ((Z["z4"] + Z["z6"]) / 2 - z) / dx ** 2
    E = ((Z["z2"] + Z["z8"]) / 2 - z) / dy ** 2
    F = (-Z["z1"] + Z["z3"] + Z["z7"] - Z["z9"]) / (4 * dx * dy)
    G = (-Z["z4"] + Z["z6"]) / (2 * dx)
    H = (Z["z2"] - Z["z8"]) / (2 * dy)
    p = G ** 2 + H ** 2
    with np.errstate(invalid="ignore", divide="ignore"):
        prof = np.where(p > 1e-12, -2 * (D * G ** 2 + E * H ** 2 + F * G * H) / p, 0.0)
        plan = np.where(p > 1e-12, 2 * (D * H ** 2 + E * G ** 2 - F * G * H) / p, 0.0)
    relief = maximum_filter(z, 11) - minimum_filter(z, 11)
    return {"roughness_m": rough.astype(np.float32), "tri_m": tri.astype(np.float32),
            "curv_profile": (prof * 100).astype(np.float32),   # 1/100 m
            "curv_plan": (plan * 100).astype(np.float32),
            "relief_1km_m": relief.astype(np.float32)}


def hydrology_layers(dem_path_clip, slope_deg, transform, stream_km2: float = 1.0):
    """pysheds D8: akış birikimi → dere ağı (≥ stream_km2) → dereye uzaklık + TWI."""
    if not hasattr(np, "in1d"):          # pysheds 0.5, NumPy ≥ 2.4 uyumluluğu
        np.in1d = np.isin
    from pysheds.grid import Grid
    from scipy.ndimage import distance_transform_edt
    grid = Grid.from_raster(str(dem_path_clip))
    dem = grid.read_raster(str(dem_path_clip))
    filled = grid.fill_depressions(grid.fill_pits(dem))
    flat = grid.resolve_flats(filled)
    fdir = grid.flowdir(flat)
    acc = np.asarray(grid.accumulation(fdir), dtype=np.float64)
    h = acc.shape[0]
    dx_km, dy_km = _pixel_size_km(transform, h)
    cell_km2 = dx_km * dy_km
    streams = acc * cell_km2 >= stream_km2
    dist = distance_transform_edt(~streams, sampling=(dy_km, dx_km)).astype(np.float32)
    sca = (acc + 1) * cell_km2 * 1e6 / (dx_km * 1000)          # özgül katkı alanı (m)
    tanb = np.tan(np.deg2rad(np.clip(slope_deg, 0.1, 89)))
    twi = np.log(sca / tanb).astype(np.float32)
    return {"dist_stream_km": dist, "twi": twi, "flow_acc_km2": (acc * cell_km2).astype(np.float32)}


# --------------------------------------------------------------------------- #
# 5) Arazi örtüsü — ESA WorldCover (COG, uzaktan önizleme seviyesinden okunur)
# --------------------------------------------------------------------------- #
def worldcover_layer(transform, w, h, bounds):
    import rasterio
    from rasterio.warp import reproject, Resampling
    lon0, lat0, lon1, lat1 = bounds
    out = np.zeros((h, w), np.uint8)
    tiles = {f"N{la:02d}E{lo:03d}" for la in range(int(lat0 // 3 * 3), int(lat1 // 3 * 3) + 1, 3)
             for lo in range(int(lon0 // 3 * 3), int(lon1 // 3 * 3) + 1, 3)}
    with rasterio.Env(GDAL_HTTP_TIMEOUT=120, GDAL_DISABLE_READDIR_ON_OPEN="EMPTY_DIR",
                      CPL_VSIL_CURL_ALLOWED_EXTENSIONS=".tif"):
        for t in sorted(tiles):
            url = "/vsicurl/" + WORLDCOVER_URL.format(tile=t)
            try:
                with rasterio.open(url, overview_level=2) as src:      # ≈ 80 m
                    tmp = np.zeros((h, w), np.uint8)
                    reproject(rasterio.band(src, 1), tmp, src_transform=src.transform,
                              src_crs=src.crs, dst_transform=transform, dst_crs="EPSG:4326",
                              resampling=Resampling.mode, src_nodata=0, dst_nodata=0)
                    out = np.where(tmp > 0, tmp, out)
                    print(f"  WorldCover {t}: {int((tmp > 0).sum()):,} piksel")
            except Exception as e:
                print(f"  WorldCover {t}: atlandı ({e})")
    return out


# --------------------------------------------------------------------------- #
# 6) Zemin — ISRIC SoilGrids 2.0 (WCS)
# --------------------------------------------------------------------------- #
def soilgrids_layer(prop: str, bounds, transform, w, h, depth="15-30cm"):
    """SoilGrids özelliğini (kil/kum, g/kg) WCS ile indirip ızgaraya hizalar → %."""
    import io
    import rasterio
    from rasterio.warp import reproject, Resampling
    from urllib.request import urlopen, Request
    from .strong_motion import _ssl_context
    lon0, lat0, lon1, lat1 = bounds
    cov = f"{prop}_{depth}_mean"
    url = (SOILGRIDS_WCS.format(prop=prop) +
           f"&SERVICE=WCS&VERSION=2.0.1&REQUEST=GetCoverage&COVERAGEID={cov}"
           f"&FORMAT=GEOTIFF_INT16&SUBSET=long({lon0},{lon1})&SUBSET=lat({lat0},{lat1})"
           "&SUBSETTINGCRS=http://www.opengis.net/def/crs/EPSG/0/4326"
           "&OUTPUTCRS=http://www.opengis.net/def/crs/EPSG/0/4326")
    body = urlopen(Request(url, headers={"User-Agent": "heyelan-dijital-twin"}),
                   timeout=300, context=_ssl_context()).read()
    dst = np.full((h, w), np.nan, np.float32)
    with rasterio.MemoryFile(io.BytesIO(body)) as mem, mem.open() as src:
        reproject(rasterio.band(src, 1), dst, src_transform=src.transform, src_crs=src.crs,
                  dst_transform=transform, dst_crs="EPSG:4326", resampling=Resampling.bilinear,
                  src_nodata=src.nodata, dst_nodata=np.nan)
    return dst / 10.0     # g/kg → %
