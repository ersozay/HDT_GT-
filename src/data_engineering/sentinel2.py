"""
Sentinel-2 L2A ile envanter doğrulaması (açık veri, AWS Earth Search STAC).

Yöntem (her poligon / nokta için):
  · Ön dönem: 2022 Temmuz–Ağustos, son dönem: 2023 Temmuz–Ağustos (aynı mevsim →
    fenoloji farkı en aza iner; Şubat 2023 karı/gölgesi devre dışı kalır)
  · Her MGRS karosu için en az bulutlu sahneler; SCL ile bulut/gölge/kar/su maskesi
  · İç bölge: poligon (noktalar için r = 40 m disk); halka: 30–150 m tampon
  · Göstergeler: ΔNDVI_iç − ΔNDVI_halka (bitki kaybı < 0) ve
                 ΔParlaklık_iç − ΔParlaklık_halka (yeni yara izi > 0)

Kaynak: Copernicus Sentinel-2 L2A (ESA), COG kopyaları: AWS Open Data
(s3://sentinel-cogs), STAC: https://earth-search.aws.element84.com/v1
"""
from __future__ import annotations

import json
import threading
from dataclasses import dataclass

import numpy as np

STAC = "https://earth-search.aws.element84.com/v1/search"
VALID_SCL = (4, 5, 7)            # bitki, çıplak toprak, sınıflanmamış
GDAL_ENV = dict(GDAL_DISABLE_READDIR_ON_OPEN="EMPTY_DIR", CPL_VSIL_CURL_ALLOWED_EXTENSIONS=".tif",
                GDAL_HTTP_MULTIRANGE="YES", GDAL_HTTP_MERGE_CONSECUTIVE_RANGES="YES",
                VSI_CACHE="TRUE", GDAL_HTTP_TIMEOUT="60", GDAL_HTTP_MAX_RETRY="3",
                GDAL_HTTP_RETRY_DELAY="2")


def _post(body, retries=5):
    import time
    from urllib.request import Request, urlopen
    from .strong_motion import _ssl_context
    last = None
    for k in range(retries):
        try:
            req = Request(STAC, data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
            return json.load(urlopen(req, context=_ssl_context(), timeout=120))
        except Exception as e:        # bağlantı kopması → bekle, yeniden dene
            last = e
            time.sleep(3 * (k + 1))
    raise RuntimeError(f"STAC isteği başarısız: {last}")


def search_scenes(bbox, period: str, max_cloud: float = 5.0) -> list[dict]:
    """Dönemdeki tüm sahneler (sayfalı)."""
    body = {"collections": ["sentinel-2-l2a"], "bbox": list(bbox), "datetime": period,
            "query": {"eo:cloud_cover": {"lt": max_cloud}}, "limit": 100,
            # yanıtı küçült: yalnız gereken alanlar (fields uzantısı)
            "fields": {"include": ["id", "geometry", "properties.grid:code", "properties.eo:cloud_cover",
                                   "properties.datetime", "properties.proj:epsg",
                                   "assets.red", "assets.nir", "assets.scl", "assets.visual"],
                       "exclude": ["links"]}}
    out = []
    while True:
        d = _post(body)
        out += d["features"]
        nxt = [l for l in d.get("links", []) if l.get("rel") == "next"]
        if not nxt or not d["features"]:
            break
        body = nxt[0].get("body") or {**body, "next": nxt[0]["href"].split("next=")[-1]}
    return out


@dataclass
class Scene:
    id: str
    tile: str
    epsg: int
    cloud: float
    date: str
    footprint: object
    assets: dict


def best_scenes(features, per_tile: int = 4) -> dict[str, list[Scene]]:
    """Karo başına en az bulutlu `per_tile` sahne (ayın ortasına yakın olanlar önce)."""
    from shapely.geometry import shape
    by_tile: dict[str, list[Scene]] = {}
    for f in features:
        p = f["properties"]
        a = f["assets"]
        need = ("red", "nir", "scl", "visual")
        if not all(k in a for k in need):
            continue
        s = Scene(f["id"], p["grid:code"], int(p.get("proj:epsg") or a["red"].get("proj:epsg")),
                  float(p["eo:cloud_cover"]), p["datetime"][:10], shape(f["geometry"]),
                  {k: a[k]["href"] for k in need})
        by_tile.setdefault(s.tile, []).append(s)
    for t in by_tile:
        by_tile[t].sort(key=lambda s: (s.cloud > 1.0, s.cloud, abs(int(s.date[5:7]) - 7.5)))
        by_tile[t] = by_tile[t][:per_tile]
    return by_tile


_tls = threading.local()


def _open(href):
    import rasterio
    cache = getattr(_tls, "ds", None)
    if cache is None:
        cache = _tls.ds = {}
    if href not in cache:
        cache[href] = rasterio.open(href)
    return cache[href]


def _read(href, window, out_shape=None):
    ds = _open(href)
    return ds.read(1, window=window, out_shape=out_shape, boundless=True, fill_value=0)


def measure(scene: Scene, geom_ll, inner_r_m: float | None = None,
            ring=(30.0, 150.0)) -> dict | None:
    """
    Bir sahnede iç bölge ve halka için NDVI ve parlaklık ortalamaları.
    geom_ll: WGS84 poligon (ya da nokta + inner_r_m).
    """
    import rasterio
    from rasterio.features import rasterize
    from rasterio.windows import from_bounds
    from shapely.ops import transform as shp_transform
    import pyproj

    to_utm = pyproj.Transformer.from_crs(4326, scene.epsg, always_xy=True).transform
    g = shp_transform(to_utm, geom_ll)
    inner = g.buffer(inner_r_m) if inner_r_m else g
    outer = inner.buffer(ring[1])
    ring_g = outer.difference(inner.buffer(ring[0]))
    with rasterio.Env(**GDAL_ENV):
        ds = _open(scene.assets["red"])
        win = from_bounds(*outer.bounds, ds.transform).round_offsets().round_lengths()
        if win.width < 3 or win.height < 3:
            win = win.__class__(win.col_off, win.row_off, max(win.width, 3), max(win.height, 3))
        tr = ds.window_transform(win)
        shp = (int(win.height), int(win.width))
        red = _read(scene.assets["red"], win).astype(float)
        nir = _read(scene.assets["nir"], win).astype(float)
        grn = _read(scene.assets["green"], win).astype(float)
        blu = _read(scene.assets["blue"], win).astype(float)
        scl_ds = _open(scene.assets["scl"])
        swin = from_bounds(*outer.bounds, scl_ds.transform)
        scl = scl_ds.read(1, window=swin, out_shape=shp, boundless=True, fill_value=0)
    m_in = rasterize([(inner, 1)], out_shape=shp, transform=tr, all_touched=True, dtype="uint8").astype(bool)
    m_rg = rasterize([(ring_g, 1)], out_shape=shp, transform=tr, dtype="uint8").astype(bool)
    valid = np.isin(scl, VALID_SCL) & (red > 0) & (nir > 0)
    ndvi = (nir - red) / np.maximum(nir + red, 1)
    bright = (red + grn + blu) / 3.0 / 10000.0
    out = {}
    for key, m in (("in", m_in), ("ring", m_rg)):
        v = m & valid
        out[f"n_{key}"] = int(m.sum())
        out[f"valid_{key}"] = float(v.sum() / max(m.sum(), 1))
        out[f"ndvi_{key}"] = float(np.median(ndvi[v])) if v.sum() >= 2 else np.nan
        out[f"bright_{key}"] = float(np.median(bright[v])) if v.sum() >= 2 else np.nan
    return out


def measure_best(scenes: list[Scene], geom_ll, centroid_ll, inner_r_m=None, min_valid=0.6):
    """Karoyu kapsayan sahnelerden geçerli piksel oranı yeterli ilkini kullanır."""
    from shapely.geometry import Point
    pt = Point(centroid_ll)
    last = None
    for s in scenes:
        if not s.footprint.contains(pt):
            continue
        try:
            r = measure(s, geom_ll, inner_r_m)
        except Exception as e:          # ağ / okuma hatası → sonraki sahne
            last = {"error": str(e)[:120]}
            continue
        r["scene"] = s.id; r["date"] = s.date
        if r["valid_in"] >= min_valid and r["valid_ring"] >= min_valid:
            return r
        last = r
    return last


def measure_group(scene: Scene, geoms_ll: list, inner_r_m: float | None = None,
                  ring=(30.0, 150.0)) -> list[dict]:
    """
    Birbirine yakın birden çok poligon/nokta için TEK pencere okuyup hepsini ölçer
    (uzaktan COG istek sayısını azaltır).
    """
    import rasterio
    from rasterio.features import rasterize
    from rasterio.windows import from_bounds
    from shapely.ops import transform as shp_transform, unary_union
    import pyproj

    to_utm = pyproj.Transformer.from_crs(4326, scene.epsg, always_xy=True).transform
    inners = []
    for g in geoms_ll:
        gu = shp_transform(to_utm, g)
        inners.append(gu.buffer(inner_r_m) if inner_r_m else gu)
    outers = [g.buffer(ring[1]) for g in inners]
    bounds = unary_union(outers).bounds
    with rasterio.Env(**GDAL_ENV):
        ds = _open(scene.assets["red"])
        win = from_bounds(*bounds, ds.transform).round_offsets().round_lengths()
        tr = ds.window_transform(win)
        shp = (max(int(win.height), 3), max(int(win.width), 3))
        red = _read(scene.assets["red"], win, shp).astype(float)
        nir = _read(scene.assets["nir"], win, shp).astype(float)
        vis = _open(scene.assets["visual"]).read(window=from_bounds(*bounds, _open(scene.assets["visual"]).transform),
                                                 out_shape=(3,) + shp, boundless=True, fill_value=0).astype(float)
        scl_ds = _open(scene.assets["scl"])
        scl = scl_ds.read(1, window=from_bounds(*bounds, scl_ds.transform), out_shape=shp,
                          boundless=True, fill_value=0)
    valid = np.isin(scl, VALID_SCL) & (red > 0) & (nir > 0)
    ndvi = (nir - red) / np.maximum(nir + red, 1)
    bright = vis.mean(axis=0) / 255.0
    out = []
    for inner, outer in zip(inners, outers):
        ring_g = outer.difference(inner.buffer(ring[0]))
        m_in = rasterize([(inner, 1)], out_shape=shp, transform=tr, all_touched=True, dtype="uint8").astype(bool)
        m_rg = rasterize([(ring_g, 1)], out_shape=shp, transform=tr, dtype="uint8").astype(bool)
        r = {"scene": scene.id, "date": scene.date}
        for key, m in (("in", m_in), ("ring", m_rg)):
            v = m & valid
            r[f"n_{key}"] = int(m.sum())
            r[f"valid_{key}"] = float(v.sum() / max(m.sum(), 1))
            r[f"ndvi_{key}"] = float(np.median(ndvi[v])) if v.sum() >= 2 else np.nan
            r[f"bright_{key}"] = float(np.median(bright[v])) if v.sum() >= 2 else np.nan
        out.append(r)
    return out
