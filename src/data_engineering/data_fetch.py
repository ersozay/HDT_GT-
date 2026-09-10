"""
Gerçek Veri Edinme Modülü.

Kamuya açık, atıf verilebilir kaynaklardan gerçek veri çeker:
  - USGS ShakeMap       : 6 Şubat 2023 Kahramanmaraş depremi PGA ızgarası
  - Copernicus GLO-90 DEM: Sayısal yükseklik modeli (AWS S3, anahtarsız)
  - Macrostrat API      : Litoloji PROXY'si (CC-BY 4.0) — resmi MTA haritası DEĞİL

Not: Kocaman (2025) heyelan envanteri kamuya açık API üzerinden çekilemez;
kullanıcı tarafından sağlanmalıdır. Litoloji katmanı Macrostrat tabanlı bir
jeolojik birim PROXY'sidir; tezde bu şekilde (proxy) atıf verilmelidir.
"""
from __future__ import annotations

import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
import requests


# 6 Şubat 2023 Kahramanmaraş deprem dizisi — USGS event ID'leri
USGS_EVENTS = {
    "us6000jllz": "Mw7.8 Pazarcık (04:17 TSİ)",
    "us6000jlqa": "Mw7.5 Elbistan (13:24 TSİ)",
}

USGS_EVENT_API = "https://earthquake.usgs.gov/fdsnws/event/1/query"


def get_shakemap_grid_url(event_id: str) -> str:
    """USGS event API'sinden ShakeMap grid.xml URL'sini çeker."""
    r = requests.get(
        USGS_EVENT_API,
        params={"format": "geojson", "eventid": event_id},
        timeout=30,
    )
    r.raise_for_status()
    products = r.json()["properties"]["products"]
    shakemap = products["shakemap"][0]
    return shakemap["contents"]["download/grid.xml"]["url"]


def download_shakemap_grid(event_id: str, dest: str | Path) -> Path:
    """
    USGS ShakeMap grid.xml dosyasını indirir.

    Parametreler
    -----------
    event_id : USGS event ID (örn. 'us6000jllz')
    dest     : Kaydedilecek dosya yolu

    Döndürür
    --------
    Path : İndirilen dosya yolu
    """
    dest = Path(dest)
    dest.parent.mkdir(parents=True, exist_ok=True)

    url = get_shakemap_grid_url(event_id)
    print(f"  İndiriliyor: {USGS_EVENTS.get(event_id, event_id)}")
    print(f"  {url}")

    with requests.get(url, stream=True, timeout=120) as r:
        r.raise_for_status()
        total = int(r.headers.get("content-length", 0))
        downloaded = 0
        with open(dest, "wb") as f:
            for chunk in r.iter_content(chunk_size=1 << 16):
                f.write(chunk)
                downloaded += len(chunk)
        print(f"  Kaydedildi: {dest} ({downloaded/1e6:.1f} MB)")

    return dest


def parse_shakemap_grid(grid_xml_path: str | Path) -> dict:
    """
    ShakeMap grid.xml dosyasını ayrıştırır.

    ShakeMap XML formatı: her satır bir grid noktası; sütunlar
    LON LAT PGA PGV MMI PSA03 PSA10 PSA30 ... (grid_field ile tanımlı).

    PGA birimi: %g (yüzde g). Bu fonksiyon g'ye çevirir.

    Döndürür
    --------
    dict :
        lon, lat  : 1D koordinat dizileri
        pga_g     : PGA (g cinsinden) 1D dizi
        pgv       : PGV (cm/s)
        mmi       : Modifiye Mercalli Şiddet
        nx, ny    : Izgara boyutları
        bounds    : (lon_min, lon_max, lat_min, lat_max)
    """
    grid_xml_path = Path(grid_xml_path)

    # ShakeMap namespace
    ns = {"sm": "http://earthquake.usgs.gov/eqcenter/shakemap"}

    tree = ET.parse(grid_xml_path)
    root = tree.getroot()

    # grid_specification niteliklerini oku
    spec = root.find("sm:grid_specification", ns)
    if spec is None:
        # namespace olmadan dene
        spec = root.find("{http://earthquake.usgs.gov/eqcenter/shakemap}grid_specification")

    def _find(tag):
        el = root.find(f"sm:{tag}", ns)
        if el is None:
            el = root.find(f"{{http://earthquake.usgs.gov/eqcenter/shakemap}}{tag}")
        return el

    spec = _find("grid_specification")
    nx = int(spec.attrib["nlon"])
    ny = int(spec.attrib["nlat"])
    lon_min = float(spec.attrib["lon_min"])
    lon_max = float(spec.attrib["lon_max"])
    lat_min = float(spec.attrib["lat_min"])
    lat_max = float(spec.attrib["lat_max"])

    # grid_field sıralamasını oku (hangi sütun hangi büyüklük)
    fields = {}
    for gf in root.iter():
        if gf.tag.endswith("grid_field"):
            idx = int(gf.attrib["index"])
            name = gf.attrib["name"]
            fields[name.upper()] = idx - 1  # 1-indexli → 0-indexli

    # grid_data metnini oku
    grid_data_el = _find("grid_data")
    raw = grid_data_el.text.strip()

    data = np.fromstring(raw, sep=" ").reshape(-1, len(fields))

    lon = data[:, fields["LON"]]
    lat = data[:, fields["LAT"]]
    pga_pct_g = data[:, fields["PGA"]]  # %g cinsinden
    pga_g = pga_pct_g / 100.0

    result = {
        "lon": lon,
        "lat": lat,
        "pga_g": pga_g,
        "nx": nx,
        "ny": ny,
        "bounds": (lon_min, lon_max, lat_min, lat_max),
        "fields": fields,
    }
    if "PGV" in fields:
        result["pgv"] = data[:, fields["PGV"]]
    if "MMI" in fields:
        result["mmi"] = data[:, fields["MMI"]]

    return result


def shakemap_to_raster(
    grid: dict,
    out_tif: str | Path,
    field: str = "pga_g",
) -> Path:
    """
    Ayrıştırılmış ShakeMap ızgarasını GeoTIFF rasterına dönüştürür.

    Parametreler
    -----------
    grid    : parse_shakemap_grid çıktısı
    out_tif : Çıktı .tif yolu
    field   : Raster'a yazılacak alan ('pga_g', 'pgv', 'mmi')

    Döndürür
    --------
    Path : GeoTIFF yolu
    """
    import rasterio
    from rasterio.transform import from_bounds

    out_tif = Path(out_tif)
    out_tif.parent.mkdir(parents=True, exist_ok=True)

    nx, ny = grid["nx"], grid["ny"]
    lon_min, lon_max, lat_min, lat_max = grid["bounds"]

    # ShakeMap ızgarası: kuzeybatıdan güneydoğuya (lat azalan, lon artan)
    values = grid[field].reshape(ny, nx)

    transform = from_bounds(lon_min, lat_min, lon_max, lat_max, nx, ny)

    with rasterio.open(
        out_tif, "w",
        driver="GTiff",
        height=ny, width=nx,
        count=1, dtype="float32",
        crs="EPSG:4326",
        transform=transform,
        nodata=-9999.0,
    ) as dst:
        dst.write(values.astype("float32"), 1)

    print(f"  GeoTIFF kaydedildi: {out_tif} ({nx}×{ny} piksel)")
    return out_tif


def combine_max_pga(grids: list[dict]) -> dict:
    """
    Birden fazla deprem ShakeMap'inden piksel bazlı MAKSİMUM PGA alır.

    6 Şubat 2023'te iki büyük deprem (Mw7.8 + Mw7.5) olduğu için her
    lokasyonun maruz kaldığı en yüksek PGA envelope'u hesaplanır.

    Not: Izgaralar farklı çözünürlük/kapsama sahipse ortak ızgaraya
    enterpolasyon gerekir. Bu fonksiyon aynı ızgara varsayar; farklıysa
    resample_to_common ile önce hizalanmalıdır.
    """
    if len(grids) == 1:
        return grids[0]

    # En büyük kapsamlı ızgarayı referans al
    ref = max(grids, key=lambda g: g["nx"] * g["ny"])
    max_pga = ref["pga_g"].copy()

    # Basit yaklaşım: aynı boyutlu ızgaralarda element-wise max
    for g in grids:
        if g["pga_g"].shape == max_pga.shape:
            max_pga = np.maximum(max_pga, g["pga_g"])

    result = dict(ref)
    result["pga_g"] = max_pga
    return result


# ---------------------------------------------------------------------------
# Copernicus DEM (GLO-90) — kamuya açık, atıf verilebilir sayısal yükseklik
# AWS Open Data üzerinden anahtarsız erişim: https://copernicus-dem-90m.s3.amazonaws.com
# Atıf: ESA/Airbus Copernicus DEM (2021), GLO-90, DOI:10.5270/ESA-c5d3d65
# ---------------------------------------------------------------------------

COPERNICUS_GLO90_BASE = "https://copernicus-dem-90m.s3.amazonaws.com"


def _copernicus_tile_name(lat_tile: int, lon_tile: int) -> str:
    """
    Copernicus GLO-90 kayan tavan (floor) tile ismi üretir.

    Tile ismi enlem/boylamın tam sayı tabanına göre kurulur, örn.
    lat=37, lon=37 → 'Copernicus_DSM_COG_30_N37_00_E037_00_DEM'.
    Kuzey/Güney ve Doğu/Batı işaretleri otomatik atanır.
    """
    ns = "N" if lat_tile >= 0 else "S"
    ew = "E" if lon_tile >= 0 else "W"
    return (
        f"Copernicus_DSM_COG_30_"
        f"{ns}{abs(lat_tile):02d}_00_"
        f"{ew}{abs(lon_tile):03d}_00_DEM"
    )


def download_copernicus_dem_tiles(
    bounds: tuple[float, float, float, float],
    dest_dir: str | Path,
) -> list[Path]:
    """
    Verilen sınırları kaplayan Copernicus GLO-90 DEM tile'larını indirir.

    Parametreler
    -----------
    bounds   : (lon_min, lat_min, lon_max, lat_max) — WGS84
    dest_dir : Tile'ların kaydedileceği klasör

    Döndürür
    --------
    list[Path] : Başarıyla indirilen (veya önbellekte bulunan) tile yolları
    """
    import math

    dest_dir = Path(dest_dir)
    dest_dir.mkdir(parents=True, exist_ok=True)

    lon_min, lat_min, lon_max, lat_max = bounds
    lat_tiles = range(int(math.floor(lat_min)), int(math.floor(lat_max)) + 1)
    lon_tiles = range(int(math.floor(lon_min)), int(math.floor(lon_max)) + 1)

    tiles: list[Path] = []
    for lat_t in lat_tiles:
        for lon_t in lon_tiles:
            name = _copernicus_tile_name(lat_t, lon_t)
            url = f"{COPERNICUS_GLO90_BASE}/{name}/{name}.tif"
            out = dest_dir / f"{name}.tif"

            if out.exists() and out.stat().st_size > 0:
                tiles.append(out)
                continue

            try:
                with requests.get(url, stream=True, timeout=120) as r:
                    if r.status_code == 404:
                        # Deniz/veri yok — bu tile atlanır
                        print(f"  [atlandı] Tile yok (deniz?): {name}")
                        continue
                    r.raise_for_status()
                    with open(out, "wb") as f:
                        for chunk in r.iter_content(chunk_size=1 << 16):
                            f.write(chunk)
                mb = out.stat().st_size / 1e6
                print(f"  İndirildi: {name} ({mb:.1f} MB)")
                tiles.append(out)
            except requests.RequestException as e:
                print(f"  [hata] {name}: {e}")

    return tiles


def mosaic_and_clip_dem(
    tiles: list[Path],
    bounds: tuple[float, float, float, float],
    out_tif: str | Path,
) -> Path:
    """
    İndirilen DEM tile'larını birleştirip (mosaic) verilen sınırlara klipler.

    Parametreler
    -----------
    tiles   : download_copernicus_dem_tiles çıktısı
    bounds  : (lon_min, lat_min, lon_max, lat_max) klip penceresi
    out_tif : Çıktı GeoTIFF yolu

    Döndürür
    --------
    Path : Birleştirilmiş+kliplenmiş DEM yolu
    """
    import rasterio
    from rasterio.merge import merge
    from rasterio.windows import from_bounds as window_from_bounds

    out_tif = Path(out_tif)
    out_tif.parent.mkdir(parents=True, exist_ok=True)

    if not tiles:
        raise RuntimeError("Birleştirilecek DEM tile bulunamadı.")

    lon_min, lat_min, lon_max, lat_max = bounds

    srcs = [rasterio.open(t) for t in tiles]
    try:
        mosaic, transform = merge(
            srcs, bounds=(lon_min, lat_min, lon_max, lat_max)
        )
        profile = srcs[0].profile.copy()
    finally:
        for s in srcs:
            s.close()

    profile.update(
        driver="GTiff",
        height=mosaic.shape[1],
        width=mosaic.shape[2],
        transform=transform,
        count=1,
        dtype=mosaic.dtype,
    )

    with rasterio.open(out_tif, "w", **profile) as dst:
        dst.write(mosaic[0], 1)

    elev = mosaic[0]
    valid = elev[elev > -1000]
    print(f"\nDEM mozaiği kaydedildi: {out_tif}")
    print(f"  Boyut: {mosaic.shape[2]}×{mosaic.shape[1]} piksel")
    print(f"  Yükseklik: min={valid.min():.0f}m  "
          f"mean={valid.mean():.0f}m  max={valid.max():.0f}m")
    return out_tif


def fetch_dem(
    bounds: tuple[float, float, float, float],
    out_tif: str | Path,
    tile_cache_dir: str | Path | None = None,
) -> Path:
    """
    Copernicus GLO-90 DEM'i indirip birleştirip klipleyen tek-adım fonksiyon.

    Parametreler
    -----------
    bounds         : (lon_min, lat_min, lon_max, lat_max) — WGS84
    out_tif        : Çıktı DEM GeoTIFF yolu
    tile_cache_dir : Ham tile önbelleği (varsayılan: out_tif yanında dem_tiles/)

    Döndürür
    --------
    Path : Nihai DEM yolu
    """
    out_tif = Path(out_tif)
    if tile_cache_dir is None:
        tile_cache_dir = out_tif.parent / "dem_tiles"

    print("Copernicus GLO-90 DEM tile'ları indiriliyor...")
    tiles = download_copernicus_dem_tiles(bounds, tile_cache_dir)
    print(f"  Toplam {len(tiles)} tile hazır.")
    return mosaic_and_clip_dem(tiles, bounds, out_tif)


# ---------------------------------------------------------------------------
# Litoloji proxy — Macrostrat API (CC-BY 4.0, anahtarsız)
# Kaynak: Peters, S.E. et al. (2018) Macrostrat; alttaki jeoloji haritaları
#   1:5M Int. Geol. Map of Europe (Asch 2005) & Generalized Geology of the
#   World (Chorlton 2007). MTA 1:100k litoloji haritasının açık proxy'si.
# ---------------------------------------------------------------------------

MACROSTRAT_MAP_API = "https://macrostrat.org/api/v2/geologic_units/map"

# Macrostrat 'lith'/'name' metnindeki anahtar kelimeleri projedeki 27 sınıflık
# LITHOLOGY_TABLE ID'lerine eşleyen kurallar (sıra önemlidir: ilk eşleşen kazanır).
LITHO_KEYWORD_MAP: list[tuple[tuple[str, ...], int]] = [
    (("alluv", "unconsolidated", "colluv", "soil", "gravel"), 1),
    (("talus", "scree"),                                       12),
    (("evaporit", "gypsum", "anhydrite", "salt", "halite"),    9),
    (("marl",),                                                7),
    (("dolomit",),                                             13),
    (("limestone", "carbonate", "chalk", "travertine"),        5),
    (("mudstone", "claystone", "shale", "argillite"),          8),
    (("clay",),                                                9),
    (("siltstone", "silt"),                                    10),
    (("sandstone", "arenite", "greywacke", "siliciclastic", "flysch", "molasse"), 4),
    (("conglomerate", "breccia"),                              25),
    (("ophiolit", "serpentin", "peridotit", "ultramafic"),     20),
    (("basalt", "mafic", "gabbro", "diabase", "dolerite", "basic"), 14),
    (("andesit", "intermediate volcanic", "dacit"),            15),
    (("rhyolit", "acid volcanic", "felsic volcanic"),          22),
    (("ignimbrit", "tuff", "pyroclastic", "volcaniclastic"),   21),
    (("granit", "granodiorit", "diorit", "plutonic", "intrusive"), 16),
    (("phyllit",),                                             18),
    (("schist", "gneiss", "metamorphic", "marble", "quartzite", "slate"), 17),
    (("volcanic", "lava", "andesitic"),                        15),
    (("sedimentary",),                                         27),
]

# Litoloji metni bulunamazsa/eşleşmezse "karışık litoloji" (27)
LITHO_DEFAULT_ID = 27


def lith_text_to_class(text: str) -> int:
    """Macrostrat litoloji/isim metnini 1–27 arası proje sınıfına eşler."""
    if not text:
        return LITHO_DEFAULT_ID
    t = text.lower()
    for keywords, cls in LITHO_KEYWORD_MAP:
        if any(k in t for k in keywords):
            return cls
    return LITHO_DEFAULT_ID


def query_macrostrat_point(lon: float, lat: float, session=None) -> int:
    """
    Tek bir (lon, lat) noktası için Macrostrat'tan litoloji sınıfı sorgular.

    Macrostrat birden fazla harita kaynağını yığın olarak döndürür (data[]).
    Tüm kayıtlar taranır; sırayla 'lith' (daha detaylı) sonra 'name' metni
    27 sınıfa eşlenir ve varsayılan (27) olmayan İLK anlamlı sınıf seçilir.
    Böylece üst poligonun litolojisi boşsa alttaki kaynaklardan yararlanılır.
    Ağ hatasında varsayılan sınıf (27) döner.
    """
    sess = session or requests
    try:
        r = sess.get(
            MACROSTRAT_MAP_API,
            params={"lng": lon, "lat": lat, "format": "json"},
            timeout=30,
        )
        r.raise_for_status()
        data = r.json().get("success", {}).get("data", [])
        if not data:
            return LITHO_DEFAULT_ID
        # Önce tüm kayıtların 'lith' alanı, sonra 'name' alanı denenir
        for field in ("lith", "name"):
            for rec in data:
                text = (rec.get(field) or "").strip()
                if not text:
                    continue
                cls = lith_text_to_class(text)
                if cls != LITHO_DEFAULT_ID:
                    return cls
        return LITHO_DEFAULT_ID
    except (requests.RequestException, ValueError):
        return LITHO_DEFAULT_ID


def fetch_lithology_raster(
    bounds: tuple[float, float, float, float],
    out_tif: str | Path,
    resolution_deg: float = 0.05,
    cache_path: str | Path | None = None,
) -> Path:
    """
    Macrostrat noktasal sorgularından litoloji proxy rasterı üretir.

    Çalışma alanını `resolution_deg` adımlı bir ızgaraya böler, her hücre
    merkezini Macrostrat'a sorar, 27 sınıflık ID'ye eşler ve GeoTIFF yazar.
    Sorgular JSON önbelleğe alınır (kesinti sonrası devam edilebilir).

    Parametreler
    -----------
    bounds         : (lon_min, lat_min, lon_max, lat_max) — WGS84
    out_tif        : Çıktı litoloji GeoTIFF yolu (int16, ID = 1..27)
    resolution_deg : Izgara adımı (derece). 0.05° ≈ 5.5 km.
    cache_path     : Sorgu önbelleği JSON yolu (varsayılan: out_tif+'.cache.json')
    """
    import json
    import numpy as np
    import rasterio
    from rasterio.transform import from_origin

    out_tif = Path(out_tif)
    out_tif.parent.mkdir(parents=True, exist_ok=True)
    if cache_path is None:
        cache_path = out_tif.with_suffix(".cache.json")
    cache_path = Path(cache_path)

    cache: dict[str, int] = {}
    if cache_path.exists():
        cache = json.loads(cache_path.read_text())
        print(f"  Önbellek yüklendi: {len(cache)} nokta")

    lon_min, lat_min, lon_max, lat_max = bounds
    lons = np.arange(lon_min + resolution_deg / 2, lon_max, resolution_deg)
    lats = np.arange(lat_min + resolution_deg / 2, lat_max, resolution_deg)
    nx, ny = len(lons), len(lats)
    grid = np.full((ny, nx), LITHO_DEFAULT_ID, dtype=np.int16)

    sess = requests.Session()
    total = nx * ny
    done = 0
    print(f"Litoloji ızgarası sorgulanıyor: {nx}×{ny} = {total} hücre")
    for j, lat in enumerate(lats[::-1]):          # kuzeyden güneye (raster düzeni)
        for i, lon in enumerate(lons):
            key = f"{lon:.4f},{lat:.4f}"
            if key in cache:
                grid[j, i] = cache[key]
            else:
                cls = query_macrostrat_point(lon, lat, session=sess)
                cache[key] = cls
                grid[j, i] = cls
                done += 1
                if done % 100 == 0:
                    cache_path.write_text(json.dumps(cache))
                    print(f"  {done} yeni sorgu tamamlandı...")
    cache_path.write_text(json.dumps(cache))

    transform = from_origin(lon_min, lat_max, resolution_deg, resolution_deg)
    profile = {
        "driver": "GTiff", "height": ny, "width": nx, "count": 1,
        "dtype": "int16", "crs": "EPSG:4326", "transform": transform,
        "nodata": 0,
    }
    with rasterio.open(out_tif, "w", **profile) as dst:
        dst.write(grid, 1)

    uniq, counts = np.unique(grid, return_counts=True)
    print(f"\nLitoloji rasterı kaydedildi: {out_tif}  ({nx}×{ny})")
    print("  Sınıf dağılımı:", dict(zip(uniq.tolist(), counts.tolist())))
    return out_tif
