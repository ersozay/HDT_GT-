"""
Yağış (yağmur) verisi indirici — CHIRPS-2.0 (UCSB Climate Hazards Center).

Amaç: Çalışma alanı için aylık yağış rasterı indirip STUDY_BOUNDS'a kırpar.
Bu katman YALNIZCA GÖRSEL bağlam içindir — PINN öznitelik matrisine GİRMEZ
(model yeniden eğitilmedi). Kahramanmaraş 6 Şubat 2023 depremi bağlamında,
olay ayı (2023-02) yağışı jüriye bölgesel ıslaklık göstergesi olarak sunulur.

Kaynak: CHIRPS-2.0, anahtarsız HTTP:
  https://data.chc.ucsb.edu/products/CHIRPS-2.0/global_monthly/tifs/
  chirps-v2.0.YYYY.MM.tif.gz   (0.05°, global, mm/ay)

Ağ yoksa veya indirme başarısızsa fonksiyon açık hata verir; uygulama yağış
katmanı olmadan çalışmaya devam eder (backend `available:false` döndürür).

Referans: Funk, C. et al. (2015). The climate hazards infrared precipitation
with stations. Scientific Data 2, 150066.
"""
from __future__ import annotations

import gzip
import io
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import config

CHIRPS_MONTHLY_BASE = (
    "https://data.chc.ucsb.edu/products/CHIRPS-2.0/global_monthly/tifs"
)
RAINFALL_RASTER = config.DATA_RAW / "rainfall_chirps.tif"


def chirps_monthly_url(year: int, month: int) -> str:
    return f"{CHIRPS_MONTHLY_BASE}/chirps-v2.0.{year}.{month:02d}.tif.gz"


def fetch_rainfall_raster(
    year: int = 2023,
    month: int = 2,
    bounds: tuple[float, float, float, float] | None = None,
    out_tif: str | Path = RAINFALL_RASTER,
    timeout: int = 180,
) -> Path:
    """
    CHIRPS-2.0 aylık yağış GeoTIFF'ini indirir, STUDY_BOUNDS'a kırpar, kaydeder.

    Parametreler
    -----------
    year, month : Yağış ayı (varsayılan 2023-02, deprem ayı)
    bounds      : (lon_min, lat_min, lon_max, lat_max); None → STUDY_BOUNDS
    out_tif     : Çıktı GeoTIFF yolu

    Döndürür
    --------
    Path : yazılan raster yolu

    Hata
    ----
    RuntimeError : indirme/işleme başarısızsa (ağ yok vs.)
    """
    import numpy as np
    import requests
    import rasterio
    from rasterio.io import MemoryFile
    from rasterio.windows import from_bounds as window_from_bounds

    if bounds is None:
        # config.STUDY_BOUNDS yoksa MEMORY'deki alanı kullan
        bounds = getattr(config, "STUDY_BOUNDS", (35.7, 35.9, 39.25, 38.65))
    lon_min, lat_min, lon_max, lat_max = bounds

    out_tif = Path(out_tif)
    out_tif.parent.mkdir(parents=True, exist_ok=True)

    url = chirps_monthly_url(year, month)
    print(f"CHIRPS indiriliyor: {url}")
    try:
        r = requests.get(url, stream=True, timeout=timeout)
        r.raise_for_status()
        gz_bytes = r.content
    except requests.RequestException as e:
        raise RuntimeError(f"CHIRPS indirilemedi (ağ?): {e}") from e

    try:
        tif_bytes = gzip.decompress(gz_bytes)
    except OSError as e:
        raise RuntimeError(f"CHIRPS gzip açılamadı: {e}") from e

    with MemoryFile(tif_bytes) as mem, mem.open() as src:
        window = window_from_bounds(
            lon_min, lat_min, lon_max, lat_max, transform=src.transform
        )
        data = src.read(1, window=window)
        transform = src.window_transform(window)
        src_nodata = src.nodata if src.nodata is not None else -9999.0

        # CHIRPS nodata (-9999) → NaN
        arr = data.astype("float32")
        arr[arr == src_nodata] = np.nan
        arr[arr < 0] = np.nan

        profile = {
            "driver": "GTiff", "height": arr.shape[0], "width": arr.shape[1],
            "count": 1, "dtype": "float32", "crs": "EPSG:4326",
            "transform": transform, "nodata": float("nan"),
        }
        with rasterio.open(out_tif, "w", **profile) as dst:
            dst.write(arr, 1)

    finite = arr[np.isfinite(arr)]
    vmin = float(finite.min()) if finite.size else 0.0
    vmax = float(finite.max()) if finite.size else 0.0
    print(f"Yağış rasterı kaydedildi: {out_tif}  ({arr.shape[1]}×{arr.shape[0]})")
    print(f"  {year}-{month:02d} yağış aralığı: {vmin:.1f} – {vmax:.1f} mm")
    return out_tif


if __name__ == "__main__":
    fetch_rainfall_raster()
