"""
PGA (Tepe Yer İvmesi) Izgara İşleme.

6 Şubat 2023 Kahramanmaraş depremi (Mw 7.7 + Mw 7.6):
  - AFAD ShakeMap veya USGS ShakeMap'ten .tif raster indirilir
  - Her piksel için PGA (g cinsinden) değeri okunur
  - Her heyelan poligonuna zonal statistics ile PGA atanır
"""
from __future__ import annotations

import numpy as np
from pathlib import Path


class PGAGrid:
    """
    PGA raster verisi yönetim sınıfı.

    Kullanım
    --------
    grid = PGAGrid("pga_kahramanmaras_20230206.tif")
    pga_values = grid.extract_polygon_pga(gdf)
    pga_array  = grid.to_array()
    """

    def __init__(self, tif_path: str | Path):
        """
        Parametreler
        -----------
        tif_path : AFAD/USGS ShakeMap PGA raster dosyası (.tif)
                   g cinsinden PGA değerleri içermelidir.
        """
        self.tif_path = Path(tif_path)
        self._array: np.ndarray | None = None
        self._transform = None
        self._crs = None

    def load(self) -> "PGAGrid":
        """Raster verisini belleğe yükler."""
        import rasterio

        with rasterio.open(self.tif_path) as src:
            self._array    = src.read(1).astype(np.float32)
            self._transform = src.transform
            self._crs      = src.crs
            self._nodata   = src.nodata or -9999.0

        # Nodata değerlerini NaN'a çevir
        self._array = np.where(self._array == self._nodata, np.nan, self._array)
        return self

    @property
    def array(self) -> np.ndarray:
        if self._array is None:
            self.load()
        return self._array

    @property
    def transform(self):
        if self._transform is None:
            self.load()
        return self._transform

    def to_array(self) -> np.ndarray:
        """PGA değerlerini 2D numpy array olarak döndürür."""
        return self.array

    def extract_polygon_pga(
        self,
        gdf: "gpd.GeoDataFrame",
        stat: str = "mean",
    ) -> np.ndarray:
        """
        Her heyelan poligonuna PGA değeri atar (zonal statistics).

        Parametreler
        -----------
        gdf  : GeoDataFrame — heyelan poligonları
        stat : "mean", "max", "min" veya "median"

        Döndürür
        --------
        pga_values : (n_polygons,) ndarray — g cinsinden PGA
        """
        from rasterio.mask import mask as rio_mask
        import rasterio
        from shapely.geometry import mapping

        pga_values = np.full(len(gdf), np.nan, dtype=np.float32)

        with rasterio.open(self.tif_path) as src:
            for i, geom in enumerate(gdf.geometry):
                try:
                    out_img, _ = rio_mask(src, [mapping(geom)], crop=True)
                    data = out_img[0].astype(float)
                    data = data[data != (src.nodata or -9999.0)]
                    data = data[np.isfinite(data)]

                    if len(data) == 0:
                        continue

                    if stat == "mean":
                        pga_values[i] = np.mean(data)
                    elif stat == "max":
                        pga_values[i] = np.max(data)
                    elif stat == "min":
                        pga_values[i] = np.min(data)
                    elif stat == "median":
                        pga_values[i] = np.median(data)
                except Exception:
                    continue

        return pga_values

    def sample_at_point(self, x: float, y: float) -> float:
        """
        Tek bir koordinatta PGA değerini örnekler.

        Parametreler
        -----------
        x, y : Coğrafi koordinatlar (CRS ile uyumlu)

        Döndürür
        --------
        pga : float — g cinsinden PGA
        """
        from rasterio.transform import rowcol

        r, c = rowcol(self.transform, x, y)
        arr  = self.array
        r = int(np.clip(r, 0, arr.shape[0] - 1))
        c = int(np.clip(c, 0, arr.shape[1] - 1))
        return float(arr[r, c])

    def create_synthetic_pga_grid(
        self,
        shape: tuple[int, int],
        center_pga: float = 0.6,
        decay_km: float = 50.0,
    ) -> np.ndarray:
        """
        Gerçek ShakeMap yokken test için sentetik PGA ızgarası.

        Gauss azalım modeli ile merkezi yüksek PGA alanı simüle eder.
        Kahramanmaraş deprem merkezi koordinatları baz alınır.

        Parametreler
        -----------
        shape      : (yükseklik, genişlik) piksel boyutu
        center_pga : Merkezdeki max PGA (g)
        decay_km   : PGA'nın %37'ye düştüğü mesafe (km)
        """
        h, w = shape
        cy, cx = h // 2, w // 2  # piksel merkezi

        yy, xx = np.mgrid[0:h, 0:w]
        dist = np.sqrt((yy - cy) ** 2 + (xx - cx) ** 2)

        # Gauss azalım (piksel cinsinden, 1 piksel ≈ 0.5 km varsayımı)
        pixel_per_km = 2.0
        sigma = decay_km * pixel_per_km
        pga = center_pga * np.exp(-0.5 * (dist / sigma) ** 2)

        return pga.astype(np.float32)
