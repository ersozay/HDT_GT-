"""
Script 0: Gerçek Veri Edinme

USGS ShakeMap'ten 6 Şubat 2023 Kahramanmaraş deprem dizisi (Mw7.8 + Mw7.5)
PGA ızgaralarını indirir, GeoTIFF'e çevirir ve maksimum-PGA zarfını oluşturur.

Kullanım:
  python scripts/00_fetch_real_data.py
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from config import DATA_RAW, ensure_dirs
from src.data_engineering.data_fetch import (
    USGS_EVENTS, download_shakemap_grid,
    parse_shakemap_grid, shakemap_to_raster,
)


def build_max_pga_envelope(out_tif: Path) -> Path:
    """
    İki depremin PGA rasterını ortak ızgaraya hizalayıp element-wise
    maksimum alarak birleşik PGA zarfı üretir.

    Ortak ızgara: en geniş kapsamlı Mw7.8 rasterı referans alınır,
    Mw7.5 rasterı bu ızgaraya yeniden örneklenir (bilinear).
    """
    import rasterio
    from rasterio.warp import reproject, Resampling

    tif_78 = DATA_RAW / "pga_us6000jllz.tif"  # Mw7.8 (referans, geniş)
    tif_75 = DATA_RAW / "pga_us6000jlqa.tif"  # Mw7.5

    with rasterio.open(tif_78) as ref:
        ref_data = ref.read(1)
        ref_profile = ref.profile.copy()
        ref_transform = ref.transform
        ref_crs = ref.crs
        ref_shape = ref_data.shape

        # Mw7.5'i referans ızgaraya yeniden örnekle
        resampled_75 = np.full(ref_shape, -9999.0, dtype="float32")
        with rasterio.open(tif_75) as src75:
            reproject(
                source=rasterio.band(src75, 1),
                destination=resampled_75,
                src_transform=src75.transform,
                src_crs=src75.crs,
                dst_transform=ref_transform,
                dst_crs=ref_crs,
                resampling=Resampling.bilinear,
                dst_nodata=-9999.0,
            )

    # Nodata'ları 0'a çek, sonra element-wise max
    a = np.where(ref_data == -9999.0, 0.0, ref_data)
    b = np.where(resampled_75 == -9999.0, 0.0, resampled_75)
    max_pga = np.maximum(a, b).astype("float32")

    ref_profile.update(dtype="float32", count=1, nodata=-9999.0)
    with rasterio.open(out_tif, "w", **ref_profile) as dst:
        dst.write(max_pga, 1)

    print(f"\nBirleşik maksimum-PGA zarfı: {out_tif}")
    print(f"  PGA: min={max_pga.min():.3f}g  "
          f"mean={max_pga[max_pga>0].mean():.3f}g  max={max_pga.max():.3f}g")
    return out_tif


def run():
    ensure_dirs()

    print("=" * 60)
    print("GERÇEK VERİ EDİNME — USGS ShakeMap")
    print("6 Şubat 2023 Kahramanmaraş Deprem Dizisi")
    print("=" * 60)

    for eid in USGS_EVENTS:
        xml_path = DATA_RAW / f"shakemap_{eid}_grid.xml"
        if not xml_path.exists():
            download_shakemap_grid(eid, xml_path)
        grid = parse_shakemap_grid(xml_path)
        shakemap_to_raster(grid, DATA_RAW / f"pga_{eid}.tif")

    # Birleşik zarf
    build_max_pga_envelope(DATA_RAW / "pga_kahramanmaras_20230206.tif")

    print("\n" + "=" * 60)
    print("Gerçek PGA verisi hazır.")
    print("Kaynak: USGS ShakeMap (DOI: 10.5066/F7W957B2)")
    print("Atıf: Worden et al. (2020), USGS ShakeMap System")
    print("=" * 60)


if __name__ == "__main__":
    run()
