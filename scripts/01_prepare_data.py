"""
Script 1: Veri Hazırlama Pipeline'ı

Adımlar:
  1. KML envanterini yükle (2596 poligon)
  2. DEM'den slope/aspect rasterları üret
  3. Litoloji haritasından jeomekanik parametreler ata
  4. PGA rasterını yükle ve poligonlara zonal statistics uygula
  5. Öznitelik matrisini oluştur ve kaydet

Kullanım:
  python scripts/01_prepare_data.py [--synthetic]
"""
import sys
import argparse
import numpy as np
from pathlib import Path

# Proje kökü sys.path'e ekle
sys.path.insert(0, str(Path(__file__).parent.parent))

from config import (
    DATA_RAW, DATA_PROCESSED, FIGURES_DIR, PILOT_IDS,
    Z_REF_M, DN_THRESHOLD_M, ensure_dirs
)


from src.data_engineering.feature_matrix import prepare_synthetic_data


# Çalışma alanı sınırları (Kocaman 2025 envanter sınırları + tampon)
STUDY_BOUNDS = (35.7, 35.9, 39.25, 38.65)  # (lon_min, lat_min, lon_max, lat_max)


def run_synthetic():
    """Sentetik veriyle öznitelik matrisi üretir (hızlı doğrulama için)."""
    from src.data_engineering.feature_matrix import build_feature_matrix
    from src.geomechanics.newmark import LandslideLabel

    print("Sentetik veri üretiliyor...")
    raw = prepare_synthetic_data()
    n = raw["n_pixels"]
    print(f"  {n:,} piksel üretildi")

    df, feat_cols = build_feature_matrix(
        elevation_m=raw["elevation_m"].reshape(n),
        slope_deg=raw["slope_deg"].reshape(n),
        aspect_deg=raw["aspect_deg"].reshape(n),
        litho_array=raw["litho_array"].reshape(n),
        pga_array=raw["pga_array"].reshape(n),
        u_array=raw["u_array"].reshape(n),
        z_ref=Z_REF_M,
    )
    df["label"] = LandslideLabel(df["dn_cm"].values / 100.0, threshold_m=DN_THRESHOLD_M)
    df["polygon_id"] = raw["polygon_ids"]
    return df, feat_cols


def run_real():
    """
    Gerçek katmanlardan (USGS PGA + Copernicus DEM + Macrostrat litoloji +
    Kocaman 2025 envanter) nokta-bazlı öznitelik matrisi üretir.
    Etiket = gözlemlenen heyelan (envanter poligonu içi = 1).
    """
    from src.data_engineering.real_dataset import (
        compute_slope_aspect, generate_samples, build_real_feature_matrix,
    )

    dem_path    = DATA_RAW / "EUDEM_kahramanmaras.tif"
    pga_path    = DATA_RAW / "pga_kahramanmaras_20230206.tif"
    litho_path  = DATA_RAW / "lithology_macrostrat_proxy.tif"
    inv_path    = DATA_RAW / "TJES-2024-00171_Landslides.kml"
    slope_path  = DATA_PROCESSED / "slope_deg.tif"
    aspect_path = DATA_PROCESSED / "aspect_deg.tif"

    for p in (dem_path, pga_path, litho_path, inv_path):
        if not p.exists():
            raise FileNotFoundError(f"Gerekli veri eksik: {p}")

    # 1. Eğim/bakı rasterları (önbellek: yoksa üret)
    if not (slope_path.exists() and aspect_path.exists()):
        print("DEM'den eğim/bakı türetiliyor...")
        compute_slope_aspect(dem_path, slope_path, aspect_path)
    else:
        print("Eğim/bakı rasterları önbellekten kullanılıyor.")

    # 2. Pozitif (envanter) + negatif (arka plan) örnek noktaları
    print("Örnek noktaları üretiliyor...")
    samples = generate_samples(inv_path, bounds=STUDY_BOUNDS,
                               seed=TRAIN_SEED)

    # 3. Katmanları örnekle ve öznitelik matrisi kur
    print("Katmanlar örnekleniyor ve öznitelikler hesaplanıyor...")
    df, feat_cols = build_real_feature_matrix(
        samples, dem_path, slope_path, aspect_path, pga_path, litho_path,
        z_ref=Z_REF_M,
    )
    # 03_train_pinn.py şeması ile uyum: label + polygon_id
    df["label"] = df["y"].astype(int)
    df["polygon_id"] = df["group"]
    return df, feat_cols


TRAIN_SEED = 42


def run(synthetic: bool = False):
    ensure_dirs()

    if synthetic:
        df, feat_cols = run_synthetic()
    else:
        try:
            df, feat_cols = run_real()
        except Exception as e:
            print(f"Gerçek veri işlenemedi ({e}); sentetik veriye geçiliyor...")
            df, feat_cols = run_synthetic()

    out_path = DATA_PROCESSED / "feature_matrix.parquet"
    df.to_parquet(out_path, index=False)
    print(f"\nÖznitelik matrisi kaydedildi: {out_path}")
    print(f"  Boyut: {df.shape}")
    print(f"  Heyelan oranı: {df['label'].mean():.1%}")
    print(f"  Öznitelik sayısı: {len(feat_cols)}")
    print("\nVeri hazırlama tamamlandı.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Veri hazırlama pipeline'ı")
    parser.add_argument("--synthetic", action="store_true",
                        help="Gerçek veri yerine sentetik veri kullan")
    args = parser.parse_args()
    run(synthetic=args.synthetic)
