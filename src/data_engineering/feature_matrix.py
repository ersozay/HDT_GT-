"""
Öznitelik Matrisi Oluşturucu.

Tüm öznitelik grupları:
  - Topografik  : DEM, Slope, Aspect (sin/cos)
  - Jeomekanik  : c', φ', γ, u
  - Sismik      : PGA, Ky, Dn
  - Litoloji    : One-Hot (27 sınıf)
  - Morfolojik  : Compactness, Elongation (isteğe bağlı)
"""
from __future__ import annotations

import numpy as np
import pandas as pd

# Litoloji one-hot sütun isimleri hariç temel öznitelik sütunları
FEATURE_COLUMNS = [
    # Topografik
    "elevation_m",
    "slope_deg",
    "aspect_sin",
    "aspect_cos",
    # Jeomekanik
    "cohesion_kpa",
    "friction_deg",
    "unit_weight",
    "pore_pressure_kpa",
    # Sismik
    "pga_g",
    "ky_g",
    "dn_cm",
    # Litoloji ID (sayısal, one-hot ile ayrıca eklenir)
    "litho_id",
]

LITHO_ONEHOT_COLS = [f"litho_{i}" for i in range(1, 28)]  # litho_1 ... litho_27

ALL_FEATURE_COLUMNS = FEATURE_COLUMNS + LITHO_ONEHOT_COLS


def build_feature_matrix(
    elevation_m: np.ndarray,
    slope_deg: np.ndarray,
    aspect_deg: np.ndarray,
    litho_array: np.ndarray,
    pga_array: np.ndarray,
    u_array: np.ndarray | None = None,
    z_ref: float = 5.0,
    include_onehot: bool = True,
) -> tuple[pd.DataFrame, list[str]]:
    """
    Piksel bazlı öznitelik matrisi oluşturur.

    Parametreler
    -----------
    elevation_m : 2D array — DEM yüksekliği (m)
    slope_deg   : 2D array — eğim (derece)
    aspect_deg  : 2D array — bakı (derece, 0=K, 90=D, 180=G, 270=B)
    litho_array : 2D int array — 1–27 litoloji sınıfı
    pga_array   : 2D float array — PGA (g cinsinden)
    u_array     : 2D float array — boşluk suyu basıncı (kPa), None → 0
    z_ref       : Hesap derinliği (m), varsayılan 5 m
    include_onehot: Litoloji one-hot sütunlarını ekle

    Döndürür
    --------
    df       : pd.DataFrame öznitelik matrisi (n_pixels, n_features)
    feat_cols: Kullanılan öznitelik sütun isimleri listesi
    """
    from ..geomechanics.lithology import get_litho_arrays
    from ..geomechanics.mohr_coulomb import StaticFS, CriticalAcceleration
    from ..geomechanics.newmark import NewmarkDisplacement

    shape = elevation_m.shape
    n = elevation_m.size

    flat = lambda arr: arr.ravel().astype(float)

    elevation_f = flat(elevation_m)
    slope_f     = flat(slope_deg)
    aspect_f    = flat(aspect_deg)
    litho_f     = flat(litho_array).astype(int)
    pga_f       = flat(pga_array)
    u_f         = flat(u_array) if u_array is not None else np.zeros(n)

    # Aspect döngüselliğini kır: sin ve cos bileşenlerine dönüştür
    aspect_sin = np.sin(np.deg2rad(aspect_f))
    aspect_cos = np.cos(np.deg2rad(aspect_f))

    # Litoloji parametreleri
    c_arr, phi_arr, gamma_arr = get_litho_arrays(litho_array)
    c_f     = flat(c_arr)
    phi_f   = flat(phi_arr)
    gamma_f = flat(gamma_arr)

    # Jeomekanik hesaplamalar
    fs_static = StaticFS(c_f, phi_f, gamma_f, z_ref, slope_f, u_f)
    ky        = CriticalAcceleration(fs_static, slope_f)
    dn_m      = NewmarkDisplacement(ky, pga_f)
    dn_cm     = dn_m * 100.0  # m → cm

    data = {
        "elevation_m":      elevation_f,
        "slope_deg":        slope_f,
        "aspect_sin":       aspect_sin,
        "aspect_cos":       aspect_cos,
        "cohesion_kpa":     c_f,
        "friction_deg":     phi_f,
        "unit_weight":      gamma_f,
        "pore_pressure_kpa": u_f,
        "pga_g":            pga_f,
        "ky_g":             ky,
        "dn_cm":            dn_cm,
        "litho_id":         litho_f,
    }

    df = pd.DataFrame(data)
    feat_cols = list(FEATURE_COLUMNS)

    if include_onehot:
        for lid in range(1, 28):
            df[f"litho_{lid}"] = (litho_f == lid).astype(np.float32)
        feat_cols = FEATURE_COLUMNS + LITHO_ONEHOT_COLS

    return df, feat_cols


def add_morphometric_features(
    df: pd.DataFrame,
    gdf: "gpd.GeoDataFrame",
    pixel_polygon_ids: np.ndarray,
) -> pd.DataFrame:
    """
    Morfolojik şekil metrikleri ekler (Compactness, Elongation).

    ⚠️ Bu metrikler yalnızca mekanizma öneri niteliğindedir;
    DEM + saha verisi olmadan kesin tür ataması yapılmaz.

    Parametreler
    -----------
    df               : Öznitelik matrisi DataFrame
    gdf              : Heyelan poligon GeoDataFrame
    pixel_polygon_ids: Her piksel için poligon ID'si

    Döndürür
    --------
    df : Compactness ve Elongation sütunları eklenmiş DataFrame
    """
    compactness_map = {}
    elongation_map  = {}

    for _, row in gdf.iterrows():
        geom = row.geometry
        if geom is None:
            continue
        area      = geom.area
        perimeter = geom.length

        comp = (4 * np.pi * area) / (perimeter ** 2) if perimeter > 0 else np.nan

        mbr = geom.minimum_rotated_rectangle
        if mbr is not None:
            coords = list(mbr.exterior.coords)
            sides  = [
                np.sqrt((coords[i+1][0]-coords[i][0])**2 + (coords[i+1][1]-coords[i][1])**2)
                for i in range(len(coords)-1)
            ]
            sides.sort()
            elong = sides[-1] / sides[-2] if sides[-2] > 0 else 1.0
        else:
            elong = np.nan

        pid = row.get("polygon_id", row.name)
        compactness_map[pid] = comp
        elongation_map[pid]  = elong

    df = df.copy()
    df["compactness"]  = pd.Series(pixel_polygon_ids).map(compactness_map).values
    df["elongation"]   = pd.Series(pixel_polygon_ids).map(elongation_map).values
    return df


def normalize_features(
    X_train: np.ndarray,
    X_test: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, dict]:
    """
    Train istatistiklerine göre standardize (z-score) normalize eder.
    Litoloji one-hot sütunlarına normalize uygulanmaz (zaten 0/1).

    Döndürür
    --------
    X_train_norm, X_test_norm, scaler_params (mean, std)
    """
    mean = X_train.mean(axis=0)
    std  = X_train.std(axis=0)
    std  = np.where(std < 1e-8, 1.0, std)

    X_train_norm = (X_train - mean) / std
    X_test_norm  = (X_test  - mean) / std

    return X_train_norm, X_test_norm, {"mean": mean, "std": std}


def prepare_synthetic_data(n_pixels: int = 50_000, seed: int = 42) -> dict:
    """
    Gerçek veri olmadan test için sentetik veri seti üretir.

    n_pixels heyelan envanterindeki toplam piksel sayısını simüle eder.
    """
    import numpy as np
    rng = np.random.RandomState(seed)

    # Topografik öznitelikler
    elevation_m = rng.uniform(200, 2500, n_pixels).astype(np.float32)
    slope_deg   = rng.exponential(scale=15, size=n_pixels).clip(0, 70).astype(np.float32)
    aspect_deg  = rng.uniform(0, 360, n_pixels).astype(np.float32)

    # Litoloji (27 sınıf, gerçek dağılıma yakın)
    litho_probs = np.array([0.15, 0.10, 0.08, 0.07, 0.06,  # 1-5
                             0.05, 0.05, 0.05, 0.04, 0.04,  # 6-10
                             0.04, 0.03, 0.03, 0.03, 0.03,  # 11-15
                             0.02, 0.02, 0.02, 0.02, 0.02,  # 16-20
                             0.02, 0.02, 0.01, 0.01, 0.01,  # 21-25
                             0.01, 0.01])                    # 26-27
    litho_probs /= litho_probs.sum()
    litho_array = rng.choice(np.arange(1, 28), size=n_pixels, p=litho_probs).astype(int)

    # PGA (6 Şubat 2023 Kahramanmaraş depremi — 0.1g–1.2g arası)
    pga_array = rng.lognormal(mean=-0.8, sigma=0.5, size=n_pixels).clip(0.05, 1.5).astype(np.float32)

    # Boşluk suyu basıncı (kuru koşul başlangıcı)
    u_array = np.zeros(n_pixels, dtype=np.float32)

    # Poligon ID'leri (2596 poligon simülasyonu)
    polygon_ids = rng.randint(0, 2596, size=n_pixels).astype(int)

    return {
        "elevation_m": elevation_m,
        "slope_deg":   slope_deg,
        "aspect_deg":  aspect_deg,
        "litho_array": litho_array,
        "pga_array":   pga_array,
        "u_array":     u_array,
        "polygon_ids": polygon_ids,
        "n_pixels":    n_pixels,
    }

