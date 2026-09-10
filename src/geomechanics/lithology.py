"""
27 litoloji sınıfı için jeomekanik parametre tablosu.

Değerler: Literatür başlangıç tahminleri (MTA 2023 Kahramanmaraş jeoloji
raporları ile FEM kalibrasyon aşamasında güncellenmelidir).

Her litoloji sınıfı:
  c_prime   : Efektif kohezyon (kPa)
  phi_prime : Efektif içsel sürtünme açısı (derece)
  gamma     : Birim hacim ağırlık (kN/m³)
  E_mpa     : Elastisite modülü (MPa) – FEM için
  nu        : Poisson oranı – FEM için
  description: Kısa açıklama
"""

LITHOLOGY_TABLE: dict[int, dict] = {
    1:  {"c_prime": 0.0,  "phi_prime": 28.0, "gamma": 17.0, "E_mpa": 5.0,   "nu": 0.35, "description": "Alüvyon (kuru)"},
    2:  {"c_prime": 5.0,  "phi_prime": 25.0, "gamma": 18.0, "E_mpa": 8.0,   "nu": 0.33, "description": "Alüvyon (ıslak)"},
    3:  {"c_prime": 10.0, "phi_prime": 30.0, "gamma": 18.5, "E_mpa": 15.0,  "nu": 0.30, "description": "Kumtaşı (zayıf)"},
    4:  {"c_prime": 20.0, "phi_prime": 35.0, "gamma": 20.0, "E_mpa": 50.0,  "nu": 0.25, "description": "Kumtaşı (sağlam)"},
    5:  {"c_prime": 30.0, "phi_prime": 38.0, "gamma": 22.0, "E_mpa": 100.0, "nu": 0.22, "description": "Kireçtaşı (orta)"},
    6:  {"c_prime": 50.0, "phi_prime": 40.0, "gamma": 25.0, "E_mpa": 300.0, "nu": 0.20, "description": "Kireçtaşı (masif)"},
    7:  {"c_prime": 15.0, "phi_prime": 28.0, "gamma": 19.0, "E_mpa": 25.0,  "nu": 0.30, "description": "Marn"},
    8:  {"c_prime": 8.0,  "phi_prime": 22.0, "gamma": 18.0, "E_mpa": 10.0,  "nu": 0.35, "description": "Kil taşı"},
    9:  {"c_prime": 5.0,  "phi_prime": 18.0, "gamma": 17.5, "E_mpa": 6.0,   "nu": 0.38, "description": "Kil (yüksek plastisiteli)"},
    10: {"c_prime": 12.0, "phi_prime": 24.0, "gamma": 18.5, "E_mpa": 18.0,  "nu": 0.32, "description": "Silttaşı"},
    11: {"c_prime": 0.0,  "phi_prime": 32.0, "gamma": 16.5, "E_mpa": 3.0,   "nu": 0.35, "description": "Yamaç molozu / kolüvyon"},
    12: {"c_prime": 0.0,  "phi_prime": 35.0, "gamma": 17.5, "E_mpa": 4.0,   "nu": 0.33, "description": "Talus (kireçtaşı döküntüsü)"},
    13: {"c_prime": 25.0, "phi_prime": 36.0, "gamma": 21.0, "E_mpa": 80.0,  "nu": 0.23, "description": "Dolomit"},
    14: {"c_prime": 40.0, "phi_prime": 42.0, "gamma": 26.0, "E_mpa": 500.0, "nu": 0.18, "description": "Bazalt"},
    15: {"c_prime": 35.0, "phi_prime": 40.0, "gamma": 25.0, "E_mpa": 400.0, "nu": 0.20, "description": "Andezit"},
    16: {"c_prime": 20.0, "phi_prime": 32.0, "gamma": 22.0, "E_mpa": 120.0, "nu": 0.22, "description": "Granit (alterasyonlu)"},
    17: {"c_prime": 15.0, "phi_prime": 30.0, "gamma": 20.5, "E_mpa": 60.0,  "nu": 0.25, "description": "Şist"},
    18: {"c_prime": 10.0, "phi_prime": 25.0, "gamma": 19.5, "E_mpa": 30.0,  "nu": 0.28, "description": "Fillit"},
    19: {"c_prime": 5.0,  "phi_prime": 20.0, "gamma": 18.0, "E_mpa": 12.0,  "nu": 0.35, "description": "Serpantinit (ayrışmış)"},
    20: {"c_prime": 45.0, "phi_prime": 38.0, "gamma": 24.0, "E_mpa": 250.0, "nu": 0.20, "description": "Metabazalt / Ofiyolit"},
    21: {"c_prime": 18.0, "phi_prime": 28.0, "gamma": 20.0, "E_mpa": 40.0,  "nu": 0.27, "description": "Tüf (volkanik)"},
    22: {"c_prime": 22.0, "phi_prime": 33.0, "gamma": 21.5, "E_mpa": 70.0,  "nu": 0.24, "description": "Riyolit"},
    23: {"c_prime": 3.0,  "phi_prime": 16.0, "gamma": 17.0, "E_mpa": 4.0,   "nu": 0.40, "description": "Karstik çözünme dolgusu"},
    24: {"c_prime": 0.0,  "phi_prime": 30.0, "gamma": 20.0, "E_mpa": 10.0,  "nu": 0.30, "description": "Kırıklı-çatlaklı kaya kütlesi"},
    25: {"c_prime": 28.0, "phi_prime": 36.0, "gamma": 22.5, "E_mpa": 150.0, "nu": 0.22, "description": "Konglomera (sağlam)"},
    26: {"c_prime": 8.0,  "phi_prime": 22.0, "gamma": 18.5, "E_mpa": 15.0,  "nu": 0.33, "description": "Konglomera (zayıf çimentolu)"},
    27: {"c_prime": 12.0, "phi_prime": 26.0, "gamma": 19.0, "E_mpa": 20.0,  "nu": 0.30, "description": "Karışık litoloji / Formasyonsuz"},
}


def get_litho_params(litho_id: int) -> dict:
    """
    Litoloji ID'sine göre jeomekanik parametreleri döndürür.

    Parametreler
    -----------
    litho_id : 1–27 arası litoloji sınıf numarası

    Döndürür
    --------
    dict : c_prime, phi_prime, gamma, E_mpa, nu, description
    """
    if litho_id not in LITHOLOGY_TABLE:
        raise KeyError(f"Geçersiz litoloji ID: {litho_id}. 1–27 arası bir değer girin.")
    return LITHOLOGY_TABLE[litho_id]


def get_litho_arrays(litho_array):
    """
    2D litoloji raster array'den c', φ', γ dizileri oluşturur.

    Parametreler
    -----------
    litho_array : np.ndarray (int) — 1–27 arası litoloji ID matrisi

    Döndürür
    --------
    c_arr, phi_arr, gamma_arr : np.ndarray (float)
    """
    import numpy as np

    c_arr    = np.zeros_like(litho_array, dtype=float)
    phi_arr  = np.zeros_like(litho_array, dtype=float)
    gamma_arr = np.zeros_like(litho_array, dtype=float)

    for lid, params in LITHOLOGY_TABLE.items():
        mask = litho_array == lid
        c_arr[mask]     = params["c_prime"]
        phi_arr[mask]   = params["phi_prime"]
        gamma_arr[mask] = params["gamma"]

    return c_arr, phi_arr, gamma_arr
