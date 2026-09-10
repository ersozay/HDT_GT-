"""
Proje Merkezi Konfigürasyonu.

Fizik Tabanlı Dijital Twin Hibrit Heyelan Modellemesi
Zeynep Ersöz · Trakya Üniversitesi · Haziran 2026
"""
from pathlib import Path

# Proje kök dizini
PROJECT_ROOT = Path(__file__).parent

# Veri dizinleri
DATA_RAW       = PROJECT_ROOT / "data" / "raw"
DATA_PROCESSED = PROJECT_ROOT / "data" / "processed"

# Çıktı dizinleri
OUTPUTS_DIR    = PROJECT_ROOT / "outputs"
FIGURES_DIR    = OUTPUTS_DIR / "figures"
MODELS_DIR     = OUTPUTS_DIR / "models"

# Veri dosyaları (gerçek dosya yolları buraya eklenecek)
INVENTORY_KML  = DATA_RAW / "TJES-2024-00171_Landslides.kml"
PGA_RASTER     = DATA_RAW / "pga_kahramanmaras_20230206.tif"
DEM_RASTER     = DATA_RAW / "EUDEM_kahramanmaras.tif"
# NOT: Litoloji katmanı resmi MTA 1:100.000 haritası DEĞİL; Macrostrat API
# (CC-BY 4.0) tabanlı bir jeolojik birim PROXY'sidir. Tezde bu şekilde atıf verilmeli.
# Kaynak: Macrostrat.org geologic map API (Peters et al., 2018).
LITHO_RASTER   = DATA_RAW / "lithology_macrostrat_proxy.tif"

# Pilot nokta ID'leri
PILOT_IDS = [1327, 1353]

# Jeomekanik sabitler
Z_REF_M     = 5.0    # Referans hesap derinliği (m)
DN_THRESHOLD_M = 0.10  # Heyelan etiketi için Dn eşiği (m = 10 cm)

# PGA senaryoları
PGA_SCENARIOS = [0.2, 0.4, 0.6, 0.8]  # g cinsinden

# Eğitim hiper-parametreleri (varsayılan)
TRAIN_CONFIG = {
    "epochs":        100,
    "batch_size":    512,
    "lr":            1e-3,
    "weight_decay":  1e-4,
    "patience":      15,
    "lambda_phys":   1.0,
    "max_lambda":    5.0,
    "test_ratio":    0.2,
    "n_splits":      5,
    "random_state":  42,
}

# Fizik ceza eşikleri
PHYS_SLOPE_THRESHOLD = 10.0   # derece
PHYS_PGA_THRESHOLD   = 0.10   # g

# Focal Loss parametreleri
# alpha = heyelan (pozitif) sınıf ağırlığı. 0.25 azınlık heyelan sınıfını
# bastırıp recall'ı düşürüyordu; %28.6 pozitif oranı + gamma odaklaması ile
# dengeli 0.5 tercih edildi (heyelan kaçırmayı azaltmak için).
FOCAL_ALPHA = 0.5
FOCAL_GAMMA = 2.0

# Karar eşiği doğrulama setinde F1-maks ile seçilir (aşağıdaki aralıkta taranır)
THRESHOLD_SEARCH = (0.10, 0.90, 0.01)  # (min, max, step)

# Litoloji sayısı
N_LITHOLOGY_CLASSES = 27


def ensure_dirs():
    """Gerekli dizinleri oluşturur."""
    for d in [DATA_RAW, DATA_PROCESSED, FIGURES_DIR, MODELS_DIR]:
        d.mkdir(parents=True, exist_ok=True)


if __name__ == "__main__":
    ensure_dirs()
    print("Dizinler oluşturuldu:")
    for d in [DATA_RAW, DATA_PROCESSED, FIGURES_DIR, MODELS_DIR]:
        print(f"  {d}")
