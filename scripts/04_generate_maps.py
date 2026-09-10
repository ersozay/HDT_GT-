"""
Script 4: Dinamik Risk Haritaları Üretimi (11 İl Upscaling)

Adımlar:
  1. Eğitilmiş PINN modelini yükle
  2. 11 il için piksel bazlı öznitelik matrisi oluştur
  3. Farklı PGA senaryolarında (0.2g, 0.4g, 0.6g, 0.8g) risk haritası üret
  4. Analitik Ky ve PINN olasılık haritalarını karşılaştır
  5. Grafikleri kaydet

Kullanım:
  python scripts/04_generate_maps.py [--synthetic]
"""
import sys
import argparse
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from config import (
    DATA_PROCESSED, DATA_RAW, FIGURES_DIR, MODELS_DIR,
    PGA_SCENARIOS, ensure_dirs,
)
from src.visualization.risk_maps import (
    plot_scenario_grid, plot_risk_comparison
)


def generate_synthetic_regional_map(
    shape: tuple[int, int] = (500, 600),
    seed: int = 42,
) -> dict:
    """
    11 il bölgesi için sentetik alan verileri üretir.
    Gerçek proje: rasterio ile GeoTIFF yüklenecek.
    """
    rng = np.random.RandomState(seed)
    h, w = shape

    # DEM ve slope (dağlık Kahramanmaraş topoğrafyası simülasyonu)
    x = np.linspace(0, 2 * np.pi, w)
    y = np.linspace(0, 2 * np.pi, h)
    X, Y = np.meshgrid(x, y)

    elevation_m = (1000 + 500 * np.sin(X) * np.cos(Y / 2) +
                   200 * rng.randn(h, w)).clip(100, 2800).astype(np.float32)

    # Eğim: gradyan hesabı
    dy, dx = np.gradient(elevation_m, 30.0, 30.0)  # 30m çözünürlük
    slope_deg = np.degrees(np.arctan(np.sqrt(dx**2 + dy**2))).astype(np.float32)

    # Litoloji (basit desen)
    litho_array = np.ones((h, w), dtype=int)
    litho_array[:h//3, :] = 5    # Kireçtaşı kuzey
    litho_array[h//3:2*h//3, :] = 11  # Kolüvyon orta
    litho_array[2*h//3:, :] = 7  # Marn güney

    # PGA merkezde yüksek (Kahramanmaraş episenter)
    cy, cx = h * 0.4, w * 0.5
    dist   = np.sqrt(((np.arange(h)[:, None] - cy)  / (h * 0.3))**2 +
                     ((np.arange(w)[None, :] - cx)   / (w * 0.3))**2)
    pga_array = (0.7 * np.exp(-dist) + 0.1 + 0.05 * rng.rand(h, w)).astype(np.float32)

    return {
        "elevation_m": elevation_m,
        "slope_deg":   slope_deg,
        "litho_array": litho_array,
        "pga_array":   pga_array,
        "shape":       shape,
    }


def compute_ky_map(elevation_m, slope_deg, litho_array, z_ref: float = 5.0) -> np.ndarray:
    """Piksel bazlı Ky haritasını hesaplar."""
    from src.geomechanics.lithology import get_litho_arrays
    from src.geomechanics.mohr_coulomb import StaticFS, CriticalAcceleration

    c_arr, phi_arr, gamma_arr = get_litho_arrays(litho_array)
    u_arr = np.zeros_like(slope_deg)

    fs = StaticFS(c_arr, phi_arr, gamma_arr, z_ref, slope_deg, u_arr)
    ky = CriticalAcceleration(fs, slope_deg)
    return ky.astype(np.float32)


def run(synthetic: bool = True):
    ensure_dirs()

    print("Bölgesel veri hazırlanıyor (11 il)...")

    if synthetic:
        data = generate_synthetic_regional_map()
        print(f"  Alan: {data['shape']} piksel (sentetik)")
    else:
        # Gerçek veri: DATA_RAW klasöründen raster yükle
        try:
            import rasterio
            with rasterio.open(DATA_RAW / "EUDEM_kahramanmaras.tif") as src:
                elev = src.read(1).astype(np.float32)
            data = {"elevation_m": elev, "shape": elev.shape}
            print(f"  Alan: {elev.shape} piksel (gerçek)")
        except Exception as e:
            print(f"Gerçek veri yüklenemedi ({e}), sentetik kullanılıyor...")
            data = generate_synthetic_regional_map()

    elev  = data["elevation_m"]
    slope = data["slope_deg"] if "slope_deg" in data else np.zeros_like(elev)
    litho = data["litho_array"] if "litho_array" in data else np.ones_like(elev, dtype=int)

    # Ky haritası
    print("Ky (kritik ivme) haritası hesaplanıyor...")
    ky_map = compute_ky_map(elev, slope, litho)
    print(f"  Ky istatistikleri: min={ky_map.min():.3f}g | "
          f"mean={ky_map.mean():.3f}g | max={ky_map.max():.3f}g")

    # Risk haritaları
    print("\nDinamik risk haritaları oluşturuluyor...")
    fig = plot_scenario_grid(
        ky_map,
        scenarios=PGA_SCENARIOS,
        save_path=FIGURES_DIR / "dynamic_risk_maps.png",
        title="Farklı PGA Senaryolarında Heyelan Risk Haritası — Kahramanmaraş 11 İl",
    )
    print(f"  Risk haritaları kaydedildi: {FIGURES_DIR / 'dynamic_risk_maps.png'}")

    # PINN ile Risk Haritası Tahmini ve Karşılaştırma
    print("\nPINN modeli ile bölgesel risk tahmini ve karşılaştırma yapılıyor...")
    import torch
    import pickle
    from src.pinn.model import LandslidePINN
    from src.data_engineering.feature_matrix import build_feature_matrix

    scaler_path = MODELS_DIR / "scaler.pkl"
    model_path  = MODELS_DIR / "best_model.pt"

    if scaler_path.exists() and model_path.exists():
        try:
            # Scaler yükle
            scaler = pickle.loads(scaler_path.read_bytes())
            
            # Model yükle
            ckpt = torch.load(model_path, map_location="cpu")
            in_features = len(scaler["mean"])
            model = LandslidePINN(in_features=in_features)
            model.load_state_dict(ckpt["model_state"])
            model.eval()

            # 0.4g deprem senaryosu için öznitelik matrisi
            h, w = elev.shape
            pga_scenario = 0.4
            pga_scenario_arr = np.full((h, w), pga_scenario, dtype=np.float32)
            
            # Aspect'i DEM gradyanından çıkaralım
            dy, dx = np.gradient(elev, 30.0, 30.0)
            aspect_deg = np.degrees(np.arctan2(dy, -dx)) % 360

            df_reg, feat_cols = build_feature_matrix(
                elevation_m = elev,
                slope_deg   = slope,
                aspect_deg  = aspect_deg,
                litho_array = litho,
                pga_array   = pga_scenario_arr,
                z_ref       = 5.0,
            )

            # Standardize et
            X_reg = df_reg[feat_cols].values.astype(np.float32)
            X_reg_n = (X_reg - scaler["mean"]) / scaler["std"]

            # Tahmin üret
            X_reg_t = torch.tensor(X_reg_n, dtype=torch.float32)
            with torch.no_grad():
                pinn_prob_flat = model.predict_proba(X_reg_t).numpy().squeeze()
            
            pinn_prob_map = pinn_prob_flat.reshape((h, w))

            # Analitik vs PINN karşılaştırma haritası
            plot_risk_comparison(
                ky_map_analytical=ky_map,
                pinn_prob_map=pinn_prob_map,
                pga_scenario=pga_scenario,
                save_path=FIGURES_DIR / "risk_comparison_0.4g.png"
            )
            print(f"  Analitik vs PINN karşılaştırma haritası kaydedildi: {FIGURES_DIR / 'risk_comparison_0.4g.png'}")
        except Exception as e:
            print(f"  PINN karşılaştırma haritası çizilirken hata oluştu: {e}")
    else:
        print("  Eğitilmiş model veya scaler bulunamadı. Lütfen önce 03_train_pinn.py çalıştırın.")

    print("\nRisk haritası üretimi tamamlandı.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Dinamik risk haritası üretimi")
    parser.add_argument("--synthetic", action="store_true", default=True)
    parser.add_argument("--real", action="store_true")
    args = parser.parse_args()

    run(synthetic=not args.real)
