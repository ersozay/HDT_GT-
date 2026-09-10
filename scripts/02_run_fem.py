"""
Script 2: FEM Lokal Analiz (Pilot Noktalar ID 1327 ve ID 1353)

Adımlar:
  1. Her pilot nokta için jeomekanik parametreleri al
  2. DEM'den şev profili çıkar ve mesh oluştur
  3. Geostatic step çalıştır
  4. 6 Şubat 2023 sismik kayıt uygula
  5. Von Mises, Dn, Ky çıktılarını kaydet
  6. PGA-FS eğrisini çiz

Kullanım:
  python scripts/02_run_fem.py [--slope-id 1327]
"""
import sys
import argparse
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from config import (
    DATA_RAW, DATA_PROCESSED, FIGURES_DIR, PILOT_IDS, ensure_dirs
)
from src.geomechanics.lithology import LITHOLOGY_TABLE
from src.fem.sfepy_solver import SlopeFEMSolver
from src.visualization.pga_fs_curve import plot_pga_fs_curve, plot_pga_fs_multi


# Pilot nokta jeomekanik parametreleri (litoloji ID'sine göre atanacak)
PILOT_PARAMS = {
    1327: {
        "litho_id":   11,    # Yamaç molozu / kolüvyon
        "theta_deg":  28.0,  # DEM'den ölçülen eğim
        "z_m":        5.0,   # Hesap derinliği
        "actual_pga": 0.58,  # 6 Şubat 2023 AFAD ShakeMap
    },
    1353: {
        "litho_id":   7,     # Marn
        "theta_deg":  22.0,
        "z_m":        8.0,
        "actual_pga": 0.45,
    },
}


def synthetic_seismic_record(pga_g: float, duration_s: float = 60.0, dt: float = 0.01) -> np.ndarray:
    """
    Basit sentetik ivme-zaman kaydı (gerçek kayıt yokken).
    Gerçek proje: AFAD'dan Pazarcık ve Elbistan kayıtları kullanılmalı.
    """
    t = np.arange(0, duration_s, dt)
    # Sinusoidal + Gaussian envelope
    envelope = np.exp(-((t - 15) ** 2) / (2 * 5**2))
    freq     = 2.5  # Hz (baskın frekans yaklaşımı)
    record   = pga_g * envelope * np.sin(2 * np.pi * freq * t)
    return record


def run_fem_for_slope(slope_id: int, params: dict):
    """Tek bir pilot nokta için FEM analizi."""
    litho = LITHOLOGY_TABLE[params["litho_id"]]

    print(f"\n{'='*50}")
    print(f"ID {slope_id} FEM Analizi")
    print(f"  Litoloji   : {litho['description']}")
    print(f"  c' = {litho['c_prime']} kPa | φ' = {litho['phi_prime']}° | γ = {litho['gamma']} kN/m³")
    print(f"  θ = {params['theta_deg']}° | z = {params['z_m']} m")
    print(f"  Gerçek PGA = {params['actual_pga']} g")

    # FEM çözücü
    solver = SlopeFEMSolver(
        slope_id=slope_id,
        litho_params=litho,
        output_dir=DATA_PROCESSED / "fem",
    )

    # Sismik kayıt
    seismic_ts = synthetic_seismic_record(pga_g=params["actual_pga"])

    # Geostatic
    geo_result = solver.run_geostatic()
    print(f"  Geostatic σ_v = {geo_result['sigma_v_kpa']:.1f} kPa")

    # Sismik analiz
    fem_result = solver.run_seismic(seismic_ts)
    print(f"  FEM Ky     = {fem_result.ky_fem:.3f} g")
    print(f"  FEM Dn_max = {fem_result.dn_max * 100:.1f} cm")
    print(f"  Label      = {'HEYELAN (1)' if fem_result.label else 'KARARLI (0)'}")

    # 2D Kontur ve Yer Değiştirme Profili Grafiklerini Üret
    print(f"  ID {slope_id} için FEM konturları ve yer değiştirme profili çiziliyor...")
    from src.visualization.fem_contours import plot_fem_contours_2d, plot_displacement_profile
    
    n_pts = len(fem_result.von_mises)
    x_coords = np.linspace(0, 200, n_pts)
    # Eğim açısına göre yükseklik profili oluştur (y_coords)
    theta_rad = np.deg2rad(params["theta_deg"])
    y_coords = np.zeros(n_pts)
    for i, x in enumerate(x_coords):
        if x < 50:
            y_coords[i] = 50.0
        elif x > 150:
            y_coords[i] = 50.0 - 100.0 * np.tan(theta_rad)
        else:
            y_coords[i] = 50.0 - (x - 50.0) * np.tan(theta_rad)
            
    plot_fem_contours_2d(
        x_coords=x_coords,
        y_coords=y_coords,
        von_mises=fem_result.von_mises,
        eq_plastic_strain=fem_result.eq_plastic_strain,
        displacement=fem_result.displacement_magnitude,
        slope_id=slope_id,
        save_dir=FIGURES_DIR,
    )
    
    plot_displacement_profile(
        x_coords=x_coords,
        displacement_y=fem_result.displacement_y,
        slope_id=slope_id,
        dn_threshold=0.10,
        save_path=FIGURES_DIR / f"displacement_profile_ID{slope_id}.png",
    )

    return fem_result



def run(slope_ids: list[int] | None = None):
    ensure_dirs()

    if slope_ids is None:
        slope_ids = PILOT_IDS

    fem_results = {}
    for sid in slope_ids:
        if sid not in PILOT_PARAMS:
            print(f"ID {sid} için parametre tanımı bulunamadı, atlanıyor.")
            continue
        result = run_fem_for_slope(sid, PILOT_PARAMS[sid])
        fem_results[sid] = result

    # PGA-FS eğrisi çiz
    print("\nPGA–FS eğrileri çiziliyor...")

    slope_params_list = []
    for sid in slope_ids:
        if sid not in PILOT_PARAMS:
            continue
        p = PILOT_PARAMS[sid]
        litho = LITHOLOGY_TABLE[p["litho_id"]]
        slope_params_list.append({
            "slope_id": sid,
            "c":        litho["c_prime"],
            "phi_deg":  litho["phi_prime"],
            "gamma":    litho["gamma"],
            "z":        p["z_m"],
            "theta_deg": p["theta_deg"],
        })

        # Bireysel eğri
        fig = plot_pga_fs_curve(
            slope_id=sid,
            c=litho["c_prime"],
            phi_deg=litho["phi_prime"],
            gamma=litho["gamma"],
            z=p["z_m"],
            theta_deg=p["theta_deg"],
            actual_pga=p["actual_pga"],
            save_path=FIGURES_DIR / f"pga_fs_curve_ID{sid}.png",
        )

    # Karşılaştırmalı eğri
    if len(slope_params_list) > 1:
        plot_pga_fs_multi(
            slope_params_list,
            save_path=FIGURES_DIR / "pga_fs_comparison.png",
        )

    print(f"\nGrafikler kaydedildi: {FIGURES_DIR}")
    print("FEM analizi tamamlandı.")

    return fem_results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="FEM lokal analiz")
    parser.add_argument("--slope-id", type=int, nargs="+",
                        help="Analiz edilecek slope ID'leri (varsayılan: tümü)")
    args = parser.parse_args()
    run(slope_ids=args.slope_id)
