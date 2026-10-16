"""
Script 4: Senaryo Risk Haritaları — "Dijital İkiz" ne-olursa analizi

Gerçek raster yığınına (Copernicus DEM + Macrostrat litoloji) bölge genelinde
tekdüze PGA senaryoları (0.2 / 0.4 / 0.6 / 0.8 g) uygulanır; her senaryoda Dn
Jibson (2007) ile yeniden hesaplanır ve PINN v2 olasılık haritası üretilir.
Ayrıca 0.4 g senaryosunda saf fizik (Newmark Dn ≥ 10 cm) ile PINN karşılaştırılır.

v1'deki bu betik varsayılan olarak SENTETİK arazi kullanıyordu; v2 yalnız
gerçek katmanlarla çalışır.

Kullanım:
  python3 scripts/04_generate_maps.py [--res 0.01]
"""
import sys
import argparse
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from config import FIGURES_DIR, MODELS_DIR, PGA_SCENARIOS, Z_REF_M, DN_THRESHOLD_M, ensure_dirs
from src.data_engineering.regional_grid import load_regional_stack, stack_features
from src.pinn.io import load_checkpoint


def run(res: float = 0.01):
    ensure_dirs()
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    lm = load_checkpoint(MODELS_DIR / "best_model.pt", MODELS_DIR / "scaler.pkl",
                         MODELS_DIR / "threshold.json")
    stack = load_regional_stack(res)
    h, w = stack.shape
    lon0, lat0, lon1, lat1 = stack.bounds
    ext = [lon0, lon1, lat0, lat1]

    maps, dn_maps = {}, {}
    for pg in PGA_SCENARIOS:
        df = stack_features(stack, Z_REF_M, pga_override=pg)
        p = lm.predict(df[lm.feat_cols].to_numpy(np.float32)).reshape(h, w)
        maps[pg] = np.where(stack.valid, p, np.nan)
        dn_maps[pg] = np.where(stack.valid, df["dn_cm"].to_numpy().reshape(h, w), np.nan)
        print(f"PGA {pg:.1f} g → P ≥ eşik alan oranı "
              f"{np.nanmean(maps[pg] >= lm.threshold) * 100:.1f}%")

    fig, axes = plt.subplots(2, 2, figsize=(13, 9.5), sharex=True, sharey=True)
    for ax, pg in zip(axes.ravel(), PGA_SCENARIOS):
        im = ax.imshow(maps[pg], extent=ext, origin="upper", cmap="RdYlGn_r", vmin=0, vmax=1,
                       interpolation="nearest")
        ax.set_title(f"Senaryo PGA = {pg:.1f} g (bölge geneli)", loc="left", fontsize=11)
    fig.colorbar(im, ax=axes, shrink=0.7, label="P(heyelan) — PINN v2")
    fig.suptitle("Dijital ikiz senaryoları — gerçek arazi + litoloji, tekdüze PGA", x=0.01, ha="left")
    fig.savefig(FIGURES_DIR / "dynamic_risk_maps.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    pg = 0.4 if 0.4 in maps else PGA_SCENARIOS[0]
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.6), sharey=True)
    axes[0].imshow(np.where(np.isfinite(dn_maps[pg]), dn_maps[pg] >= DN_THRESHOLD_M * 100, np.nan),
                   extent=ext, origin="upper", cmap="Greys", vmin=0, vmax=1, interpolation="nearest")
    axes[0].set_title(f"Newmark (Jibson 2007): Dn ≥ 10 cm — PGA {pg} g", loc="left")
    im = axes[1].imshow(maps[pg], extent=ext, origin="upper", cmap="RdYlGn_r", vmin=0, vmax=1,
                        interpolation="nearest")
    axes[1].set_title(f"PINN v2 P(heyelan) — PGA {pg} g", loc="left")
    fig.colorbar(im, ax=axes[1], shrink=0.8)
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / f"risk_comparison_{pg}g.png", dpi=150)
    plt.close(fig)
    print(f"Figürler: {FIGURES_DIR / 'dynamic_risk_maps.png'}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Senaryo risk haritaları (gerçek veri)")
    ap.add_argument("--res", type=float, default=0.01)
    run(ap.parse_args().res)
