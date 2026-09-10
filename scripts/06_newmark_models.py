"""
Script 6: Newmark Kalıcı Yer Değiştirme Modelleri — Pilot Yamaçlar

Danışman (Aytuk) isteği doğrultusunda, iki pilot heyelan için (ID 1327 ve
ID 1353) Newmark rijit-blok kalıcı yer değiştirmesini iki farklı bağımsız
değişkene karşı çizer:

  Sol panel : Dₙ vs PGA (amax)  — Jibson (2007) ve Ambraseys & Menu (1988)
  Sağ panel : Dₙ vs Arias Yoğunluğu Iₐ — Jibson (2007) Denklem 9

Her panelde 10 cm "Serviceability Limit" (Jibson 2007 kritik eşik) yatay
çizgisi vurgulanır. Pilotların kritik (akma) ivmesi aᵧ, GERÇEK envanter
merkez koordinatlarındaki dokümante edilmiş yamaç geometrisi ve litoloji
sınıfından Mohr–Coulomb + Newmark ile hesaplanır (uydurma yok).

BİLİMSEL DÜRÜSTLÜK NOTLARI (figüre de yazılır):
  · aᵧ, dokümante tez pilot geometrisiyle (θ, litoloji) hesaplanır; PGA ise
    USGS ShakeMap zarfından pilot merkezinde örneklenir.
  · Iₐ ekseni bir SENARYO taramasıdır — pilot için gerçek kayıt zaman-serisi
    integrali (Iₐ = π/2g ∫a(t)²dt) mevcut değildir.
  · Ampirik regresyonların tahmin bandı vardır; eğriler nokta-kestirimlerdir.

Kullanım:
  python3 scripts/06_newmark_models.py
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

import config
from src.geomechanics.mohr_coulomb import StaticFS, CriticalAcceleration
from src.geomechanics.newmark import NewmarkDisplacement, NewmarkDisplacementIa
from src.geomechanics.lithology import LITHOLOGY_TABLE

# Dokümante tez pilot tanımları (MEMORY / tez metni ile tutarlı)
#   ID 1327 : yamaç molozu / kolüvyon (litho 11), θ = 28°
#   ID 1353 : marn (litho 7),                      θ = 22°
PILOTS = [
    {"id": "1327", "litho": 11, "theta": 28.0, "color": "#c026d3"},  # magenta
    {"id": "1353", "litho": 7,  "theta": 22.0, "color": "#0284c7"},  # sky
]

DN_LIMIT_CM = config.DN_THRESHOLD_M * 100.0  # 10 cm


def sample_raster(path, lon, lat):
    import rasterio
    with rasterio.open(path) as ds:
        return float(list(ds.sample([(lon, lat)]))[0][0])


def pilot_centroid(gdf, pid):
    c = gdf[gdf["Name"] == pid].geometry.iloc[0].centroid
    return float(c.x), float(c.y)


def compute_pilots():
    """Her pilot için aᵧ (g), gözlemlenen PGA ve etiketleri hazırlar."""
    import geopandas as gpd
    gdf = gpd.read_file(config.INVENTORY_KML)

    out = []
    for p in PILOTS:
        lith = LITHOLOGY_TABLE[p["litho"]]
        fs = StaticFS(lith["c_prime"], lith["phi_prime"], lith["gamma"],
                      config.Z_REF_M, p["theta"])
        ky = float(CriticalAcceleration(fs, p["theta"]))
        try:
            lon, lat = pilot_centroid(gdf, p["id"])
            pga_obs = sample_raster(config.PGA_RASTER, lon, lat)
        except Exception:
            lon = lat = pga_obs = float("nan")
        out.append({
            **p,
            "desc": lith["description"],
            "fs_static": float(fs),
            "ky": ky,
            "lon": lon, "lat": lat,
            "pga_obs": float(pga_obs),
        })
    return out


def make_figure(pilots):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, (axL, axR) = plt.subplots(1, 2, figsize=(15, 6.2))
    fig.suptitle(
        "Newmark Kalıcı Yer Değiştirmesi — Pilot Yamaçlar (ID 1327 · ID 1353)",
        fontsize=15, fontweight="bold", y=0.99,
    )

    # ---------------- Sol panel: Dn vs PGA ----------------
    pga = np.linspace(0.05, 1.40, 200)
    for p in pilots:
        ky = p["ky"]
        dn_j = NewmarkDisplacement(ky, pga, method="jibson2007") * 100.0
        dn_a = NewmarkDisplacement(ky, pga, method="ambraseys1988") * 100.0
        axL.plot(pga, dn_j, color=p["color"], lw=2.4,
                 label=f"ID {p['id']} · Jibson 2007  (aᵧ={ky:.3f}g)")
        axL.plot(pga, dn_a, color=p["color"], lw=2.0, ls="--",
                 label=f"ID {p['id']} · Ambraseys 1988")
        # gözlemlenen PGA işareti
        if np.isfinite(p["pga_obs"]):
            axL.axvline(p["pga_obs"], color=p["color"], lw=1.0, ls=":", alpha=0.6)

    axL.axhline(DN_LIMIT_CM, color="#dc2626", lw=1.8, ls="-.", alpha=0.9)
    axL.text(1.38, DN_LIMIT_CM * 1.15, "10 cm — Serviceability Limit (Jibson 2007)",
             ha="right", va="bottom", fontsize=9.5, color="#dc2626", fontweight="bold")
    axL.set_xlabel("Tepe Yer İvmesi  PGA  (g)", fontsize=11)
    axL.set_ylabel("Kalıcı Yer Değiştirme  Dₙ  (cm)", fontsize=11)
    axL.set_title("Dₙ – PGA (oran-tabanlı bağıntılar)", fontsize=12)
    axL.set_yscale("log")
    axL.set_ylim(0.1, 1100)
    axL.grid(True, which="both", alpha=0.25)
    axL.legend(fontsize=8.4, loc="lower right", framealpha=0.92)

    # ---------------- Sağ panel: Dn vs Ia ----------------
    ia = np.linspace(0.1, 30.0, 200)  # m/s (senaryo taraması)
    for p in pilots:
        ky = p["ky"]
        dn_ia = NewmarkDisplacementIa(ky, ia) * 100.0
        axR.plot(ia, dn_ia, color=p["color"], lw=2.4,
                 label=f"ID {p['id']} · Jibson 2007 (Iₐ)  (aᵧ={ky:.3f}g)")

    axR.axhline(DN_LIMIT_CM, color="#dc2626", lw=1.8, ls="-.", alpha=0.9)
    axR.text(29.5, DN_LIMIT_CM * 1.15, "10 cm — Serviceability Limit",
             ha="right", va="bottom", fontsize=9.5, color="#dc2626", fontweight="bold")
    axR.set_xlabel("Arias Yoğunluğu  Iₐ  (m/s)  [senaryo taraması]", fontsize=11)
    axR.set_ylabel("Kalıcı Yer Değiştirme  Dₙ  (cm)", fontsize=11)
    axR.set_title("Dₙ – Arias Yoğunluğu (Jibson 2007, Denk. 9)", fontsize=12)
    axR.set_yscale("log")
    axR.set_ylim(0.1, 1100)
    axR.grid(True, which="both", alpha=0.25)
    axR.legend(fontsize=8.6, loc="lower right", framealpha=0.92)

    # ---------------- Duyarlılık kutusu (aᵧ/PGA) ----------------
    lines = ["Duyarlilik (a_y / PGA_gozlem):"]
    for p in pilots:
        if np.isfinite(p["pga_obs"]) and p["pga_obs"] > 0:
            ratio = p["ky"] / p["pga_obs"]
            tag = "a_y>PGA -> tetiklenme yok" if ratio >= 1 else "a_y<PGA -> kayma"
            lines.append(
                f"  ID {p['id']} ({p['desc']}): a_y={p['ky']:.3f}g, "
                f"PGA={p['pga_obs']:.3f}g → oran={ratio:.2f}  [{tag}]"
            )
    # en düşük oranlı (en hassas) pilotu belirt
    valid = [p for p in pilots if np.isfinite(p["pga_obs"]) and p["pga_obs"] > 0]
    if valid:
        most = min(valid, key=lambda p: p["ky"] / p["pga_obs"])
        lines.append(f"  → En hassas (en düşük oran): ID {most['id']}")
    box = "\n".join(lines)
    fig.text(0.012, 0.012, box, fontsize=8.6, va="bottom", ha="left",
             family="monospace",
             bbox=dict(boxstyle="round,pad=0.5", fc="#fef9c3", ec="#ca8a04", alpha=0.95))

    fig.text(0.99, 0.012,
             "Not: aᵧ dokümante pilot geometrisiyle (θ, litoloji) hesaplandı; "
             "PGA USGS ShakeMap zarfından örneklendi.\nIₐ ekseni senaryo taramasıdır "
             "(gerçek kayıt zaman-serisi integrali değil).",
             fontsize=7.6, va="bottom", ha="right", color="#64748b", style="italic")

    fig.tight_layout(rect=[0, 0.10, 1, 0.965])
    out = config.FIGURES_DIR / "newmark_arias_models.png"
    config.FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=140, bbox_inches="tight")
    plt.close(fig)
    return out


def main():
    pilots = compute_pilots()
    print("Pilot hesapları:")
    for p in pilots:
        print(f"  ID {p['id']} · {p['desc']} · θ={p['theta']}° · "
              f"FS={p['fs_static']:.3f} · aᵧ={p['ky']:.4f}g · "
              f"PGA_gözlem={p['pga_obs']:.3f}g")
    out = make_figure(pilots)
    print(f"\nFigür kaydedildi: {out}")


if __name__ == "__main__":
    main()
