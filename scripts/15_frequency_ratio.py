"""
Script 15: Frekans Oranı (FR) Analizi — Kocaman ve ark. (2025) yöntemiyle

FR = (sınıftaki heyelan piksel oranı) / (sınıfın alan oranı)

Envanter poligonları DEM ızgarasına (0.000833° ≈ 90 m) dökülür; her faktör
sınıfında heyelan pikseli / toplam piksel oranı hesaplanır.

ÖNEMLİ: Litoloji FR'si Macrostrat PROXY'si (~5 km, 1:5 M) ile hesaplanır ve
"proxy-FR" olarak etiketlenir. Kocaman ve ark. Tablo 4'teki MTA FR değerleri
(örn. peridotit 6,70) bu proxy ile ÜRETİLEMEZ; MTA haritası gelince yeniden
hesaplanmalıdır.

Çıktı: outputs/results/frequency_ratio.json, outputs/figures/frequency_ratio.png
"""
import sys
import json
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))

from config import OUTPUTS_DIR, FIGURES_DIR
from src.data_engineering.extra_layers import fine_grid, LAYER_DIR
from src.data_engineering.regional_grid import (STUDY_BOUNDS, SLOPE_PATH, DEM_PATH, LITHO_PATH,
                                                reproject_to_grid, rasterize_inventory)
from src.geomechanics.lithology import LITHOLOGY_TABLE

LC = {10: "Ağaç", 20: "Çalılık", 30: "Çayır", 40: "Tarım", 50: "Yapılaşma", 60: "Çıplak/seyrek",
      80: "Su", 90: "Sulak alan"}


def fr_table(cls, ls, valid, labels=None, order=None):
    tot = valid.sum(); tot_ls = (ls & valid).sum()
    rows = []
    keys = order if order is not None else sorted(np.unique(cls[valid]))
    for k in keys:
        m = valid & (cls == k)
        if m.sum() == 0:
            continue
        a = m.sum() / tot; b = (ls & m).sum() / max(tot_ls, 1)
        rows.append({"sinif": labels.get(k, str(k)) if labels else str(k),
                     "alan_%": round(100 * a, 2), "heyelan_%": round(100 * b, 2),
                     "FR": round(b / a, 2) if a > 0 else None, "piksel": int(m.sum())})
    return pd.DataFrame(rows)


def run():
    t, w, h = fine_grid(STUDY_BOUNDS)
    ls = rasterize_inventory(t, w, h, all_touched=True).astype(bool)
    dem = reproject_to_grid(DEM_PATH, t, w, h, "bilinear")
    valid = np.isfinite(dem) & (dem > 0)
    slope = reproject_to_grid(SLOPE_PATH, t, w, h, "bilinear")
    lith = reproject_to_grid(LITHO_PATH, t, w, h, "nearest")
    out = {"note": "Envanter: Kocaman ve ark. (2025), 90 m ızgara. Litoloji FR'si Macrostrat "
                   "PROXY'sidir (MTA değil) — Kocaman Tablo 4 ile doğrudan karşılaştırılamaz.",
           "landslide_pixels": int((ls & valid).sum()), "tables": {}}

    sbins = [0, 5, 10, 15, 20, 25, 30, 35, 40, 90]
    sc = np.digitize(slope, sbins) - 1
    out["tables"]["eğim (°)"] = fr_table(sc, ls, valid & np.isfinite(slope),
                                         {i: f"{sbins[i]}–{sbins[i+1]}" for i in range(len(sbins) - 1)},
                                         list(range(len(sbins) - 1)))
    lith_i = np.where(np.isfinite(lith), lith, 0).astype(int)
    out["tables"]["litoloji — Macrostrat PROXY"] = fr_table(
        lith_i, ls, valid & (lith_i > 0), {k: v["description"] for k, v in LITHOLOGY_TABLE.items()})
    for name, path, bins in [("faya uzaklık (km, GEM)", LAYER_DIR / "dist_fault_km.tif", [0, 2, 5, 10, 20, 40, 300]),
                             ("2023 kırılmasına uzaklık (km)", LAYER_DIR / "dist_rupture_km.tif", [0, 2, 5, 10, 20, 40, 300]),
                             ("yükseklik (m)", DEM_PATH, [0, 500, 750, 1000, 1250, 1500, 2000, 4000])]:
        a = reproject_to_grid(path, t, w, h, "bilinear")
        c = np.digitize(a, bins) - 1
        out["tables"][name] = fr_table(c, ls, valid & np.isfinite(a),
                                       {i: f"{bins[i]}–{bins[i+1]}" for i in range(len(bins) - 1)},
                                       list(range(len(bins) - 1)))
    lc = reproject_to_grid(LAYER_DIR / "landcover.tif", t, w, h, "nearest")
    lc = np.where(np.isfinite(lc), lc, 0).astype(int)
    out["tables"]["arazi örtüsü (ESA WorldCover)"] = fr_table(lc, ls, valid & (lc > 0), LC)

    for k, df in out["tables"].items():
        print(f"\n== {k}\n{df.to_string(index=False)}")
    js = {**out, "tables": {k: v.to_dict(orient="records") for k, v in out["tables"].items()}}
    (OUTPUTS_DIR / "results" / "frequency_ratio.json").write_text(json.dumps(js, indent=2, ensure_ascii=False))
    _plot(out["tables"])


def _plot(tables):
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    keys = list(tables)
    fig, axes = plt.subplots(2, 3, figsize=(15, 8.5))
    for ax, k in zip(axes.ravel(), keys):
        df = tables[k]
        col = ["#e34948" if v and v > 1 else "#a3a29c" for v in df["FR"]]
        ax.barh(df["sinif"], df["FR"], color=col, edgecolor="white", linewidth=1.5)
        ax.axvline(1, color="#0b0b0b", ls=":", lw=1)
        ax.set_title(k, loc="left", fontsize=10.5)
        ax.tick_params(labelsize=8)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
    fig.suptitle("Frekans Oranı (FR > 1 kırmızı: heyelana yatkın) — Kocaman ve ark. (2025) yöntemi",
                 x=0.01, ha="left")
    fig.tight_layout(); fig.savefig(FIGURES_DIR / "frequency_ratio.png", dpi=150); plt.close(fig)


if __name__ == "__main__":
    run()
