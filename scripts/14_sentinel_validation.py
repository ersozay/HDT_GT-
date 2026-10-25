"""
Script 14: Envanter Doğrulaması — Sentinel-2 (2022 yaz → 2023 yaz)

Danışman önerisi üç adım:
  (1) Poligon gerçek mi?  Her poligonda ΔNDVI ve Δparlaklık, halkaya göre.
      Eşikler, dik arazideki rastgele arka plan noktalarının değişim dağılımından
      (yüzdelik) alınır → "doğrulandı / sinyal yok / belirsiz".
  (2) Negatif etiket kirliliği: dik ve tarım dışı arka plan noktalarında aynı ölçüm;
      güçlü yara izi sinyali veren noktalar "olası kaçmış heyelan" olarak işaretlenir.
  (3) Göz kontrolü için rastgele poligonlardan ön/son gerçek renk paftaları.

Çıktılar
  outputs/results/sentinel_polygons.csv, sentinel_negatives.csv, sentinel_summary.json
  outputs/figures/sentinel_change_dist.png, sentinel_chips.png

Kullanım:
  python3 scripts/14_sentinel_validation.py [--neg 4000] [--workers 24]
"""
import sys
import json
import time
import argparse
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))

from config import DATA_PROCESSED, OUTPUTS_DIR, FIGURES_DIR, INVENTORY_KML
from src.data_engineering.feature_matrix import load_feature_table
from src.data_engineering.regional_grid import STUDY_BOUNDS
from src.data_engineering import sentinel2 as S2

R = OUTPUTS_DIR / "results"
PRE = "2022-07-01T00:00:00Z/2022-08-31T23:59:59Z"
POST = "2023-07-01T00:00:00Z/2023-08-31T23:59:59Z"


def all_scenes(scenes_by_tile):
    """Tüm karoların sahneleri, en az bulutlu önce."""
    out = [s for sc in scenes_by_tile.values() for s in sc]
    return sorted(out, key=lambda s: (s.cloud > 1.0, s.cloud))


def run_pairs(items, pre_sc, post_sc, workers, inner_r=None, label="", cell_deg=0.02):
    """
    items: [(key, geom_ll)] → DataFrame. Yakın öğeler ~2 km hücrelerde gruplanır;
    her hücre için grubu tamamen kapsayan, geçerli piksel oranı yeterli ilk sahne kullanılır.
    """
    from shapely.geometry import MultiPoint
    groups = {}
    for k, g in items:
        c = g.representative_point() if inner_r is None else g
        groups.setdefault((round(c.y / cell_deg), round(c.x / cell_deg)), []).append((k, g, c))
    t0 = time.time()
    rows = {}

    def solve(scenes, members):
        hull = MultiPoint([c for _, _, c in members]).envelope.buffer(0.003)
        res = {}
        for s in scenes:
            todo = [m for m in members if m[0] not in res or res[m[0]]["valid_in"] < 0.6]
            if not todo:
                break
            if not s.footprint.contains(hull):
                continue
            try:
                out = S2.measure_group(s, [g for _, g, _ in todo], inner_r)
            except Exception:
                continue
            for (k, _, _), r in zip(todo, out):
                if k not in res or r["valid_in"] > res[k]["valid_in"]:
                    res[k] = r
        return res

    def job(members):
        return members, solve(pre_sc, members), solve(post_sc, members)

    with ThreadPoolExecutor(workers) as ex:
        futs = [ex.submit(job, m) for m in groups.values()]
        done = 0
        for f in as_completed(futs):
            members, a, b = f.result()
            for k, _, _ in members:
                row = {"key": k}
                for tag, r in (("pre", a.get(k)), ("post", b.get(k))):
                    for kk, vv in (r or {}).items():
                        row[f"{tag}_{kk}"] = vv
                rows[k] = row
            done += 1
            if done % 100 == 0:
                print(f"  {label}: {done}/{len(groups)} hücre ({time.time()-t0:.0f} s)", flush=True)
    df = pd.DataFrame(rows.values())
    for v in ("ndvi", "bright"):
        if f"pre_{v}_in" in df and f"post_{v}_in" in df:
            df[f"d_{v}_in"] = df[f"post_{v}_in"] - df[f"pre_{v}_in"]
            df[f"d_{v}_ring"] = df[f"post_{v}_ring"] - df[f"pre_{v}_ring"]
            df[f"rel_d_{v}"] = df[f"d_{v}_in"] - df[f"d_{v}_ring"]
    return df


def run(n_neg=4000, workers=24, seed=0, sample=None):
    import geopandas as gpd
    from shapely.geometry import Point
    R.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    print("STAC sahne araması…")
    pre = S2.best_scenes(S2.search_scenes(STUDY_BOUNDS, PRE))
    post = S2.best_scenes(S2.search_scenes(STUDY_BOUNDS, POST))
    print(f"  ön: {len(pre)} karo, son: {len(post)} karo")
    pre_sc, post_sc = all_scenes(pre), all_scenes(post)

    inv = gpd.read_file(INVENTORY_KML)
    inv = inv[inv.geom_type.isin(["Polygon", "MultiPolygon"])].reset_index(drop=True)
    items = list(enumerate(inv.geometry))
    if sample:
        rs = np.random.RandomState(seed)
        items = [items[i] for i in sorted(rs.choice(len(items), sample, replace=False))]
        n_neg = min(n_neg, 2 * sample)
    pol = run_pairs(items, pre_sc, post_sc, workers, label="poligon")
    pol["polygon_id"] = pol["key"].astype(int)
    pol["area_m2"] = inv.to_crs(32637).area.values[pol["polygon_id"]]

    # Arka plan: dik (≥15°) ve tarım dışı negatifler + karşılaştırma için aynı sayıda düz/rastgele
    df = load_feature_table(DATA_PROCESSED / "feature_matrix.parquet")
    neg = df[df["label"] == 0]
    steep = neg[(neg["slope_deg"] >= 15) & (neg.get("landcover", 30) != 40)]
    rng = np.random.RandomState(seed)
    pick = steep.sample(min(n_neg, len(steep)), random_state=seed)
    items = [(int(i), Point(r.lon, r.lat)) for i, r in pick.iterrows()]
    negm = run_pairs(items, pre_sc, post_sc, workers, inner_r=40.0, label="negatif")
    negm["row"] = negm["key"].astype(int)
    negm["slope_deg"] = df.loc[negm["row"], "slope_deg"].values

    pol.to_csv(R / "sentinel_polygons.csv", index=False)
    negm.to_csv(R / "sentinel_negatives.csv", index=False)
    summary = classify(pol, negm)
    (R / "sentinel_summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False))
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    _plot_dist(pol, negm, summary)
    if not sample:
        _chips(inv, pol, pre_sc, post_sc)
    print(f"Süre: {time.time()-t0:.0f} s")


def classify(pol, negm, q=0.05, min_valid=0.6, min_px=4):
    """Eşikler arka plan dağılımından; sonuç sınıfları poligon tablosuna yazılır."""
    ok_n = (negm["pre_valid_in"] >= min_valid) & (negm["post_valid_in"] >= min_valid)
    bg = negm[ok_n]
    thr_ndvi = float(bg["rel_d_ndvi"].quantile(q))
    thr_bright = float(bg["rel_d_bright"].quantile(1 - q))

    usable = ((pol["pre_valid_in"] >= min_valid) & (pol["post_valid_in"] >= min_valid) &
              (pol["pre_n_in"] >= min_px) & pol["rel_d_ndvi"].notna())
    signal = (pol["rel_d_ndvi"] <= thr_ndvi) | (pol["rel_d_bright"] >= thr_bright)
    sparse = pol["pre_ndvi_in"] < 0.2                 # zaten çıplak/seyrek → NDVI duyarsız
    pol["s2_class"] = np.where(~usable, "belirsiz",
                      np.where(signal, "doğrulandı", np.where(sparse, "belirsiz", "sinyal_yok")))
    sig_n = ok_n & ((negm["rel_d_ndvi"] <= thr_ndvi) | (negm["rel_d_bright"] >= thr_bright))
    negm["s2_flag"] = np.where(~ok_n, "belirsiz", np.where(sig_n, "olası_kaçmış", "temiz"))
    pol.to_csv(R / "sentinel_polygons.csv", index=False)
    negm.to_csv(R / "sentinel_negatives.csv", index=False)

    vc = pol["s2_class"].value_counts()
    by_size = pol.assign(sz=pd.cut(pol["area_m2"], [0, 1000, 5000, 20000, 1e12],
                                   labels=["<1k", "1–5k", "5–20k", ">20k m²"]))
    tab = by_size[by_size["s2_class"] != "belirsiz"].groupby("sz", observed=True)["s2_class"].apply(
        lambda s: round(float((s == "doğrulandı").mean()), 3)).to_dict()
    return {
        "method": "Sentinel-2 L2A, 2022 Tem–Ağu → 2023 Tem–Ağu; iç−halka (30–150 m) değişimi; "
                  f"eşik = dik arka plan noktalarının %{int(q*100)}/%{int((1-q)*100)} yüzdeliği",
        "thresholds": {"rel_d_ndvi_max": round(thr_ndvi, 4), "rel_d_bright_min": round(thr_bright, 4)},
        "polygons": {k: int(v) for k, v in vc.items()},
        "polygons_share_confirmed_of_usable": round(float((pol["s2_class"] == "doğrulandı").sum() /
                                                          max((pol["s2_class"] != "belirsiz").sum(), 1)), 4),
        "confirmed_share_by_size": {str(k): v for k, v in tab.items()},
        "negatives": {k: int(v) for k, v in negm["s2_flag"].value_counts().items()},
        "negatives_contamination_rate": round(float((negm["s2_flag"] == "olası_kaçmış").sum() /
                                                    max((negm["s2_flag"] != "belirsiz").sum(), 1)), 4),
        "note": f"Arka plan eşiği tanım gereği ~%{int(2*q*100)} yanlış pozitif içerir; "
                "kirlilik oranı bu taban değerle karşılaştırılmalıdır.",
    }


def _plot_dist(pol, negm, summary):
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    INK, MUTED, BLUE, GRAY = "#0b0b0b", "#52514e", "#2a78d6", "#a3a29c"
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.6))
    for ax, col, thr, title in [
        (axes[0], "rel_d_ndvi", summary["thresholds"]["rel_d_ndvi_max"], "Göreli ΔNDVI (iç − halka)"),
        (axes[1], "rel_d_bright", summary["thresholds"]["rel_d_bright_min"], "Göreli Δparlaklık (iç − halka)")]:
        a = negm[col].dropna(); b = pol[col].dropna()
        lo, hi = np.nanpercentile(np.r_[a, b], [1, 99])
        bins = np.linspace(lo, hi, 60)
        ax.hist(a, bins, density=True, color=GRAY, alpha=0.8)
        ax.hist(b, bins, density=True, color=BLUE, alpha=0.6)
        ax.axvline(thr, color=INK, ls=":", lw=1)
        ax.set_title(title, loc="left", fontsize=11, color=INK)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
    axes[0].text(0.02, 0.95, "mavi: envanter poligonları\ngri: dik arka plan noktaları",
                 transform=axes[0].transAxes, va="top", fontsize=9, color=INK)
    fig.tight_layout(); fig.savefig(FIGURES_DIR / "sentinel_change_dist.png", dpi=160); plt.close(fig)


def _chips(inv, pol, pre_sc, post_sc, n=8, seed=1):
    """Göz kontrolü: her sınıftan rastgele poligonlar için ön/son gerçek renk."""
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import rasterio
    from rasterio.windows import from_bounds
    from shapely.geometry import Point
    from shapely.ops import transform as shp_transform
    import pyproj
    picks = []
    for cls in ("doğrulandı", "sinyal_yok", "belirsiz"):
        sub = pol[(pol["s2_class"] == cls) & (pol["area_m2"] > 3000)]
        picks += [(cls, int(i)) for i in sub.sample(min(n // 2 + 1 if cls != "doğrulandı" else n // 2,
                                                       len(sub)), random_state=seed)["polygon_id"]]
    fig, axes = plt.subplots(len(picks), 2, figsize=(6, 2.9 * len(picks)))
    for r, (cls, pid) in enumerate(picks):
        g = inv.geometry.iloc[pid]; c = g.representative_point()
        for col, scs, lab in ((0, pre_sc, "2022 yaz"), (1, post_sc, "2023 yaz")):
            ax = axes[r, col]; ax.axis("off")
            sc = [s for s in scs if s.footprint.contains(Point(c.x, c.y).buffer(0.003))]
            if not sc:
                continue
            s = sc[0]
            gu = shp_transform(pyproj.Transformer.from_crs(4326, s.epsg, always_xy=True).transform, g)
            box = gu.buffer(250).bounds
            try:
                with rasterio.Env(**S2.GDAL_ENV), rasterio.open(s.assets["visual"]) as ds:
                    win = from_bounds(*box, ds.transform)
                    img = ds.read(window=win, boundless=True, fill_value=0).transpose(1, 2, 0)
            except Exception:
                continue
            ax.imshow(np.clip(img.astype(float) / 255 * 1.4, 0, 1), extent=(box[0], box[2], box[1], box[3]))
            xs, ys = (gu.exterior.xy if hasattr(gu, "exterior") else gu.geoms[0].exterior.xy)
            ax.plot(xs, ys, color="#ff3b30", lw=1.2)
            ax.set_title(f"ID {pid} · {cls} · {lab} ({s.date})", fontsize=7.5)
    fig.tight_layout(); fig.savefig(FIGURES_DIR / "sentinel_chips.png", dpi=130); plt.close(fig)


def _from_gee_table(path, key):
    """GEE dışa aktarımı (gee/ndvi_change.js) → classify() biçimi."""
    g = pd.read_csv(path)
    df = pd.DataFrame({"key": g[key].astype(int)})
    for tag in ("pre", "post"):
        for v in ("ndvi", "bright"):
            df[f"{tag}_{v}_in"] = g[f"{v}_{tag}_in"]
            df[f"{tag}_{v}_ring"] = g[f"{v}_{tag}_ring"]
        n = g[f"ndvi_{tag}_n_in"].fillna(0)
        df[f"{tag}_n_in"] = n
        df[f"{tag}_valid_in"] = (n >= 2).astype(float)
        df[f"{tag}_valid_ring"] = g[f"ndvi_{tag}_ring"].notna().astype(float)
    for v in ("ndvi", "bright"):
        df[f"d_{v}_in"] = df[f"post_{v}_in"] - df[f"pre_{v}_in"]
        df[f"d_{v}_ring"] = df[f"post_{v}_ring"] - df[f"pre_{v}_ring"]
        df[f"rel_d_{v}"] = df[f"d_{v}_in"] - df[f"d_{v}_ring"]
    return df


def run_from_gee(folder):
    import geopandas as gpd
    folder = Path(folder)
    inv = gpd.read_file(INVENTORY_KML)
    inv = inv[inv.geom_type.isin(["Polygon", "MultiPolygon"])].reset_index(drop=True)
    pol = _from_gee_table(folder / "s2_polygons.csv", "pid")
    pol["polygon_id"] = pol["key"]
    pol["area_m2"] = inv.to_crs(32637).area.values[pol["polygon_id"]]
    negm = _from_gee_table(folder / "s2_negatives.csv", "row")
    negm["row"] = negm["key"]
    summary = classify(pol, negm)
    summary["source"] = "Google Earth Engine — COPERNICUS/S2_SR_HARMONIZED medyan kompozitleri"
    (R / "sentinel_summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False))
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    _plot_dist(pol, negm, summary)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--neg", type=int, default=4000)
    ap.add_argument("--workers", type=int, default=24)
    ap.add_argument("--sample", type=int, default=None, help="rastgele poligon örneklemi (hızlı tahmin)")
    ap.add_argument("--from-gee", default=None,
                    help="GEE CSV klasörü (gee/ndvi_change.js çıktısı) — ağdan okuma yapılmaz")
    a = ap.parse_args()
    if a.from_gee:
        run_from_gee(a.from_gee)
    else:
        run(a.neg, a.workers, sample=a.sample)
