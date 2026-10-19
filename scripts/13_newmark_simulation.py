"""
Script 13: Poligon Bazında Kayma Simülasyonu — Gerçek İvme Kayıtlarıyla Newmark

Her heyelan poligonu (Kocaman 2025, n = 2596) ve her eğitim noktası için:
  1. Kᵧ: poligondaki örnek noktalarının medyanı (Mohr-Coulomb sonsuz şev, z = 5 m)
  2. Her deprem (Pazarcık Mw 7.7, Elbistan Mw 7.5) için en yakın AFAD istasyonunun
     işlenmiş ivme kaydı (ESM), noktanın ShakeMap PGA'sına ölçeklenir
  3. Rijit-blok Newmark zaman-tanım integrasyonu (Jibson 1993); iki yatay bileşen
     × iki polarite → en büyük kalıcı yer değiştirme
  4. İki depremden büyük olanı (ardışık olaylar için üst sınır) raporlanır

Çıktılar
  outputs/results/newmark_th_polygons.json   — poligon başına Dₙ(t) (2 s adım) + nihai Dₙ
  outputs/results/newmark_th_validation.json — zaman-tanım vs Jibson (2007) regresyonu
  outputs/figures/newmark_th_validation.png, newmark_th_pilots.png

Varsayımlar (tezde belirtilmeli): en yakın istasyon kaydı + PGA ölçeklemesi frekans
içeriğini istasyondan taşır; rijit blok, tek yönlü kayma; Kᵧ litoloji proxy'sine dayanır.

Kullanım:
  python3 scripts/13_newmark_simulation.py
"""
import sys
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))

from config import DATA_RAW, DATA_PROCESSED, OUTPUTS_DIR, FIGURES_DIR, DN_THRESHOLD_M
from src.data_engineering.feature_matrix import load_feature_table
from src.data_engineering.real_dataset import sample_raster
from src.data_engineering.strong_motion import EVENTS, horizontal_records
from src.geomechanics.newmark import NewmarkDisplacement
from src.geomechanics.newmark_ts import rigid_block_both_directions

SM_DIR = DATA_RAW / "strong_motion"
KY_FLOOR = 0.02          # Kᵧ ≈ 0 (statik marjinal) bloklar için alt sınır (g)
DT_TARGET = 0.01         # s — kayıtlar 100 Hz'e seyreltilir (Newmark için yeterli)
HIST_STEP = 2.0          # s — animasyon için anlık görüntü aralığı


def load_records(event):
    """{istasyon: (lat, lon, [bileşenler], dt, pga)} — yatay bileşenler, 100 Hz."""
    out = {}
    for f in sorted((SM_DIR / event).glob("TK.*..HN*.ASC")):
        st = f.name.split(".")[1]
        out.setdefault(st, []).append(f)
    recs = {}
    for st, files in out.items():
        hz = horizontal_records(files)
        if not hz:
            continue
        step = max(1, int(round(DT_TARGET / hz[0].dt)))
        comps = [r.acc_g[::step] for r in hz]
        n = min(len(c) for c in comps)
        comps = [c[:n] for c in comps]
        recs[st] = (hz[0].lat, hz[0].lon, comps, hz[0].dt * step, max(r.pga_g for r in hz))
    return recs


def nearest_station(lat, lon, recs):
    codes = list(recs)
    la = np.array([recs[c][0] for c in codes]); lo = np.array([recs[c][1] for c in codes])
    d = np.hypot((lat[:, None] - la[None]) * 111.0,
                 (lon[:, None] - lo[None]) * 111.0 * np.cos(np.deg2rad(lat[:, None])))
    i = d.argmin(axis=1)
    return np.array(codes)[i], d[np.arange(len(lat)), i]


def simulate(lat, lon, ky, pga_site, recs, history=False):
    """Noktaları en yakın istasyona göre gruplayıp vektörel olarak çözer."""
    st, dist = nearest_station(lat, lon, recs)
    dn = np.zeros(len(lat)); hist = {}
    times = None
    for code in np.unique(st):
        m = st == code
        _, _, comps, dt, pga_rec = recs[code]
        scale = np.clip(pga_site[m] / pga_rec, 0.05, 20.0)
        d, t, h = rigid_block_both_directions(comps, dt, np.maximum(ky[m], KY_FLOOR), scale,
                                              HIST_STEP if history else None)
        dn[m] = d
        if history:
            hist[code] = (np.where(m)[0], t, h)
            times = t if times is None or len(t) > len(times) else times
    H = None
    if history:
        H = np.zeros((len(times), len(lat)))
        for idx, t, h in hist.values():
            n = min(len(t), len(times))
            H[:n, idx] = h[:n]
            H[n:, idx] = h[n - 1]
    return dn, st, dist, times, H


def run():
    t0 = time.time()
    recs = {ev: load_records(ev) for ev in EVENTS}
    print({ev: len(r) for ev, r in recs.items()}, "istasyon kaydı yüklendi")
    df = load_feature_table(DATA_PROCESSED / "feature_matrix.parquet")
    lat, lon = df["lat"].to_numpy(), df["lon"].to_numpy()
    ky = df["ky_g"].to_numpy()
    pga_ev = {"pazarcik": sample_raster(DATA_RAW / "pga_us6000jllz.tif", lon, lat),
              "elbistan": sample_raster(DATA_RAW / "pga_us6000jlqa.tif", lon, lat)}

    # ---- Eğitim noktaları: zaman-tanım Dₙ vs regresyon ------------------------
    dn_ev = {}
    for ev in EVENTS:
        p = np.nan_to_num(pga_ev[ev], nan=0.01)
        dn_ev[ev], _, _, _, _ = simulate(lat, lon, ky, p, recs[ev])
        print(f"  {ev}: noktalar çözüldü ({time.time()-t0:.0f} s)", flush=True)
    dn_th = np.maximum(dn_ev["pazarcik"], dn_ev["elbistan"]) * 100
    dn_reg = df["dn_cm"].to_numpy()
    y = df["label"].to_numpy()
    from sklearn.metrics import roc_auc_score
    thr = DN_THRESHOLD_M * 100
    val = {
        "auc_dn_timehistory": round(float(roc_auc_score(y, np.log10(dn_th + 0.1))), 4),
        "auc_dn_regression_jibson2007": round(float(roc_auc_score(y, np.log10(dn_reg + 0.1))), 4),
        "hit_rate_landslide_points_th": round(float((dn_th[y == 1] >= thr).mean()), 4),
        "hit_rate_landslide_points_reg": round(float((dn_reg[y == 1] >= thr).mean()), 4),
        "false_rate_background_th": round(float((dn_th[y == 0] >= thr).mean()), 4),
        "false_rate_background_reg": round(float((dn_reg[y == 0] >= thr).mean()), 4),
        "dominant_event_share_pazarcik": round(float((dn_ev["pazarcik"] >= dn_ev["elbistan"])[y == 1].mean()), 4),
        "n_points": int(len(y)),
        "note": "En yakın AFAD istasyonu kaydı (ESM, MP) ShakeMap PGA'sına ölçeklendi; "
                f"Kᵧ alt sınırı {KY_FLOOR} g; iki olaydan büyük Dₙ.",
    }
    print(json.dumps(val, indent=2, ensure_ascii=False))

    # ---- Poligonlar: Dₙ(t) animasyon verisi -----------------------------------
    pos = df[df["label"] == 1]
    g = pos.groupby("polygon_id").agg(lat=("lat", "median"), lon=("lon", "median"),
                                       ky=("ky_g", "median"), slope=("slope_deg", "median"))
    plat, plon, pky = g["lat"].to_numpy(), g["lon"].to_numpy(), g["ky"].to_numpy()
    res = {}
    for ev in EVENTS:
        p = np.nan_to_num(sample_raster(DATA_RAW / f"pga_{EVENTS[ev]['usgs_id']}.tif", plon, plat), nan=0.01)
        d, st, dist, times, H = simulate(plat, plon, pky, p, recs[ev], history=True)
        res[ev] = (d, st, dist, times, H, p)
    use_p = res["pazarcik"][0] >= res["elbistan"][0]
    final = np.where(use_p, res["pazarcik"][0], res["elbistan"][0]) * 100
    T = max(len(res[e][3]) for e in EVENTS)
    hist = np.zeros((T, len(g)))
    for ev, mask in (("pazarcik", use_p), ("elbistan", ~use_p)):
        H = res[ev][4]
        hist[:len(H), mask] = H[:, mask] * 100
        hist[len(H):, mask] = H[-1, mask] * 100
    times = (np.arange(T) * HIST_STEP).tolist()
    ids = [int(i) for i in g.index]
    out = {
        "times_s": times,
        "polygon_ids": ids,
        "final_dn_cm": {str(i): round(float(v), 2) for i, v in zip(ids, final)},
        "event": {str(i): ("pazarcik" if u else "elbistan") for i, u in zip(ids, use_p)},
        "ky_g": {str(i): round(float(v), 3) for i, v in zip(ids, pky)},
        "pga_g": {str(i): round(float(res["pazarcik" if u else "elbistan"][5][k]), 3)
                  for k, (i, u) in enumerate(zip(ids, use_p))},
        "station": {str(i): str(res["pazarcik" if u else "elbistan"][1][k])
                    for k, (i, u) in enumerate(zip(ids, use_p))},
        "station_dist_km": {str(i): round(float(res["pazarcik" if u else "elbistan"][2][k]), 1)
                            for k, (i, u) in enumerate(zip(ids, use_p))},
        "history_cm": np.round(hist, 1).T.tolist(),        # [poligon][zaman]
        "dn_threshold_cm": thr,
        "summary": {
            "n_polygons": len(ids),
            "share_triggered_10cm": round(float((final >= thr).mean()), 4),
            "median_dn_cm": round(float(np.median(final)), 2),
            "share_pazarcik_dominant": round(float(use_p.mean()), 4),
        },
    }
    R = OUTPUTS_DIR / "results"
    (R / "newmark_th_polygons.json").write_text(json.dumps(out, ensure_ascii=False))
    val["polygons"] = out["summary"]
    (R / "newmark_th_validation.json").write_text(json.dumps(val, indent=2, ensure_ascii=False))
    print("Poligon özeti:", out["summary"])
    _plots(dn_th, dn_reg, y, out, recs)
    print(f"Süre: {time.time()-t0:.0f} s")


def _plots(dn_th, dn_reg, y, out, recs):
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    INK, MUTED, BLUE, ORANGE = "#0b0b0b", "#52514e", "#2a78d6", "#eb6834"
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.8), sharey=True)
    bins = np.linspace(-1, 3.1, 42)
    for ax, d, title in [(axes[0], dn_reg, "Jibson (2007) regresyonu (PGA)"),
                         (axes[1], dn_th, "Zaman-tanım Newmark (gerçek AFAD kayıtları)")]:
        ld = np.log10(d + 0.1)
        ax.hist(ld[y == 0], bins, color="#a3a29c", alpha=0.75, density=True)
        ax.hist(ld[y == 1], bins, color=BLUE, alpha=0.75, density=True)
        ax.axvline(np.log10(10.1), color=INK, ls=":", lw=1)
        ax.text(np.log10(10.1), ax.get_ylim()[1] * 0.92, " 10 cm", color=INK, fontsize=9)
        ax.set_title(title, loc="left", fontsize=11, color=INK)
        ax.set_xlabel("log₁₀(Dₙ + 0,1)  [cm]", color=MUTED)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
    axes[0].set_ylabel("Yoğunluk", color=MUTED)
    axes[1].text(0.98, 0.95, "mavi: heyelan noktaları\ngri: arka plan", transform=axes[1].transAxes,
                 ha="right", va="top", fontsize=9, color=INK)
    fig.tight_layout(); fig.savefig(FIGURES_DIR / "newmark_th_validation.png", dpi=160); plt.close(fig)

    ids = out["polygon_ids"]; H = out["history_cm"]; t = out["times_s"]
    fig, ax = plt.subplots(figsize=(9, 5))
    for pid, col in ((1327, "#c026d3"), (1353, "#0284c7")):
        if pid in ids:
            k = ids.index(pid)
            ax.plot(t, H[k], color=col, lw=2.2)
            ax.text(t[-1] + 1, H[k][-1], f"ID {pid} · {out['station'][str(pid)]} istasyonu "
                    f"({out['event'][str(pid)]}) · Kᵧ={out['ky_g'][str(pid)]} g", fontsize=8.5, va="center")
    ax.axhline(out["dn_threshold_cm"], color=MUTED, ls=":", lw=1)
    ax.set_xlabel("Deprem başlangıcından süre (s)", color=MUTED)
    ax.set_ylabel("Kalıcı yer değiştirme Dₙ (cm)", color=MUTED)
    ax.set_title("Pilot yamaçlar — zaman-tanım Newmark kayması", loc="left", fontsize=11.5, color=INK)
    ax.set_xlim(0, t[-1] * 1.7)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    fig.tight_layout(); fig.savefig(FIGURES_DIR / "newmark_th_pilots.png", dpi=160); plt.close(fig)


if __name__ == "__main__":
    run()
