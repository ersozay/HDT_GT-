"""
Script 11: 6 Şubat 2023 ivme kayıtlarını indir (ESM / AFAD TK ağı)

Çalışma alanındaki tüm açık TK istasyonları için Pazarcık (Mw 7.7) ve Elbistan
(Mw 7.5) depremlerinin işlenmiş (MP) ivme kayıtlarını indirir ve bir özet
tablosu yazar: istasyon, konum, bileşen PGA'ları, Arias şiddeti.

Çıktılar:
  data/raw/strong_motion/<olay>/TK.<istasyon>..HN?.*.ASC
  outputs/results/strong_motion_stations.csv

Kullanım:
  python3 scripts/11_fetch_strong_motion.py
"""
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))

from config import DATA_RAW, OUTPUTS_DIR
from src.data_engineering.regional_grid import STUDY_BOUNDS
from src.data_engineering.strong_motion import (
    EVENTS, list_stations, fetch_record, horizontal_records, arias_intensity,
)

SM_DIR = DATA_RAW / "strong_motion"


def run():
    # İstasyon araması alanın biraz dışına taşar (sınırdaki poligonlar için)
    lon0, lat0, lon1, lat1 = STUDY_BOUNDS
    stations = list_stations((lon0 - 0.3, lat0 - 0.3, lon1 + 0.3, lat1 + 0.3))
    print(f"ESM'de {len(stations)} açık TK istasyonu bulundu.")
    rows = []
    for ev in EVENTS:
        n_ok = 0
        for i, st in enumerate(stations, 1):
            files = fetch_record(ev, st["code"], SM_DIR)
            if not files:
                continue
            hz = horizontal_records(files)
            if not hz:
                continue
            n_ok += 1
            rows.append({
                "event": ev, "station": st["code"], "lat": st["lat"], "lon": st["lon"],
                "pga_g": max(r.pga_g for r in hz),
                "arias_ms": max(arias_intensity(r.acc_g, r.dt) for r in hz),
                "components": ",".join(r.component for r in hz),
                "dt_s": hz[0].dt, "duration_s": round(len(hz[0].acc_g) * hz[0].dt, 1),
                "vs30": hz[0].header.get("VS30_M/S", ""),
                "epi_km": hz[0].header.get("EPICENTRAL_DISTANCE_KM", ""),
            })
            if i % 10 == 0:
                print(f"  [{ev}] {i}/{len(stations)} istasyon tarandı, {n_ok} kayıt", flush=True)
        print(f"{ev}: {n_ok} istasyonda kayıt indirildi.")
    df = pd.DataFrame(rows).sort_values(["event", "pga_g"], ascending=[True, False])
    out = OUTPUTS_DIR / "results" / "strong_motion_stations.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)
    print(df.groupby("event")[["pga_g", "arias_ms"]].describe().round(3).T.to_string())
    print(f"\nÖzet: {out}")


if __name__ == "__main__":
    run()
