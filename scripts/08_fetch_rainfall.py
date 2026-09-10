"""
Script 8: Yağış (CHIRPS-2.0) rasterini indir.

Çalışma alanı için aylık yağış rasterı indirir ve data/raw/rainfall_chirps.tif
olarak kaydeder. Bu katman YALNIZCA GÖRSEL bağlam içindir; PINN modeline girmez.

Kullanım:
  python3 scripts/08_fetch_rainfall.py [--year 2023] [--month 2]
"""
import sys
import argparse
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data_engineering.rainfall_fetch import fetch_rainfall_raster


def main():
    ap = argparse.ArgumentParser(description="CHIRPS-2.0 yağış rasteri indir")
    ap.add_argument("--year", type=int, default=2023)
    ap.add_argument("--month", type=int, default=2)
    args = ap.parse_args()
    try:
        out = fetch_rainfall_raster(year=args.year, month=args.month)
        print(f"\nTamam: {out}")
    except RuntimeError as e:
        print(f"\nUYARI: Yağış indirilemedi — {e}")
        print("Uygulama yağış katmanı olmadan çalışmaya devam eder.")
        sys.exit(1)


if __name__ == "__main__":
    main()
