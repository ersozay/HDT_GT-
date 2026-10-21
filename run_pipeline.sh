#!/usr/bin/env bash
# Heyelan Dijital Twin — v2 analiz hattını baştan sona çalıştırır.
# Kullanım: ./run_pipeline.sh            (veri hazırsa ~15 dk, Apple M-serisi CPU)
#           ./run_pipeline.sh --prepare  (öznitelik matrisini de yeniden üretir)
set -euo pipefail
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
mkdir -p outputs/logs

step() { echo ""; echo "▶ $1"; }

if [[ "${1:-}" == "--prepare" ]]; then
  step "01 · Öznitelik matrisi (gerçek katmanlar)"
  python3 scripts/01_prepare_data.py
fi
step "12 · Açık kaynak ek katmanlar (ShakeMap, faylar, DEM türevleri, WorldCover, SoilGrids)"
python3 scripts/12_build_layers.py
step "11 · AFAD ivme kayıtları (ESM)"
python3 scripts/11_fetch_strong_motion.py
step "09 · Mekânsal blok CV karşılaştırması (15 model × 5 kat)"
python3 scripts/09_benchmark.py | tee outputs/logs/benchmark.log
python3 scripts/09_benchmark.py --models logreg_L,rf_L,gbm_L,gbm_mono_L,mlp_L,pinn_v2_L | tee -a outputs/logs/benchmark.log
step "03 · Nihai PINN v2 + katmanlar eğitimi + SHAP + metrics.json"
python3 scripts/03_train_pinn.py --features v3 | tee outputs/logs/train_v3.log
step "05 · Bölgesel duyarlılık haritası"
python3 scripts/05_susceptibility_map.py
step "10 · Kritik PGA: belirsizlik, tasarım tablosu, ampirik doğrulama"
python3 scripts/10_critical_pga.py | tee outputs/logs/critical_pga.log
step "02 · FEM-SSR pilot analizi"
python3 scripts/02_run_fem.py | tee outputs/logs/fem.log
step "13 · Poligon bazında kayma simülasyonu (zaman-tanım Newmark)"
python3 scripts/13_newmark_simulation.py
step "15/16 · Frekans Oranı ve etiket kirliliği duyarlılığı"
python3 scripts/15_frequency_ratio.py
caffeinate -i -s python3 scripts/16_label_noise.py --budget 100000
step "04 · Senaryo haritaları"
python3 scripts/04_generate_maps.py
step "06/07 · Newmark modelleri ve Wenchuan fizik-transferi"
python3 scripts/06_newmark_models.py
python3 scripts/07_wenchuan_validation.py
# Sentinel-2 doğrulaması (yavaş bağlantıda GEE yolu: gee/ndvi_change.js → --from-gee):
#   caffeinate -i -s python3 scripts/14_sentinel_validation.py
step "Rapor"
python3 docs/build_report.py
step "Testler"
python3 -m pytest tests -q
echo ""; echo "✔ Tamamlandı. Uygulama: ./run_app.sh"
