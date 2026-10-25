# Heyelan Dijital İkizi — Kahramanmaraş 6 Şubat 2023

Zeynep Ersöz · Trakya Üniversitesi

Bu proje, depremle tetiklenen heyelanlar için fizik kısıtlı hibrit bir dijital ikiz kurar ve şu soruyu yanıtlar: **bir yamaç, hangi yer ivmesinde (PGA) ve hangi zemin nemi koşulunda kaymaya başlar?** Bileşenleri:
- **Mohr-Coulomb + Newmark:** analitik formül ve gerçek AFAD ivme kayıtlarıyla zaman-tanım alanında çözüm
- **FEM:** Kayma Dayanımı Azaltma (SSR) yöntemi
- **PINN:** üç fizik yolu kısıtına sahip sinir ağı (PGA, eğim, doygunluk)
- **Envanter:** Kocaman ve ark. (2025), 2.596 heyelan

## Ana sonuçlar (mekânsal blok çapraz doğrulama)

| | Değer |
|---|---|
| ROC-AUC (PINN v2 + açık katmanlar) | **0,965 ± 0,013**; dik arazide 0,909 |
| Fiziksel tutarlılık ihlali | **%0–0,1** (kısıtsız RF / GBM / MLP: %18–73) |
| Heyelan lehine başlangıç eşiği (LR = 1) | **≈ 0,08 g** (varyantlar arası 0,04–0,09 g); envanterden ampirik değer 0,085 g |
| Envanter uydu doğrulaması (Sentinel-2, örneklem) | Kullanılabilir poligonların %53'ünde bağımsız heyelan izi (yöntem taban değeri ≈%10) |
| Kritik PGA (P = 0,5, en dik %5 yamaç) | 0,59 g (model aralığı 0,50–0,80 g); örnekleme tasarımına duyarlı (0,35–0,73 g), aralık olarak raporlanmalı |
| Yağış etkisi (6 Şubat 2023 zemin nemi) | Newmark AUC 0,62 → 0,70; tam doygunlukta Kritik PGA 0,59 → 0,43 g |
| Litoloji proxy'si çıkarıldığında | AUC 0,967 — kaba litoloji sonuçları belirlemiyor |

Ayrıntılar: [docs/V2_SONUCLAR.md](docs/V2_SONUCLAR.md) · Yayın hazırlığı ve lisanslar: [docs/YAYIN_HAZIRLIK.md](docs/YAYIN_HAZIRLIK.md) · Teknik rapor: `Heyelan_Dijital_Twin_Rapor.docx`

## Çalıştırma

```bash
pip install -r requirements.txt
./run_pipeline.sh          # tüm analiz (veri → model → Kritik PGA → simülasyon → rapor)
python3 -m pytest tests -q
./run_app.sh               # web uygulaması: http://localhost:5173
```

Uygulamanın sayfaları:
- **Tahmin:** zemin doygunluğu kaydırıcısıyla nokta tahmini
- **Kritik PGA:** kuru, 6 Şubat 2023 ve bugünkü nem senaryoları
- **Harita:** envanter, faylar, 2023 kırılması, ivme istasyonları, ek katmanlar ve deprem boyunca kayma animasyonu
- **Sonuçlar:** 16 modelli karşılaştırma ve tüm figürler

## Betikler

| # | Betik | Ne yapar |
|---|---|---|
| 01 | `prepare_data` | Envanterden örnek noktaları ve özniteliklerini üretir |
| 02 | `run_fem` | FEM-SSR pilot analizi |
| 03 | `train_pinn` | Nihai model, SHAP |
| 04/05 | Haritalar | Senaryo ve duyarlılık haritaları |
| 09 | `benchmark` | 16 model × 5 kat mekânsal CV |
| 10 | `critical_pga` | Belirsizlik, tasarım tablosu, ampirik doğrulama |
| 11 | `fetch_strong_motion` | AFAD ivme kayıtları (ESM) |
| 12 | `build_layers` | Açık kaynak katmanlar |
| 13 | `newmark_simulation` | Poligon bazında kayma simülasyonu |
| 14 | `sentinel_validation` | Sentinel-2 envanter doğrulaması (GEE yolu: `gee/ndvi_change.js`) |
| 15 | `frequency_ratio` | Frekans Oranı analizi |
| 16 | `label_noise` | Etiket kirliliği (PU) duyarlılığı |
| 17 | `rainfall_scenario` | Yağış senaryosu (ERA5-Land + boşluk suyu basıncı) |

## Atıf

[CITATION.cff](CITATION.cff) dosyasına bakın. Kullanılan verilerin kaynakları ve lisansları [docs/YAYIN_HAZIRLIK.md](docs/YAYIN_HAZIRLIK.md) içinde listelenir.
