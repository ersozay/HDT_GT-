# Yayına Hazırlık — Veri, Lisans, Yeniden Üretilebilirlik ve Açık Kalanlar

Bu belge, tez ve olası bir makale için hakemlerin soracağı "veri nereden, hangi
lisansla, nasıl yeniden üretilir, sınırlar neler" sorularını tek yerde yanıtlar.

## 1. Veri kaynakları ve lisansları

| Veri | Kaynak | Lisans / kullanım koşulu | Depoda mı? |
|---|---|---|---|
| Heyelan envanteri (2.596 poligon) | Kocaman ve ark. (2025), TJES 34(1), doi:10.55730/1300-0985.1945; ek veri doi:10.48623/aperta.274144 | CC BY 4.0 (makale) — atıf | Hayır (`data/raw`, indirilebilir) |
| PGA, PGV, MMI, Vs30, kırılma modeli | USGS ShakeMap us6000jllz, us6000jlqa | ABD kamu malı | Hayır (script 00 / 12) |
| DEM (≈90 m) | Copernicus DEM GLO-90 | Copernicus açık lisansı — atıf | Hayır |
| Aktif faylar | GEM Global Active Faults (Styron & Pagani 2020) | CC BY-SA 4.0 | Hayır (script 12) |
| Arazi örtüsü | ESA WorldCover 2021 v200 | CC BY 4.0 | Hayır (script 12) |
| Kil / kum | ISRIC SoilGrids 2.0 | CC BY 4.0 | Hayır (script 12) |
| Litoloji (proxy) | Macrostrat API (Peters ve ark. 2018) | CC BY 4.0 | Hayır |
| İvme kayıtları | AFAD TK ağı (doi:10.7914/SN/TK), ESM v3.0 (doi:10.13127/esm.3) | Ağ varsayılan lisansı (D) — atıf | Hayır (script 11) |
| Toprak nemi, yağış | ERA5-Land (Muñoz-Sabater ve ark. 2021), Open-Meteo API | Copernicus lisansı / CC BY 4.0 | Hayır (script 17) |
| Sentinel-2 L2A | ESA Copernicus (AWS / Google Earth Engine) | Copernicus açık lisansı | Hayır |
| **MTA jeoloji haritası** | MTA Genel Müdürlüğü | **Ücretli, yazılı izin gerekir** — kullanılmadı; resmi talep taslağı hazır | — |

**Bilinçli olarak kullanılmayanlar:** MTA Yerbilimleri Harita Görüntüleyicisi'nden
kazıma (web scraping) — MTA koşulları yazılı izinsiz çoğaltmayı yasaklar; bu, yayın
engeli yaratırdı. `~/Downloads/Kahramanmaras_2023_Landslides_QGIS.zip` içindeki
öznitelikler — gerçek verilerle korelasyonu ≈ 0 (uydurma), bkz. V2_SONUCLAR §0.6.

## 2. Veri ve kod erişilebilirlik beyanı (makaleye eklenecek metin)

> Bu çalışmanın kodu ve eğitilmiş modelleri https://github.com/ersozay/HDT_GT- adresinde
> MIT lisansıyla açıktır. Kullanılan tüm girdi verileri açık kaynaklardan
> (USGS ShakeMap, Copernicus DEM, ESA WorldCover, ISRIC SoilGrids, GEM Global Active
> Faults, ESM/AFAD ivme kayıtları, ERA5-Land, Macrostrat) ve Kocaman ve ark. (2025)
> envanterinden `run_pipeline.sh` ile yeniden üretilir. Ara ürünler (sonuç tabloları,
> figürler, model ağırlıkları) depodaki `outputs/` klasöründedir.

## 3. Yeniden üretim

```bash
pip install -r requirements.txt
./run_pipeline.sh            # Uzun adımlarda makinenin uykuya geçmesini caffeinate engeller
python3 -m pytest tests -q   # fizik, FEM kıyas problemleri, mekânsal CV, Newmark analitik testleri
```

Tohumlar sabittir; CPU'da çok iş parçacıklı eğitim tam deterministik olmadığından
sonuçlar ±0,005 AUC ve ±0,05 g Kritik PGA içinde değişebilir.

## 4. Hakem sorularına hazır yanıtlar

| Olası soru | Yanıt / kanıt |
|---|---|
| Uzamsal sızıntı var mı? | 0,25° blok CV; rastgele poligon ayrımı 0,963'ü 0,947'ye düşürüyor (V2 §2) |
| Basit modellerden iyi mi? | 16 modelli karşılaştırma; doğrulukta eşdeğer, fiziksel tutarlılıkta üstün |
| "PINN" gerçekten fizik içeriyor mu? | 3 fizik yolu (PGA, eğim, doygunluk) üzerinde kolokasyon kısıtı; ihlal %0–0,1 |
| Litoloji proxy'si sonuçları belirliyor mu? | **Hayır:** litolojisiz ablasyon AUC 0,967 vs 0,965 (V2 §0.11) |
| Envanter eksik/hatalı mı? | Bağımsız yeniden hesaplama (V2 §0.8); etiket kirliliği duyarlılığı (V2 §0.9); Sentinel-2 hattı |
| Kritik PGA neden 0,5'te? | P = 0,5 örneklemeye bağlı; asıl sağlam bulgu LR = 1 başlangıcı (≈0,04–0,09 g), ampirik FR = 0,085 g |
| Yağış etkisi? | ERA5-Land olay nemiyle Newmark AUC 0,62 → 0,70; senaryo m = 0 → 1 ile Kritik PGA 0,59 → 0,43 g (V2 §0.10) |
| FEM doğrulandı mı? | Griffiths & Lane Örnek 1: FS 1,35 (B&M 1,38); kohezyonsuz şevde tanφ/tanθ |
| Newmark zaman-tanım doğru mu? | Dikdörtgen darbe analitik çözümüyle ‰2 içinde (test) |

## 5. Açık kalanlar (yayın engeli değil, "gelecek çalışma" / ek)

1. **MTA 1:100 000 litoloji** — resmi talep yazısı hazır. Ablasyon gösterdiği için
   ana sonuçları değiştirmesi beklenmiyor; yalnız litoloji bazlı tasarım tablosu ve
   Newmark (saf fizik) Kᵧ'si iyileşir. O zamana kadar tasarım tablosu "proxy, gösterge" diye verilir.
2. **Sentinel-2 poligon doğrulaması** — kod ve GEE betiği hazır; bu makinedeki bağlantı
   uzaktan okumaya yetmediği için tüm envanter Google Earth Engine'de koşturulmalı
   (`gee/ndvi_change.js`, ~10 dk). Sonuç `--from-gee` ile içe alınır.
3. **Tek deprem** — model başka bir olayda yeniden eğitilmedi (Wenchuan yalnız fizik-transferi).
4. **FEM** 2B idealize kesit ve psödo-statik yük; gerçek profil ve dinamik analiz gelecek iş.
