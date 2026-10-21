# Heyelan Dijital İkizi — v2: Değişiklikler, Sonuçlar ve Savunma Notları

Ekim 2026 · v1 (Temmuz 2026 teknik rapor) üzerine yapılan revizyon.

Bu belge, v1'de bir tez jürisinin sorgulayabileceği noktaları, bunların nasıl
giderildiğini ve yeni sayısal sonuçları özetler. Tüm sayılar
`outputs/results/` altındaki JSON/CSV dosyalarından gelir ve
`./run_pipeline.sh` ile yeniden üretilebilir.


---

## 0. Güncelleme (19 Ekim 2026): açık kaynak katmanlar, gerçek ivme kayıtları, kayma simülasyonu

Bu bölüm aşağıdaki v2 sonuçlarının üzerine eklenen çalışmayı özetler. Nihai
model artık **PINN v2 + açık katmanlar** (`pinn_v2_L`, 54 öznitelik).

### 0.1 Yeni veri katmanları (tamamı açık kaynak, `scripts/12_build_layers.py`)

| Katman | Kaynak | Tek başına AUC | Not |
|---|---|---|---|
| Arazi pürüzlülük indeksi (TRI) | Copernicus DEM | **0,920** | eğimden (0,898) daha ayırt edici |
| Yerel rölyef (~1 km) | Copernicus DEM | 0,884 | |
| Vs30 | USGS ShakeMap | 0,847 | topoğrafyadan türetilmiş Vs30 |
| 2023 kırılmasına uzaklık | USGS rupture.json | 0,801 | heyelanlarda medyan 10,5 km — **modelde yok**, olaya özgü |
| Aktif faya uzaklık | GEM Global Active Faults | 0,765 | heyelanlarda medyan **7,15 km** — Kocaman (2025): ort. 7 km |
| TWI, dereye uzaklık | DEM + pysheds | 0,748 / 0,590 | |
| Arazi örtüsü | ESA WorldCover 2021 | 0,697 | tarım alanında heyelan oranı ≈ %0 |
| Kil / kum (15–30 cm) | ISRIC SoilGrids 2.0 | 0,596 / 0,637 | |
| PGV, MMI | USGS ShakeMap | 0,78 / 0,79 | **modelde yok**, olaya özgü |

Olaya özgü sarsıntı göstergeleri (PGV, MMI, 2023 kırılmasına uzaklık) bilinçli
olarak modele alınmadı: Kritik PGA taramasında PGA değişirken bunlar sabit
kalacağı için sarsıntı etkisini PGA'dan "çalarlardı". Haritada analiz katmanı
olarak sunulurlar.

Litoloji açık kaynaktan iyileştirilemedi: Macrostrat bu bölgede yalnız 1:5 M
harita sunuyor. Duyarlılık haritasındaki dikdörtgen bloklar bu ~5 km'lik proxy'nin
izidir; MTA (1:100 000 / 1:500 000) ya da Kocaman ve ark.'nın 27 birimli haritası
en büyük kalan iyileştirme fırsatıdır.

### 0.2 Karşılaştırma (mekânsal blok CV, 5 kat)

| Model | ROC-AUC | Dik arazi AUC | F1 | Brier | Fizik ihlali |
|---|---|---|---|---|---|
| PINN v2 | 0,947 ± 0,015 | 0,887 | 0,783 | 0,097 | %0,8 |
| **PINN v2 + katmanlar** | **0,965 ± 0,013** | **0,908** | **0,829** | **0,078** | **%0,0** |
| GBM + monoton + katmanlar | 0,966 ± 0,010 | 0,906 | 0,833 | 0,075 | %0,0 |
| GBM + katmanlar | 0,964 ± 0,010 | 0,903 | 0,828 | 0,079 | %31,6 |
| Random Forest + katmanlar | 0,966 ± 0,009 | 0,905 | 0,836 | 0,088 | %18,2 |
| Lojistik regresyon + katmanlar | 0,965 ± 0,011 | 0,903 | 0,832 | 0,068 | %0,0 |

Nihai model (ayrık test blokları): AUC 0,965, dik arazi 0,911, F1 0,837,
Kappa 0,777, Brier 0,072, ihlal %0,07.

### 0.3 Kritik PGA (nihai model)

| Tanım | Değer |
|---|---|
| P = 0,5 (6 model ortalama eğrisi) | **0,62 g** (model aralığı 0,49–0,75 g) |
| Başlangıç, LR = 1 | **0,093 g** |
| Ampirik başlangıç, gözlenen FR = 1 | **0,085 g** |
| Örnekleme oranı 1 / 1,5 / 2,5 → P = 0,5 | 0,45 / 0,45 / 0,74 g |

**En sağlam bulgu:** heyelan lehine başlangıç eşiği (LR = 1 ≈ 0,09 g) envanterden
bağımsız hesaplanan ampirik değerle (FR = 1 ≈ 0,085 g) örtüşüyor. P = 0,5
düzeyindeki değer ise hem modele (v2'de 0,38 g, katmanlı modelde 0,62 g) hem
örnekleme oranına duyarlı; tezde aralık olarak verilmeli.

### 0.4 Gerçek ivme kayıtları ve kayma simülasyonu

- **Kayıtlar** (`scripts/11_fetch_strong_motion.py`): ESM v3.0 (INGV/ORFEUS)
  üzerinden AFAD TK ağının işlenmiş (MP) kayıtları — Pazarcık 69, Elbistan 67
  istasyon. Token gerekmez. En büyük PGA 2,25 g (TK.4614 Pazarcık).
  Atıf: ESM https://doi.org/10.13127/esm.3 ; AFAD TK ağı https://doi.org/10.7914/SN/TK
- **Zaman-tanım Newmark** (`src/geomechanics/newmark_ts.py`, Jibson 1993):
  analitik dikdörtgen darbe çözümüyle binde 2 içinde doğrulandı.
- **Poligon simülasyonu** (`scripts/13_newmark_simulation.py`): her poligon
  için en yakın istasyon kaydı ShakeMap PGA'sına ölçeklenip iki depremle ayrı
  ayrı çözüldü. Poligonların %21'i 10 cm'yi aşıyor; %94'ünde Pazarcık depremi
  belirleyici (Kocaman ve ark.: heyelanların çoğu ilk depremle).
- **Kayıt vs regresyon:** heyelan/arka plan ayrımı AUC 0,697 (zaman-tanım) vs
  0,692 (Jibson 2007); heyelan noktalarında 10 cm'yi aşma oranı %22,3 vs %18,7,
  arka planda yanlış alarm %4,6 vs %5,9. Saf fizik tek başına zayıf kalıyor;
  bunun ana nedeni Kᵧ'nin litoloji proxy'sine dayanması.
- Uygulamada **Harita → Kayma Simülasyonu → Depremi oynat** ile poligonların
  deprem boyunca renk değiştirdiği animasyon izlenebilir.

### 0.5 Harita sayfası

Yeni katmanlar: Kocaman (2025) envanter poligonları, aktif faylar (GEM),
6 Şubat 2023 kırılma izleri (USGS), AFAD ivme istasyonları (ESM) ve seçilebilir
ek rasterlar (PGV, MMI, Vs30, kırılmaya/faya/dereye uzaklık, TRI, rölyef, TWI,
kil, arazi örtüsü) lejantlarıyla. Noktaya tıklayınca faya uzaklık, Vs30, PGV,
pürüzlülük ve kil oranı da gösterilir.

### 0.6 Veri bütünlüğü uyarısı

`~/Downloads/Kahramanmaras_2023_Landslides_QGIS.zip` içindeki GeoPackage'ın
**poligon geometrileri gerçek** (Kocaman envanteriyle birebir), ancak
**öznitelikleri uydurmadır**: eğim, yükseklik, PGA ve faya uzaklık değerlerinin
gerçek rasterlarla sıra korelasyonu ≈ 0 (0,03; −0,01; 0,01; 0,01). Litoloji,
heyelan türü, risk skoru sütunları da gerçek veriye dayanmıyor; "Fault_Lines"
katmanı üç noktalı şematik çizgilerdir. Bu dosya tezde veya uygulamada
**kullanılmamalı**. Pilot QGIS projesi (`ID1327_ID1353_Geotechnical_Pro.qgz`)
ise yalnızca referans verir; içindeki GeoPackage diskte bulunamadı.


### 0.8 Danışman önerisi: envanterin doğrulanması (üç adım)

**(a) Poligon gerçek mi? — Sentinel-2, 2022 yaz → 2023 yaz.**
Her poligonda ve 4.000 dik/tarım-dışı arka plan noktasında ΔNDVI ve Δparlaklık,
30–150 m'lik halkaya göre ölçülür. Eşikler arka plan dağılımının %5/%95
yüzdeliklerinden gelir; poligonlar "doğrulandı / sinyal yok / belirsiz"
(zaten çıplak ya da bulutlu) olarak sınıflanır. Saf gürültüyle yapılan kontrol
testinde beklendiği gibi yalnız ≈%10 "doğrulandı" çıkıyor.
Bu makinedeki bağlantı (≈4 KB/s) uzaktan COG okumaya yetmediği için hesap
**Google Earth Engine**'e taşındı: `gee/ndvi_change.js` (kurulum adımları dosyanın
başında), yükleme paketleri `gee/inventory_upload.zip`, `gee/negatives_upload.zip`
(`scripts/14` ile yeniden üretilebilir). Sonuç CSV'leri gelince:
`python3 scripts/14_sentinel_validation.py --from-gee outputs/gee`.
Bağlantı iyiyse doğrudan `python3 scripts/14_sentinel_validation.py` de çalışır.

**(b) Litoloji FR — yalnız PROXY.** `scripts/15_frequency_ratio.py`, Kocaman ve
ark. yöntemiyle (sınıftaki heyelan piksel oranı / sınıf alan oranı) FR üretir.
Litoloji FR'si Macrostrat proxy'siyle hesaplanır ve öyle etiketlenir; Kocaman
Tablo 4'teki MTA değerleri (peridotit 6,70; mermer-şist 6,36) bu proxy ile
üretilemez. Diğer faktörler (90 m, 2.596 poligon):

| Faktör | Sınıf | FR |
|---|---|---|
| Eğim | 0–5° / 15–20° / 30–35° / >40° | 0,06 / 2,16 / 9,07 / 35,9 |
| Faya uzaklık (GEM) | 0–2 / 2–5 / 20–40 / >40 km | 3,49 / 2,31 / 0,63 / 0,02 |
| 2023 kırılmasına uzaklık | 0–2 / 2–5 / 5–10 / >40 km | 6,32 / 5,12 / 3,46 / 0,17 |
| Yükseklik | 750–1000 / 1000–1250 / >2000 m | 2,31 / 3,41 / 0,30 |
| Arazi örtüsü | çalılık / ağaç / çıplak / tarım | 3,83 / 2,38 / 2,23 / 0,03 |
| Litoloji (PROXY) | granit / andezit / kil / silttaşı | 3,61 / 2,14 / 1,97 / 0,21 |

**(c) "Heyelan yok" kesin yokluk değil (PU öğrenme).** `scripts/16_label_noise.py`
dört eğitim varyantını aynı test bloklarında karşılaştırır (bkz. 0.9).
Ek referans: Elkan C. & Noto K. (2008). Learning classifiers from only positive and unlabeled data. KDD.

### 0.9 "Heyelan yok" etiketinin kirliliğine duyarlılık (`scripts/16_label_noise.py`)

Negatiflerin heyelanlara medyan uzaklığı 24,4 km; %2,1'i 1 km'den, %5,2'si 2 km'den yakın.
Aynı mekânsal test bloklarında (5 kat, tüm negatiflerle) dört eğitim varyantı:

| Eğitim varyantı | ROC-AUC | Dik arazi AUC | Kritik PGA (P = 0,5) | Başlangıç (LR = 1) | Elkan–Noto c |
|---|---|---|---|---|---|
| Temel (negatifler ≥ 200 m) | 0,965 ± 0,014 | 0,906 | 0,57 ± 0,07 g | 0,06 ± 0,03 g | 0,69 |
| Negatifler ≥ 1 km | 0,965 ± 0,013 | 0,908 | 0,52 ± 0,06 g | 0,06 ± 0,02 g | 0,70 |
| Negatifler ≥ 2 km | 0,965 ± 0,014 | 0,907 | 0,46 ± 0,05 g | 0,09 ± 0,02 g | 0,71 |
| PU-casus (%5 şüpheli negatif çıkarıldı) | 0,967 ± 0,011 | 0,915 | 0,35 ± 0,03 g | 0,04 ± 0,03 g | 0,80 |

**Yorum.** Ayırma gücü etiket kirliliğine dayanıklı; PU-casus temizliği dik arazide
küçük bir iyileşme getiriyor. Başlangıç eşiği (LR = 1 ≈ 0,04–0,09 g) tüm varyantlarda
aynı aralıkta ve ampirik FR = 1 değeriyle (0,085 g) uyumlu — tezin sağlam bulgusu.
P = 0,5 düzeyindeki Kritik PGA ise negatif kümesine göre 0,35–0,57 g arasında kayıyor;
örnekleme oranı duyarlılığıyla birlikte bu, o sayının tek değer olarak verilmemesi
gerektiğini gösteriyor.

Not: bu ortamda uzun işler makine uykuya geçince duruyor; `caffeinate -i -s python3 …`
ile çalıştırılmalı (betik kaldığı yerden devam eder; önbellek `outputs/cache/`).

### 0.7 Referanslar (eklenen)

- Kocaman S. ve ark. (2025). Landslides triggered by the 6 February 2023
  Kahramanmaraş earthquakes (Türkiye). Turkish J. Earth Sci. 34(1): 47–67.
  https://doi.org/10.55730/1300-0985.1945
- Li L. ve ark. (2026). Digital Twin Modeling for Landslide Risk Scenarios in
  Mountainous Regions. Sensors 26: 421. https://doi.org/10.3390/s26020421 —
  bizim FEM kesitimiz de onlarınki gibi H = 20 m; onların "topuktan 1/10
  yükseklikte duyarlı bölge" bulgusu FEM-SSR'deki topuk yoğunlaşmasıyla uyumlu.
- Jibson R.W. (1993). Transp. Res. Rec. 1411: 9–17.
- Allen T.I. & Wald D.J. (2009) / Wald & Allen (2007) — topoğrafik Vs30 (ShakeMap).
- Styron R. & Pagani M. (2020). The GEM Global Active Faults Database. Earthquake Spectra 36(S1).
- Zanaga D. ve ark. (2022). ESA WorldCover 10 m 2021 v200.
- Poggio L. ve ark. (2021). SoilGrids 2.0. SOIL 7: 217–240.
- Riley S.J. ve ark. (1999). Terrain ruggedness index. Intermountain J. Sci. 5.
- Zevenbergen L.W. & Thorne C.R. (1987). Earth Surf. Process. Landf. 12: 47–56.

---

## 1. v1'de tespit edilen sorunlar

| # | Sorun | Etkisi | v2 çözümü |
|---|---|---|---|
| 1 | Değerlendirme **rastgele poligon ayrımı** ile yapılıyordu. Koseismik heyelanlar kümelendiğinden test poligonunun komşusu eğitimdeydi. | AUC iyimser (0.963) | 0.25° **mekânsal blok CV** (5 kat, StratifiedGroupKFold) + blok-bazlı iç doğrulama |
| 2 | **Karşılaştırma (baseline) yoktu.** "Hibrit daha iyi" savı test edilmemişti. | Ana sav kanıtsız | 9 model: Newmark, LR, RF, GBM, monoton GBM, MLP, PINN v1, PINN v2, ablasyon |
| 3 | "PINN" kaybındaki fizik terimi yalnız *eğim < 10° ve PGA < 0.1 g* bölgesinde aktifti — bölgede neredeyse hiç tetiklenmiyordu. | Model pratikte düz MLP; **%60 örnekte PGA artınca olasılık düşüyordu** | **Fizik-yolu kolokasyon kısıtı** (§3) |
| 4 | `anneal_lambda` λ'yı kendi üzerine biriktiriyordu (ilk ~10 epoch'ta λ_max'a doyuyordu); erken durdurma büyüyen λ'yı içeren toplam kayıpla yapılıyordu. | Eğitim dinamiği belgelenenle uyumsuz | Doğrusal λ(t) = λ₀ + (λ_max − λ₀)·t; erken durdurma saf focal val kaybıyla |
| 5 | Kritik PGA taramasında (`05_susceptibility_map.py`) PGA değişiyor ama **Dₙ sabit kalıyordu**. | PGA–Dₙ fiziksel tutarsızlığı | Tarama her PGA'da Dₙ'yi Jibson (2007) ile yeniden hesaplar |
| 6 | Öznitelikler: `litho_id` sayısal (ordinal) verilmişti; `pore_pressure` hep 0; `dn_cm` 0–1000 aşırı çarpık. | Gereksiz/kötü koşullu girdiler | 37 öznitelik: one-hot litoloji, log₁₀(Dₙ+0.1), sabit sütun yok |
| 7 | Kritik PGA ≈ 0.55 g **tek sayı**, belirsizliksiz; P = 0.5 eşiğinin örnekleme oranına bağlılığı test edilmemişti. | Jüri sorusu: "neden 0.5?" | 6 modelli belirsizlik, örnekleme oranı duyarlılığı, ampirik doğrulama (§4) |
| 8 | **FEM analitik yedekti** (SfePy kurulu değil; konturlar temsiliydi). | Başlıktaki "FEM" bileşeni zayıf | Griffiths & Lane (1999) SSR çözücüsü yazıldı ve literatürle doğrulandı (§5) |
| 9 | `backend/metrics.json` elle yazılmıştı; `04_generate_maps.py` varsayılan olarak **sentetik** arazi kullanıyordu; `run_app.sh` backend'i yanlış portta (8000, proxy 8010) başlatıyordu. | Tutarsızlık / sentetik figür riski | metrics.json eğitimden otomatik yazılır; 04 gerçek veriyle; port düzeltildi |
| 10 | Test yoktu. | Kod güvenilirliği belgelenmemiş | 19 birim/kıyas testi (`tests/`) |

---

## 2. Karşılaştırmalı değerlendirme (mekânsal blok CV, 5 kat)

`scripts/09_benchmark.py` · `outputs/results/benchmark_summary.json`

| Model | ROC-AUC | Dik arazi AUC¹ | F1 | Brier | PGA-monotonluk ihlali² |
|---|---|---|---|---|---|
| Newmark (saf fizik, Dₙ) | 0.689 ± 0.098 | 0.661 | 0.500 | 0.245 | %0 |
| Lojistik regresyon | 0.934 ± 0.015 | 0.858 | 0.750 | 0.096 | %0.7 |
| Random Forest | 0.942 ± 0.018 | 0.870 | 0.789 | 0.104 | **%72.9** |
| Gradient Boosting | 0.946 ± 0.014 | 0.880 | 0.786 | 0.092 | **%56.1** |
| GBM + monoton kısıt | 0.951 ± 0.010 | 0.888 | 0.799 | 0.083 | %0 |
| MLP (fiziksiz) | 0.944 ± 0.013 | 0.884 | 0.771 | 0.101 | **%57.5** |
| PINN v1 | 0.947 ± 0.018 | 0.888 | 0.788 | 0.095 | **%59.9** |
| **PINN v2 (önerilen)** | **0.947 ± 0.016** | **0.887** | 0.783 | 0.097 | **%0.8** |
| PINN v2 – fizik girdisiz (ablasyon) | 0.947 ± 0.018 | 0.890 | 0.790 | 0.095 | %64.3 |

¹ Yalnız eğim ≥ 15° test örnekleri — rastgele arka plan negatiflerinin çoğu
düzlük olduğundan eğim tek başına AUC ≈ 0.90 verir; dik arazide ayırma gücü
modelin eğim dışında ne öğrendiğini gösterir.
² Test örneklerinde PGA 0.05 → 1.20 g taranır (Kᵧ sabit, Dₙ yeniden hesaplanır);
P'nin herhangi bir adımda 0.01'den fazla düştüğü örnek oranı.

**Referans:** Aynı PINN v2, v1'in rastgele-poligon protokolünde AUC **0.963**
verir. v1'in 0.963'ü modelden değil, iyimser protokolden gelir; mekânsal CV
ile gerçekçi değer **0.947 ± 0.016**'dır.

### Yorum (tezde kullanılabilir)

1. **Öğrenen modellerin ayırma gücü istatistiksel olarak ayırt edilemez**
   (farklar < 1 std). Saf fizik (Newmark) tek başına zayıftır (0.69).
2. **Ayırt edici olan fiziksel tutarlılıktır.** Kısıtsız RF, GBM, MLP ve
   PINN v1 örneklerin %56–73'ünde PGA arttıkça heyelan olasılığını *düşürür*.
   Bu modellerle yapılan bir Kritik PGA taraması fiziksel olarak anlamsızdır.
3. Fizik kısıtı **doğruluktan ödün vermeden** ihlali ~%0'a indirir
   (PINN v2: 0.947 AUC, %0.8 ihlal).
4. Ablasyon: Kᵧ ve Dₙ girdileri AUC'ye katkı vermez — eğim, litoloji ve PGA'nın
   deterministik fonksiyonları oldukları için ağ bunları zaten öğrenebilir.
   Fizik girdilerinin değeri doğruluk değil, **fizik kısıtını kurmayı ve
   yorumlanabilirliği** mümkün kılmalarıdır.
5. Monoton GBM, PINN v2 ile aynı düzeydedir. Bu tezin savını güçlendirir:
   **katkı, öğrenici türünden çok fizik kısıtının kendisindedir.** PINN'in
   ek avantajı, kısıtın doğrudan Mohr-Coulomb + Newmark zincirinden
   türetilmesi ve kolokasyon noktalarıyla gözlem aralığı dışına (yüksek PGA)
   uzatılabilmesidir.

### Nihai model (ayrık mekânsal test blokları)

`scripts/03_train_pinn.py` · ROC-AUC 0.943 · dik arazi AUC 0.887 · F1 0.787 ·
Kappa 0.706 · Brier 0.093 · ihlal %0.6 · karar eşiği 0.43 (val F1-maks).

SHAP (GradientExplainer, litoloji tek grup): **eğim > PGA > yükseklik >
litoloji > Kᵧ** > bakı > γ, φ′, c′ > log Dₙ.

---

## 3. Fizik-yolu kolokasyon kısıtı (PINN v2)

`src/pinn/physics_paths.py`, `src/pinn/loss.py`

    L = Focal + λ_s · L_stabilite + λ_m · mean( ReLU( P(x_a) − P(x_b) ) )

Her eğitim adımında kolokasyon çiftleri (x_a → x_b) üretilir:

- **PGA yolu:** PGA_b = PGA_a + U(0.02, 0.15) g, PGA_a ~ U(0.02, 1.30) g;
  Dₙ, Jibson (2007) ile yeniden hesaplanır.
- **Eğim yolu:** θ_b = θ_a + U(1°, 6°); FS, Kᵧ ve Dₙ Mohr-Coulomb sonsuz şev
  + Newmark ile yeniden hesaplanır.

Öznitelikler birbirinden bağımsız oynatılmaz — fizik zinciri korunur. Kısıt,
dağıtılacak deterministik ağ (dropout kapalı) üzerinde değerlendirilir.
λ_m = 12 (5 ile 12 arası duyarlılık denendi; 12'de ihlal ~%0, AUC değişmez).

Neden türev cezası değil? ∂P/∂x üzerinden autograd cezası denendi; BatchNorm
ve dropout eğitim modunda kısıtın "kandırılmasına" izin verdi, erken durdurma
1–2. epoch'ta takıldı ve AUC 0.92'ye düştü. Kolokasyon çifti yaklaşımı hem
daha kararlı hem de fiziksel olarak doğrudur (yönlü türev fizik yolu boyunca).

---

## 4. Kritik PGA

`scripts/10_critical_pga.py` · `outputs/results/critical_pga.json`

En dik %5 hücre (eğim ≥ 22.3°, n = 7 394; 0.008° ızgara):

| Tanım | Değer | Belirsizlik |
|---|---|---|
| **P = 0.5** (v1 tanımı), 6 modelin ortalama eğrisi | **0.38 g** | tekil modeller 0.26–0.48 g; hücre bootstrap %95: 0.37–0.39 g |
| **Başlangıç: LR = 1** (P = π = 0.286) | 0.06 g | modeller 0.02–0.10 g |
| **Ampirik başlangıç**: gözlenen Frekans Oranı = 1 | 0.085 g | doğrudan envanterden |
| Newmark rijit-blok (Dₙ ≥ 10 cm hücrelerin %50'si) | 1.29 g | — |

### Örnekleme oranı duyarlılığı (her oran için 3 tohum)

| Neg/Poz oranı | Kritik PGA (P = 0.5) | Başlangıç (LR = 1) | Test AUC |
|---|---|---|---|
| 1.0 | 0.28 ± 0.07 g | 0.29 g | 0.942 |
| 1.5 | 0.24 ± 0.05 g | 0.08 g | 0.940 |
| 2.5 (kullanılan) | 0.44 ± 0.11 g | 0.08 g | 0.943 |

Not: CPU'da çok iş parçacıklı eğitim tam deterministik değildir; tekrar
koşularda bu değerler ±0.05 g oynayabilir. Sonuç değişmez: oran 1–1.5 iken
≈ 0.24–0.28 g, oran 2.5 iken ≈ 0.44 g.

### Savunmada nasıl anlatılmalı

- P = 0.5 eşiği, olabilirlik oranı LR = N_neg/N_poz'a karşılık gelir; bu yüzden
  **Kritik PGA (P = 0.5) örnekleme tasarımına bağlıdır**. AUC değişmez. Bu, susceptibility literatüründe bilinen bir
  durumdur ve tezde açıkça belirtilmelidir (oran 1–1.5 → ≈ 0.25 g; 2.5 → ≈ 0.44 g).
- **Önerilen ifade:** "En dik %5 yamaçlarda heyelan olasılığı, PGA ≈ 0.38 g
  civarında (model belirsizliği 0.26–0.48 g) heyelan sınıfına geçer. Dik
  yamaçların arka plan araziye göre heyelana yatkın hale gelmesi (FR = 1)
  çok daha düşük, ≈ 0.06–0.09 g'de başlar; bu, envanterden bağımsız olarak
  hesaplanan ampirik değerle (0.085 g) uyumludur."
- v1'in 0.55 g değeri, tek bir modelin P = 0.5 geçişiydi ve Dₙ'nin taramada
  sabit tutulması hatasını içeriyordu.
- Newmark rijit-blok (1.29 g) çok muhafazakârdır: litoloji proxy'sinin
  dayanım parametreleri ve rijit-blok varsayımı tetiklenmeyi az tahmin eder.
  FEM (§5) bu farkın bir kısmını açıklar.

### Tasarım tablosu (eğim × litoloji, P = 0.5)

`outputs/results/critical_pga_table.csv` · Fiziksel olarak beklenen eğilim
görülür: eğim arttıkça Kritik PGA düşer (andezit 0.64 → 0.32 g, granit
0.41 → 0.22 g, kil 0.39 → 0.15 g, şist 0.70 → 0.44 g).

### Ampirik doğrulama

Envanter poligonları 90 m'de rasterlaştırılıp 0.008° ızgaraya alan oranı
olarak aktarıldı. Eğim × PGA sınıflarında gözlenen heyelan yoğunluğu ile
model olasılığı arasında **Spearman ρ = 0.88** (p ≈ 10⁻⁸, 24 sınıf).

---

## 5. FEM — Kayma Dayanımı Azaltma (SSR)

`src/fem/ssr_fem.py`, `scripts/02_run_fem.py` · `outputs/results/fem_pilots.json`

Griffiths & Lane (1999) elasto-viskoplastik algoritması: 8 düğümlü dörtgen
elemanlar, 2×2 indirgenmiş integrasyon, Mohr-Coulomb (ψ = 0), yerçekimi
tek adımda; FS = viskoplastik iterasyonun yakınsamadığı en küçük SRF.
Psödo-statik k_h ile K_y,FEM (SRF = 1'de göçme) hesaplanır.

**Doğrulama** (`tests/test_fem.py`):
- Mohr-Coulomb ölçütü üç eksenli basınçta kapalı-form çözümle birebir.
- Griffiths & Lane Örnek 1 (β = 26.57°, φ′ = 20°, c′/γH = 0.05): **FS = 1.35**
  (Bishop & Morgenstern 1.38, G&L FEM ≈ 1.4).
- Kohezyonsuz şev: FS_FEM → tanφ/tanθ.

**Pilot sonuçları** (H = 20 m, D = 1.5 idealize kesit):

| Pilot | FS_FEM | FS sonsuz şev | K_y,FEM | K_y sonsuz şev | PGA | Dₙ (K_y FEM / sonsuz) |
|---|---|---|---|---|---|---|
| ID 1327 kolüvyon, θ = 28°, c′ = 0 | 1.19 | 1.18 | 0.080 g | 0.082 g | 0.58 g | 20.0 / 19.0 cm → tetiklenir |
| ID 1353 marn, θ = 22°, c′ = 15 kPa | 1.91 | 1.60 | 0.279 g | 0.225 g | 0.45 g | 0.3 / 0.9 cm → tetiklenmez |

Kohezyonsuz yamaçta FEM ve sonsuz şev örtüşür (yüzeye paralel mekanizma).
Kohezyonlu yamaçta FEM daha derin dairesel bir mekanizma bulur ve K_y'yi
%24 daha yüksek verir. Plastik gerinme konturları topuktan geçen kayma yüzeyini
gösterir (`fem_1353_plastic_strain.png`).

---

## 6. Kalan kısıtlar (tezde belirtilmeli)

1. **Litoloji proxy'si** (Macrostrat, ~5 km çözünürlük, bölgede 12 sınıf).
   c′, φ′, γ literatür sınıf ortalamaları. Resmi MTA 1:25 000/1:100 000
   haritası en büyük iyileştirme fırsatıdır.
2. **Negatif örnekleme**: rastgele arka plan, envanterden 200 m tampon. Mutlak
   olasılık değil, göreli duyarlılık üretir (bkz. §4).
3. **FEM** 2B idealize kesittir; dinamik zaman-tanım analizi değil, psödo-statik.
   Gerçek pilot profili DEM'den çıkarılarak geliştirilebilir.
4. **Newmark** yalnız PGA tabanlı (Jibson 2007 Denk. 6); Arias şiddeti için
   istasyon kayıtları gerekir.
5. Tek deprem (Kahramanmaraş 2023). Wenchuan karşılaştırması yalnız fizik-transferidir;
   PINN başka bir depremde yeniden eğitilmedi.

---

## 7. Yeniden üretim

```bash
./run_pipeline.sh            # ~15 dk (Apple M-serisi, CPU)
python3 -m pytest tests -q   # 19 test
./run_app.sh                 # http://localhost:5173
```

v1 model ve figürleri `outputs/models_v1_baseline/` altında saklanır (git dışı).
