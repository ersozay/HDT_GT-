# Heyelan Dijital İkizi — v2: Değişiklikler, Sonuçlar ve Savunma Notları

Ekim 2026 · v1 (Temmuz 2026 teknik rapor) üzerine yapılan revizyon.

Bu belge, v1'de bir tez jürisinin sorgulayabileceği noktaları, bunların nasıl
giderildiğini ve yeni sayısal sonuçları özetler. Tüm sayılar
`outputs/results/` altındaki JSON/CSV dosyalarından gelir ve
`./run_pipeline.sh` ile yeniden üretilebilir.

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
