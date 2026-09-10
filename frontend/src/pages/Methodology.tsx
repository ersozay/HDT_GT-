import "katex/dist/katex.min.css";
import { BlockMath, InlineMath } from "react-katex";
import { ReactNode } from "react";
import { Card, CardTitle } from "../components/Card";
import PipelineDiagram from "../components/PipelineDiagram";

function Section({
  step,
  title,
  hint,
  children,
}: {
  step: string;
  title: string;
  hint?: string;
  children: ReactNode;
}) {
  return (
    <Card>
      <div className="mb-3 flex items-center gap-3">
        <span className="grid h-8 w-8 shrink-0 place-items-center rounded-lg bg-brand-600 text-sm font-bold text-white">
          {step}
        </span>
        <div>
          <h3 className="text-base font-semibold text-slate-900 dark:text-white">
            {title}
          </h3>
          {hint && (
            <p className="text-xs text-slate-500 dark:text-slate-400">{hint}</p>
          )}
        </div>
      </div>
      <div className="space-y-3 text-sm leading-relaxed text-slate-600 dark:text-slate-300">
        {children}
      </div>
    </Card>
  );
}

/** Denklem kutusu — koyu/açık temada okunur, ortalanmış. */
function Eq({ children, note }: { children: string; note?: string }) {
  return (
    <div className="my-2 overflow-x-auto rounded-lg border border-slate-200 bg-slate-50 px-4 py-3 dark:border-slate-700 dark:bg-slate-900/60">
      <BlockMath math={children} />
      {note && (
        <p className="mt-2 text-center text-xs text-slate-400 dark:text-slate-500">
          {note}
        </p>
      )}
    </div>
  );
}

function Sym({ children }: { children: string }) {
  return <InlineMath math={children} />;
}

export default function Methodology() {
  return (
    <div className="space-y-6">
      {/* Başlık */}
      <div>
        <h1 className="text-2xl font-bold tracking-tight text-slate-900 dark:text-white">
          Yöntem &amp; Denklemler
        </h1>
        <p className="mt-1 max-w-3xl text-sm text-slate-500 dark:text-slate-400">
          Sismik tetiklemeli heyelan duyarlılığı; klasik jeoteknik (Mohr–Coulomb
          limit denge + Newmark rijit-blok) ile fizik-bilgili sinir ağının (PINN)
          hibrit birleşimiyle hesaplanır. Aşağıda, bu uygulamayı besleyen{" "}
          <span className="font-medium text-slate-700 dark:text-slate-200">
            gerçek kaynak denklemleri
          </span>{" "}
          verilmiştir.
        </p>
      </div>

      {/* CBS–Newmark–PINN entegrasyon diyagramı (danışman isteği 1) */}
      <Card>
        <CardTitle hint="CBS raster → fizik (kᵧ, Dₙ) → PINN → kritik PGA → harita">
          Entegrasyon Akışı (CBS + Newmark + PINN)
        </CardTitle>
        <PipelineDiagram />
      </Card>

      {/* 1 — Statik güvenlik katsayısı */}
      <Section
        step="1"
        title="Statik Güvenlik Katsayısı (Mohr–Coulomb, sonsuz şev)"
        hint="src/geomechanics/mohr_coulomb.py · StaticFS"
      >
        <p>
          Sonsuz-şev limit denge modelinde, kayma yüzeyindeki direnç
          gerilmesinin kayma gerilmesine oranı güvenlik katsayısını (
          <Sym>{"FS"}</Sym>) verir:
        </p>
        <Eq note="c′ efektif kohezyon, φ′ efektif içsel sürtünme açısı, γ birim hacim ağırlık, z derinlik, θ şev eğimi, u boşluk suyu basıncı">
          {String.raw`FS = \frac{c' + \left(\gamma z\cos^{2}\theta - u\right)\tan\phi'}{\gamma z\,\sin\theta\,\cos\theta}`}
        </Eq>
        <p>
          <Sym>{"FS>1"}</Sym> stabil, <Sym>{"FS\\le 1"}</Sym> ise statik
          koşulda dahi kararsız şevi gösterir. Uygulamadaki{" "}
          <em>“statik sınırda”</em> uyarısı bu koşuldan üretilir.
        </p>
      </Section>

      {/* 2 — Dinamik / sözde-statik */}
      <Section
        step="2"
        title="Dinamik (Sözde-Statik) Güvenlik Katsayısı"
        hint="src/geomechanics/mohr_coulomb.py · DynamicFS"
      >
        <p>
          Deprem ivmesi, yatay sismik katsayı{" "}
          <Sym>{"k_h = a_{max}/g = \\text{PGA}/g"}</Sym> ile sözde-statik ek
          kuvvet olarak eklenir:
        </p>
        <Eq note="Payda büyüdükçe FS düşer → deprem şevi zayıflatır">
          {String.raw`FS_{din} = \frac{c + \left(\gamma z\cos^{2}\theta - u\right)\tan\phi}{\gamma z\,\sin\theta\,\cos\theta + k_h\,\gamma z\cos^{2}\theta}`}
        </Eq>
      </Section>

      {/* 3 — Kritik ivme (akma) */}
      <Section
        step="3"
        title="Kritik (Akma) İvmesi kᵧ"
        hint="src/geomechanics/mohr_coulomb.py · CriticalAcceleration"
      >
        <p>
          Şevin kaymaya başladığı eşik ivme (yield acceleration),{" "}
          <Sym>{"FS_{din}=1"}</Sym> koşulundan türetilir:
        </p>
        <Eq note="ky negatif çıkarsa 0’a kırpılır (statikçe zaten kararsız şev)">
          {String.raw`k_y = \left(FS_{statik} - 1\right)\,\sin\theta \quad,\quad k_y \ge 0`}
        </Eq>
        <p>
          <Sym>{"k_y"}</Sym> (birimi <Sym>{"g"}</Sym>), bir şevin depreme
          dayanımının doğrudan ölçüsüdür; Newmark deplasman entegralinin girdisidir.
        </p>
      </Section>

      {/* 4 — Newmark deplasmanı */}
      <Section
        step="4"
        title="Newmark Kalıcı Deplasmanı (Jibson 2007, Denk. 6)"
        hint="src/geomechanics/newmark.py · NewmarkDisplacement"
      >
        <p>
          Rijit-blok modelinde, sarsıntı boyunca <Sym>{"a_{max}>k_y"}</Sym>{" "}
          olan anlarda biriken kalıcı yer değiştirme <Sym>{"D_n"}</Sym>,
          Jibson (2007) regresyonuyla kestirilir:
        </p>
        <Eq note="Dn santimetre; oran r = ky/amax sayısal olarak [0.01, 1) aralığına kırpılır">
          {String.raw`\log_{10}(D_n) = 0.215 + \log_{10}\!\left[\left(1-\frac{k_y}{a_{max}}\right)^{2.341}\left(\frac{k_y}{a_{max}}\right)^{-1.438}\right]`}
        </Eq>
        <p>
          <Sym>{"a_{max}\\le k_y"}</Sym> ise deplasman sıfırdır (tetiklenme yok).
        </p>
      </Section>

      {/* 4a — Ambraseys & Menu (1988) */}
      <Section
        step="4a"
        title="Alternatif Newmark Bağıntısı (Ambraseys & Menu 1988)"
        hint="src/geomechanics/newmark.py · method='ambraseys1988'"
      >
        <p>
          Aynı akma-ivmesi oranını (<Sym>{"r = k_y/a_{max}"}</Sym>) kullanan,
          bağımsız bir ampirik regresyon. İki bağıntının farkı, rijit-blok
          kestiriminin <em>tahmin bandını</em> (belirsizliğini) gösterir:
        </p>
        <Eq note="Dn santimetre; Ambraseys & Menu (1988) tipik olarak Jibson'dan biraz daha büyük Dn verir">
          {String.raw`\log_{10}(D_n) = 0.90 + \log_{10}\!\left[\left(1-r\right)^{2.53}\,r^{-1.09}\right]`}
        </Eq>
        <p>
          Uygulamada her iki eğri <span className="font-medium">Kritik PGA</span>{" "}
          sayfasında birlikte çizilir; danışmanın önerisiyle eklenmiştir.
        </p>
      </Section>

      {/* 4b — Arias yoğunluğu tabanlı Newmark */}
      <Section
        step="4b"
        title="Arias Yoğunluğu Tabanlı Newmark (Jibson 2007, Denk. 9)"
        hint="src/geomechanics/newmark.py · NewmarkDisplacementIa"
      >
        <p>
          Arias yoğunluğu <Sym>{"I_a"}</Sym>, yer hareketinin toplam enerji
          içeriğini (süre + genlik) tek bir ölçüde birleştirir:
        </p>
        <Eq note="a(t) yer ivmesi zaman-serisi, Td sarsıntı süresi; birim m/s">
          {String.raw`I_a = \frac{\pi}{2g}\int_{0}^{T_d} a(t)^{2}\,dt`}
        </Eq>
        <p>
          Jibson (2007), <Sym>{"D_n"}</Sym>'i hem <Sym>{"I_a"}</Sym> hem de akma
          ivmesi cinsinden verir — PGA yerine enerji-tabanlı bir kestirim sağlar:
        </p>
        <Eq note="Ia m/s, ky g cinsinden; ID 1327 & 1353 için Ia–Dn eğrileri Kritik PGA figüründe">
          {String.raw`\log_{10}(D_n) = 0.561\,\log_{10} I_a - 3.833\,\log_{10} k_y - 1.474`}
        </Eq>
        <p className="text-xs text-slate-400 dark:text-slate-500">
          Not: Uygulamadaki <Sym>{"I_a"}</Sym> eğrileri bir{" "}
          <em>senaryo taramasıdır</em> (Kahramanmaraş kaydının gerçek zaman-serisi
          entegrali değildir); enerji-duyarlılık davranışını göstermek içindir.
        </p>
      </Section>

      {/* 5 — Heyelan etiketi */}
      <Section
        step="5"
        title="Heyelan Etiketi (Eğitim Hedefi)"
        hint="src/geomechanics/newmark.py · LandslideLabel"
      >
        <p>
          İkili sınıf etiketi, kritik deplasman eşiği{" "}
          <Sym>{"D_c = 0.10\\,\\text{m}"}</Sym> (10 cm, Jibson 2007) ile üretilir:
        </p>
        <Eq>
          {String.raw`y = \begin{cases} 1 & D_n \ge 0.10\ \text{m} \\ 0 & D_n < 0.10\ \text{m}\end{cases}`}
        </Eq>
        <p>
          Bu fiziksel etiket, aşağıdaki PINN’in denetimli öğrenme hedefidir —
          yani ağ, salt istatistiği değil <em>mekaniği</em> öğrenir.
        </p>
      </Section>

      {/* 6 — PINN */}
      <Section
        step="6"
        title="Fizik-Bilgili Sinir Ağı (PINN)"
        hint="src/pinn/model.py · LandslidePINN"
      >
        <p>
          39 öznitelik (eğim, yükseklik, bakı, litoloji parametreleri,{" "}
          <Sym>{"k_y"}</Sym>, PGA…) girişli, Swish aktivasyonlu ileri-beslemeli ağ:
        </p>
        <Eq note="Katman genişlikleri; çıkış sigmoid ile P(heyelan) ∈ (0,1)">
          {String.raw`39 \rightarrow 256 \rightarrow 128 \rightarrow 64 \rightarrow 32 \rightarrow 1`}
        </Eq>
        <p>
          Kayıp fonksiyonu, dengesiz sınıflar için <strong>Focal Loss</strong>{" "}
          ile mekaniğe aykırı tahminleri cezalandıran bir{" "}
          <strong>fizik cezası</strong> teriminin toplamıdır:
        </p>
        <Eq note="α=0.5 (azınlık heyelan sınıfı için), γ=2 odak parametresi; λ fizik ağırlığı">
          {String.raw`\mathcal{L} = \underbrace{-\alpha\,(1-p_t)^{\gamma}\log p_t}_{\text{Focal}} + \lambda\,\underbrace{\mathcal{L}_{\text{fizik}}}_{k_y\text{–PGA tutarlılığı}}`}
        </Eq>
        <p>
          Karar eşiği, doğrulama kümesinde <Sym>{"F_1"}</Sym> maksimize edilerek
          seçilir (<Sym>{"\\approx 0.49"}</Sym>).
        </p>
      </Section>

      {/* 7 — Kritik PGA çıkarımı */}
      <Section
        step="7"
        title="Kritik PGA Çıkarımı"
        hint="Uygulamanın çekirdek sorusu"
      >
        <p>
          Sabit bir şev için PGA taranır; modelin heyelan olasılığının eşiği ilk
          aştığı değer <strong>kritik PGA (ML)</strong> olarak alınır:
        </p>
        <Eq>
          {String.raw`\text{PGA}_{krit} = \min\{\,\text{PGA} : P_{\theta}(\text{heyelan}\mid \text{PGA}) \ge \tau\,\}`}
        </Eq>
        <p>
          Paralelde, aynı şev için Newmark <Sym>{"D_n=10\\,\\text{cm}"}</Sym>{" "}
          eşiğini veren PGA, fiziksel karşılaştırma eğrisi olarak çizilir
          (bkz. <span className="font-medium">Kritik PGA</span> sayfası).
        </p>
      </Section>

      {/* Dürüst kısıtlar */}
      <Card className="border-amber-300/60 bg-amber-50 dark:border-amber-500/30 dark:bg-amber-500/10">
        <CardTitle hint="Bilimsel şeffaflık — tez savunmasında açıkça belirtilmelidir">
          Model Kısıtları
        </CardTitle>
        <ul className="list-disc space-y-1.5 pl-5 text-sm text-amber-900 dark:text-amber-200">
          <li>
            <strong>Litoloji bir proxy’dir</strong> (Macrostrat, CC-BY) — resmî
            MTA jeoloji haritası değildir. Kohezyon/sürtünme değerleri sınıf-ortalamasıdır.
          </li>
          <li>
            <strong>FEM analitik yedektir</strong>: SfePy kurulu olmadığından
            gerilme alanı, Mohr–Coulomb + Newmark ile analitik hesaplanır;
            von-Mises/plastik konturlar sentetiktir.
          </li>
          <li>
            Newmark <strong>rijit-blok</strong> varsayar (iç deformasyon yok) —
            ML’den daha muhafazakârdır; iki eğrinin farkı bu yüzden anlamlıdır.
          </li>
        </ul>
      </Card>

      {/* Literatür konumlandırma */}
      <Card>
        <CardTitle hint="Bu tezin yayınlanmış güncel çalışmalar içindeki yeri">
          Literatürdeki Konum
        </CardTitle>
        <div className="space-y-3 text-sm leading-relaxed text-slate-600 dark:text-slate-300">
          <p>
            Bu çalışma, aynı 2023 Kahramanmaraş depremi için yürütülen{" "}
            <strong className="text-slate-700 dark:text-slate-200">
              Wang vd. (2026, Nature — Communications Earth &amp; Environment)
            </strong>{" "}
            çalışmasıyla <em>aynı fiziksel temeli</em> (Mohr–Coulomb) ve{" "}
            <em>aynı USGS bileşik PGA zarfını</em> (onların maks. ≈ 1.27 g; bizde 1.31 g)
            paylaşır. Wang vd. şev dayanımı değişimini <strong>InSAR</strong> yüzey
            deformasyonu tersinimi ile kestirir; bu tez ise şev dayanımını doğrudan
            <strong> kritik PGA</strong>'ya çeviren bir <strong>PINN + Newmark</strong>{" "}
            hibridine odaklanır. İki yaklaşım birbirini tamamlar.
          </p>
          <p>
            Kavramsal <strong>Dijital İkiz</strong> çerçevesi Brocca vd. (2026,
            Frontiers in Science) ile uyumludur; problem tanımı ise{" "}
            <strong>AFAD-UDAP 2026</strong> öncelikli araştırma temalarıyla (deprem,
            heyelan, dijital ikiz, makine öğrenmesi, duyarlılık haritalama, PGA)
            doğrudan örtüşür.
          </p>
        </div>
      </Card>

      {/* Referanslar */}
      <Card>
        <CardTitle>Temel Referanslar</CardTitle>
        <ul className="space-y-1.5 text-sm text-slate-600 dark:text-slate-300">
          <li>
            Wang, Y., Dahal, A., van Westen, C.J., Görüm, T. &amp; Tanyaş, H.
            (2026). <em>New and existing settlements built on hillslopes weakened
            by the 2023 Kahramanmaraş earthquake sequence.</em> Communications
            Earth &amp; Environment 7:521.
          </li>
          <li>
            Brocca, L. <em>et al.</em> (2026). <em>A Digital Twin of the
            terrestrial water cycle.</em> Frontiers in Science.
          </li>
          <li>
            Jibson, R.W. (2007). <em>Regression models for estimating
            coseismic landslide displacement.</em> Engineering Geology 91(2–4).
          </li>
          <li>
            Ambraseys, N.N. &amp; Menu, J.M. (1988). <em>Earthquake-induced
            ground displacements.</em> Earthquake Eng. Struct. Dyn. 16(7).
          </li>
          <li>
            Newmark, N.M. (1965). <em>Effects of earthquakes on dams and
            embankments.</em> Géotechnique 15(2).
          </li>
          <li>
            Raissi, Perdikaris &amp; Karniadakis (2019).{" "}
            <em>Physics-informed neural networks.</em> J. Comput. Phys. 378.
          </li>
          <li className="text-xs text-slate-400 dark:text-slate-500">
            AFAD-UDAP 2026 Çağrı Metni — Ulusal Deprem Araştırma Programı
            (öncelikli temalar: deprem, heyelan, dijital ikiz, makine öğrenmesi, PGA).
          </li>
        </ul>
      </Card>
    </div>
  );
}
