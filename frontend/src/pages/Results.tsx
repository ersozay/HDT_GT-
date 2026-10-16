import { useEffect, useState } from "react";
import { api, Benchmark, CriticalPgaReport, FemPilot, Summary } from "../api/client";
import { Card, CardTitle } from "../components/Card";
import MetricCard from "../components/MetricCard";
import Spinner from "../components/Spinner";

const FIGURES = [
  { key: "benchmark_auc", label: "Model Karşılaştırması", desc: "Mekânsal blok CV — ROC-AUC (tüm test ve yalnız dik arazi), 5 kat ort. ± std" },
  { key: "benchmark_consistency", label: "Doğruluk × Fizik", desc: "AUC ile PGA-monotonluk ihlali düzlemi — sol üst en iyi" },
  { key: "benchmark_pga_response", label: "PGA Tepkisi", desc: "Kısıtsız modellerde PGA artınca olasılık düşebiliyor; PINN v2 fizikle tutarlı" },
  { key: "critical_pga_uncertainty", label: "Kritik PGA", desc: "6 model (nihai + 5 CV katı) ortalama eğrisi, belirsizlik bandı ve Newmark referansı" },
  { key: "critical_pga_design_table", label: "Tasarım Tablosu", desc: "Eğim sınıfı × litoloji için Kritik PGA" },
  { key: "empirical_validation", label: "Ampirik Doğrulama", desc: "Gözlenen Frekans Oranı ile model olabilirlik oranı, aynı ızgarada" },
  { key: "shap_summary", label: "SHAP Önem", desc: "Özniteliklerin model kararına katkısı (litoloji tek grup)" },
  { key: "training_history", label: "Eğitim Geçmişi", desc: "Kayıp, doğruluk ve fizik ceza bileşenleri" },
  { key: "confusion_matrix", label: "Karışıklık Matrisi", desc: "Mekânsal test bloklarında sınıflandırma" },
  { key: "susceptibility_map_real", label: "Duyarlılık Haritası", desc: "Bölgesel PINN v2 çıktısı" },
  { key: "fem_1353_plastic_strain", label: "FEM · 1353 kayma yüzeyi", desc: "FEM-SSR birikmiş plastik kayma gerinmesi" },
  { key: "fem_1353_displacement", label: "FEM · 1353 mekanizma", desc: "Göçme anındaki artımsal yer değiştirme" },
  { key: "fem_1327_plastic_strain", label: "FEM · 1327 kayma yüzeyi", desc: "Kohezyonsuz kolüvyonda yüzeysel kayma" },
  { key: "pga_fs_curve_ID1353", label: "FS–kₕ · 1353", desc: "Psödo-statik FS ve FEM kritik ivmesi" },
];

const ORDER = ["pinn_v2", "gbm_mono", "pinn_v1", "mlp", "gbm", "rf", "logreg", "newmark", "pinn_v2_nophysfeat"];
const f3 = (v?: number | null) => (v == null ? "—" : v.toFixed(3));
const f2 = (v?: number | null) => (v == null ? "—" : v.toFixed(2));

export default function Results() {
  const [s, setS] = useState<Summary | null>(null);
  const [b, setB] = useState<Benchmark | null>(null);
  const [c, setC] = useState<CriticalPgaReport | null>(null);
  const [fem, setFem] = useState<FemPilot[] | null>(null);
  const [tab, setTab] = useState(FIGURES[0].key);

  useEffect(() => {
    api.summary().then(setS).catch(() => {});
    api.benchmark().then(setB).catch(() => {});
    api.criticalReport().then(setC).catch(() => {});
    api.fem().then(setFem).catch(() => {});
  }, []);

  const active = FIGURES.find((f) => f.key === tab)!;
  const cp = c?.critical_pga;
  const cv = s?.cv;

  return (
    <div className="space-y-6">
      <div>
        <h1 className="text-2xl font-bold tracking-tight text-slate-900 dark:text-white">
          Model Sonuçları ve Değerlendirme
        </h1>
        <p className="mt-1 text-sm text-slate-500 dark:text-slate-400">
          Mekânsal blok (0.25° ≈ 25 km) ayrımı: test blokları coğrafi olarak eğitimden ayrıktır;
          komşu heyelanlardan sızıntı yoktur. Karar eşiği yalnız doğrulama bloklarında seçilir.
        </p>
      </div>

      {s ? (
        <div className="grid gap-4 sm:grid-cols-2 lg:grid-cols-3">
          <MetricCard
            label="ROC–AUC (CV)"
            value={cv?.roc_auc ? cv.roc_auc.mean.toFixed(3) : s.roc_auc.toFixed(3)}
            accent="brand"
            sub={cv?.roc_auc ? `± ${cv.roc_auc.std.toFixed(3)} · 5 mekânsal kat · ayrık test ${s.roc_auc.toFixed(3)}` : "Ayrık test blokları"}
          />
          <MetricCard
            label="Dik arazide AUC"
            value={(cv?.roc_auc_steep?.mean ?? s.roc_auc_steep ?? 0).toFixed(3)}
            accent="sky"
            sub="Yalnız eğim ≥ 15° — eğim dışı ayırma gücü"
          />
          <MetricCard
            label="Fizik ihlali"
            value={`${(100 * (cv?.mono_violation_rate?.mean ?? s.mono_violation_rate ?? 0)).toFixed(1)}%`}
            accent="emerald"
            sub="PGA artınca P düşen örnek oranı (kısıtsız MLP ≈ %57)"
          />
          <MetricCard label="F1 Skoru" value={s.f1.toFixed(2)} accent="amber" sub={`Recall ${(s.recall * 100).toFixed(0)}% · Precision ${(s.precision * 100).toFixed(0)}%`} />
          <MetricCard label="Kappa" value={(s.kappa ?? 0).toFixed(2)} accent="violet" sub={`Brier ${(s as Summary & { brier?: number }).brier?.toFixed(3) ?? "—"}`} />
          <MetricCard
            label="Kritik PGA"
            value={f2(cp?.critical_pga_g ?? s.critical_pga_ml)}
            unit="g"
            accent="rose"
            sub={cp?.ensemble_range_g ? `6 model aralığı ${f2(cp.ensemble_range_g[0])}–${f2(cp.ensemble_range_g[1])} g` : "ML tetiklenme eşiği"}
          />
        </div>
      ) : (
        <Spinner />
      )}

      {/* Model karşılaştırması */}
      <Card>
        <CardTitle hint="5 katlı mekânsal blok CV, ortalama ± std">Karşılaştırmalı Değerlendirme</CardTitle>
        {b ? (
          <div className="overflow-x-auto">
            <table className="w-full min-w-[720px] text-sm">
              <thead>
                <tr className="border-b border-slate-200 text-left text-xs uppercase tracking-wide text-slate-500 dark:border-slate-800 dark:text-slate-400">
                  <th className="py-2 pr-3">Model</th>
                  <th className="py-2 pr-3 text-right">ROC-AUC</th>
                  <th className="py-2 pr-3 text-right">Dik arazi AUC</th>
                  <th className="py-2 pr-3 text-right">PR-AUC</th>
                  <th className="py-2 pr-3 text-right">F1</th>
                  <th className="py-2 pr-3 text-right">Brier</th>
                  <th className="py-2 text-right">PGA-monotonluk ihlali</th>
                </tr>
              </thead>
              <tbody>
                {ORDER.filter((k) => b.models[k]).map((k) => {
                  const m = b.models[k];
                  const hi = k === "pinn_v2";
                  const viol = m.mono_violation_rate?.mean;
                  return (
                    <tr
                      key={k}
                      className={`border-b border-slate-100 dark:border-slate-800/60 ${hi ? "bg-brand-50 font-semibold dark:bg-brand-900/20" : ""}`}
                    >
                      <td className="py-2 pr-3 text-slate-800 dark:text-slate-100">{m.label}</td>
                      <td className="py-2 pr-3 text-right tabular-nums">
                        {f3(m.roc_auc?.mean)} <span className="text-xs text-slate-400">± {f3(m.roc_auc?.std)}</span>
                      </td>
                      <td className="py-2 pr-3 text-right tabular-nums">{f3(m.roc_auc_steep?.mean)}</td>
                      <td className="py-2 pr-3 text-right tabular-nums">{f3(m.pr_auc?.mean)}</td>
                      <td className="py-2 pr-3 text-right tabular-nums">{f3(m.f1?.mean)}</td>
                      <td className="py-2 pr-3 text-right tabular-nums">{f3(m.brier?.mean)}</td>
                      <td className={`py-2 text-right tabular-nums ${viol != null && viol > 0.1 ? "text-rose-600 dark:text-rose-300" : "text-emerald-700 dark:text-emerald-300"}`}>
                        {viol == null ? "—" : `${(100 * viol).toFixed(1)}%`}
                      </td>
                    </tr>
                  );
                })}
              </tbody>
            </table>
            <p className="mt-3 text-xs leading-relaxed text-slate-500 dark:text-slate-400">
              Öğrenen modeller benzer ayırma gücüne ulaşır (fark &lt; 1 std); ayırt edici olan fiziksel
              tutarlılıktır. Kısıtsız RF/GBM/MLP örneklerin %56–73'ünde PGA arttıkça heyelan olasılığını
              düşürür — Kritik PGA taraması için kabul edilemez. Fizik kısıtlı modeller (PINN v2, monoton GBM)
              bu ihlali ~%0'a indirir.
              {b.reference_polygon_split_pinn_v2 && (
                <>
                  {" "}Aynı PINN v2, v1'in rastgele-poligon ayrımında AUC{" "}
                  {f3(b.reference_polygon_split_pinn_v2.roc_auc)} verir — v1'in 0.963'ü bu iyimser protokolden gelir.
                </>
              )}
            </p>
          </div>
        ) : (
          <p className="text-sm text-slate-500">Karşılaştırma sonucu yok — scripts/09_benchmark.py çalıştırın.</p>
        )}
      </Card>

      {/* Kritik PGA */}
      {c && cp && (
        <div className="grid gap-4 lg:grid-cols-2">
          <Card>
            <CardTitle hint={cp.definition}>Kritik PGA ve Belirsizlik</CardTitle>
            <dl className="grid grid-cols-2 gap-x-4 gap-y-3 text-sm">
              <dt className="text-slate-500 dark:text-slate-400">Kritik PGA (P = 0.5)</dt>
              <dd className="text-right font-semibold tabular-nums">{f2(cp.critical_pga_g)} g</dd>
              <dt className="text-slate-500 dark:text-slate-400">Tekil model aralığı</dt>
              <dd className="text-right tabular-nums">
                {cp.ensemble_range_g ? `${f2(cp.ensemble_range_g[0])}–${f2(cp.ensemble_range_g[1])} g` : "—"}
              </dd>
              <dt className="text-slate-500 dark:text-slate-400">Hücre bootstrap %95</dt>
              <dd className="text-right tabular-nums">
                {cp.bootstrap_95ci_g ? `${f2(cp.bootstrap_95ci_g[0])}–${f2(cp.bootstrap_95ci_g[1])} g` : "—"}
              </dd>
              <dt className="text-slate-500 dark:text-slate-400">Başlangıç (LR = 1)</dt>
              <dd className="text-right tabular-nums">{f2(cp.onset_lr1_g)} g</dd>
              <dt className="text-slate-500 dark:text-slate-400">Ampirik başlangıç (FR = 1)</dt>
              <dd className="text-right tabular-nums">{f2(c.empirical_validation.empirical_critical_pga_fr1_g)} g</dd>
              <dt className="text-slate-500 dark:text-slate-400">Newmark (Dn ≥ 10 cm, %50)</dt>
              <dd className="text-right tabular-nums">{f2(cp.newmark_critical_pga_g)} g</dd>
              <dt className="text-slate-500 dark:text-slate-400">Gözlem–model Spearman ρ</dt>
              <dd className="text-right tabular-nums">{f2(c.empirical_validation.spearman_binned)}</dd>
            </dl>
            {c.neg_ratio_sensitivity && (
              <div className="mt-4">
                <p className="mb-2 text-xs font-semibold uppercase tracking-wide text-slate-500 dark:text-slate-400">
                  Örnekleme oranı duyarlılığı (3 tohum)
                </p>
                <table className="w-full text-sm">
                  <thead>
                    <tr className="text-left text-xs text-slate-500 dark:text-slate-400">
                      <th className="py-1">Neg/Poz</th>
                      <th className="py-1 text-right">Kritik PGA (P=0.5)</th>
                      <th className="py-1 text-right">Başlangıç (LR=1)</th>
                      <th className="py-1 text-right">AUC</th>
                    </tr>
                  </thead>
                  <tbody>
                    {c.neg_ratio_sensitivity.map((r) => (
                      <tr key={r.neg_pos_ratio} className="border-t border-slate-100 dark:border-slate-800/60">
                        <td className="py-1 tabular-nums">{r.neg_pos_ratio}</td>
                        <td className="py-1 text-right tabular-nums">
                          {f2(r["critical_pga_at_0.5_mean"])} ± {f2(r["critical_pga_at_0.5_sd"])} g
                        </td>
                        <td className="py-1 text-right tabular-nums">{f2(r.critical_pga_onset_lr1_mean)} g</td>
                        <td className="py-1 text-right tabular-nums">{f3(r.test_roc_auc_mean)}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
                <p className="mt-2 text-xs leading-relaxed text-slate-500 dark:text-slate-400">
                  P = 0.5 eşiği LR = Neg/Poz oranına karşılık geldiğinden örnekleme tasarımına bağlıdır;
                  ayırma gücü (AUC) ise değişmez. Kritik PGA bu nedenle aralığıyla birlikte raporlanır.
                </p>
              </div>
            )}
          </Card>

          <Card>
            <CardTitle hint="P = 0.5 · 6 model ortalama eğrisi · — : 1.5 g'ye kadar ulaşılmaz">
              Tasarım Tablosu — Eğim × Litoloji
            </CardTitle>
            {c.design_table && c.design_table.length ? (
              <DesignTable rows={c.design_table} />
            ) : (
              <p className="text-sm text-slate-500">Tablo yok.</p>
            )}
          </Card>
        </div>
      )}

      {/* FEM */}
      {fem && fem.length > 0 && (
        <Card>
          <CardTitle hint="Griffiths & Lane (1999) elasto-viskoplastik SSR · 2B idealize kesit (H = 20 m)">
            FEM Pilot Analizi — Sonsuz Şev ile Karşılaştırma
          </CardTitle>
          <div className="overflow-x-auto">
            <table className="w-full min-w-[680px] text-sm">
              <thead>
                <tr className="border-b border-slate-200 text-left text-xs uppercase tracking-wide text-slate-500 dark:border-slate-800 dark:text-slate-400">
                  <th className="py-2 pr-3">Pilot</th>
                  <th className="py-2 pr-3 text-right">FS FEM</th>
                  <th className="py-2 pr-3 text-right">FS sonsuz şev</th>
                  <th className="py-2 pr-3 text-right">Kᵧ FEM</th>
                  <th className="py-2 pr-3 text-right">Kᵧ sonsuz şev</th>
                  <th className="py-2 pr-3 text-right">PGA</th>
                  <th className="py-2 text-right">Dₙ (Kᵧ FEM / sonsuz)</th>
                </tr>
              </thead>
              <tbody>
                {fem.map((p) => (
                  <tr key={p.slope_id} className="border-b border-slate-100 dark:border-slate-800/60">
                    <td className="py-2 pr-3">
                      <span className="font-semibold">ID {p.slope_id}</span>{" "}
                      <span className="text-slate-500 dark:text-slate-400">
                        {p.lithology}, θ = {p.theta_deg}°, c′ = {p.c_kpa} kPa, φ′ = {p.phi_deg}°
                      </span>
                    </td>
                    <td className="py-2 pr-3 text-right tabular-nums">{f2(p.fs_fem)}</td>
                    <td className="py-2 pr-3 text-right tabular-nums">{f2(p.fs_infinite_slope)}</td>
                    <td className="py-2 pr-3 text-right tabular-nums">{f3(p.ky_fem_g)} g</td>
                    <td className="py-2 pr-3 text-right tabular-nums">{f3(p.ky_infinite_slope_g)} g</td>
                    <td className="py-2 pr-3 text-right tabular-nums">{f2(p.actual_pga_g)} g</td>
                    <td className="py-2 text-right tabular-nums">
                      {p.dn_newmark_ky_fem_cm.toFixed(1)} / {p.dn_newmark_ky_infinite_cm.toFixed(1)} cm
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </Card>
      )}

      {/* Figür sekmeleri */}
      <Card>
        <div className="mb-4 flex flex-wrap gap-2">
          {FIGURES.map((f) => (
            <button
              key={f.key}
              onClick={() => setTab(f.key)}
              className={`rounded-lg px-3 py-1.5 text-sm font-medium transition ${
                tab === f.key
                  ? "bg-brand-600 text-white"
                  : "bg-slate-100 text-slate-600 hover:bg-slate-200 dark:bg-slate-800 dark:text-slate-300 dark:hover:bg-slate-700"
              }`}
            >
              {f.label}
            </button>
          ))}
        </div>
        <p className="mb-3 text-sm text-slate-500 dark:text-slate-400">{active.desc}</p>
        <div className="overflow-hidden rounded-xl border border-slate-200 bg-white dark:border-slate-800 dark:bg-slate-950">
          <img
            src={api.figureUrl(active.key)}
            alt={active.label}
            className="mx-auto max-h-[560px] w-full object-contain"
            loading="lazy"
          />
        </div>
      </Card>

      <div className="grid gap-4 lg:grid-cols-2">
        <Card>
          <CardTitle hint="Öznitelik önem sıralaması (SHAP, GradientExplainer)">Model Yorumu</CardTitle>
          {s && (
            <div className="space-y-2">
              {s.shap_order.map((feat, i) => (
                <div key={feat} className="flex items-center gap-3">
                  <span className="grid h-6 w-6 place-items-center rounded-md bg-brand-100 text-xs font-bold text-brand-700 dark:bg-brand-900/40 dark:text-brand-200">
                    {i + 1}
                  </span>
                  <span className="text-sm text-slate-700 dark:text-slate-200">{feat}</span>
                </div>
              ))}
            </div>
          )}
          <p className="mt-4 text-sm leading-relaxed text-slate-500 dark:text-slate-400">
            Eğim baskın etken; ardından PGA, yükseklik ve litoloji gelir. Fizik türetilmiş girdiler
            (Kᵧ, Dₙ) ayırma gücüne az katkı verir — eğim, litoloji ve PGA'nın deterministik
            fonksiyonlarıdır — ancak fizik kısıtının kurulmasını ve yorumlanabilirliği sağlarlar.
          </p>
        </Card>

        <Card className="border-amber-200 dark:border-amber-900/50">
          <CardTitle hint="Bilimsel şeffaflık">Kısıtlar ve Dürüstlük Notu</CardTitle>
          <ul className="space-y-3 text-sm leading-relaxed text-slate-600 dark:text-slate-300">
            {s &&
              Object.entries(s.notes).map(([k, v]) => (
                <li key={k} className="flex gap-2">
                  <span className="text-amber-500">◆</span>
                  <span>{v}</span>
                </li>
              ))}
          </ul>
          {s && (
            <div className="mt-4 rounded-lg bg-slate-50 px-3 py-2 text-xs text-slate-500 dark:bg-slate-800/50 dark:text-slate-400">
              Veri: {s.earthquake} · {s.n_samples.toLocaleString("tr-TR")} örnek · {s.n_features} öznitelik ·
              ayrım: {s.split} · model: {s.model_version ?? "pinn_v1"}
            </div>
          )}
        </Card>
      </div>
    </div>
  );
}

function DesignTable({ rows }: { rows: CriticalPgaReport["design_table"] & {} }) {
  const classes = Array.from(new Set(rows.map((r) => r.slope_class))).sort(
    (a, b) => (rows.find((r) => r.slope_class === a)!.slope_lo - rows.find((r) => r.slope_class === b)!.slope_lo),
  );
  const lithos = Array.from(new Set(rows.map((r) => r.litho))).sort();
  const cell = (l: string, cl: string) => rows.find((r) => r.litho === l && r.slope_class === cl);
  return (
    <div className="overflow-x-auto">
      <table className="w-full text-sm">
        <thead>
          <tr className="border-b border-slate-200 text-left text-xs uppercase tracking-wide text-slate-500 dark:border-slate-800 dark:text-slate-400">
            <th className="py-2 pr-3">Litoloji</th>
            {classes.map((cl) => (
              <th key={cl} className="py-2 pr-3 text-right">
                Eğim {cl}
              </th>
            ))}
          </tr>
        </thead>
        <tbody>
          {lithos.map((l) => (
            <tr key={l} className="border-b border-slate-100 dark:border-slate-800/60">
              <td className="py-1.5 pr-3 text-slate-700 dark:text-slate-200">{l}</td>
              {classes.map((cl) => {
                const r = cell(l, cl);
                return (
                  <td key={cl} className="py-1.5 pr-3 text-right tabular-nums" title={r ? `n = ${r.n_cells} hücre · aralık ${f2(r.range_lo_g)}–${f2(r.range_hi_g)} g` : ""}>
                    {r ? (r.critical_pga_g == null ? "—" : `${r.critical_pga_g.toFixed(2)} g`) : <span className="text-slate-300 dark:text-slate-600">·</span>}
                  </td>
                );
              })}
            </tr>
          ))}
        </tbody>
      </table>
      <p className="mt-2 text-xs text-slate-500 dark:text-slate-400">
        · : bu sınıfta yeterli hücre yok (&lt; 40). Hücrenin üzerine gelince model aralığı görünür.
      </p>
    </div>
  );
}
