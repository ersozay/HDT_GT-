import { useEffect, useState } from "react";
import { api, Summary } from "../api/client";
import { Card, CardTitle } from "../components/Card";
import MetricCard from "../components/MetricCard";
import Spinner from "../components/Spinner";

const FIGURES = [
  { key: "shap_summary", label: "SHAP Önem", desc: "Özniteliklerin model kararına katkısı" },
  { key: "training_history", label: "Eğitim Geçmişi", desc: "Kayıp ve doğruluk eğrileri" },
  { key: "confusion_matrix", label: "Karışıklık Matrisi", desc: "Test kümesi sınıflandırma sonuçları" },
  { key: "critical_pga_curve_real", label: "Kritik PGA Eğrisi", desc: "ML vs Newmark tetiklenme eşiği" },
  { key: "susceptibility_map_real", label: "Duyarlılık Haritası", desc: "Bölgesel PINN çıktısı" },
];

export default function Results() {
  const [s, setS] = useState<Summary | null>(null);
  const [tab, setTab] = useState(FIGURES[0].key);

  useEffect(() => {
    api.summary().then(setS).catch(() => {});
  }, []);

  const active = FIGURES.find((f) => f.key === tab)!;

  return (
    <div className="space-y-6">
      <div>
        <h1 className="text-2xl font-bold tracking-tight text-slate-900 dark:text-white">
          Model Sonuçları ve Değerlendirme
        </h1>
        <p className="mt-1 text-sm text-slate-500 dark:text-slate-400">
          Poligon tabanlı 3’lü ayrım (train/val/test) ile veri sızıntısı önlenmiş,
          bağımsız test kümesindeki başarımlar.
        </p>
      </div>

      {/* Metrikler */}
      {s ? (
        <div className="grid gap-4 sm:grid-cols-2 lg:grid-cols-3">
          <MetricCard label="ROC–AUC" value={s.roc_auc.toFixed(3)} accent="brand" sub="Ayırt edicilik" />
          <MetricCard label="Doğruluk" value={`${(s.accuracy * 100).toFixed(0)}%`} accent="sky" sub="Genel isabet" />
          <MetricCard label="Duyarlılık" value={`${(s.recall * 100).toFixed(0)}%`} accent="emerald" sub="Recall (heyelan)" />
          <MetricCard label="Kesinlik" value={`${(s.precision * 100).toFixed(0)}%`} accent="brand" sub="Precision (heyelan)" />
          <MetricCard label="F1 Skoru" value={s.f1.toFixed(2)} accent="amber" sub="Denge ölçütü" />
          <MetricCard label="Kritik PGA" value={s.critical_pga_ml.toFixed(2)} unit="g" accent="rose" sub="ML tetiklenme eşiği" />
        </div>
      ) : (
        <Spinner />
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
            className="mx-auto max-h-[520px] w-full object-contain"
            loading="lazy"
          />
        </div>
      </Card>

      {/* Yöntem + dürüst kısıtlar */}
      <div className="grid gap-4 lg:grid-cols-2">
        <Card>
          <CardTitle hint="Öznitelik önem sıralaması">Model Yorumu</CardTitle>
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
            Eğim baskın etken; ardından yükseklik, PGA ve kritik ivme gelir.
            Bu sıralama fiziksel beklentiyle (dik yamaçlar + yüksek sismik yük)
            uyumludur.
          </p>
        </Card>

        <Card className="border-amber-200 dark:border-amber-900/50">
          <CardTitle hint="Bilimsel şeffaflık">Kısıtlar ve Dürüstlük Notu</CardTitle>
          <ul className="space-y-3 text-sm leading-relaxed text-slate-600 dark:text-slate-300">
            <li className="flex gap-2">
              <span className="text-amber-500">◆</span>
              <span>
                <strong>Litoloji:</strong>{" "}
                {s?.notes.lithology ??
                  "Macrostrat API tabanlı jeolojik birim proxy'si; resmi MTA haritası değildir."}
              </span>
            </li>
            <li className="flex gap-2">
              <span className="text-amber-500">◆</span>
              <span>
                <strong>FEM:</strong> {s?.notes.fem}
              </span>
            </li>
            <li className="flex gap-2">
              <span className="text-amber-500">◆</span>
              <span>
                Kritik PGA (ML ≈ {s?.critical_pga_ml.toFixed(2)} g), rijit-blok Newmark
                eğrisinden daha az muhafazakârdır; ML, mukavemet kaybı ve
                süreksizlikleri de yakalar.
              </span>
            </li>
          </ul>
          {s && (
            <div className="mt-4 rounded-lg bg-slate-50 px-3 py-2 text-xs text-slate-500 dark:bg-slate-800/50 dark:text-slate-400">
              Veri: {s.earthquake} · {s.n_samples.toLocaleString("tr-TR")} örnek ·
              {s.n_features} öznitelik · ayrım: {s.split}
            </div>
          )}
        </Card>
      </div>
    </div>
  );
}
