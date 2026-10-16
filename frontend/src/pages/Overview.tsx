import { useEffect, useState } from "react";
import { Link } from "react-router-dom";
import MetricCard from "../components/MetricCard";
import { Card, CardTitle } from "../components/Card";
import { api, Summary } from "../api/client";

export default function Overview() {
  const [s, setS] = useState<Summary | null>(null);
  useEffect(() => {
    api.summary().then(setS).catch(() => {});
  }, []);
  const auc = s?.cv?.roc_auc?.mean ?? s?.roc_auc;
  const crit = s?.critical_pga_ml;
  const rng = (s as (Summary & { critical_pga?: { ensemble_range_g?: [number, number] } }) | null)
    ?.critical_pga?.ensemble_range_g;
  return (
    <div className="space-y-8">
      {/* Hero */}
      <section className="relative overflow-hidden rounded-3xl border border-slate-200 bg-gradient-to-br from-white to-slate-50 p-8 shadow-sm dark:border-slate-800 dark:from-slate-900 dark:to-slate-950 md:p-12">
        <div className="absolute -right-16 -top-16 h-64 w-64 rounded-full bg-brand-500/10 blur-3xl" />
        <div className="relative max-w-3xl">
          <span className="inline-flex items-center gap-1.5 rounded-full border border-brand-200 bg-brand-50 px-3 py-1 text-xs font-semibold text-brand-700 dark:border-brand-800 dark:bg-brand-900/30 dark:text-brand-200">
            Fizik + Yapay Zekâ Hibrit Model
          </span>
          <h1 className="mt-4 text-3xl font-extrabold tracking-tight text-slate-900 dark:text-white md:text-4xl">
            Deprem Kaynaklı Heyelan Dijital İkizi
          </h1>
          <p className="mt-3 text-base leading-relaxed text-slate-600 dark:text-slate-300">
            Kahramanmaraş 6 Şubat 2023 (Mw 7.8 + 7.5) depremi için geliştirilen bu
            dijital ikiz,{" "}
            <strong className="text-slate-800 dark:text-slate-100">
              “Bir yamaç hangi yer ivmesinde (PGA) kaymaya başlar?”
            </strong>{" "}
            sorusunu yanıtlar. Mohr–Coulomb limit denge, Newmark yer değiştirme
            ve fizik-bilgili sinir ağı (PINN) yaklaşımlarını tek bir modelde
            birleştirir.
          </p>
          <div className="mt-6 flex flex-wrap gap-3">
            <Link to="/predictor" className="btn-primary">
              Tahmin yap
              <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"><path d="M5 12h14M13 6l6 6-6 6" /></svg>
            </Link>
            <Link
              to="/map"
              className="btn border border-slate-200 bg-white text-slate-700 hover:bg-slate-100 dark:border-slate-700 dark:bg-slate-900 dark:text-slate-200 dark:hover:bg-slate-800"
            >
              Haritayı incele
            </Link>
          </div>
        </div>
      </section>

      {/* Anahtar sayılar */}
      <section>
        <div className="grid gap-4 sm:grid-cols-2 lg:grid-cols-4">
          <MetricCard
            label="Kritik PGA (ML)"
            value={crit != null ? crit.toFixed(2) : "…"}
            unit="g"
            accent="amber"
            sub={rng ? `Belirsizlik ${rng[0].toFixed(2)}–${rng[1].toFixed(2)} g · en dik %5 yamaç` : "Heyelan tetiklenme eşiği"}
          />
          <MetricCard
            label="ROC–AUC"
            value={auc != null ? auc.toFixed(3) : "…"}
            accent="brand"
            sub="Mekânsal blok CV (5 kat)"
          />
          <MetricCard
            label="Fizik tutarlılığı"
            value={s?.mono_violation_rate != null ? `%${(100 * (1 - s.mono_violation_rate)).toFixed(1)}` : "…"}
            accent="emerald"
            sub="PGA ↑ ⇒ olasılık düşmeyen örnek oranı"
          />
          <MetricCard
            label="Örnek Sayısı"
            value={s ? `${(s.n_samples / 1000).toFixed(1)}K` : "…"}
            accent="sky"
            sub={s ? `%${(100 * s.landslide_ratio).toFixed(1)} heyelan · ${s.n_features} öznitelik` : ""}
          />
        </div>
      </section>

      {/* Yöntem özeti */}
      <section className="grid gap-4 lg:grid-cols-3">
        <Card>
          <CardTitle hint="Sismik yükte kayma güvenliği">
            1 · Fizik Çekirdeği
          </CardTitle>
          <p className="text-sm leading-relaxed text-slate-600 dark:text-slate-300">
            Sonsuz yamaç Mohr–Coulomb modeliyle her nokta için{" "}
            <strong>kritik ivme K<sub>y</sub></strong> hesaplanır; Newmark/Jibson
            (2007) ile deprem boyunca birikimli{" "}
            <strong>yer değiştirme D<sub>n</sub></strong> bulunur. D<sub>n</sub> ≥ 10
            cm kayma göstergesidir.
          </p>
        </Card>
        <Card>
          <CardTitle hint="Fizik-bilgili sinir ağı">
            2 · PINN Modeli
          </CardTitle>
          <p className="text-sm leading-relaxed text-slate-600 dark:text-slate-300">
            256→128→64→32→1 mimarili ağ, eğim/yükseklik/bakı/litoloji/PGA ve
            fizikten türetilen Kᵧ, Dₙ dahil {s?.n_features ?? 37} öznitelikten heyelan
            olasılığını öğrenir. Fizik-yolu kısıtı, PGA veya eğim arttıkça olasılığın
            düşmemesini kayıp fonksiyonu içinde zorunlu kılar.
          </p>
        </Card>
        <Card>
          <CardTitle hint="Bölgesel duyarlılık">
            3 · Dijital İkiz
          </CardTitle>
          <p className="text-sm leading-relaxed text-slate-600 dark:text-slate-300">
            Model tüm çalışma alanına uygulanarak duyarlılık haritası üretilir.
            Haritaya tıklayarak herhangi bir konum için anlık tahmin alınabilir;
            senaryo PGA değerleri değiştirilerek risk sınanabilir.
          </p>
        </Card>
      </section>

      {/* Kaynak veri şeridi */}
      <section className="rounded-2xl border border-slate-200 bg-slate-50 px-5 py-4 text-xs text-slate-500 dark:border-slate-800 dark:bg-slate-900/50 dark:text-slate-400">
        <span className="font-semibold text-slate-600 dark:text-slate-300">
          Veri kaynakları:
        </span>{" "}
        USGS ShakeMap (PGA zarfı) · Copernicus GLO-90 DEM · Macrostrat litoloji
        proxy · Kocaman (2025) heyelan envanteri (2.596 poligon).
      </section>
    </div>
  );
}
