import { useEffect, useState } from "react";
import { api, Validation as ValidationData } from "../api/client";
import { Card, CardTitle } from "../components/Card";
import Spinner from "../components/Spinner";

export default function Validation() {
  const [v, setV] = useState<ValidationData | null>(null);
  const [err, setErr] = useState<string | null>(null);

  useEffect(() => {
    api
      .validation()
      .then(setV)
      .catch((e) => setErr(String(e)));
  }, []);

  if (err) {
    return (
      <div className="rounded-xl bg-rose-50 p-4 text-sm text-rose-700 dark:bg-rose-900/30 dark:text-rose-200">
        Doğrulama verisi alınamadı: {err}
      </div>
    );
  }
  if (!v) {
    return (
      <div className="flex h-96 items-center justify-center">
        <Spinner label="Doğrulama yükleniyor…" />
      </div>
    );
  }

  return (
    <div className="space-y-6">
      <div>
        <h1 className="text-2xl font-bold tracking-tight text-slate-900 dark:text-white">
          {v.title}
        </h1>
        <p className="mt-1 max-w-3xl text-sm text-slate-500 dark:text-slate-400">
          {v.method}
        </p>
      </div>

      {/* Yayınlanmış olay künyesi */}
      <div className="grid gap-4 sm:grid-cols-2 lg:grid-cols-4">
        {Object.entries(v.published).map(([k, val]) => (
          <div
            key={k}
            className="rounded-xl border border-slate-200 p-4 dark:border-slate-800"
          >
            <div className="text-[11px] font-semibold uppercase tracking-wide text-slate-400">
              {labelize(k)}
            </div>
            <div className="mt-1 text-sm font-medium text-slate-700 dark:text-slate-200">
              {val}
            </div>
          </div>
        ))}
      </div>

      {/* Yayınlanmış bağımsız çalışma — aynı deprem (Wang vd. 2026, Nature) */}
      <Card>
        <CardTitle hint="Aynı 2023 Kahramanmaraş depremi · bağımsız yayınlanmış çalışma">
          Bağımsız Yayınlanmış Kıyas — Wang vd. (2026, Nature)
        </CardTitle>
        <div className="grid gap-4 lg:grid-cols-2">
          <div className="space-y-2.5 text-sm leading-relaxed text-slate-600 dark:text-slate-300">
            <p>
              <strong className="text-slate-700 dark:text-slate-200">
                Wang, Dahal, van Westen, Görüm &amp; Tanyaş (2026)
              </strong>{" "}
              — <em>Communications Earth &amp; Environment</em> (Nature Portfolio),
              7:521 — aynı 2023 Kahramanmaraş dizisini (Pazarcık M<sub>w</sub>7.8 +
              Elbistan M<sub>w</sub>7.5) inceler.
            </p>
            <p>
              Ortak temeller bu tezle örtüşür: <strong>Mohr–Coulomb</strong>{" "}
              yenilme ölçütü ve <strong>USGS bileşik PGA zarfı</strong>{" "}
              (onlarda maks. ≈ 1.27 g, bu projede 1.31 g). Ayrıca litoloji-tabanlı
              şev birimleri ve Wenchuan (2008) miras etkisi referansı paylaşılır.
            </p>
            <p className="text-xs text-slate-500 dark:text-slate-400">
              Fark: Wang vd. şev dayanımı değişimini <strong>InSAR</strong>{" "}
              deformasyonu tersinimi + Hoek–Brown ile kestirir; bu tez dayanımı
              <strong> PINN + Newmark</strong> ile <strong>kritik PGA</strong>'ya
              çevirir. Yaklaşımlar tamamlayıcıdır.
            </p>
          </div>
          <div className="grid content-start gap-3">
            <div className="rounded-xl border border-slate-200 p-3 dark:border-slate-800">
              <div className="text-[11px] font-semibold uppercase tracking-wide text-slate-400">
                Çalışma alanı (Wang vd.)
              </div>
              <div className="mt-1 text-sm font-medium text-slate-700 dark:text-slate-200">
                51.287 km² · 2023 Kahramanmaraş
              </div>
            </div>
            <div className="grid grid-cols-2 gap-3">
              <div className="rounded-xl border border-rose-200 bg-rose-50 p-3 dark:border-rose-500/30 dark:bg-rose-500/10">
                <div className="text-[11px] font-semibold uppercase tracking-wide text-rose-500 dark:text-rose-300">
                  Zayıflayan şev
                </div>
                <div className="mt-1 text-sm font-semibold text-rose-700 dark:text-rose-200">
                  ~%59 (~13.000 km²)
                </div>
              </div>
              <div className="rounded-xl border border-emerald-200 bg-emerald-50 p-3 dark:border-emerald-500/30 dark:bg-emerald-500/10">
                <div className="text-[11px] font-semibold uppercase tracking-wide text-emerald-600 dark:text-emerald-300">
                  Güçlenen şev
                </div>
                <div className="mt-1 text-sm font-semibold text-emerald-700 dark:text-emerald-200">
                  ~%7 (~1.000 km²)
                </div>
              </div>
            </div>
            <div className="rounded-xl border border-amber-300/60 bg-amber-50 p-3 text-xs leading-relaxed text-amber-900 dark:border-amber-500/30 dark:bg-amber-500/10 dark:text-amber-200">
              <strong>Kısıt:</strong> Bu, ortak fiziksel temel ve girdi (PGA
              zarfı, Mohr–Coulomb) üzerinden bir <em>literatür konumlandırmasıdır</em>;
              modelimizin çıktısının Wang vd. dayanım haritasıyla sayısal
              piksel-bazlı çapraz doğrulaması yapılmamıştır.
            </div>
          </div>
        </div>
      </Card>

      {/* Figür */}
      <Card>
        <CardTitle hint="Global Newmark eğrileri · yayınlanmış istatistikle karşılaştırma">
          Wenchuan Fizik-Transfer Figürü
        </CardTitle>
        <div className="overflow-hidden rounded-xl border border-slate-200 bg-white dark:border-slate-800">
          <img
            src={api.figureUrl("wenchuan_validation")}
            alt="Wenchuan fizik-transfer doğrulama figürü"
            className="w-full"
            loading="lazy"
          />
        </div>
      </Card>

      {/* Karşılaştırma tablosu */}
      <Card>
        <CardTitle hint={`Dₙ eşiği ${v.dn_threshold_cm} cm`}>
          Karşılaştırma Tablosu
        </CardTitle>
        <div className="overflow-x-auto">
          <table className="w-full min-w-[640px] text-sm">
            <thead>
              <tr className="border-b border-slate-200 text-left text-xs uppercase tracking-wide text-slate-400 dark:border-slate-700">
                <th className="py-2 pr-3">PGA (g)</th>
                <th className="py-2 pr-3">Yamaç sınıfı</th>
                <th className="py-2 pr-3 tabular-nums">kᵧ (g)</th>
                <th className="py-2 pr-3 tabular-nums">Dₙ · Jibson</th>
                <th className="py-2 pr-3 tabular-nums">Dₙ · Ambraseys</th>
                <th className="py-2 pr-3">Tetiklenme</th>
              </tr>
            </thead>
            <tbody>
              {v.comparison.map((c, i) => {
                const trig = c.triggered_jibson || c.triggered_ambraseys;
                return (
                  <tr
                    key={i}
                    className="border-b border-slate-100 dark:border-slate-800/60"
                  >
                    <td className="py-2 pr-3 font-semibold tabular-nums text-slate-700 dark:text-slate-200">
                      {c.pga_g.toFixed(1)}
                    </td>
                    <td className="py-2 pr-3 text-slate-600 dark:text-slate-300">
                      {c.class}
                    </td>
                    <td className="py-2 pr-3 tabular-nums text-slate-500 dark:text-slate-400">
                      {c.ky_g.toFixed(2)}
                    </td>
                    <td className="py-2 pr-3 tabular-nums text-sky-600 dark:text-sky-300">
                      {c.dn_jibson_cm.toFixed(1)} cm
                    </td>
                    <td className="py-2 pr-3 tabular-nums text-violet-600 dark:text-violet-300">
                      {c.dn_ambraseys_cm.toFixed(1)} cm
                    </td>
                    <td className="py-2 pr-3">
                      <span
                        className={`inline-block rounded-full px-2 py-0.5 text-[11px] font-medium ${
                          trig
                            ? "bg-rose-100 text-rose-700 dark:bg-rose-900/40 dark:text-rose-200"
                            : "bg-emerald-100 text-emerald-700 dark:bg-emerald-900/40 dark:text-emerald-200"
                        }`}
                      >
                        {trig ? "Dₙ ≥ 10 cm" : "Dₙ < 10 cm"}
                      </span>
                    </td>
                  </tr>
                );
              })}
            </tbody>
          </table>
        </div>
        <p className="mt-3 text-xs leading-relaxed text-slate-500 dark:text-slate-400">
          Beklenen davranış: yüksek PGA + dik/zayıf yamaç (düşük kᵧ) → büyük Dₙ →
          yaygın tetiklenme; bu, Wenchuan'da gözlenen ~197.000 heyelanın dik yamaç +
          yüksek PGA bölgelerinde yoğunlaşmasıyla <em>niteliksel olarak</em> tutarlıdır.
        </p>
      </Card>

      {/* Kaynaklar + kısıtlar */}
      <div className="grid gap-6 lg:grid-cols-2">
        <Card>
          <CardTitle>Kaynaklar</CardTitle>
          <ul className="space-y-1.5 text-sm text-slate-600 dark:text-slate-300">
            {v.sources.map((s, i) => (
              <li key={i}>{s}</li>
            ))}
          </ul>
        </Card>

        <Card className="border-amber-300/60 bg-amber-50 dark:border-amber-500/30 dark:bg-amber-500/10">
          <CardTitle hint="Bilimsel şeffaflık">Kısıtlar</CardTitle>
          <ul className="list-disc space-y-1.5 pl-5 text-sm text-amber-900 dark:text-amber-200">
            {v.caveats.map((c, i) => (
              <li key={i}>{c}</li>
            ))}
          </ul>
        </Card>
      </div>
    </div>
  );
}

function labelize(k: string): string {
  const map: Record<string, string> = {
    event: "Olay",
    landslide_count: "Heyelan sayısı",
    peak_pga_g: "Tepe PGA",
    note: "Not",
  };
  return map[k] ?? k;
}
