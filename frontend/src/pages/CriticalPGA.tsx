import { useEffect, useMemo, useState } from "react";
import {
  ResponsiveContainer,
  ComposedChart,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  Legend,
  ReferenceLine,
  ReferenceArea,
} from "recharts";
import { api, Lithology, CriticalResult, NewmarkCurves } from "../api/client";
import { Card, CardTitle } from "../components/Card";
import MetricCard from "../components/MetricCard";
import Spinner from "../components/Spinner";
import { useTheme } from "../theme";

const DN_LIMIT_CM = 10; // Jibson 2007 — kayma göstergesi eşiği

/** dn_cm dizisinde 10 cm'i ilk aşan PGA'yı doğrusal ara-değerle bul. */
function newmarkCritical(pga: number[], dn: number[]): number | null {
  for (let i = 1; i < dn.length; i++) {
    if (dn[i] >= DN_LIMIT_CM) {
      const x0 = pga[i - 1], x1 = pga[i];
      const y0 = dn[i - 1], y1 = dn[i];
      if (y1 === y0) return x1;
      return +(x0 + ((DN_LIMIT_CM - y0) * (x1 - x0)) / (y1 - y0)).toFixed(2);
    }
  }
  return null;
}

export default function CriticalPGA() {
  const { theme } = useTheme();
  const dark = theme === "dark";
  const [litho, setLitho] = useState<Lithology[]>([]);
  const [slope, setSlope] = useState(35);
  const [elevation, setElevation] = useState(1200);
  const [lithoId, setLithoId] = useState(11);
  const [res, setRes] = useState<CriticalResult | null>(null);
  const [nm, setNm] = useState<NewmarkCurves | null>(null);
  const [loading, setLoading] = useState(false);

  useEffect(() => {
    api.lithology().then(setLitho).catch(() => {});
  }, []);

  useEffect(() => {
    setLoading(true);
    const t = setTimeout(() => {
      Promise.all([
        api.criticalPga({ slope, elevation, aspect: 180, litho_id: lithoId }),
        api.newmarkCurves({ slope, elevation, aspect: 180, litho_id: lithoId }),
      ])
        .then(([c, n]) => {
          setRes(c);
          setNm(n);
        })
        .catch(() => {})
        .finally(() => setLoading(false));
    }, 250);
    return () => clearTimeout(t);
  }, [slope, elevation, lithoId]);

  // Ambraseys eğrisini PGA'ya göre ara-değerleyerek ML/Jibson gridine hizala
  const ambAt = useMemo(() => {
    if (!nm) return (_p: number) => NaN;
    const xs = nm.pga, ys = nm.dn_ambraseys_cm;
    return (p: number) => {
      if (p <= xs[0]) return ys[0];
      if (p >= xs[xs.length - 1]) return ys[ys.length - 1];
      for (let i = 1; i < xs.length; i++) {
        if (xs[i] >= p) {
          const t = (p - xs[i - 1]) / (xs[i] - xs[i - 1]);
          return ys[i - 1] + t * (ys[i] - ys[i - 1]);
        }
      }
      return ys[ys.length - 1];
    };
  }, [nm]);

  const data =
    res?.pga.map((p, i) => ({
      pga: p,
      prob: res.prob[i] * 100,
      dn: res.dn_cm[i],
      dnAmb: ambAt(p),
    })) ?? [];

  const newmarkCrit = useMemo(
    () => (res ? newmarkCritical(res.pga, res.dn_cm) : null),
    [res]
  );
  const ambraseysCrit = nm?.critical_pga_ambraseys ?? null;

  const grid = dark ? "#1e293b" : "#e2e8f0";
  const axis = dark ? "#94a3b8" : "#64748b";

  return (
    <div className="space-y-6">
      <div>
        <h1 className="text-2xl font-bold tracking-tight text-slate-900 dark:text-white">
          Kritik PGA Analizi — ML vs Fizik
        </h1>
        <p className="mt-1 max-w-3xl text-sm text-slate-500 dark:text-slate-400">
          Seçilen yamaç için PGA arttıkça hem <strong className="text-brand-600 dark:text-brand-300">yapay
          zekâ olasılığını</strong> (P) hem de <strong className="text-sky-600 dark:text-sky-300">Newmark
          rijit-blok yer değiştirmesini</strong> (D<sub>n</sub>) birlikte gösterir. İki
          bağımsız kritik eşik karşılaştırılır: ML karar eşiği ve fiziksel D<sub>n</sub> = 10 cm.
        </p>
      </div>

      <div className="grid gap-6 lg:grid-cols-3">
        {/* Kontroller */}
        <Card className="lg:col-span-1">
          <CardTitle hint="Yamaç seçimi">Parametreler</CardTitle>
          <div className="space-y-5">
            <div>
              <div className="mb-1 flex justify-between text-sm">
                <span className="font-medium text-slate-600 dark:text-slate-300">Eğim</span>
                <span className="font-semibold tabular-nums text-slate-700 dark:text-slate-200">{slope}°</span>
              </div>
              <input type="range" className="input-range" min={0} max={70} step={1} value={slope} onChange={(e) => setSlope(Number(e.target.value))} />
            </div>
            <div>
              <div className="mb-1 flex justify-between text-sm">
                <span className="font-medium text-slate-600 dark:text-slate-300">Yükseklik</span>
                <span className="font-semibold tabular-nums text-slate-700 dark:text-slate-200">{elevation} m</span>
              </div>
              <input type="range" className="input-range" min={0} max={3000} step={50} value={elevation} onChange={(e) => setElevation(Number(e.target.value))} />
            </div>
            <div>
              <label className="mb-1 block text-sm font-medium text-slate-600 dark:text-slate-300">Litoloji</label>
              <select
                className="w-full rounded-xl border border-slate-200 bg-white px-3 py-2 text-sm text-slate-700 outline-none focus:border-brand-500 dark:border-slate-700 dark:bg-slate-900 dark:text-slate-200"
                value={lithoId}
                onChange={(e) => setLithoId(Number(e.target.value))}
              >
                {litho.map((l) => (
                  <option key={l.id} value={l.id}>
                    {l.id} — {l.description}
                  </option>
                ))}
              </select>
            </div>

            <div className="grid gap-3 pt-2">
              <MetricCard
                label="Kritik PGA · ML"
                value={res?.critical_pga != null ? res.critical_pga.toFixed(2) : "—"}
                unit="g"
                accent="amber"
                sub={res?.critical_pga != null ? "P, karar eşiğini aşar" : "Bu aralıkta eşik aşılmıyor"}
              />
              <MetricCard
                label="Kritik PGA · Newmark (Jibson)"
                value={newmarkCrit != null ? newmarkCrit.toFixed(2) : "—"}
                unit="g"
                accent="sky"
                sub={newmarkCrit != null ? "Dₙ, 10 cm'i aşar" : "Dₙ < 10 cm (kayma yok)"}
              />
              <MetricCard
                label="Kritik PGA · Ambraseys (1988)"
                value={ambraseysCrit != null ? ambraseysCrit.toFixed(2) : "—"}
                unit="g"
                accent="violet"
                sub={ambraseysCrit != null ? "Dₙ, 10 cm'i aşar" : "Dₙ < 10 cm (kayma yok)"}
              />
            </div>
          </div>
        </Card>

        {/* Grafik */}
        <Card className="lg:col-span-2">
          <div className="flex items-center justify-between">
            <CardTitle hint="P(heyelan) ve Newmark Dₙ vs PGA">İkili Eğri</CardTitle>
            {loading && <Spinner label="Hesaplanıyor…" />}
          </div>
          <div className="h-[380px] w-full">
            <ResponsiveContainer>
              <ComposedChart data={data} margin={{ top: 8, right: 8, bottom: 12, left: -8 }}>
                <CartesianGrid stroke={grid} strokeDasharray="3 3" />
                <XAxis
                  dataKey="pga"
                  type="number"
                  domain={[0, "dataMax"]}
                  tickFormatter={(v) => v.toFixed(1)}
                  stroke={axis}
                  fontSize={12}
                  label={{ value: "PGA (g)", position: "insideBottom", offset: -4, fill: axis, fontSize: 12 }}
                />
                <YAxis
                  yAxisId="p"
                  domain={[0, 100]}
                  stroke="#2c807b"
                  fontSize={12}
                  tickFormatter={(v) => `${v}%`}
                />
                <YAxis
                  yAxisId="dn"
                  orientation="right"
                  domain={[0, 12]}
                  allowDataOverflow
                  stroke="#0284c7"
                  fontSize={12}
                  tickFormatter={(v) => `${v}cm`}
                />
                <Tooltip
                  contentStyle={{
                    background: dark ? "#0f172a" : "#fff",
                    border: `1px solid ${grid}`,
                    borderRadius: 12,
                    fontSize: 12,
                  }}
                  labelFormatter={(v) => `PGA: ${Number(v).toFixed(2)} g`}
                  formatter={(val: number, name) =>
                    name === "P(heyelan)"
                      ? [`${val.toFixed(1)}%`, "P(heyelan)"]
                      : [`${val.toFixed(2)} cm`, name]
                  }
                />
                <Legend wrapperStyle={{ fontSize: 12, paddingTop: 4 }} />

                {res && (
                  <ReferenceArea yAxisId="p" y1={res.threshold * 100} y2={100} fill="#dc2626" fillOpacity={0.05} />
                )}
                {res && (
                  <ReferenceLine
                    yAxisId="p"
                    y={res.threshold * 100}
                    stroke="#dc2626"
                    strokeDasharray="5 4"
                    label={{ value: `ML eşik ${(res.threshold * 100).toFixed(0)}%`, fill: "#dc2626", fontSize: 10, position: "insideTopLeft" }}
                  />
                )}
                <ReferenceLine
                  yAxisId="dn"
                  y={DN_LIMIT_CM}
                  stroke="#0284c7"
                  strokeDasharray="5 4"
                  label={{ value: "Dₙ = 10 cm", fill: "#0284c7", fontSize: 10, position: "insideBottomRight" }}
                />
                {res?.critical_pga != null && (
                  <ReferenceLine yAxisId="p" x={res.critical_pga} stroke="#f59e0b" strokeWidth={2}
                    label={{ value: `ML ${res.critical_pga.toFixed(2)}g`, fill: "#f59e0b", fontSize: 10, position: "top" }} />
                )}
                {newmarkCrit != null && (
                  <ReferenceLine yAxisId="p" x={newmarkCrit} stroke="#0284c7" strokeWidth={2}
                    label={{ value: `Jibson ${newmarkCrit.toFixed(2)}g`, fill: "#0284c7", fontSize: 10, position: "insideTopRight" }} />
                )}
                {ambraseysCrit != null && (
                  <ReferenceLine yAxisId="p" x={ambraseysCrit} stroke="#7c3aed" strokeWidth={2}
                    label={{ value: `Ambraseys ${ambraseysCrit.toFixed(2)}g`, fill: "#7c3aed", fontSize: 10, position: "bottom" }} />
                )}

                <Line yAxisId="p" name="P(heyelan)" type="monotone" dataKey="prob" stroke="#2c807b" strokeWidth={2.5} dot={false} activeDot={{ r: 5 }} />
                <Line yAxisId="dn" name="Dₙ · Jibson 2007" type="monotone" dataKey="dn" stroke="#0284c7" strokeWidth={2.5} strokeDasharray="6 3" dot={false} activeDot={{ r: 5 }} />
                <Line yAxisId="dn" name="Dₙ · Ambraseys 1988" type="monotone" dataKey="dnAmb" stroke="#7c3aed" strokeWidth={2.5} strokeDasharray="2 3" dot={false} activeDot={{ r: 5 }} />
              </ComposedChart>
            </ResponsiveContainer>
          </div>
          <p className="mt-3 rounded-lg bg-slate-50 px-3 py-2 text-xs leading-relaxed text-slate-500 dark:bg-slate-800/50 dark:text-slate-400">
            <strong className="text-slate-600 dark:text-slate-300">Yorum:</strong> ML modeli
            genellikle daha düşük PGA'da risk işaretler; saf rijit-blok Newmark ise
            tetiklenmeyi daha geç (daha muhafazakâr) verir. İki ampirik Newmark bağıntısı
            —<span className="text-sky-600 dark:text-sky-300"> Jibson (2007)</span> ve
            <span className="text-violet-600 dark:text-violet-300"> Ambraseys &amp; Menu (1988)</span>—
            tahmin bandını gösterir; Ambraseys tipik olarak biraz daha büyük yer
            değiştirme verir. Aradaki fark, mukavemet kaybı, topografik büyütme ve
            süreksizliklerin ML tarafından örtük olarak yakalanmasından kaynaklanır.
          </p>
        </Card>
      </div>

      {/* Pilot yamaçlar — Newmark & Arias Yoğunluğu bilimsel figürü (danışman isteği) */}
      <Card>
        <CardTitle hint="Pilot ID 1327 (yamaç molozu · θ28°) ve ID 1353 (marn · θ22°) — dokümante geometri">
          Pilot Yamaçlar · Dₙ – PGA ve Dₙ – Arias Yoğunluğu
        </CardTitle>
        <div className="overflow-hidden rounded-xl border border-slate-200 bg-white dark:border-slate-800">
          <img
            src={api.figureUrl("newmark_arias_models")}
            alt="Pilot yamaçlar için Newmark kalıcı yer değiştirmesi — PGA ve Arias Yoğunluğu panelleri, 10 cm serviceability limit"
            className="w-full"
            loading="lazy"
          />
        </div>
        <div className="mt-3 grid gap-3 sm:grid-cols-3">
          <div className="rounded-lg bg-slate-50 px-3 py-2 text-xs leading-relaxed text-slate-600 dark:bg-slate-800/50 dark:text-slate-300">
            <strong className="text-slate-700 dark:text-slate-200">Sol panel:</strong> Dₙ – PGA;
            iki ampirik bağıntı (Jibson 2007 · Ambraseys &amp; Menu 1988). Her pilotun akma ivmesi
            aᵧ, Mohr–Coulomb + Newmark ile <em>gerçek</em> dokümante geometriden hesaplandı.
          </div>
          <div className="rounded-lg bg-slate-50 px-3 py-2 text-xs leading-relaxed text-slate-600 dark:bg-slate-800/50 dark:text-slate-300">
            <strong className="text-slate-700 dark:text-slate-200">Sağ panel:</strong> Dₙ – Arias
            Yoğunluğu Iₐ (Jibson 2007, Denk. 9). Kırmızı çizgi 10 cm{" "}
            <span className="font-medium text-rose-600 dark:text-rose-300">Serviceability Limit</span>.
          </div>
          <div className="rounded-lg bg-amber-50 px-3 py-2 text-xs leading-relaxed text-amber-900 dark:bg-amber-500/10 dark:text-amber-200">
            <strong>Kısıt:</strong> Iₐ ekseni senaryo taramasıdır (gerçek kayıt zaman-serisi
            integrali değil); ampirik regresyonların tahmin bandı vardır.
          </div>
        </div>
      </Card>
    </div>
  );
}
