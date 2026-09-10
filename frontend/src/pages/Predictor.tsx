import { useEffect, useState } from "react";
import {
  api,
  Lithology,
  PointResult,
  PointInput,
  InventoryItem,
  LatLonResult,
} from "../api/client";
import { Card, CardTitle } from "../components/Card";
import Gauge from "../components/Gauge";
import RiskBadge from "../components/RiskBadge";
import Spinner from "../components/Spinner";

function Slider({
  label,
  value,
  min,
  max,
  step,
  unit,
  onChange,
}: {
  label: string;
  value: number;
  min: number;
  max: number;
  step: number;
  unit: string;
  onChange: (v: number) => void;
}) {
  return (
    <div>
      <div className="mb-1 flex items-center justify-between">
        <label className="text-sm font-medium text-slate-600 dark:text-slate-300">
          {label}
        </label>
        <span className="rounded-md bg-slate-100 px-2 py-0.5 text-sm font-semibold tabular-nums text-slate-700 dark:bg-slate-800 dark:text-slate-200">
          {value}
          <span className="ml-0.5 text-xs font-normal text-slate-400">{unit}</span>
        </span>
      </div>
      <input
        type="range"
        className="input-range"
        min={min}
        max={max}
        step={step}
        value={value}
        onChange={(e) => onChange(Number(e.target.value))}
      />
    </div>
  );
}

export default function Predictor() {
  const [litho, setLitho] = useState<Lithology[]>([]);
  const [input, setInput] = useState<PointInput>({
    slope: 35,
    elevation: 1200,
    aspect: 180,
    litho_id: 11,
    pga: 0.6,
  });
  const [result, setResult] = useState<PointResult | null>(null);
  const [loading, setLoading] = useState(false);
  const [err, setErr] = useState<string | null>(null);

  // Envanter / koordinat ile otomatik doldurma (istek 11)
  const [inv, setInv] = useState<InventoryItem[]>([]);
  const [invId, setInvId] = useState("");
  const [lat, setLat] = useState("");
  const [lon, setLon] = useState("");
  const [autoLoading, setAutoLoading] = useState(false);
  const [autoNote, setAutoNote] = useState<string | null>(null);

  useEffect(() => {
    api.lithology().then(setLitho).catch(() => {});
    api.inventory().then(setInv).catch(() => {});
  }, []);

  const applyLatLon = (r: LatLonResult, label: string) => {
    if (!r.in_bounds) {
      setAutoNote(`${label}: çalışma alanı dışında — öznitelik alınamadı.`);
      return;
    }
    setInput({
      slope: Math.round(r.slope),
      elevation: Math.round(r.elevation),
      aspect: Math.round(r.aspect),
      litho_id: r.litho_id,
      pga: Number(r.pga.toFixed(2)),
    });
    setAutoNote(
      `${label} → ${r.lat.toFixed(3)}, ${r.lon.toFixed(3)} · ${r.litho_name}`,
    );
  };

  const loadInventory = (id: string) => {
    if (!id) return;
    setAutoLoading(true);
    setAutoNote(null);
    api
      .inventoryPoint(id)
      .then((r) => applyLatLon(r, `Envanter ID ${id}`))
      .catch(() => setAutoNote(`Envanter ID ${id} bulunamadı.`))
      .finally(() => setAutoLoading(false));
  };

  const loadCoord = () => {
    const la = Number(lat);
    const lo = Number(lon);
    if (!Number.isFinite(la) || !Number.isFinite(lo)) {
      setAutoNote("Geçerli lat/lon girin.");
      return;
    }
    setAutoLoading(true);
    setAutoNote(null);
    api
      .mapPoint(la, lo)
      .then((r) => applyLatLon(r, "Koordinat"))
      .catch(() => setAutoNote("Koordinat için veri alınamadı."))
      .finally(() => setAutoLoading(false));
  };

  // girdi değişince otomatik tahmin (debounce)
  useEffect(() => {
    setLoading(true);
    setErr(null);
    const t = setTimeout(() => {
      api
        .predictPoint(input)
        .then(setResult)
        .catch((e) => setErr(String(e)))
        .finally(() => setLoading(false));
    }, 250);
    return () => clearTimeout(t);
  }, [input]);

  const set = (k: keyof PointInput, v: number) =>
    setInput((s) => ({ ...s, [k]: v }));

  const presets = [
    { name: "Pilot 1327 · Yamaç molozu", slope: 28, elevation: 1100, aspect: 200, litho_id: 11, pga: 0.6 },
    { name: "Pilot 1353 · Marn", slope: 22, elevation: 1000, aspect: 160, litho_id: 7, pga: 0.6 },
    { name: "Düşük risk · Alüvyon düzlük", slope: 4, elevation: 500, aspect: 90, litho_id: 1, pga: 0.3 },
  ];

  return (
    <div className="space-y-6">
      <div>
        <h1 className="text-2xl font-bold tracking-tight text-slate-900 dark:text-white">
          Nokta Tahmini
        </h1>
        <p className="mt-1 text-sm text-slate-500 dark:text-slate-400">
          Yamaç parametrelerini değiştirin; model anlık olarak heyelan olasılığını,
          kritik ivmeyi (K<sub>y</sub>) ve Newmark yer değiştirmesini (D<sub>n</sub>)
          hesaplar.
        </p>
      </div>

      <div className="grid gap-6 lg:grid-cols-2">
        {/* Girdi paneli */}
        <Card>
          <CardTitle hint="Envanter ID · koordinat · manuel">Girdiler</CardTitle>

          {/* Otomatik doldurma: envanter ID ara / koordinat gir (istek 11) */}
          <div className="mb-5 space-y-3 rounded-xl border border-slate-200 bg-slate-50 p-3 dark:border-slate-700 dark:bg-slate-800/50">
            <div>
              <label className="mb-1 block text-xs font-semibold text-slate-500 dark:text-slate-400">
                Envanter ID ile seç ({inv.length} kayıt)
              </label>
              <div className="flex gap-2">
                <input
                  list="inv-list"
                  value={invId}
                  onChange={(e) => setInvId(e.target.value)}
                  onKeyDown={(e) => e.key === "Enter" && loadInventory(invId)}
                  placeholder="örn. 1353"
                  className="w-full rounded-lg border border-slate-200 bg-white px-3 py-1.5 text-sm text-slate-700 outline-none focus:border-brand-500 dark:border-slate-700 dark:bg-slate-900 dark:text-slate-200"
                />
                <datalist id="inv-list">
                  {inv.slice(0, 500).map((it) => (
                    <option key={it.id} value={it.id} />
                  ))}
                </datalist>
                <button
                  onClick={() => loadInventory(invId)}
                  className="rounded-lg bg-brand-600 px-3 py-1.5 text-sm font-medium text-white transition hover:bg-brand-500"
                >
                  Getir
                </button>
              </div>
            </div>
            <div>
              <label className="mb-1 block text-xs font-semibold text-slate-500 dark:text-slate-400">
                Koordinat gir (lat / lon)
              </label>
              <div className="flex gap-2">
                <input
                  value={lat}
                  onChange={(e) => setLat(e.target.value)}
                  onKeyDown={(e) => e.key === "Enter" && loadCoord()}
                  placeholder="enlem"
                  className="w-full rounded-lg border border-slate-200 bg-white px-3 py-1.5 text-sm text-slate-700 outline-none focus:border-brand-500 dark:border-slate-700 dark:bg-slate-900 dark:text-slate-200"
                />
                <input
                  value={lon}
                  onChange={(e) => setLon(e.target.value)}
                  onKeyDown={(e) => e.key === "Enter" && loadCoord()}
                  placeholder="boylam"
                  className="w-full rounded-lg border border-slate-200 bg-white px-3 py-1.5 text-sm text-slate-700 outline-none focus:border-brand-500 dark:border-slate-700 dark:bg-slate-900 dark:text-slate-200"
                />
                <button
                  onClick={loadCoord}
                  className="rounded-lg bg-brand-600 px-3 py-1.5 text-sm font-medium text-white transition hover:bg-brand-500"
                >
                  Getir
                </button>
              </div>
            </div>
            {autoLoading && <Spinner label="Öznitelikler alınıyor…" />}
            {autoNote && (
              <p className="text-[11px] leading-relaxed text-slate-500 dark:text-slate-400">
                {autoNote}
              </p>
            )}
          </div>

          <div className="mb-2 text-xs font-semibold uppercase tracking-wide text-slate-400">
            Hazır senaryolar
          </div>
          <div className="mb-5 flex flex-wrap gap-2">
            {presets.map((p) => (
              <button
                key={p.name}
                onClick={() => setInput({ slope: p.slope, elevation: p.elevation, aspect: p.aspect, litho_id: p.litho_id, pga: p.pga })}
                className="rounded-lg border border-slate-200 bg-slate-50 px-2.5 py-1 text-xs font-medium text-slate-600 transition hover:border-brand-400 hover:text-brand-600 dark:border-slate-700 dark:bg-slate-800 dark:text-slate-300 dark:hover:border-brand-500 dark:hover:text-brand-300"
                title="Tez pilot noktası — hazır senaryo"
              >
                {p.name}
              </button>
            ))}
          </div>
          <div className="space-y-5">
            <Slider label="Eğim" value={input.slope} min={0} max={70} step={1} unit="°" onChange={(v) => set("slope", v)} />
            <Slider label="Yükseklik" value={input.elevation} min={0} max={3000} step={50} unit="m" onChange={(v) => set("elevation", v)} />
            <Slider label="Bakı" value={input.aspect} min={0} max={359} step={1} unit="°" onChange={(v) => set("aspect", v)} />
            <Slider label="PGA (yer ivmesi)" value={input.pga} min={0.05} max={1.5} step={0.05} unit="g" onChange={(v) => set("pga", v)} />
            <div>
              <label className="mb-1 block text-sm font-medium text-slate-600 dark:text-slate-300">
                Litoloji (jeolojik birim)
              </label>
              <select
                className="w-full rounded-xl border border-slate-200 bg-white px-3 py-2 text-sm text-slate-700 outline-none focus:border-brand-500 dark:border-slate-700 dark:bg-slate-900 dark:text-slate-200"
                value={input.litho_id}
                onChange={(e) => set("litho_id", Number(e.target.value))}
              >
                {litho.map((l) => (
                  <option key={l.id} value={l.id}>
                    {l.id} — {l.description}
                  </option>
                ))}
              </select>
            </div>
          </div>
        </Card>

        {/* Sonuç paneli */}
        <Card>
          <div className="flex items-center justify-between">
            <CardTitle hint="Model çıktısı">Sonuç</CardTitle>
            {loading && <Spinner label="Hesaplanıyor…" />}
          </div>

          {err ? (
            <div className="rounded-xl bg-rose-50 p-4 text-sm text-rose-700 dark:bg-rose-900/30 dark:text-rose-200">
              Hata: {err}
            </div>
          ) : result ? (
            <div className="space-y-5">
              <Gauge value={result.p} threshold={result.threshold} />
              <div className="flex justify-center">
                <RiskBadge p={result.p} threshold={result.threshold} />
              </div>
              <div className="grid grid-cols-2 gap-3">
                <div className="rounded-xl border border-slate-200 p-4 dark:border-slate-800">
                  <div className="text-xs font-semibold uppercase text-slate-500 dark:text-slate-400">
                    Kritik ivme K<sub>y</sub>
                  </div>
                  <div className="mt-1 text-2xl font-bold text-slate-900 dark:text-white">
                    {result.ky.toFixed(3)}
                    <span className="ml-1 text-sm font-medium text-slate-400">g</span>
                  </div>
                </div>
                <div className="rounded-xl border border-slate-200 p-4 dark:border-slate-800">
                  <div className="text-xs font-semibold uppercase text-slate-500 dark:text-slate-400">
                    Newmark D<sub>n</sub>
                  </div>
                  <div className="mt-1 text-2xl font-bold text-slate-900 dark:text-white">
                    {result.dn_cm.toFixed(1)}
                    <span className="ml-1 text-sm font-medium text-slate-400">cm</span>
                  </div>
                </div>
              </div>
              {result.static_marginal && (
                <p className="rounded-lg bg-amber-50 px-3 py-2 text-xs text-amber-700 dark:bg-amber-900/30 dark:text-amber-200">
                  ⚠ Statik güvenlik zaten sınırda — bu yamaç deprem olmadan da
                  marjinal kararlıdır.
                </p>
              )}
            </div>
          ) : (
            <Spinner />
          )}
        </Card>
      </div>
    </div>
  );
}
