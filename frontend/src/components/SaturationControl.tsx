import { useState } from "react";
import { api, Moisture } from "../api/client";

/**
 * Yağış senaryosu: su tablası oranı m (0 kuru – 1 yüzeye kadar doygun).
 * u = m·γw·z·cos²θ → Kᵧ düşer, Dₙ artar. Hazır seçenekler: 6 Şubat 2023 nemi
 * (ERA5-Land) ve bugünkü / 3 günlük tahmin (Open-Meteo).
 */
export default function SaturationControl({
  value,
  onChange,
  lat = 37.752,
  lon = 37.979,
}: {
  value: number;
  onChange: (v: number) => void;
  lat?: number;
  lon?: number;
}) {
  const [m, setM] = useState<Moisture | null>(null);
  const [busy, setBusy] = useState(false);
  const load = (pick: (x: Moisture) => number | null | undefined) => {
    setBusy(true);
    api
      .moisture(lat, lon)
      .then((x) => {
        setM(x);
        const v = pick(x);
        if (v != null) onChange(Number(v.toFixed(2)));
      })
      .catch(() => {})
      .finally(() => setBusy(false));
  };
  const btn =
    "rounded-lg border border-slate-200 px-2.5 py-1 text-xs font-medium text-slate-600 hover:bg-slate-100 disabled:opacity-50 dark:border-slate-700 dark:text-slate-300 dark:hover:bg-slate-800";
  return (
    <div>
      <div className="mb-1 flex items-center justify-between">
        <label className="text-sm font-medium text-slate-600 dark:text-slate-300">
          Zemin doygunluğu (yağış) m
        </label>
        <span className="rounded-md bg-sky-50 px-2 py-0.5 text-sm font-semibold tabular-nums text-sky-700 dark:bg-sky-900/30 dark:text-sky-200">
          {value.toFixed(2)}
        </span>
      </div>
      <input
        type="range"
        className="input-range"
        min={0}
        max={1}
        step={0.05}
        value={value}
        onChange={(e) => onChange(Number(e.target.value))}
      />
      <div className="mt-2 flex flex-wrap gap-1.5">
        <button className={btn} onClick={() => onChange(0)}>Kuru</button>
        <button className={btn} disabled={busy} onClick={() => load((x) => x.event_2023_02_06)}>
          6 Şubat 2023
        </button>
        <button className={btn} disabled={busy} onClick={() => load((x) => x.now)}>
          Bugün
        </button>
        <button className={btn} disabled={busy} onClick={() => load((x) => x.forecast_3d_max ?? x.now)}>
          3 gün tahmini
        </button>
      </div>
      <p className="mt-1.5 text-[11px] leading-relaxed text-slate-400 dark:text-slate-500">
        u = m·γ<sub>w</sub>·z·cos²θ (yamaca paralel sızma). m ≈ kök bölgesi doygunluğu.
        {m && m.forecast_3d_precip_mm != null && <> Önümüzdeki 3 gün yağış: {m.forecast_3d_precip_mm} mm.</>}
        {m?.live_error && <> Canlı tahmin alınamadı.</>}
      </p>
    </div>
  );
}
