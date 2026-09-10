import { ReactNode } from "react";

export default function MetricCard({
  label,
  value,
  unit,
  sub,
  accent = "brand",
  icon,
}: {
  label: string;
  value: string | number;
  unit?: string;
  sub?: string;
  accent?: "brand" | "amber" | "rose" | "emerald" | "sky" | "violet";
  icon?: ReactNode;
}) {
  const accents: Record<string, string> = {
    brand: "text-brand-600 dark:text-brand-300",
    amber: "text-amber-600 dark:text-amber-300",
    rose: "text-rose-600 dark:text-rose-300",
    emerald: "text-emerald-600 dark:text-emerald-300",
    sky: "text-sky-600 dark:text-sky-300",
    violet: "text-violet-600 dark:text-violet-300",
  };
  return (
    <div className="card p-5">
      <div className="flex items-start justify-between">
        <span className="text-xs font-semibold uppercase tracking-wide text-slate-500 dark:text-slate-400">
          {label}
        </span>
        {icon && <span className={accents[accent]}>{icon}</span>}
      </div>
      <div className="mt-2 flex items-baseline gap-1">
        <span className={`text-3xl font-extrabold tracking-tight ${accents[accent]}`}>
          {value}
        </span>
        {unit && (
          <span className="text-sm font-semibold text-slate-400 dark:text-slate-500">
            {unit}
          </span>
        )}
      </div>
      {sub && <p className="mt-1 text-xs text-slate-500 dark:text-slate-400">{sub}</p>}
    </div>
  );
}
