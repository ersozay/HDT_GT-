export default function RiskBadge({ p, threshold = 0.49 }: { p: number; threshold?: number }) {
  let label: string, cls: string;
  if (p >= 0.75) {
    label = "Yüksek Risk";
    cls = "bg-rose-100 text-rose-700 dark:bg-rose-900/40 dark:text-rose-200";
  } else if (p >= threshold) {
    label = "Orta Risk";
    cls = "bg-amber-100 text-amber-700 dark:bg-amber-900/40 dark:text-amber-200";
  } else {
    label = "Düşük Risk";
    cls = "bg-emerald-100 text-emerald-700 dark:bg-emerald-900/40 dark:text-emerald-200";
  }
  return (
    <span className={`inline-flex items-center gap-1.5 rounded-full px-3 py-1 text-sm font-semibold ${cls}`}>
      <span className="h-2 w-2 rounded-full bg-current" />
      {label}
    </span>
  );
}
