import { useEffect, useState } from "react";

/** Yarım daire (180°) olasılık göstergesi — 0..1 arası P(heyelan). */
export default function Gauge({
  value,
  threshold = 0.49,
  size = 240,
}: {
  value: number; // 0..1
  threshold?: number;
  size?: number;
}) {
  const [anim, setAnim] = useState(0);
  useEffect(() => {
    const id = requestAnimationFrame(() => setAnim(value));
    return () => cancelAnimationFrame(id);
  }, [value]);

  const w = size;
  const h = size / 2 + 24;
  const cx = w / 2;
  const cy = size / 2 + 6;
  const r = size / 2 - 16;
  const stroke = 16;

  // yarım daire uzunluğu
  const arcLen = Math.PI * r;
  const pct = Math.max(0, Math.min(1, anim));

  // renk: eşiğin altı yeşil→sarı, üstü turuncu→kırmızı
  const color =
    value >= 0.75 ? "#dc2626" : value >= threshold ? "#f59e0b" : "#16a34a";

  // eşik işareti açısı (180° sol=0, sağ=1)
  const thAngle = Math.PI * (1 - threshold);
  const thx = cx + r * Math.cos(thAngle);
  const thy = cy - r * Math.sin(thAngle);

  const describeArc = (frac: number) => {
    const a = Math.PI * (1 - frac);
    const x = cx + r * Math.cos(a);
    const y = cy - r * Math.sin(a);
    const large = frac > 0.5 ? 1 : 0;
    return `M ${cx - r} ${cy} A ${r} ${r} 0 ${large} 1 ${x} ${y}`;
  };

  return (
    <div className="flex flex-col items-center">
      <svg width={w} height={h} viewBox={`0 0 ${w} ${h}`}>
        {/* zemin yay */}
        <path
          d={`M ${cx - r} ${cy} A ${r} ${r} 0 1 1 ${cx + r} ${cy}`}
          fill="none"
          className="stroke-slate-200 dark:stroke-slate-800"
          strokeWidth={stroke}
          strokeLinecap="round"
        />
        {/* değer yayı */}
        <path
          d={`M ${cx - r} ${cy} A ${r} ${r} 0 1 1 ${cx + r} ${cy}`}
          fill="none"
          stroke={color}
          strokeWidth={stroke}
          strokeLinecap="round"
          strokeDasharray={arcLen}
          strokeDashoffset={arcLen * (1 - pct)}
          style={{ transition: "stroke-dashoffset 900ms cubic-bezier(.22,1,.36,1)" }}
        />
        {/* eşik çizgisi */}
        <circle cx={thx} cy={thy} r={4} className="fill-slate-400 dark:fill-slate-500" />
        {/* merkez değer */}
        <text
          x={cx}
          y={cy - 6}
          textAnchor="middle"
          className="fill-slate-900 dark:fill-white"
          style={{ fontSize: 40, fontWeight: 800 }}
        >
          {Math.round(value * 100)}
          <tspan style={{ fontSize: 18, fontWeight: 700 }}>%</tspan>
        </text>
        <text
          x={cx}
          y={cy + 16}
          textAnchor="middle"
          className="fill-slate-500 dark:fill-slate-400"
          style={{ fontSize: 12, fontWeight: 600 }}
        >
          P(heyelan)
        </text>
      </svg>
      <div className="text-xs text-slate-400 dark:text-slate-500">
        Karar eşiği: {(threshold * 100).toFixed(0)}%
      </div>
    </div>
  );
}
