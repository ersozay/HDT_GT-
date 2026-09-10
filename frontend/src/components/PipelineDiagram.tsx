/**
 * PipelineDiagram — Newmark'ın CBS (GIS) tabanlı dinamik sisteme entegrasyonunu
 * gösteren SVG akış şeması (danışmanın 1. isteği). Koyu/açık tema uyumlu:
 * renkler currentColor + Tailwind sınıflarıyla değil, tema-nötr paletle verilir;
 * metinler CSS değişkeni yerine doğrudan okunur tonlarda.
 */

interface NodeDef {
  id: string;
  x: number;
  y: number;
  w: number;
  h: number;
  title: string;
  sub?: string;
  fill: string;
  stroke: string;
}

// Üç sütunlu akış: Rasterlar → Öznitelik/Fizik → Model/Çıktı
const NODES: NodeDef[] = [
  // Sütun 1 — CBS raster girdileri
  { id: "dem", x: 20, y: 20, w: 150, h: 46, title: "DEM (GLO-90)", sub: "eğim · bakı · yükseklik", fill: "#ecfdf5", stroke: "#10b981" },
  { id: "pga", x: 20, y: 84, w: 150, h: 46, title: "PGA rasteri", sub: "USGS ShakeMap zarfı", fill: "#fff7ed", stroke: "#f59e0b" },
  { id: "litho", x: 20, y: 148, w: 150, h: 46, title: "Litoloji rasteri", sub: "c′, φ′, γ (sınıf tablo)", fill: "#faf5ff", stroke: "#a855f7" },
  { id: "rain", x: 20, y: 212, w: 150, h: 46, title: "Yağış (CHIRPS)", sub: "yalnız görsel — modele girmez", fill: "#eff6ff", stroke: "#3b82f6" },

  // Sütun 2 — fizik / öznitelik
  { id: "fs", x: 230, y: 52, w: 160, h: 46, title: "Mohr–Coulomb", sub: "StaticFS → kᵧ", fill: "#f0fdfa", stroke: "#2c807b" },
  { id: "newmark", x: 230, y: 148, w: 160, h: 46, title: "Newmark rijit-blok", sub: "Jibson · Ambraseys → Dₙ", fill: "#f0fdfa", stroke: "#2c807b" },

  // Sütun 3 — model / çıktı
  { id: "pinn", x: 450, y: 52, w: 160, h: 46, title: "PINN (39→…→1)", sub: "Focal + fizik cezası", fill: "#eef2ff", stroke: "#6366f1" },
  { id: "crit", x: 450, y: 148, w: 160, h: 46, title: "Kritik PGA", sub: "P≥τ  ·  Dₙ=10 cm", fill: "#fef2f2", stroke: "#ef4444" },
  { id: "map", x: 450, y: 226, w: 160, h: 46, title: "Duyarlılık haritası", sub: "GeoTIFF + web overlay", fill: "#f8fafc", stroke: "#64748b" },
];

interface EdgeDef {
  from: string;
  to: string;
  dashed?: boolean;
}

const EDGES: EdgeDef[] = [
  { from: "dem", to: "fs" },
  { from: "litho", to: "fs" },
  { from: "pga", to: "newmark" },
  { from: "litho", to: "newmark" },
  { from: "fs", to: "newmark" },
  { from: "fs", to: "pinn" },
  { from: "newmark", to: "pinn" },
  { from: "pinn", to: "crit" },
  { from: "newmark", to: "crit" },
  { from: "crit", to: "map" },
  { from: "pinn", to: "map" },
  { from: "rain", to: "map", dashed: true },
];

function byId(id: string) {
  return NODES.find((n) => n.id === id)!;
}

/** Sağ-kenar merkezinden hedef sol-kenar merkezine yatay yönelimli path. */
function edgePath(a: NodeDef, b: NodeDef): string {
  const x1 = a.x + a.w;
  const y1 = a.y + a.h / 2;
  const x2 = b.x;
  const y2 = b.y + b.h / 2;
  const mx = (x1 + x2) / 2;
  return `M ${x1} ${y1} C ${mx} ${y1}, ${mx} ${y2}, ${x2} ${y2}`;
}

export default function PipelineDiagram() {
  return (
    <div className="overflow-x-auto">
      <svg
        viewBox="0 0 630 290"
        className="h-auto w-full min-w-[560px] text-slate-700 dark:text-slate-200"
        role="img"
        aria-label="CBS tabanlı Newmark–PINN entegrasyon akış şeması"
      >
        <defs>
          <marker
            id="arrow"
            viewBox="0 0 10 10"
            refX="9"
            refY="5"
            markerWidth="7"
            markerHeight="7"
            orient="auto-start-reverse"
          >
            <path d="M 0 0 L 10 5 L 0 10 z" fill="#94a3b8" />
          </marker>
        </defs>

        {/* Sütun başlıkları */}
        <text x="95" y="12" textAnchor="middle" className="fill-slate-400" fontSize="10" fontWeight="600">
          CBS RASTER GİRDİLERİ
        </text>
        <text x="310" y="42" textAnchor="middle" className="fill-slate-400" fontSize="10" fontWeight="600">
          FİZİK ÇEKİRDEĞİ
        </text>
        <text x="530" y="42" textAnchor="middle" className="fill-slate-400" fontSize="10" fontWeight="600">
          MODEL &amp; ÇIKTI
        </text>

        {/* Kenarlar */}
        {EDGES.map((e, i) => (
          <path
            key={i}
            d={edgePath(byId(e.from), byId(e.to))}
            fill="none"
            stroke="#94a3b8"
            strokeWidth={1.4}
            strokeDasharray={e.dashed ? "4 3" : undefined}
            markerEnd="url(#arrow)"
            opacity={e.dashed ? 0.6 : 0.9}
          />
        ))}

        {/* Düğümler */}
        {NODES.map((n) => (
          <g key={n.id}>
            <rect
              x={n.x}
              y={n.y}
              width={n.w}
              height={n.h}
              rx={8}
              fill={n.fill}
              stroke={n.stroke}
              strokeWidth={1.6}
            />
            <text
              x={n.x + n.w / 2}
              y={n.y + (n.sub ? 20 : n.h / 2 + 4)}
              textAnchor="middle"
              fontSize="11.5"
              fontWeight="700"
              fill="#0f172a"
            >
              {n.title}
            </text>
            {n.sub && (
              <text
                x={n.x + n.w / 2}
                y={n.y + 35}
                textAnchor="middle"
                fontSize="9"
                fill="#475569"
              >
                {n.sub}
              </text>
            )}
          </g>
        ))}
      </svg>
      <p className="mt-2 text-center text-[11px] text-slate-400 dark:text-slate-500">
        Kesikli ok: yağış katmanı yalnız görsel bağlamdır; PINN'e girdi değildir.
        Rasterlar CBS'de örneklenir → fizik (kᵧ, Dₙ) hesaplanır → PINN olasılığı ve
        kritik PGA üretir → harita olarak sunulur.
      </p>
    </div>
  );
}
