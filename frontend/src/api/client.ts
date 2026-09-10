// Tip güvenli API istemcisi — vite proxy /api → :8000

export interface PointResult {
  p: number;
  ky: number;
  dn_cm: number;
  label: number;
  threshold: number;
  static_marginal: boolean;
}

export interface CriticalResult {
  pga: number[];
  prob: number[];
  dn_cm: number[];
  critical_pga: number | null;
  threshold: number;
}

export interface Lithology {
  id: number;
  description: string;
  c_prime: number;
  phi_prime: number;
  gamma: number;
}

export interface LatLonResult extends PointResult {
  lat: number;
  lon: number;
  in_bounds: boolean;
  slope: number;
  elevation: number;
  aspect: number;
  pga: number;
  litho_id: number;
  litho_name: string;
}

export interface MapMeta {
  bounds: [[number, number], [number, number]];
  center: [number, number];
  width: number;
  height: number;
  vmin: number;
  vmax: number;
  threshold: number;
}

export interface Summary {
  roc_auc: number;
  accuracy: number;
  recall: number;
  precision: number;
  f1: number;
  critical_pga_ml: number;
  n_samples: number;
  landslide_ratio: number;
  n_features: number;
  threshold: number;
  shap_order: string[];
  split: string;
  earthquake: string;
  notes: { lithology: string; fem: string };
  model_epoch: number;
  model_val_auc: number;
  dn_threshold_cm: number;
}

async function post<T>(url: string, body: unknown): Promise<T> {
  const r = await fetch(url, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
  if (!r.ok) throw new Error(`${url} → ${r.status}`);
  return r.json();
}

async function get<T>(url: string): Promise<T> {
  const r = await fetch(url);
  if (!r.ok) throw new Error(`${url} → ${r.status}`);
  return r.json();
}

export interface PointInput {
  slope: number;
  elevation: number;
  aspect: number;
  litho_id: number;
  pga: number;
}

export interface InventoryItem {
  id: string;
  lat: number;
  lon: number;
}

export interface LithologyLegendItem {
  id: number;
  description: string;
  color: string;
}

export interface RainfallMeta {
  available: boolean;
  bounds?: [[number, number], [number, number]];
  vmin?: number;
  vmax?: number;
  unit?: string;
  note?: string;
}

export interface NewmarkCurves {
  ky: number;
  fs_static: number;
  pga: number[];
  dn_jibson_cm: number[];
  dn_ambraseys_cm: number[];
  critical_pga_jibson: number | null;
  critical_pga_ambraseys: number | null;
  dn_threshold_cm: number;
}

export interface Validation {
  title: string;
  method: string;
  published: Record<string, string>;
  dn_threshold_cm: number;
  comparison: {
    pga_g: number;
    class: string;
    ky_g: number;
    dn_jibson_cm: number;
    dn_ambraseys_cm: number;
    triggered_jibson: boolean;
    triggered_ambraseys: boolean;
  }[];
  sources: string[];
  caveats: string[];
}

export const api = {
  predictPoint: (b: PointInput) => post<PointResult>("/api/predict/point", b),
  criticalPga: (b: Omit<PointInput, "pga"> & { pga_min?: number; pga_max?: number; step?: number }) =>
    post<CriticalResult>("/api/predict/critical-pga", b),
  newmarkCurves: (b: Omit<PointInput, "pga"> & { pga_min?: number; pga_max?: number; step?: number }) =>
    post<NewmarkCurves>("/api/predict/newmark-curves", b),
  lithology: () => get<Lithology[]>("/api/lithology"),
  lithologyLegend: () => get<LithologyLegendItem[]>("/api/map/lithology/legend"),
  lithologyMeta: () => get<{ bounds: [[number, number], [number, number]] }>("/api/map/lithology/meta"),
  rainfallMeta: () => get<RainfallMeta>("/api/map/rainfall/meta"),
  mapMeta: () => get<MapMeta>("/api/map/meta"),
  mapPoint: (lat: number, lon: number) =>
    get<LatLonResult>(`/api/map/point?lat=${lat}&lon=${lon}`),
  inventory: () => get<InventoryItem[]>("/api/inventory"),
  inventoryPoint: (id: string) => get<LatLonResult>(`/api/inventory/${encodeURIComponent(id)}`),
  summary: () => get<Summary>("/api/results/summary"),
  validation: () => get<Validation>("/api/results/validation"),
  figureUrl: (name: string) => `/api/figures/${name}`,
  overlayUrl: () => `/api/map/overlay.png`,
  lithologyOverlayUrl: () => `/api/map/lithology.png`,
  rainfallOverlayUrl: () => `/api/map/rainfall.png`,
};
