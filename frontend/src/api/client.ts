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

export interface MeanStd {
  mean: number;
  std: number;
}

export interface BenchmarkModel {
  label: string;
  roc_auc?: MeanStd;
  pr_auc?: MeanStd;
  f1?: MeanStd;
  kappa?: MeanStd;
  brier?: MeanStd;
  roc_auc_steep?: MeanStd;
  mono_violation_rate?: MeanStd;
}

export interface Benchmark {
  protocol: Record<string, string>;
  models: Record<string, BenchmarkModel>;
  reference_polygon_split_pinn_v2?: Record<string, number>;
}

export interface DesignRow {
  slope_class: string;
  slope_lo: number;
  litho_id: number;
  litho: string;
  n_cells: number;
  critical_pga_g: number | null;
  range_lo_g: number | null;
  range_hi_g: number | null;
  onset_lr1_g: number | null;
}

export interface CriticalPgaReport {
  critical_pga: {
    definition: string;
    critical_pga_g: number | null;
    ensemble_range_g: [number, number] | null;
    bootstrap_95ci_g: [number, number] | null;
    onset_lr1_g: number | null;
    newmark_critical_pga_g: number | null;
    newmark_max_fraction: number;
    steep_slope_threshold_deg: number;
    n_cells: number;
  };
  empirical_validation: {
    empirical_critical_pga_fr1_g: number | null;
    spearman_binned: number;
    spearman_cells: number;
    landslide_area_prevalence: number;
  };
  neg_ratio_sensitivity:
    | {
        neg_pos_ratio: number;
        "critical_pga_at_0.5_mean": number | null;
        "critical_pga_at_0.5_sd": number | null;
        critical_pga_onset_lr1_mean: number | null;
        test_roc_auc_mean: number;
        n_seeds: number;
      }[]
    | null;
  design_table?: DesignRow[];
}

export interface FemPilot {
  slope_id: number;
  lithology: string;
  theta_deg: number;
  c_kpa: number;
  phi_deg: number;
  fs_fem: number;
  ky_fem_g: number;
  fs_infinite_slope: number;
  ky_infinite_slope_g: number;
  actual_pga_g: number;
  dn_newmark_ky_infinite_cm: number;
  dn_newmark_ky_fem_cm: number;
  triggered_ky_infinite: boolean;
  triggered_ky_fem: boolean;
}

export interface Summary {
  model_version?: string;
  roc_auc_steep?: number;
  pr_auc?: number;
  kappa?: number;
  mono_violation_rate?: number;
  cv?: Record<string, MeanStd>;
  v1_reference?: { roc_auc: number; split: string; note: string };
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
  notes: { lithology: string; fem: string; critical_pga?: string; evaluation?: string };
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
  benchmark: () => get<Benchmark>("/api/results/benchmark"),
  criticalReport: () => get<CriticalPgaReport>("/api/results/critical-pga"),
  fem: () => get<FemPilot[]>("/api/results/fem"),
  figureUrl: (name: string) => `/api/figures/${name}`,
  overlayUrl: () => `/api/map/overlay.png`,
  lithologyOverlayUrl: () => `/api/map/lithology.png`,
  rainfallOverlayUrl: () => `/api/map/rainfall.png`,
};
