import { useEffect, useMemo, useRef, useState } from "react";
import {
  MapContainer,
  TileLayer,
  ImageOverlay,
  useMapEvents,
  Marker,
  Popup,
  Rectangle,
  LayersControl,
  GeoJSON,
  CircleMarker,
  Tooltip,
} from "react-leaflet";
import L from "leaflet";
import {
  api,
  MapMeta,
  LatLonResult,
  LithologyLegendItem,
  RainfallMeta,
  MapLayerInfo,
  MapLayerMeta,
  NewmarkSim,
} from "../api/client";
import { Card, CardTitle } from "../components/Card";
import RiskBadge from "../components/RiskBadge";
import Spinner from "../components/Spinner";

// Leaflet varsayılan ikon yolu düzeltmesi (Vite)
import markerIcon from "leaflet/dist/images/marker-icon.png";
import markerIcon2x from "leaflet/dist/images/marker-icon-2x.png";
import markerShadow from "leaflet/dist/images/marker-shadow.png";
const DefaultIcon = L.icon({
  iconUrl: markerIcon,
  iconRetinaUrl: markerIcon2x,
  shadowUrl: markerShadow,
  iconSize: [25, 41],
  iconAnchor: [12, 41],
  popupAnchor: [1, -34],
});
L.Marker.prototype.options.icon = DefaultIcon;

function ClickHandler({ onClick }: { onClick: (lat: number, lon: number) => void }) {
  useMapEvents({
    click(e) {
      onClick(e.latlng.lat, e.latlng.lng);
    },
  });
  return null;
}

export default function MapView() {
  const [meta, setMeta] = useState<MapMeta | null>(null);
  const [opacity, setOpacity] = useState(0.7);
  const [showOverlay, setShowOverlay] = useState(true);
  const [pick, setPick] = useState<{ lat: number; lon: number } | null>(null);
  const [pt, setPt] = useState<LatLonResult | null>(null);
  const [loading, setLoading] = useState(false);
  const [legend, setLegend] = useState<LithologyLegendItem[]>([]);
  const [rain, setRain] = useState<RainfallMeta | null>(null);
  const [layerList, setLayerList] = useState<MapLayerInfo[]>([]);
  const [extra, setExtra] = useState<string>("");
  const [extraMeta, setExtraMeta] = useState<MapLayerMeta | null>(null);
  const [extraOpacity, setExtraOpacity] = useState(0.75);
  const [inventory, setInventory] = useState<GeoJSON.FeatureCollection | null>(null);
  const [faults, setFaults] = useState<GeoJSON.FeatureCollection | null>(null);
  const [stations, setStations] = useState<GeoJSON.FeatureCollection | null>(null);
  const [sim, setSim] = useState<NewmarkSim | null>(null);
  const [frame, setFrame] = useState<number>(-1); // -1 → simülasyon kapalı
  const [playing, setPlaying] = useState(false);
  const invRef = useRef<L.GeoJSON | null>(null);

  useEffect(() => {
    api.mapMeta().then(setMeta).catch(() => {});
    api.lithologyLegend().then(setLegend).catch(() => {});
    api.rainfallMeta().then(setRain).catch(() => {});
    api.mapLayers().then(setLayerList).catch(() => {});
    api.inventoryGeojson().then(setInventory).catch(() => {});
    api.faultsGeojson().then(setFaults).catch(() => {});
    api.stationsGeojson().then(setStations).catch(() => {});
    api.newmarkSim().then(setSim).catch(() => {});
  }, []);

  const simIndex = useMemo(() => {
    const m = new Map<string, number>();
    sim?.polygon_ids.forEach((id, i) => m.set(String(id), i));
    return m;
  }, [sim]);

  // Oynatma döngüsü
  useEffect(() => {
    if (!playing || !sim) return;
    const t = window.setInterval(() => {
      setFrame((f) => {
        const n = f + 1;
        if (n >= sim.times_s.length) {
          setPlaying(false);
          return sim.times_s.length - 1;
        }
        return n;
      });
    }, 180);
    return () => window.clearInterval(t);
  }, [playing, sim]);

  // Kare değişince poligon renklerini güncelle (katmanı yeniden kurmadan)
  useEffect(() => {
    const layer = invRef.current;
    if (!layer) return;
    layer.eachLayer((l) => {
      const f = (l as L.Polygon).feature as GeoJSON.Feature | undefined;
      const id = String((f?.properties as { id?: string })?.id ?? "");
      (l as L.Path).setStyle(polyStyle(sim, simIndex, frame, id));
    });
  }, [frame, sim, simIndex]);

  useEffect(() => {
    setExtraMeta(null);
    if (extra) api.mapLayerMeta(extra).then(setExtraMeta).catch(() => {});
  }, [extra]);

  const handleClick = (lat: number, lon: number) => {
    setPick({ lat, lon });
    setPt(null);
    setLoading(true);
    api
      .mapPoint(lat, lon)
      .then(setPt)
      .catch(() => {})
      .finally(() => setLoading(false));
  };

  if (!meta) {
    return (
      <div className="flex h-96 items-center justify-center">
        <Spinner label="Harita yükleniyor…" />
      </div>
    );
  }

  const bounds = meta.bounds as [[number, number], [number, number]];

  return (
    <div className="space-y-6">
      <div>
        <h1 className="text-2xl font-bold tracking-tight text-slate-900 dark:text-white">
          Bölgesel Duyarlılık Haritası
        </h1>
        <p className="mt-1 text-sm text-slate-500 dark:text-slate-400">
          Renkli katman, PINN modelinin ürettiği heyelan duyarlılığıdır (kırmızı =
          yüksek). Herhangi bir konuma <strong>tıklayarak</strong> anlık tahmin
          alın.
        </p>
      </div>

      <div className="grid gap-6 lg:grid-cols-3">
        {/* Harita */}
        <div className="lg:col-span-2">
          <div className="card overflow-hidden p-0">
            <MapContainer
              center={meta.center as [number, number]}
              zoom={8}
              minZoom={7}
              maxBounds={[
                [bounds[0][0] - 0.5, bounds[0][1] - 0.5],
                [bounds[1][0] + 0.5, bounds[1][1] + 0.5],
              ]}
              style={{ height: "520px", width: "100%" }}
              scrollWheelZoom
            >
              <LayersControl position="topright">
                <LayersControl.BaseLayer checked name="Topografik">
                  <TileLayer
                    url="https://{s}.tile.opentopomap.org/{z}/{x}/{y}.png"
                    attribution="© OpenTopoMap"
                  />
                </LayersControl.BaseLayer>
                <LayersControl.BaseLayer name="Sokak (OSM)">
                  <TileLayer
                    url="https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png"
                    attribution="© OpenStreetMap"
                  />
                </LayersControl.BaseLayer>

                {/* Litoloji — kategorik renk katmanı (istek 10) */}
                <LayersControl.Overlay name="Litoloji (kategorik)">
                  <ImageOverlay
                    url={api.lithologyOverlayUrl()}
                    bounds={bounds}
                    opacity={0.75}
                  />
                </LayersControl.Overlay>

                {inventory && (
                  <LayersControl.Overlay checked name="Heyelan envanteri (Kocaman 2025)">
                    <GeoJSON
                      ref={invRef}
                      data={inventory}
                      style={(f) => polyStyle(sim, simIndex, frame, String((f?.properties as { id?: string })?.id ?? ""))}
                      onEachFeature={(f, layer) => {
                        const p = f.properties as { id?: string; area_m2?: number; dn_th_cm?: number | null };
                        layer.bindTooltip(
                          `Heyelan ${p.id ?? ""} · ${Math.round(p.area_m2 ?? 0).toLocaleString("tr-TR")} m²` +
                            (p.dn_th_cm != null ? ` · Dₙ ${p.dn_th_cm.toFixed(1)} cm` : ""),
                          { sticky: true },
                        );
                      }}
                    />
                  </LayersControl.Overlay>
                )}
                {faults && (
                  <LayersControl.Overlay checked name="Faylar + 2023 kırılması">
                    <GeoJSON
                      data={faults}
                      style={(f) =>
                        (f?.properties as { kind?: string })?.kind === "rupture_2023"
                          ? { color: "#b91c1c", weight: 4, opacity: 0.9 }
                          : { color: "#111827", weight: 1.5, opacity: 0.8, dashArray: "4 3" }
                      }
                      onEachFeature={(f, layer) => {
                        const p = f.properties as { kind?: string; slip_type?: string };
                        layer.bindTooltip(
                          p.kind === "rupture_2023"
                            ? "6 Şubat 2023 kırılması (USGS)"
                            : `Aktif fay (GEM)${p.slip_type ? " · " + p.slip_type : ""}`,
                          { sticky: true },
                        );
                      }}
                    />
                  </LayersControl.Overlay>
                )}
                {stations && (
                  <LayersControl.Overlay name="AFAD ivme istasyonları (ESM)">
                    <>
                      {stations.features.map((f) => {
                        const [lon, lat] = (f.geometry as GeoJSON.Point).coordinates;
                        const p = f.properties as Record<string, number | string>;
                        const pga = Number(p.pga_pazarcik ?? p.pga_elbistan ?? 0);
                        return (
                          <CircleMarker
                            key={String(p.station)}
                            center={[lat, lon]}
                            radius={4 + Math.min(10, pga * 8)}
                            pathOptions={{ color: "#ffffff", weight: 1.5, fillColor: "#1d4ed8", fillOpacity: 0.85 }}
                          >
                            <Tooltip>
                              <div className="text-xs">
                                <b>TK.{p.station}</b>
                                <br />
                                Pazarcık PGA: {p.pga_pazarcik != null ? `${Number(p.pga_pazarcik).toFixed(2)} g` : "—"}
                                <br />
                                Elbistan PGA: {p.pga_elbistan != null ? `${Number(p.pga_elbistan).toFixed(2)} g` : "—"}
                              </div>
                            </Tooltip>
                          </CircleMarker>
                        );
                      })}
                    </>
                  </LayersControl.Overlay>
                )}

                <LayersControl.Overlay name="Duyarlılık — 6 Şubat 2023 zemin nemiyle (yağış senaryosu)">
                  <ImageOverlay url="/api/map/overlay_wet.png" bounds={bounds} opacity={0.8} />
                </LayersControl.Overlay>

                {/* Yağış — yalnız görsel bağlam (istek 9) */}
                {rain?.available && (
                  <LayersControl.Overlay name="Yağış (CHIRPS · görsel)">
                    <ImageOverlay
                      url={api.rainfallOverlayUrl()}
                      bounds={(rain.bounds ?? bounds) as [[number, number], [number, number]]}
                      opacity={0.7}
                    />
                  </LayersControl.Overlay>
                )}
              </LayersControl>

              {extra && extraMeta && (
                <ImageOverlay
                  key={extra}
                  url={api.mapLayerUrl(extra)}
                  bounds={extraMeta.bounds}
                  opacity={extraOpacity}
                />
              )}

              {showOverlay && (
                <ImageOverlay
                  url={api.overlayUrl()}
                  bounds={bounds}
                  opacity={opacity}
                />
              )}

              {/* çalışma alanı sınırı */}
              <Rectangle
                bounds={bounds}
                pathOptions={{ color: "#2c807b", weight: 2, fill: false, dashArray: "6 4" }}
              />

              <ClickHandler onClick={handleClick} />

              {pick && (
                <Marker position={[pick.lat, pick.lon]}>
                  <Popup>
                    {loading ? (
                      <Spinner label="Tahmin…" />
                    ) : pt ? (
                      <div className="space-y-1 text-xs">
                        <div className="font-semibold">
                          {pt.lat.toFixed(3)}, {pt.lon.toFixed(3)}
                        </div>
                        {pt.in_bounds ? (
                          <>
                            <div>P(heyelan): <b>{(pt.p * 100).toFixed(0)}%</b></div>
                            <div>Eğim: {pt.slope.toFixed(1)}° · PGA: {pt.pga.toFixed(2)}g</div>
                            <div>Litoloji: {pt.litho_name}</div>
                            <div>Kᵧ: {pt.ky.toFixed(3)}g · Dₙ: {pt.dn_cm.toFixed(1)}cm</div>
                          </>
                        ) : (
                          <div className="text-rose-600">Çalışma alanı dışında</div>
                        )}
                      </div>
                    ) : (
                      <div className="text-xs text-rose-600">Veri alınamadı</div>
                    )}
                  </Popup>
                </Marker>
              )}
            </MapContainer>
          </div>
        </div>

        {/* Kontrol + sonuç paneli */}
        <div className="space-y-4">
          <Card>
            <CardTitle hint="Katman ayarları">Görünüm</CardTitle>
            <label className="flex items-center gap-2 text-sm text-slate-600 dark:text-slate-300">
              <input
                type="checkbox"
                checked={showOverlay}
                onChange={(e) => setShowOverlay(e.target.checked)}
                className="accent-brand-500"
              />
              Duyarlılık katmanı
            </label>
            <div className="mt-4">
              <div className="mb-1 flex justify-between text-sm">
                <span className="font-medium text-slate-600 dark:text-slate-300">Opaklık</span>
                <span className="font-semibold tabular-nums text-slate-700 dark:text-slate-200">
                  {Math.round(opacity * 100)}%
                </span>
              </div>
              <input
                type="range"
                className="input-range"
                min={0}
                max={1}
                step={0.05}
                value={opacity}
                onChange={(e) => setOpacity(Number(e.target.value))}
              />
            </div>
            {/* renk skalası — sayısal */}
            <div className="mt-5">
              <div className="mb-1 text-[11px] font-medium text-slate-500 dark:text-slate-400">
                Heyelan duyarlılığı P
              </div>
              <div className="h-3 w-full rounded-full bg-gradient-to-r from-emerald-500 via-amber-400 to-rose-600" />
              <div className="mt-1 flex justify-between text-[11px] tabular-nums text-slate-400">
                <span>{meta.vmin.toFixed(2)}</span>
                <span>{((meta.vmin + meta.vmax) / 2).toFixed(2)}</span>
                <span>{meta.vmax.toFixed(2)}</span>
              </div>
              <div className="mt-2 flex items-center gap-1.5 text-[11px] text-slate-400">
                <span className="inline-block h-0 w-4 border-t-2 border-dashed border-brand-500" />
                Çalışma alanı sınırı
              </div>
            </div>
            <p className="mt-4 text-[11px] leading-relaxed text-slate-400 dark:text-slate-500">
              Sağ üstteki katman seçiciden envanter, faylar, 2023 kırılması, ivme
              istasyonları, litoloji ve yağış katmanlarını açıp kapatabilirsiniz.
            </p>
          </Card>

          {sim && (
            <Card>
              <CardTitle hint="Gerçek AFAD ivme kayıtları · rijit-blok Newmark (zaman-tanım)">
                Kayma Simülasyonu
              </CardTitle>
              <div className="flex items-center gap-2">
                <button
                  className="btn-primary px-3 py-1.5 text-sm"
                  onClick={() => {
                    if (frame < 0 || frame >= sim.times_s.length - 1) setFrame(0);
                    setPlaying((p) => !p);
                  }}
                >
                  {playing ? "Duraklat" : frame >= 0 ? "Devam / Yeniden" : "Depremi oynat"}
                </button>
                {frame >= 0 && (
                  <button
                    className="rounded-lg px-3 py-1.5 text-sm text-slate-500 hover:bg-slate-100 dark:hover:bg-slate-800"
                    onClick={() => {
                      setPlaying(false);
                      setFrame(-1);
                    }}
                  >
                    Kapat
                  </button>
                )}
              </div>
              {frame >= 0 && (
                <div className="mt-3 space-y-2">
                  <input
                    type="range"
                    className="input-range"
                    min={0}
                    max={sim.times_s.length - 1}
                    value={frame}
                    onChange={(e) => {
                      setPlaying(false);
                      setFrame(Number(e.target.value));
                    }}
                  />
                  <div className="flex justify-between text-xs tabular-nums text-slate-500 dark:text-slate-400">
                    <span>t = {sim.times_s[frame].toFixed(0)} s</span>
                    <span>
                      Dₙ ≥ {sim.dn_threshold_cm} cm:{" "}
                      <b className="text-rose-600 dark:text-rose-300">
                        {sim.history_cm.filter((h) => h[frame] >= sim.dn_threshold_cm).length}
                      </b>{" "}
                      / {sim.polygon_ids.length} poligon
                    </span>
                  </div>
                </div>
              )}
              <ul className="mt-3 space-y-1 text-xs text-slate-600 dark:text-slate-300">
                <li className="flex items-center gap-2"><span className="inline-block h-3 w-3 rounded-sm" style={{ background: SIM_COLORS[0] }} />Dₙ &lt; 1 cm</li>
                <li className="flex items-center gap-2"><span className="inline-block h-3 w-3 rounded-sm" style={{ background: SIM_COLORS[1] }} />1–10 cm</li>
                <li className="flex items-center gap-2"><span className="inline-block h-3 w-3 rounded-sm" style={{ background: SIM_COLORS[2] }} />≥ 10 cm (tetiklendi)</li>
              </ul>
              <p className="mt-3 text-[11px] leading-relaxed text-slate-400 dark:text-slate-500">
                Her poligonda en yakın AFAD istasyonunun kaydı, poligonun ShakeMap PGA&apos;sına
                ölçeklendi. Sonda poligonların %{(100 * sim.summary.share_triggered_10cm).toFixed(0)}&apos;i 10 cm&apos;yi
                aşıyor; %{(100 * sim.summary.share_pazarcik_dominant).toFixed(0)}&apos;inde Pazarcık depremi belirleyici.
              </p>
            </Card>
          )}

          {layerList.length > 0 && (
            <Card>
              <CardTitle hint="Açık kaynak koşullandırıcılar ve yer hareketi">Ek Katman</CardTitle>
              <select
                className="w-full rounded-lg border border-slate-200 bg-white px-3 py-2 text-sm text-slate-700 dark:border-slate-700 dark:bg-slate-900 dark:text-slate-200"
                value={extra}
                onChange={(e) => setExtra(e.target.value)}
              >
                <option value="">— Yok —</option>
                {layerList.map((l) => (
                  <option key={l.id} value={l.id}>
                    {l.title}
                  </option>
                ))}
              </select>
              {extra && extraMeta && (
                <div className="mt-3 space-y-2">
                  {extraMeta.gradient ? (
                    <>
                      <div
                        className="h-3 w-full rounded-full"
                        style={{ background: `linear-gradient(to right, ${extraMeta.gradient.join(",")})` }}
                      />
                      <div className="flex justify-between text-[11px] tabular-nums text-slate-400">
                        <span>{extraMeta.vmin?.toFixed(1)}</span>
                        <span>
                          {extraMeta.vmax?.toFixed(1)} {extraMeta.unit}
                        </span>
                      </div>
                    </>
                  ) : (
                    <ul className="grid grid-cols-2 gap-1">
                      {extraMeta.legend?.map((l) => (
                        <li key={l.code} className="flex items-center gap-1.5 text-xs text-slate-600 dark:text-slate-300">
                          <span className="inline-block h-3 w-3 rounded-sm ring-1 ring-black/10" style={{ backgroundColor: l.color }} />
                          {l.label}
                        </li>
                      ))}
                    </ul>
                  )}
                  <label className="block text-xs text-slate-500 dark:text-slate-400">
                    Opaklık {Math.round(extraOpacity * 100)}%
                    <input
                      type="range"
                      className="input-range mt-1"
                      min={0}
                      max={1}
                      step={0.05}
                      value={extraOpacity}
                      onChange={(e) => setExtraOpacity(Number(e.target.value))}
                    />
                  </label>
                  <p className="text-[11px] text-slate-400 dark:text-slate-500">Kaynak: {extraMeta.source}</p>
                </div>
              )}
            </Card>
          )}

          {/* Kategorik litoloji lejantı (istek 10) */}
          {legend.length > 0 && (
            <Card>
              <CardTitle hint={`${legend.length} sınıf`}>Litoloji Lejantı</CardTitle>
              <ul className="space-y-1.5">
                {legend.map((l) => (
                  <li key={l.id} className="flex items-center gap-2 text-xs">
                    <span
                      className="inline-block h-3.5 w-3.5 flex-shrink-0 rounded-sm ring-1 ring-black/10 dark:ring-white/10"
                      style={{ backgroundColor: l.color }}
                    />
                    <span className="text-slate-600 dark:text-slate-300">
                      {l.description}
                    </span>
                  </li>
                ))}
              </ul>
              <p className="mt-3 text-[11px] leading-relaxed text-slate-400 dark:text-slate-500">
                Litoloji proxy (Macrostrat, MTA değil) — rasterda gerçekte bulunan
                sınıflar gösterilir.
              </p>
            </Card>
          )}

          {/* Yağış — dürüst not (istek 9) */}
          <Card>
            <CardTitle hint="CHIRPS-2.0">Yağış Katmanı</CardTitle>
            {rain?.available ? (
              <div className="space-y-2 text-xs text-slate-600 dark:text-slate-300">
                <div className="h-3 w-full rounded-full bg-gradient-to-r from-sky-100 via-sky-400 to-sky-800" />
                <div className="flex justify-between tabular-nums text-[11px] text-slate-400">
                  <span>{rain.vmin?.toFixed(0)}</span>
                  <span>{rain.vmax?.toFixed(0)} {rain.unit ?? "mm"}</span>
                </div>
                <p className="text-[11px] leading-relaxed text-amber-600 dark:text-amber-400">
                  Yalnızca görsel bağlam — PINN modeline <b>girmez</b>. Model yağış
                  olmadan eğitildi.
                </p>
              </div>
            ) : (
              <p className="text-xs text-slate-400 dark:text-slate-500">
                Yağış rasteri mevcut değil. Uygulama yağış katmanı olmadan çalışır.
              </p>
            )}
          </Card>

          <Card>
            <CardTitle hint="Son seçilen konum">Nokta Tahmini</CardTitle>
            {loading ? (
              <Spinner label="Hesaplanıyor…" />
            ) : pt && pt.in_bounds ? (
              <div className="space-y-3">
                <div className="flex items-center justify-between">
                  <span className="text-3xl font-extrabold text-slate-900 dark:text-white">
                    {(pt.p * 100).toFixed(0)}%
                  </span>
                  <RiskBadge p={pt.p} threshold={pt.threshold} />
                </div>
                <dl className="grid grid-cols-2 gap-2 text-sm">
                  <Info k="Eğim" v={`${pt.slope.toFixed(1)}°`} />
                  <Info k="Yükseklik" v={`${pt.elevation.toFixed(0)} m`} />
                  <Info k="PGA" v={`${pt.pga.toFixed(2)} g`} />
                  <Info k="Bakı" v={`${pt.aspect.toFixed(0)}°`} />
                  <Info k="Kᵧ" v={`${pt.ky.toFixed(3)} g`} />
                  <Info k="Dₙ" v={`${pt.dn_cm.toFixed(1)} cm`} />
                </dl>
                <div className="rounded-lg bg-slate-50 px-3 py-2 text-xs text-slate-500 dark:bg-slate-800/50 dark:text-slate-400">
                  {pt.litho_name}
                </div>
                {pt.layers && (
                  <dl className="grid grid-cols-2 gap-2 text-sm">
                    {pt.layers.dist_fault_km != null && <Info k="Faya uzaklık" v={`${pt.layers.dist_fault_km.toFixed(1)} km`} />}
                    {pt.layers.dist_rupture_km != null && <Info k="2023 kırılması" v={`${pt.layers.dist_rupture_km.toFixed(1)} km`} />}
                    {pt.layers.vs30 != null && <Info k="Vs30" v={`${pt.layers.vs30.toFixed(0)} m/s`} />}
                    {pt.layers.pgv_cms != null && <Info k="PGV" v={`${pt.layers.pgv_cms.toFixed(0)} cm/s`} />}
                    {pt.layers.tri_m != null && <Info k="Pürüzlülük (TRI)" v={`${pt.layers.tri_m.toFixed(1)} m`} />}
                    {pt.layers.clay_pct != null && <Info k="Kil" v={`%${pt.layers.clay_pct.toFixed(0)}`} />}
                  </dl>
                )}
              </div>
            ) : pt && !pt.in_bounds ? (
              <p className="text-sm text-rose-600 dark:text-rose-300">
                Seçilen konum çalışma alanı dışında.
              </p>
            ) : (
              <p className="text-sm text-slate-400 dark:text-slate-500">
                Haritada bir noktaya tıklayın.
              </p>
            )}
          </Card>
        </div>
      </div>
    </div>
  );
}

const SIM_COLORS = ["#a3a29c", "#eda100", "#e34948"];

function polyStyle(sim: NewmarkSim | null, idx: Map<string, number>, frame: number, id: string): L.PathOptions {
  if (!sim || frame < 0) {
    return { color: "#7f1d1d", weight: 1, fillColor: "#ef4444", fillOpacity: 0.35 };
  }
  const i = idx.get(id);
  const dn = i == null ? 0 : sim.history_cm[i][frame];
  const c = dn >= sim.dn_threshold_cm ? SIM_COLORS[2] : dn >= 1 ? SIM_COLORS[1] : SIM_COLORS[0];
  return { color: c, weight: dn >= sim.dn_threshold_cm ? 2 : 1, fillColor: c, fillOpacity: dn >= 1 ? 0.75 : 0.3 };
}

function Info({ k, v }: { k: string; v: string }) {
  return (
    <div className="rounded-lg border border-slate-200 px-2.5 py-1.5 dark:border-slate-800">
      <dt className="text-[11px] font-medium text-slate-400">{k}</dt>
      <dd className="font-semibold text-slate-700 dark:text-slate-200">{v}</dd>
    </div>
  );
}
