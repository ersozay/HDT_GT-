import { useEffect, useState } from "react";
import {
  MapContainer,
  TileLayer,
  ImageOverlay,
  useMapEvents,
  Marker,
  Popup,
  Rectangle,
  LayersControl,
} from "react-leaflet";
import L from "leaflet";
import {
  api,
  MapMeta,
  LatLonResult,
  LithologyLegendItem,
  RainfallMeta,
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

  useEffect(() => {
    api.mapMeta().then(setMeta).catch(() => {});
    api.lithologyLegend().then(setLegend).catch(() => {});
    api.rainfallMeta().then(setRain).catch(() => {});
  }, []);

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
              Katman seçici (sağ üst) ile <b>Litoloji</b> ve <b>Yağış</b>{" "}
              katmanlarını açıp kapatabilirsiniz.
            </p>
          </Card>

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

function Info({ k, v }: { k: string; v: string }) {
  return (
    <div className="rounded-lg border border-slate-200 px-2.5 py-1.5 dark:border-slate-800">
      <dt className="text-[11px] font-medium text-slate-400">{k}</dt>
      <dd className="font-semibold text-slate-700 dark:text-slate-200">{v}</dd>
    </div>
  );
}
