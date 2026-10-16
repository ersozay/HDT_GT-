"""
AI Çıkarım Servisi (HTTP'den bağımsız saf Python katmanı)
=========================================================

Eğitilmiş PINN modelini, fizik hesaplarını ve raster örneklemeyi tek bir
`LandslideModel` sınıfında toplar. Backend (FastAPI) ve Streamlit bu servisi
kullanır; tahmin mantığı tek yerde tutulur.

Genel API:
    model = get_model()
    model.predict_point(slope=45, elevation=1200, aspect=180, litho_id=11, pga=0.8)
    model.critical_pga(slope=35, elevation=1200, aspect=180, litho_id=11)
    model.predict_latlon(lat=37.5, lon=37.2)
    model.map_meta();  model.overlay_png()
    model.lithology_list()
"""
from __future__ import annotations

import io
import json
import threading
from functools import lru_cache
from pathlib import Path

import numpy as np

# Proje kökünü içe aktarma yoluna ekle
_ROOT = Path(__file__).resolve().parent.parent
import sys
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from config import (
    MODELS_DIR, DATA_PROCESSED, DATA_RAW, PGA_RASTER, DEM_RASTER, LITHO_RASTER,
    INVENTORY_KML, Z_REF_M, DN_THRESHOLD_M,
)
from src.geomechanics.lithology import LITHOLOGY_TABLE
from src.geomechanics.mohr_coulomb import StaticFS, CriticalAcceleration
from src.geomechanics.newmark import NewmarkDisplacement
from src.data_engineering.feature_matrix import build_feature_matrix

SLOPE_RASTER = DATA_PROCESSED / "slope_deg.tif"
ASPECT_RASTER = DATA_PROCESSED / "aspect_deg.tif"
SUSC_RASTER = DATA_PROCESSED / "susceptibility_pinn.tif"
RAINFALL_RASTER = DATA_RAW / "rainfall_chirps.tif"


class LandslideModel:
    """Eğitilmiş PINN + fizik + raster örnekleme sarmalayıcısı (tek sefer yüklenir)."""

    def __init__(self):
        import torch
        from src.pinn.io import load_checkpoint

        self._torch = torch
        self._lock = threading.Lock()

        # --- Model + ölçekleyici + eşik (v2 checkpoint'i hepsini içerir) ---
        lm = load_checkpoint(MODELS_DIR / "best_model.pt",
                             scaler_path=MODELS_DIR / "scaler.pkl",
                             threshold_path=MODELS_DIR / "threshold.json")
        self.model = lm.model
        self.n_features = len(lm.feat_cols)
        self.epoch = int(lm.meta.get("epoch", -1))
        self.val_auc = float(lm.meta.get("val_auc", 0.0))
        self.model_version = str(lm.meta.get("model_version", "pinn_v1"))
        self.threshold = lm.threshold
        self._feat_cols = lm.feat_cols
        self._mean = lm.mean
        self._std = lm.std

        self._overlay_cache: bytes | None = None
        self._litho_overlay_cache: bytes | None = None
        self._litho_legend_cache: list[dict] | None = None
        self._rainfall_overlay_cache: bytes | None = None
        self._inventory_cache: list[dict] | None = None
        self._inventory_index: dict[str, tuple[float, float]] | None = None
        self._rasters: dict = {}

    # ------------------------------------------------------------------ #
    # Çekirdek tahmin
    # ------------------------------------------------------------------ #
    def _features(self, slope, elevation, aspect, litho_id, pga, z_ref):
        a = lambda v: np.array([[v]], dtype=float)
        df, _ = build_feature_matrix(
            elevation_m=a(elevation), slope_deg=a(slope), aspect_deg=a(aspect),
            litho_array=np.array([[int(litho_id)]]), pga_array=a(pga),
            u_array=None, z_ref=z_ref, include_onehot=True,
        )
        X = df[self._feat_cols].values.astype(np.float32)
        return df, X

    def predict_point(self, *, slope, elevation, aspect, litho_id, pga, z_ref=Z_REF_M) -> dict:
        """Tek yamaç için heyelan olasılığı, Ky ve Dn döndürür."""
        torch = self._torch
        df, X = self._features(slope, elevation, aspect, litho_id, pga, z_ref)
        Xn = (X - self._mean) / self._std
        with self._lock, torch.no_grad():
            p = float(self.model(torch.tensor(Xn, dtype=torch.float32)).numpy().ravel()[0])
        ky = float(df["ky_g"].iloc[0])
        dn = float(df["dn_cm"].iloc[0])
        return {
            "p": p,
            "ky": ky,
            "dn_cm": dn,
            "label": int(p >= self.threshold),
            "threshold": self.threshold,
            "static_marginal": bool(ky <= 0.02),
        }

    def critical_pga(self, *, slope, elevation, aspect=180, litho_id,
                     pga_min=0.05, pga_max=1.50, step=0.05, z_ref=Z_REF_M) -> dict:
        """PGA taraması: her PGA için P(heyelan) ve kritik eşik (P≥threshold)."""
        from src.evaluation.metrics import critical_pga_from_curve
        torch = self._torch
        grid = np.round(np.arange(pga_min, pga_max + 1e-9, step), 3)
        n = len(grid)
        full = lambda v: np.full((1, n), v, dtype=float)
        df, _ = build_feature_matrix(
            elevation_m=full(elevation), slope_deg=full(slope), aspect_deg=full(aspect),
            litho_array=np.full((1, n), int(litho_id)), pga_array=grid.reshape(1, n),
            u_array=None, z_ref=z_ref, include_onehot=True,
        )
        Xn = (df[self._feat_cols].values.astype(np.float32) - self._mean) / self._std
        with self._lock, torch.no_grad():
            probs = self.model(torch.tensor(Xn, dtype=torch.float32)).numpy().ravel()
        dns = df["dn_cm"].to_numpy(float)
        crit = critical_pga_from_curve(grid, probs, self.threshold)
        return {
            "pga": [float(x) for x in grid],
            "prob": [float(x) for x in probs],
            "dn_cm": [float(x) for x in dns],
            "critical_pga": crit,
            "threshold": self.threshold,
        }

    # ------------------------------------------------------------------ #
    # Raster örnekleme (koordinattan tahmin)
    # ------------------------------------------------------------------ #
    def _raster(self, path):
        import rasterio
        key = str(path)
        if key not in self._rasters:
            self._rasters[key] = rasterio.open(path)
        return self._rasters[key]

    def _sample(self, path, lon, lat, default=np.nan):
        try:
            ds = self._raster(path)
            val = list(ds.sample([(lon, lat)]))[0][0]
            if val is None:
                return default
            v = float(val)
            if np.isnan(v):
                return default
            return v
        except Exception:
            return default

    def predict_latlon(self, lat: float, lon: float) -> dict:
        """Bir koordinat için rasterlardan öznitelik örnekleyip tahmin üretir."""
        slope = self._sample(SLOPE_RASTER, lon, lat, default=0.0)
        aspect = self._sample(ASPECT_RASTER, lon, lat, default=180.0)
        elevation = self._sample(DEM_RASTER, lon, lat, default=0.0)
        pga = self._sample(PGA_RASTER, lon, lat, default=0.05)
        litho = self._sample(LITHO_RASTER, lon, lat, default=27.0)

        slope = float(np.clip(np.nan_to_num(slope, nan=0.0), 0.0, 89.0))
        aspect = float(np.nan_to_num(aspect, nan=180.0))
        elevation = float(np.nan_to_num(elevation, nan=0.0))
        pga = float(np.clip(np.nan_to_num(pga, nan=0.05), 0.001, 3.0))
        litho_id = int(litho) if (1 <= int(litho) <= 27) else 27

        in_bounds = bool(elevation > 0 and pga > 0.02)
        res = self.predict_point(slope=slope, elevation=elevation, aspect=aspect,
                                 litho_id=litho_id, pga=pga)
        res.update({
            "lat": lat, "lon": lon, "in_bounds": in_bounds,
            "slope": slope, "elevation": elevation, "aspect": aspect,
            "pga": pga, "litho_id": litho_id,
            "litho_name": LITHOLOGY_TABLE.get(litho_id, {}).get("description", "?"),
        })
        return res

    # ------------------------------------------------------------------ #
    # Harita meta + overlay
    # ------------------------------------------------------------------ #
    def map_meta(self) -> dict:
        import rasterio
        with rasterio.open(SUSC_RASTER) as ds:
            b = ds.bounds
            arr = ds.read(1)
        finite = arr[np.isfinite(arr)]
        return {
            # Leaflet [ [south, west], [north, east] ]
            "bounds": [[float(b.bottom), float(b.left)], [float(b.top), float(b.right)]],
            "center": [float((b.bottom + b.top) / 2), float((b.left + b.right) / 2)],
            "width": int(ds.width) if False else int(arr.shape[1]),
            "height": int(arr.shape[0]),
            "vmin": float(finite.min()) if finite.size else 0.0,
            "vmax": float(finite.max()) if finite.size else 1.0,
            "threshold": self.threshold,
        }

    def overlay_png(self) -> bytes:
        """Susceptibility rasterını renklendirilmiş şeffaf PNG'ye çevirir (cache'li)."""
        if self._overlay_cache is not None:
            return self._overlay_cache
        import rasterio
        import matplotlib
        matplotlib.use("Agg")
        from matplotlib.colors import Normalize
        from PIL import Image

        with rasterio.open(SUSC_RASTER) as ds:
            arr = ds.read(1).astype(float)

        mask = np.isfinite(arr)
        norm = Normalize(vmin=0.0, vmax=1.0)
        cmap = matplotlib.colormaps["RdYlGn_r"]  # düşük=yeşil, yüksek=kırmızı
        rgba = cmap(norm(np.nan_to_num(arr, nan=0.0)))
        rgba[..., 3] = np.where(mask, 0.78, 0.0)  # NaN → şeffaf
        img = (rgba * 255).astype(np.uint8)

        buf = io.BytesIO()
        Image.fromarray(img, mode="RGBA").save(buf, format="PNG")
        self._overlay_cache = buf.getvalue()
        return self._overlay_cache

    # ------------------------------------------------------------------ #
    # Litoloji katmanı (kategorik renkli overlay + lejant)
    # ------------------------------------------------------------------ #
    def _litho_color_map(self, class_ids: list[int]) -> dict[int, str]:
        """Her litoloji sınıfına sabit, ayırt edilebilir bir renk atar (hex)."""
        import matplotlib
        cmap = matplotlib.colormaps["tab20"]
        colors = {}
        for i, cid in enumerate(sorted(class_ids)):
            r, g, b, _ = cmap(i % 20)
            colors[cid] = "#{:02x}{:02x}{:02x}".format(
                int(r * 255), int(g * 255), int(b * 255)
            )
        return colors

    def lithology_legend(self) -> list[dict]:
        """Litoloji rasterinde GERÇEKTE bulunan sınıfları renk+açıklama ile döndürür."""
        if self._litho_legend_cache is not None:
            return self._litho_legend_cache
        import rasterio
        with rasterio.open(LITHO_RASTER) as ds:
            arr = ds.read(1)
            nodata = ds.nodata
        vals = np.unique(arr)
        present = [int(v) for v in vals
                   if (nodata is None or v != nodata) and 1 <= int(v) <= 27]
        colors = self._litho_color_map(present)
        legend = [
            {
                "id": cid,
                "description": LITHOLOGY_TABLE.get(cid, {}).get("description", f"Sınıf {cid}"),
                "color": colors[cid],
            }
            for cid in present
        ]
        self._litho_legend_cache = legend
        return legend

    def lithology_overlay_png(self) -> bytes:
        """Litoloji rasterini kategorik renkli şeffaf PNG'ye çevirir (cache'li)."""
        if self._litho_overlay_cache is not None:
            return self._litho_overlay_cache
        import rasterio
        from PIL import Image

        with rasterio.open(LITHO_RASTER) as ds:
            arr = ds.read(1)
            nodata = ds.nodata

        legend = self.lithology_legend()
        color_hex = {item["id"]: item["color"] for item in legend}

        h, w = arr.shape
        rgba = np.zeros((h, w, 4), dtype=np.uint8)
        for cid, hexv in color_hex.items():
            r = int(hexv[1:3], 16); g = int(hexv[3:5], 16); b = int(hexv[5:7], 16)
            mask = arr == cid
            rgba[mask, 0] = r; rgba[mask, 1] = g; rgba[mask, 2] = b
            rgba[mask, 3] = 200  # yarı-şeffaf
        # nodata / sınıf dışı → şeffaf (alpha 0 zaten)

        buf = io.BytesIO()
        Image.fromarray(rgba, mode="RGBA").save(buf, format="PNG")
        self._litho_overlay_cache = buf.getvalue()
        return self._litho_overlay_cache

    def lithology_meta(self) -> dict:
        import rasterio
        with rasterio.open(LITHO_RASTER) as ds:
            b = ds.bounds
        return {"bounds": [[float(b.bottom), float(b.left)],
                           [float(b.top), float(b.right)]]}

    # ------------------------------------------------------------------ #
    # Yağış katmanı (görsel; modele girmez)
    # ------------------------------------------------------------------ #
    def rainfall_meta(self) -> dict:
        import rasterio
        if not RAINFALL_RASTER.exists():
            return {"available": False}
        with rasterio.open(RAINFALL_RASTER) as ds:
            b = ds.bounds
            arr = ds.read(1).astype(float)
        finite = arr[np.isfinite(arr)]
        return {
            "available": True,
            "bounds": [[float(b.bottom), float(b.left)],
                       [float(b.top), float(b.right)]],
            "vmin": float(finite.min()) if finite.size else 0.0,
            "vmax": float(finite.max()) if finite.size else 0.0,
            "unit": "mm/ay",
            "note": "CHIRPS-2.0 (UCSB); yalnız görsel bağlam — PINN'e girmez.",
        }

    def rainfall_overlay_png(self) -> bytes | None:
        """Yağış rasterini mavi colormap RGBA PNG'ye çevirir; yoksa None."""
        if not RAINFALL_RASTER.exists():
            return None
        if self._rainfall_overlay_cache is not None:
            return self._rainfall_overlay_cache
        import rasterio
        import matplotlib
        from matplotlib.colors import Normalize
        from PIL import Image

        with rasterio.open(RAINFALL_RASTER) as ds:
            arr = ds.read(1).astype(float)
        mask = np.isfinite(arr)
        finite = arr[mask]
        vmin = float(finite.min()) if finite.size else 0.0
        vmax = float(finite.max()) if finite.size else 1.0
        norm = Normalize(vmin=vmin, vmax=vmax)
        cmap = matplotlib.colormaps["Blues"]
        rgba = cmap(norm(np.nan_to_num(arr, nan=vmin)))
        rgba[..., 3] = np.where(mask, 0.72, 0.0)
        img = (rgba * 255).astype(np.uint8)

        buf = io.BytesIO()
        Image.fromarray(img, mode="RGBA").save(buf, format="PNG")
        self._rainfall_overlay_cache = buf.getvalue()
        return self._rainfall_overlay_cache

    # ------------------------------------------------------------------ #
    # Heyelan envanteri (ID ile nokta seçimi)
    # ------------------------------------------------------------------ #
    def _load_inventory(self):
        if self._inventory_cache is not None:
            return
        import geopandas as gpd
        gdf = gpd.read_file(INVENTORY_KML)
        items, index = [], {}
        for _, row in gdf.iterrows():
            name = str(row.get("Name", "")).strip()
            geom = row.geometry
            if geom is None or not name:
                continue
            c = geom.centroid
            lon, lat = float(c.x), float(c.y)
            items.append({"id": name, "lat": lat, "lon": lon})
            index[name] = (lat, lon)
        self._inventory_cache = items
        self._inventory_index = index

    def inventory_list(self) -> list[dict]:
        """Envanterdeki tüm heyelan poligonlarının {id, lat, lon} listesi."""
        self._load_inventory()
        return self._inventory_cache

    def inventory_point(self, inv_id: str) -> dict:
        """Envanter ID'sinden koordinat bul → rasterlardan tahmin üret."""
        self._load_inventory()
        key = str(inv_id).strip()
        if key not in self._inventory_index:
            raise KeyError(f"Envanter ID bulunamadı: {inv_id!r}")
        lat, lon = self._inventory_index[key]
        res = self.predict_latlon(lat, lon)
        res["inventory_id"] = key
        return res

    # ------------------------------------------------------------------ #
    # Newmark eğrileri (Jibson + Ambraseys, 10 cm eşiği)
    # ------------------------------------------------------------------ #
    def newmark_curves(self, *, slope, elevation=1000.0, aspect=180, litho_id,
                       pga_min=0.05, pga_max=1.50, step=0.05, z_ref=Z_REF_M) -> dict:
        """
        Bir yamaç için aᵧ hesaplar ve PGA taramasında Jibson + Ambraseys Dₙ
        dizilerini + her biri için 10 cm kritik PGA'sını döndürür.
        """
        lith = LITHOLOGY_TABLE.get(int(litho_id), LITHOLOGY_TABLE[27])
        fs = float(StaticFS(lith["c_prime"], lith["phi_prime"], lith["gamma"],
                            z_ref, slope))
        ky = float(CriticalAcceleration(fs, slope))

        grid = np.round(np.arange(pga_min, pga_max + 1e-9, step), 3)
        dn_j = NewmarkDisplacement(ky, grid, "jibson2007") * 100.0
        dn_a = NewmarkDisplacement(ky, grid, "ambraseys1988") * 100.0

        def crossing(dn):
            over = np.where(dn >= DN_THRESHOLD_M * 100.0)[0]
            return float(grid[over[0]]) if len(over) else None

        return {
            "ky": ky,
            "fs_static": fs,
            "pga": [float(x) for x in grid],
            "dn_jibson_cm": [float(x) for x in dn_j],
            "dn_ambraseys_cm": [float(x) for x in dn_a],
            "critical_pga_jibson": crossing(dn_j),
            "critical_pga_ambraseys": crossing(dn_a),
            "dn_threshold_cm": DN_THRESHOLD_M * 100.0,
        }

    # ------------------------------------------------------------------ #
    # Yardımcılar
    # ------------------------------------------------------------------ #
    def lithology_list(self) -> list[dict]:
        return [
            {"id": lid, "description": v["description"],
             "c_prime": v["c_prime"], "phi_prime": v["phi_prime"], "gamma": v["gamma"]}
            for lid, v in sorted(LITHOLOGY_TABLE.items())
        ]

    def info(self) -> dict:
        return {
            "n_features": self.n_features,
            "epoch": self.epoch,
            "val_auc": self.val_auc,
            "threshold": self.threshold,
            "dn_threshold_cm": DN_THRESHOLD_M * 100,
            "model_version": self.model_version,
        }


@lru_cache(maxsize=1)
def get_model() -> LandslideModel:
    """Süreç başına tek model örneği döndürür (lazy singleton)."""
    return LandslideModel()
