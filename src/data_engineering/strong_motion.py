"""
Kuvvetli yer hareketi (ivme) kayıtları — ESM (Engineering Strong Motion Database).

6 Şubat 2023 depremlerinin AFAD (TK ağı) ivme kayıtları ESM web servisinden
token gerektirmeden indirilir. "MP" (manually processed) sürüm kullanılır:
taban çizgisi düzeltilmiş, Butterworth bant geçiren filtreli (Paolucci ve ark.
2011) ivme, cm/s².

Kaynaklar / atıf:
  Luzi L. ve ark. (2020/2026). Engineering Strong Motion Database (ESM),
    version 3.0. INGV. https://doi.org/10.13127/esm.3
  AFAD (1973). Turkish National Strong Motion Network. https://doi.org/10.7914/SN/TK

Servisler:
  istasyon : https://esm-db.eu/fdsnws/station/1/query
  kayıt    : https://esm-db.eu/esmws/eventdata/1/query?eventid=..&station=..&format=ascii
"""
from __future__ import annotations

import io
import time
import zipfile
from dataclasses import dataclass, field
from pathlib import Path
from urllib.parse import urlencode
from urllib.request import Request, urlopen

import numpy as np

G_CMS2 = 980.665

# ESM olay kimlikleri (fdsnws/event ile bulundu)
EVENTS = {
    "pazarcik": {"esm_id": "INT-20230206_0000008", "usgs_id": "us6000jllz", "mw": 7.7,
                 "origin": "2023-02-06T01:17:35", "lat": 37.2186, "lon": 37.0726},
    "elbistan": {"esm_id": "INT-20230206_0000222", "usgs_id": "us6000jlqa", "mw": 7.5,
                 "origin": "2023-02-06T10:24:48", "lat": 38.0134, "lon": 37.1833},
}

ESM_STATION = "https://esm-db.eu/fdsnws/station/1/query"
ESM_EVENTDATA = "https://esm-db.eu/esmws/eventdata/1/query"


def _ssl_context():
    """python.org macOS kurulumunda sistem sertifikaları yoksa certifi deposunu kullan."""
    import ssl
    try:
        import certifi
        return ssl.create_default_context(cafile=certifi.where())
    except ImportError:
        return ssl.create_default_context()


def _get(url: str, timeout: int = 120, retries: int = 3) -> tuple[int, bytes]:
    last = None
    ctx = _ssl_context()
    for k in range(retries):
        try:
            with urlopen(Request(url, headers={"User-Agent": "heyelan-dijital-twin"}),
                         timeout=timeout, context=ctx) as r:
                return r.status, r.read()
        except Exception as e:          # ağ hatası → kısa bekle, tekrar dene
            last = e
            time.sleep(2 * (k + 1))
    raise RuntimeError(f"İndirilemedi: {url} ({last})")


def list_stations(bounds, network: str = "TK") -> list[dict]:
    """Çalışma alanındaki ESM istasyonları: [{code, lat, lon}]."""
    lon0, lat0, lon1, lat1 = bounds
    q = urlencode({"network": network, "minlatitude": lat0, "maxlatitude": lat1,
                   "minlongitude": lon0, "maxlongitude": lon1, "format": "text"})
    _, body = _get(f"{ESM_STATION}?{q}")
    out = []
    for line in body.decode().splitlines()[1:]:
        p = line.split("|")
        if len(p) > 5 and p[1] == "public":
            out.append({"code": p[2], "lat": float(p[3]), "lon": float(p[4])})
    return out


def fetch_record(event_key: str, station: str, out_dir: Path, network: str = "TK") -> list[Path]:
    """Bir olay–istasyon çifti için MP ASCII kayıtlarını indirir (önbellekli)."""
    out_dir = Path(out_dir) / event_key
    out_dir.mkdir(parents=True, exist_ok=True)
    have = sorted(out_dir.glob(f"{network}.{station}..*.ASC"))
    if have:
        return have
    marker = out_dir / f"{network}.{station}.none"
    if marker.exists():
        return []
    q = urlencode({"eventid": EVENTS[event_key]["esm_id"], "network": network,
                   "station": station, "format": "ascii", "data-type": "ACC",
                   "processing-type": "MP"})
    status, body = _get(f"{ESM_EVENTDATA}?{q}")
    if status != 200 or not body[:2] == b"PK":
        marker.write_text("kayıt yok")
        return []
    with zipfile.ZipFile(io.BytesIO(body)) as z:
        z.extractall(out_dir)
    return sorted(out_dir.glob(f"{network}.{station}..*.ASC"))


@dataclass
class Record:
    station: str
    component: str
    lat: float
    lon: float
    dt: float
    acc_g: np.ndarray
    header: dict = field(default_factory=dict)

    @property
    def pga_g(self) -> float:
        return float(np.abs(self.acc_g).max())

    @property
    def time(self) -> np.ndarray:
        return np.arange(len(self.acc_g)) * self.dt


def read_esm_ascii(path: str | Path) -> Record:
    """ESM ASCII (DYNA 1.2 başlığı + tek sütun veri) → Record (ivme g)."""
    header, data = {}, []
    with open(path, encoding="utf-8", errors="replace") as f:
        for line in f:
            if ":" in line and not data:
                k, v = line.split(":", 1)
                header[k.strip()] = v.strip()
            else:
                s = line.strip()
                if s:
                    data.append(float(s))
    acc = np.asarray(data, dtype=np.float64)
    units = header.get("UNITS", "cm/s^2")
    if units.startswith("cm"):
        acc = acc / G_CMS2
    elif units.startswith("m/s"):
        acc = acc / (G_CMS2 / 100.0)
    return Record(header.get("STATION_CODE", "?"), header.get("STREAM", "?"),
                  float(header["STATION_LATITUDE_DEGREE"]), float(header["STATION_LONGITUDE_DEGREE"]),
                  float(header["SAMPLING_INTERVAL_S"]), acc, header)


def arias_intensity(acc_g: np.ndarray, dt: float) -> float:
    """Arias şiddeti Iₐ = π/(2g) ∫ a(t)² dt  [m/s]."""
    a = acc_g * 9.80665
    return float(np.pi / (2 * 9.80665) * np.sum(a ** 2) * dt)


def horizontal_records(files: list[Path]) -> list[Record]:
    """Yatay bileşenler (E/N, ya da 1/2) — düşey (Z) hariç."""
    recs = [read_esm_ascii(p) for p in files]
    return [r for r in recs if not r.component.upper().endswith("Z")]
