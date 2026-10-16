# Heyelan Dijital Twin — Uygulama Katmanı

Kahramanmaraş 6 Şubat 2023 depremi için fizik-tabanlı hibrit heyelan dijital
ikizinin **3 katmanlı** profesyonel uygulaması.

```
┌──────────────┐     HTTP/JSON     ┌──────────────┐   Python   ┌──────────────┐
│  Frontend    │ ────────────────▶ │  Backend     │ ─────────▶ │  AI Servisi  │
│ React+Vite   │ ◀──────────────── │  FastAPI     │ ◀───────── │  (inference) │
│ TS+Tailwind  │                   │  REST API    │            │  PINN+Fizik  │
└──────────────┘                   └──────────────┘            └──────────────┘
   :5173                              :8010 (dev)                 model + raster
```

## Katmanlar

| Katman | Dizin | Sorumluluk |
|--------|-------|-----------|
| **AI Servisi** | `service/` | Saf Python çıkarım — model+scaler+raster tek sefer yüklenir; HTTP bilmez |
| **Backend** | `backend/` | FastAPI REST API — servis metotlarını uçlara açar, CORS, statik prod servis |
| **Frontend** | `frontend/` | React SPA — 5 sayfa, dark mode varsayılan, Leaflet harita, Recharts |

## Hızlı Başlangıç (geliştirme)

```bash
./run_app.sh
```

- Frontend: http://localhost:5173
- Backend:  http://localhost:8010/api/health

İlk çalıştırmada `npm install` otomatik yapılır (ağ gerekir).

## Manuel Çalıştırma

**Backend**
```bash
python3 -m uvicorn backend.main:app --reload --port 8010   # vite proxy bu portu bekler
```

**Frontend**
```bash
cd frontend
npm install     # ilk sefer
npm run dev
```

## Prodüksiyon (tek servis)

```bash
cd frontend && npm run build     # → frontend/dist
cd .. && python3 -m uvicorn backend.main:app --port 8000
```
`backend/main.py`, `frontend/dist` varsa `/` altında servis eder; her şey tek
portta (8000) çalışır.

## API Uçları

| Metod | Uç | Açıklama |
|-------|-----|----------|
| GET  | `/api/health` | Sağlık kontrolü |
| GET  | `/api/lithology` | 27 litoloji sınıfı |
| POST | `/api/predict/point` | Nokta tahmini (P, Kᵧ, Dₙ) |
| POST | `/api/predict/critical-pga` | PGA taraması → kritik PGA |
| GET  | `/api/map/meta` | Harita sınırları/merkez |
| GET  | `/api/map/overlay.png` | Duyarlılık katmanı (RGBA PNG) |
| GET  | `/api/map/point?lat=&lon=` | Koordinattan tahmin |
| GET  | `/api/results/summary` | Test + mekânsal CV metrikleri (03 betiği yazar) |
| GET  | `/api/results/benchmark` | 9 modelin mekânsal CV karşılaştırması |
| GET  | `/api/results/critical-pga` | Kritik PGA belirsizliği + tasarım tablosu |
| GET  | `/api/results/fem` | FEM-SSR pilot sonuçları |
| GET  | `/api/figures/{name}` | Figürler (whitelist) |

## Sayfalar

1. **Genel Bakış** — proje tanıtımı, anahtar sayılar, yöntem özeti
2. **Tahmin** — kaydırıcılarla anlık nokta tahmini + animasyonlu gösterge
3. **Kritik PGA** — P(heyelan) vs PGA eğrisi (Recharts), kritik eşik işaretli
4. **Harita** — Leaflet + duyarlılık overlay + tıkla-tahmin
5. **Sonuçlar** — metrikler, figür sekmeleri, dürüst kısıt notları

## Gereksinimler

- Python 3 + `fastapi uvicorn pydantic rasterio torch matplotlib pillow`
- Node ≥ 18 + npm

## Analiz Hattı (v2)

```bash
./run_pipeline.sh          # 09 → 03 → 05 → 10 → 02 → 04 → 06/07 → testler
python3 -m pytest tests -q # 19 test (fizik, FEM kıyas problemleri, mekânsal CV)
```

Ayrıntılar ve sonuçlar: [docs/V2_SONUCLAR.md](docs/V2_SONUCLAR.md)

## Bilimsel Şeffaflık

- **Litoloji** Macrostrat API proxy'sidir (resmi MTA haritası değil).
- **FEM**: Griffiths & Lane (1999) SSR çözücüsü (`src/fem/ssr_fem.py`); 2B idealize
  kesit, psödo-statik yükleme. `src/fem/sfepy_solver.py` v1 kalıntısıdır, kullanılmaz.
- **Kritik PGA** (P = 0.5) negatif örnekleme oranına duyarlıdır; aralığıyla raporlanır.

> `app.py` (Streamlit) hızlı yedek demo olarak korunmuştur.
