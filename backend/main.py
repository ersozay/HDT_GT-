"""
FastAPI uygulaması — Heyelan Dijital Twin API
=============================================

Çalıştırma (dev):
    cd "digital twin"
    python3 -m uvicorn backend.main:app --reload --port 8000

Prod'da (frontend build edilmişse) frontend/dist otomatik `/` altında sunulur.
"""
import sys
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from backend.routers import predict, map as map_router, results, figures, inventory

app = FastAPI(
    title="Heyelan Dijital Twin API",
    description="Fizik tabanlı hibrit PINN heyelan duyarlılık servisi "
                "(Kahramanmaraş 6 Şubat 2023).",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(predict.router)
app.include_router(map_router.router)
app.include_router(results.router)
app.include_router(figures.router)
app.include_router(inventory.router)


@app.get("/api/health")
def health():
    return {"status": "ok"}


@app.on_event("startup")
def _warmup():
    # Modeli süreç başlangıcında yükle (ilk istek gecikmesini önler)
    from service import get_model
    get_model()


# --- Prod: build edilmiş frontend'i servis et (varsa) ---
_DIST = _ROOT / "frontend" / "dist"
if _DIST.exists():
    from fastapi.staticfiles import StaticFiles
    app.mount("/", StaticFiles(directory=str(_DIST), html=True), name="frontend")
