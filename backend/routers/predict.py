"""Tahmin uçları: tek nokta ve kritik PGA taraması."""
from fastapi import APIRouter

from service import get_model
from backend.schemas import (
    PointRequest, PointResponse, CriticalRequest, CriticalResponse,
    NewmarkCurvesRequest, NewmarkCurvesResponse,
)

router = APIRouter(prefix="/api/predict", tags=["predict"])


@router.post("/point", response_model=PointResponse)
def predict_point(req: PointRequest):
    m = get_model()
    return m.predict_point(
        slope=req.slope, elevation=req.elevation, aspect=req.aspect,
        litho_id=req.litho_id, pga=req.pga, saturation=req.saturation,
    )


@router.post("/critical-pga", response_model=CriticalResponse)
def critical_pga(req: CriticalRequest):
    m = get_model()
    return m.critical_pga(
        slope=req.slope, elevation=req.elevation, aspect=req.aspect,
        litho_id=req.litho_id, pga_min=req.pga_min, pga_max=req.pga_max, step=req.step,
        saturation=req.saturation,
    )


@router.post("/newmark-curves", response_model=NewmarkCurvesResponse)
def newmark_curves(req: NewmarkCurvesRequest):
    m = get_model()
    return m.newmark_curves(
        slope=req.slope, elevation=req.elevation, aspect=req.aspect,
        litho_id=req.litho_id, pga_min=req.pga_min, pga_max=req.pga_max, step=req.step,
        saturation=req.saturation,
    )


@router.get("/moisture")
def moisture(lat: float, lon: float):
    """Noktada zemin doygunluğu: 6 Şubat 2023 (ERA5-Land) ve bugün + 3 gün tahmini (Open-Meteo)."""
    return get_model().moisture(lat=lat, lon=lon)
