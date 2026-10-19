"""Harita uçları: meta, susceptibility/litoloji/yağış overlay, koordinattan tahmin."""
from fastapi import APIRouter, Query, HTTPException
from fastapi.responses import Response

from service import get_model
from backend.schemas import LatLonResponse, LithologyLegendItem

router = APIRouter(prefix="/api/map", tags=["map"])

_PNG_CACHE = {"Cache-Control": "public, max-age=3600"}


@router.get("/meta")
def map_meta():
    return get_model().map_meta()


@router.get("/overlay.png")
def overlay_png():
    png = get_model().overlay_png()
    return Response(content=png, media_type="image/png", headers=_PNG_CACHE)


@router.get("/point", response_model=LatLonResponse)
def map_point(lat: float = Query(...), lon: float = Query(...)):
    return get_model().predict_latlon(lat=lat, lon=lon)


# --- Litoloji katmanı (kategorik renk) ---
@router.get("/lithology.png")
def lithology_png():
    png = get_model().lithology_overlay_png()
    return Response(content=png, media_type="image/png", headers=_PNG_CACHE)


@router.get("/lithology/meta")
def lithology_meta():
    return get_model().lithology_meta()


@router.get("/lithology/legend", response_model=list[LithologyLegendItem])
def lithology_legend():
    return get_model().lithology_legend()


# --- Yağış katmanı (görsel; modele girmez) ---
@router.get("/rainfall/meta")
def rainfall_meta():
    return get_model().rainfall_meta()


@router.get("/rainfall.png")
def rainfall_png():
    png = get_model().rainfall_overlay_png()
    if png is None:
        raise HTTPException(status_code=404, detail="Yağış rasteri yok")
    return Response(content=png, media_type="image/png", headers=_PNG_CACHE)


# --- Açık kaynak ek katmanlar (scripts/12) + vektörler ---
from service import layers as _layers


@router.get("/layers")
def layers_list():
    return _layers.available_layers()


@router.get("/layer/{name}.png")
def layer_png(name: str):
    if name not in _layers.RASTER_LAYERS:
        raise HTTPException(status_code=404, detail="Katman yok")
    return Response(content=_layers.layer_png(name), media_type="image/png", headers=_PNG_CACHE)


@router.get("/layer/{name}/meta")
def layer_meta(name: str):
    if name not in _layers.RASTER_LAYERS:
        raise HTTPException(status_code=404, detail="Katman yok")
    return _layers.layer_meta(name)


@router.get("/inventory.geojson")
def inventory_geojson():
    return _layers.inventory_geojson()


@router.get("/faults.geojson")
def faults_geojson():
    return _layers.faults_geojson()


@router.get("/stations.geojson")
def stations_geojson():
    return _layers.stations_geojson()
