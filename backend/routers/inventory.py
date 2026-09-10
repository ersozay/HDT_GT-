"""Envanter uçları: heyelan poligon listesi ve ID ile nokta tahmini."""
from fastapi import APIRouter, HTTPException

from service import get_model
from backend.schemas import InventoryItem, LatLonResponse

router = APIRouter(prefix="/api/inventory", tags=["inventory"])


@router.get("", response_model=list[InventoryItem])
def inventory_list():
    return get_model().inventory_list()


@router.get("/{inv_id}", response_model=LatLonResponse)
def inventory_point(inv_id: str):
    try:
        return get_model().inventory_point(inv_id)
    except KeyError:
        raise HTTPException(status_code=404, detail=f"Envanter ID yok: {inv_id}")
