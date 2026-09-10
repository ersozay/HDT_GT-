"""Sonuç uçları: model metrikleri özeti ve litoloji listesi."""
import json
from pathlib import Path

from fastapi import APIRouter

from service import get_model
from backend.schemas import LithologyItem

from fastapi import HTTPException

router = APIRouter(prefix="/api", tags=["results"])

_BACKEND_DIR = Path(__file__).resolve().parent.parent
_METRICS = _BACKEND_DIR / "metrics.json"
_VALIDATION = _BACKEND_DIR / "validation.json"


@router.get("/results/summary")
def results_summary():
    metrics = json.loads(_METRICS.read_text())
    info = get_model().info()
    metrics["model_epoch"] = info["epoch"]
    metrics["model_val_auc"] = round(info["val_auc"], 3)
    metrics["dn_threshold_cm"] = info["dn_threshold_cm"]
    return metrics


@router.get("/lithology", response_model=list[LithologyItem])
def lithology():
    return get_model().lithology_list()


@router.get("/results/validation")
def results_validation():
    if not _VALIDATION.exists():
        raise HTTPException(status_code=404, detail="Doğrulama verisi yok")
    return json.loads(_VALIDATION.read_text(encoding="utf-8"))
