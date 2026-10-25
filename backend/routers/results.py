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
_RESULTS = _BACKEND_DIR.parent / "outputs" / "results"


def _read_json(path: Path):
    if not path.exists():
        raise HTTPException(status_code=404, detail=f"{path.name} yok — ilgili betiği çalıştırın")
    return json.loads(path.read_text(encoding="utf-8"))


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


@router.get("/results/benchmark")
def results_benchmark():
    """Mekânsal blok CV karşılaştırması (scripts/09_benchmark.py)."""
    return _read_json(_RESULTS / "benchmark_summary.json")


@router.get("/results/critical-pga")
def results_critical_pga():
    """Kritik PGA belirsizliği + tasarım tablosu (scripts/10_critical_pga.py)."""
    import csv
    out = _read_json(_RESULTS / "critical_pga.json")
    table = _RESULTS / "critical_pga_table.csv"
    if table.exists():
        with table.open(encoding="utf-8") as f:
            rows = list(csv.DictReader(f))
        for r in rows:
            for k, v in r.items():
                try:
                    r[k] = float(v) if v not in ("", None) else None
                except ValueError:
                    pass
        out["design_table"] = rows
    return out


@router.get("/results/fem")
def results_fem():
    """FEM-SSR pilot sonuçları (scripts/02_run_fem.py)."""
    return _read_json(_RESULTS / "fem_pilots.json")


@router.get("/results/newmark-sim")
def results_newmark_sim():
    """Poligon bazında zaman-tanım Newmark simülasyonu (scripts/13)."""
    return _read_json(_RESULTS / "newmark_th_polygons.json")


@router.get("/results/newmark-sim/validation")
def results_newmark_sim_validation():
    return _read_json(_RESULTS / "newmark_th_validation.json")



@router.get("/results/rainfall")
def results_rainfall():
    """Yağış senaryosu (scripts/17)."""
    return _read_json(_RESULTS / "rainfall_scenario.json")
