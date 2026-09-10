"""Figür uçları: sonuç PNG'lerini beyaz-liste ile servis eder."""
from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse

from config import FIGURES_DIR

router = APIRouter(prefix="/api/figures", tags=["figures"])

# Yalnızca bu figürler dışarıya açılır (path traversal önlenir)
ALLOWED = {
    "shap_summary",
    "critical_pga_curve_real",
    "susceptibility_map_real",
    "confusion_matrix",
    "training_history",
    "newmark_arias_models",
    "wenchuan_validation",
}


@router.get("/{name}")
def figure(name: str):
    key = name.replace(".png", "")
    if key not in ALLOWED:
        raise HTTPException(status_code=404, detail="Figür bulunamadı")
    path = FIGURES_DIR / f"{key}.png"
    if not path.exists():
        raise HTTPException(status_code=404, detail="Dosya yok")
    return FileResponse(path, media_type="image/png")
