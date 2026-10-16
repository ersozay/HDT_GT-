"""
Mekânsal Blok Çapraz Doğrulama (Spatial Block CV).

Neden poligon-bazlı ayrım yetmez?
---------------------------------
Poligon-bazlı ayrım aynı heyelanın pikselleri arasındaki sızıntıyı önler; ancak
koseismik heyelanlar kümelenir: komşu poligonlar aynı yamaçta, aynı litolojide
ve aynı PGA hücresindedir. Rastgele poligon ayrımında test poligonunun komşusu
eğitimde kalır → başarı iyimser çıkar (Roberts ve ark. 2017; Brenning 2012).

Çözüm: çalışma alanı `block_deg` boyutlu karelere bölünür; bir bloğun TÜM
örnekleri (pozitif + negatif) aynı katmana düşer. Poligonun blok ataması
poligonun ortalama koordinatıyla yapılır (sınırı kesen poligon bölünmez).

Referanslar:
  Roberts, D.R. et al. (2017). Ecography 40(8), 913–929.
  Brenning, A. (2012). IEEE IGARSS, 5372–5375 (sperrorest).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold


def assign_spatial_blocks(
    df: pd.DataFrame,
    block_deg: float = 0.25,
    origin: tuple[float, float] = (35.7, 35.9),
) -> np.ndarray:
    """
    Her örneğe mekânsal blok kimliği atar.

    Pozitifler (label=1) poligon (polygon_id) bazında ortalama koordinata,
    negatifler kendi koordinatlarına göre bloklanır.
    """
    lon = df["lon"].to_numpy(float)
    lat = df["lat"].to_numpy(float)
    lab = df["label"].to_numpy(int)
    pid = df["polygon_id"].to_numpy()

    blon, blat = lon.copy(), lat.copy()
    pos = lab == 1
    if pos.any():
        cent = (pd.DataFrame({"pid": pid[pos], "lon": lon[pos], "lat": lat[pos]})
                .groupby("pid")[["lon", "lat"]].mean())
        blon[pos] = cent.loc[pid[pos], "lon"].to_numpy()
        blat[pos] = cent.loc[pid[pos], "lat"].to_numpy()

    ix = np.floor((blon - origin[0]) / block_deg).astype(int)
    iy = np.floor((blat - origin[1]) / block_deg).astype(int)
    return iy * 1000 + ix


def spatial_folds(
    y: np.ndarray,
    blocks: np.ndarray,
    n_splits: int = 5,
    seed: int = 42,
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Sınıf oranını dengeleyen grup-katlı (blok) ayrım: [(train_idx, test_idx), ...]."""
    sgkf = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    X_dummy = np.zeros((len(y), 1))
    return [(tr, te) for tr, te in sgkf.split(X_dummy, y, groups=blocks)]


def inner_validation_split(
    idx: np.ndarray,
    y: np.ndarray,
    blocks: np.ndarray,
    val_frac: float = 0.15,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Eğitim indekslerinden blok-bazlı doğrulama alt kümesi ayırır
    (erken durdurma + karar eşiği seçimi için; test bloğuna dokunulmaz).
    """
    n_splits = max(2, int(round(1.0 / val_frac)))
    sgkf = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    tr_rel, va_rel = next(sgkf.split(np.zeros((len(idx), 1)), y[idx], groups=blocks[idx]))
    return idx[tr_rel], idx[va_rel]


def fold_summary(y: np.ndarray, blocks: np.ndarray, folds) -> pd.DataFrame:
    rows = []
    for k, (tr, te) in enumerate(folds, 1):
        rows.append({
            "fold": k,
            "n_train": len(tr), "n_test": len(te),
            "blocks_train": len(np.unique(blocks[tr])),
            "blocks_test": len(np.unique(blocks[te])),
            "pos_rate_train": float(y[tr].mean()),
            "pos_rate_test": float(y[te].mean()),
        })
    return pd.DataFrame(rows)
