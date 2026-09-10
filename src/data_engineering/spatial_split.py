"""
Polygon-Based Spatial Split — Veri Sızıntısını Önleme.

Neden gerekli?
--------------
Piksel bazlı rastgele bölmede, aynı heyelan poligonunun komşu pikselleri
hem train hem test kümesine düşebilir. Bu, modelin komşu pikseli
ezberlediği sahte yüksek doğruluk (spatial autocorrelation leakage)
üretir.

Çözüm: GroupKFold — aynı poligonun TÜM pikselleri daima aynı kümede.

Referans: Roberts et al. (2017) Cross-validation strategies for spatial CV.
"""
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold


def polygon_based_split(
    X: np.ndarray,
    y: np.ndarray,
    polygon_ids: np.ndarray,
    test_ratio: float = 0.2,
    random_state: int = 42,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Poligon kimliğine göre train/test böler — veri sızıntısını önler.

    Parametreler
    -----------
    X           : Öznitelik matrisi (n_samples, n_features)
    y           : Hedef vektör (n_samples,)
    polygon_ids : Her örneğin hangi poligona ait olduğu (n_samples,)
    test_ratio  : Test için ayrılacak poligon oranı
    random_state: Tekrar edilebilirlik için tohum

    Döndürür
    --------
    X_train, X_test, y_train, y_test
    """
    unique_polygons = np.unique(polygon_ids)
    rng = np.random.RandomState(random_state)
    rng.shuffle(unique_polygons)

    n_test = max(1, int(len(unique_polygons) * test_ratio))
    test_polygons = set(unique_polygons[:n_test])

    test_mask  = np.isin(polygon_ids, list(test_polygons))
    train_mask = ~test_mask

    return X[train_mask], X[test_mask], y[train_mask], y[test_mask]


class PolygonGroupKFold:
    """
    Poligon tabanlı k-katlı çapraz doğrulama.

    sklearn GroupKFold'u sargılayan, spatial leakage'ı engelleyen CV.

    Kullanım
    --------
    cv = PolygonGroupKFold(n_splits=5)
    for fold, (train_idx, val_idx) in enumerate(cv.split(X, y, groups=polygon_ids)):
        X_tr, X_val = X[train_idx], X[val_idx]
    """

    def __init__(self, n_splits: int = 5):
        self.n_splits = n_splits
        self._gkf = GroupKFold(n_splits=n_splits)

    def split(self, X, y, groups):
        return self._gkf.split(X, y, groups=groups)

    def get_n_splits(self):
        return self.n_splits

    def summary(self, polygon_ids: np.ndarray, y: np.ndarray) -> pd.DataFrame:
        """Her fold için poligon sayısı ve sınıf dağılımını özetler."""
        rows = []
        X_dummy = np.zeros((len(polygon_ids), 1))

        for fold, (tr, val) in enumerate(self.split(X_dummy, y, groups=polygon_ids)):
            n_poly_tr  = len(np.unique(polygon_ids[tr]))
            n_poly_val = len(np.unique(polygon_ids[val]))
            rows.append({
                "fold": fold + 1,
                "train_polygons": n_poly_tr,
                "val_polygons":   n_poly_val,
                "train_landslide_ratio": y[tr].mean(),
                "val_landslide_ratio":   y[val].mean(),
            })
        return pd.DataFrame(rows)


def load_inventory_kml(kml_path: str) -> "gpd.GeoDataFrame":
    """
    Kocaman (2025) KML envanterini yükler ve polygon_id sütunu ekler.

    Parametreler
    -----------
    kml_path : .kml dosya yolu (TJES-2024-00171_Landslides.kml)

    Döndürür
    --------
    GeoDataFrame : 2596 heyelan poligonu
    """
    import geopandas as gpd

    gdf = gpd.read_file(kml_path)
    gdf["polygon_id"] = gdf.index.astype(int)
    return gdf
