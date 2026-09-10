"""
DEM verisinden 2D şev kesiti mesh oluşturucu.

SfePy / Gmsh için VTK/MSH formatında mesh üretir.
Pilot noktalar: ID 1327 ve ID 1353.
"""
import numpy as np
from pathlib import Path


def generate_slope_mesh_2d(
    dem_profile: np.ndarray,
    x_coords: np.ndarray,
    output_path: str | Path,
    depth_m: float = 30.0,
    element_size_m: float = 2.0,
    litho_layers: list[dict] | None = None,
) -> Path:
    """
    DEM profil kesitinden 2D üçgen eleman mesh oluşturur.

    VTK formatında çıktı verir (SfePy ile uyumlu).

    Parametreler
    -----------
    dem_profile    : 1D array — yüzey yükseklik profili (m)
    x_coords       : 1D array — yatay mesafe koordinatları (m)
    output_path    : Çıktı .vtk dosya yolu
    depth_m        : Model derinliği (m), varsayılan 30 m
    element_size_m : Hedef eleman boyutu (m)
    litho_layers   : Litoloji katman tanımı listesi
                     [{"depth_top": 0, "depth_bot": 5, "litho_id": 11}, ...]

    Döndürür
    --------
    Path : Oluşturulan .vtk dosya yolu
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Profil düğüm noktaları
    n_surface = len(x_coords)
    base_y = dem_profile.min() - depth_m

    # Yüzey + taban + kenar nokta seti
    surface_pts = np.column_stack([x_coords, dem_profile])

    x_left, x_right = x_coords[0], x_coords[-1]
    base_pts = np.array([
        [x_left,  base_y],
        [x_right, base_y],
    ])

    all_pts = np.vstack([surface_pts, base_pts])
    n_pts = len(all_pts)

    # Basit quad mesh (gerçek uygulamada Gmsh kullanılacak)
    # Şimdilik iki üçgen kapalı çokgen tanımı yazıyoruz
    cells = []
    for i in range(n_surface - 1):
        # Üst sol, üst sağ, alt sol üçgeni
        cells.append([i, i + 1, n_pts - 2])
        cells.append([i + 1, n_pts - 1, n_pts - 2])
    cells = np.array(cells)

    # VTK Legacy ASCII formatı
    vtk_lines = [
        "# vtk DataFile Version 3.0",
        f"Slope mesh - output: {output_path.stem}",
        "ASCII",
        "DATASET UNSTRUCTURED_GRID",
        f"POINTS {n_pts} float",
    ]
    for pt in all_pts:
        vtk_lines.append(f"{pt[0]:.4f} {pt[1]:.4f} 0.0000")

    n_cells = len(cells)
    vtk_lines.append(f"\nCELLS {n_cells} {n_cells * 4}")
    for tri in cells:
        vtk_lines.append(f"3 {tri[0]} {tri[1]} {tri[2]}")

    vtk_lines.append(f"\nCELL_TYPES {n_cells}")
    for _ in cells:
        vtk_lines.append("5")  # VTK_TRIANGLE = 5

    output_path.write_text("\n".join(vtk_lines))
    return output_path


def extract_slope_profile_from_dem(
    dem_array: np.ndarray,
    transform,
    point_x: float,
    point_y: float,
    profile_length_m: float = 500.0,
    n_points: int = 100,
    azimuth_deg: float = 180.0,
) -> tuple[np.ndarray, np.ndarray]:
    """
    DEM rasterından belirtilen nokta merkezli şev profili çıkarır.

    Parametreler
    -----------
    dem_array       : 2D DEM array (metre cinsinden yükseklik)
    transform       : rasterio affine transform
    point_x, point_y: Pilot nokta koordinatları (coğrafi)
    profile_length_m: Profil uzunluğu (m)
    n_points        : Örnekleme noktası sayısı
    azimuth_deg     : Profil yönü (° kuzeyden saat yönünde)

    Döndürür
    --------
    x_m      : Yatay mesafe dizisi (m)
    elevation : Yükseklik dizisi (m)
    """
    import rasterio
    from rasterio.transform import rowcol

    az_rad = np.deg2rad(azimuth_deg)
    dx_per_m = np.sin(az_rad)
    dy_per_m = np.cos(az_rad)

    half = profile_length_m / 2.0
    distances = np.linspace(-half, half, n_points)

    rows, cols, elevations = [], [], []
    for d in distances:
        gx = point_x + d * dx_per_m
        gy = point_y + d * dy_per_m
        r, c = rowcol(transform, gx, gy)
        r = int(np.clip(r, 0, dem_array.shape[0] - 1))
        c = int(np.clip(c, 0, dem_array.shape[1] - 1))
        rows.append(r)
        cols.append(c)
        elevations.append(float(dem_array[r, c]))

    x_m = distances + half  # 0'dan başlayan yatay mesafe
    return x_m, np.array(elevations)
