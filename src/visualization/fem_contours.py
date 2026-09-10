"""
FEM Kontur Görselleştirmesi.

PyVista ile 3D VTK mesh konturları (von Mises, kayma yüzeyi, yer değiştirme).
PyVista kurulu değilse Matplotlib 2D alternatifi kullanılır.
"""
from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path


def plot_fem_contours_2d(
    x_coords: np.ndarray,
    y_coords: np.ndarray,
    von_mises: np.ndarray,
    eq_plastic_strain: np.ndarray,
    displacement: np.ndarray,
    slope_id: int,
    save_dir: str | Path = "outputs/figures",
) -> list[plt.Figure]:
    """
    FEM çıktılarını 2D Matplotlib kontur haritaları olarak çizer.

    PyVista alternatifi — kurulum gerekmez.

    Parametreler
    -----------
    x_coords, y_coords  : 1D düğüm koordinatları (m)
    von_mises           : Von Mises gerilmesi (kPa)
    eq_plastic_strain   : Eşdeğer plastik kayma gerinmesi
    displacement        : Yer değiştirme büyüklüğü (m)
    slope_id            : Heyelan ID'si
    save_dir            : Grafik kayıt dizini

    Döndürür
    --------
    figures : 3 grafik listesi
    """
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    datasets = [
        (von_mises,          "Von Mises Gerilmesi (kPa)", "plasma",    "von_mises"),
        (eq_plastic_strain,  "Eşdeğer Plastik Kayma",    "hot_r",     "plastic_strain"),
        (displacement,       "Yer Değiştirme (m)",        "viridis",   "displacement"),
    ]

    figures = []
    for data, label, cmap, fname in datasets:
        fig, ax = plt.subplots(figsize=(10, 5))

        if len(x_coords) == len(data):
            sc = ax.scatter(x_coords, y_coords, c=data, cmap=cmap,
                            s=10, edgecolors="none")
        else:
            sc = ax.scatter(np.linspace(0, 200, len(data)),
                            np.zeros(len(data)),
                            c=data, cmap=cmap, s=10)

        plt.colorbar(sc, ax=ax, label=label)
        ax.set_title(f"ID {slope_id} — {label}", fontsize=13)
        ax.set_xlabel("Yatay Mesafe (m)")
        ax.set_ylabel("Yükseklik (m)")
        ax.grid(True, alpha=0.3)
        plt.tight_layout()

        save_path = save_dir / f"fem_{slope_id}_{fname}.png"
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        figures.append(fig)

    return figures


def plot_displacement_profile(
    x_coords: np.ndarray,
    displacement_y: np.ndarray,
    slope_id: int,
    dn_threshold: float = 0.10,
    save_path: str | Path | None = None,
) -> plt.Figure:
    """
    Şev boyunca kalıcı düşey yer değiştirme profili.

    Dn > 10 cm eşiği kırmızı bölge olarak işaretlenir.
    """
    fig, ax = plt.subplots(figsize=(10, 4))

    ax.plot(x_coords, displacement_y, "b-", lw=2, label="Dn (m)")
    ax.axhline(y=dn_threshold, color="red", linestyle="--", lw=1.5,
               label=f"Eşik = {dn_threshold*100:.0f} cm")

    ax.fill_between(x_coords, displacement_y, dn_threshold,
                    where=(displacement_y >= dn_threshold),
                    alpha=0.3, color="red", label="Dn > Eşik (Heyelan Bölgesi)")

    ax.set_xlabel("Şev Boyunca Mesafe (m)")
    ax.set_ylabel("Kalıcı Yer Değiştirme, Dn (m)")
    ax.set_title(f"ID {slope_id} — Sismik Kalıcı Yer Değiştirme Profili")
    ax.legend()
    ax.grid(True, alpha=0.35)
    plt.tight_layout()

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")

    return fig


def plot_fem_3d_pyvista(vtk_path: str | Path, slope_id: int):
    """
    PyVista ile 3D VTK kontur görselleştirmesi.

    Gereksinim: pip install pyvista

    Parametreler
    -----------
    vtk_path : FEM çıktısı .vtk dosya yolu
    slope_id : Heyelan ID'si
    """
    try:
        import pyvista as pv
    except ImportError:
        print("PyVista kurulu değil. pip install pyvista")
        return

    mesh = pv.read(str(vtk_path))

    plotter = pv.Plotter(shape=(1, 3), window_size=(1800, 600))

    # Von Mises
    plotter.subplot(0, 0)
    plotter.add_mesh(mesh, scalars="von_mises_stress", cmap="plasma",
                     show_scalar_bar=True,
                     scalar_bar_args={"title": "Von Mises (kPa)"})
    plotter.add_title(f"ID {slope_id} — Von Mises")

    # Plastik Gerinme
    plotter.subplot(0, 1)
    plotter.add_mesh(mesh, scalars="eq_plastic_strain", cmap="hot_r",
                     show_scalar_bar=True,
                     scalar_bar_args={"title": "Plastik Gerinme"})
    if "eq_plastic_strain" in mesh.array_names:
        contours = mesh.contour(isosurfaces=[0.05, 0.10, 0.20],
                                scalars="eq_plastic_strain")
        plotter.add_mesh(contours, color="cyan", line_width=3,
                         label="Kayma Yüzeyi")
    plotter.add_title(f"ID {slope_id} — Kayma Yüzeyi")

    # Yer Değiştirme
    plotter.subplot(0, 2)
    plotter.add_mesh(mesh, scalars="displacement", cmap="viridis",
                     show_scalar_bar=True,
                     scalar_bar_args={"title": "Yer Değiştirme (m)"})
    plotter.add_title(f"ID {slope_id} — Yer Değiştirme")

    plotter.show()
