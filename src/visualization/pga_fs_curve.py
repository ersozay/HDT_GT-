"""
PGA – Güvenlik Faktörü Korelasyon Eğrisi.

Bireysel heyelan ID'leri için FEM kalibre edilmiş PGA-FS eğrisi.
Ky (kritik ivme) kesim noktası görsel olarak işaretlenir.
"""
from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from pathlib import Path


def compute_fs_vs_pga(
    c: float,
    phi_deg: float,
    gamma: float,
    z: float,
    theta_deg: float,
    u: float = 0.0,
    pga_range: tuple[float, float] = (0.0, 1.2),
    n_points: int = 100,
) -> tuple[np.ndarray, np.ndarray, float]:
    """
    Verilen jeomekanik parametreler için PGA–FS ilişkisini hesaplar.

    Döndürür
    --------
    pga_values, fs_values, ky
    """
    from ..geomechanics.mohr_coulomb import StaticFS, DynamicFS, CriticalAcceleration

    pga_values = np.linspace(pga_range[0], pga_range[1], n_points)
    fs_values  = np.array([
        DynamicFS(c, phi_deg, gamma, z, theta_deg, pga, u)
        for pga in pga_values
    ])

    fs_static = float(StaticFS(c, phi_deg, gamma, z, theta_deg, u))
    ky        = float(CriticalAcceleration(fs_static, theta_deg))

    return pga_values, fs_values, ky


def plot_pga_fs_curve(
    slope_id: int,
    c: float,
    phi_deg: float,
    gamma: float,
    z: float,
    theta_deg: float,
    u: float = 0.0,
    actual_pga: float | None = None,
    save_path: str | Path | None = None,
    figsize: tuple[float, float] = (9, 5),
) -> plt.Figure:
    """
    Tek bir heyelan için PGA–FS eğrisi çizer.

    Parametreler
    -----------
    slope_id   : Heyelan poligon ID'si (grafik başlığı için)
    c, phi_deg, gamma, z, theta_deg : Jeomekanik parametreler
    u          : Boşluk suyu basıncı (kPa)
    actual_pga : Gerçek deprem PGA değeri (dikey çizgi)
    save_path  : Kaydedilecek dosya yolu

    Döndürür
    --------
    fig : matplotlib Figure
    """
    pga_values, fs_values, ky = compute_fs_vs_pga(
        c, phi_deg, gamma, z, theta_deg, u
    )

    fig, ax = plt.subplots(figsize=figsize)

    # Ana eğri
    ax.plot(pga_values, fs_values, "b-", lw=2.5,
            label=f"ID {slope_id} — θ={theta_deg}°, c'={c} kPa, φ'={phi_deg}°")

    # FS = 1 kritik eşik çizgisi
    ax.axhline(y=1.0, color="red", linestyle="--", lw=1.5, label="FS = 1 (Kritik Eşik)")

    # Ky dikey çizgisi
    if 0 < ky < pga_values[-1]:
        ax.axvline(x=ky, color="orange", linestyle=":", lw=2.0,
                   label=f"Ky = {ky:.3f} g (Kritik İvme)")
        ax.annotate(
            f"Ky={ky:.3f}g",
            xy=(ky, 1.0),
            xytext=(ky + 0.05, 1.2),
            fontsize=10,
            color="orange",
            arrowprops={"arrowstyle": "->", "color": "orange"},
        )

    # Gerçek PGA
    if actual_pga is not None:
        ax.axvline(x=actual_pga, color="crimson", linestyle="-.", lw=2.0,
                   label=f"Gerçek PGA = {actual_pga:.2f} g")

    # FS < 1 bölgesini vurgula
    ax.fill_between(pga_values, 0, 1, where=(fs_values <= 1),
                    alpha=0.15, color="red", label="Göçme Bölgesi (FS<1)")

    ax.set_xlabel("PGA (g)", fontsize=13)
    ax.set_ylabel("Güvenlik Faktörü (FS)", fontsize=13)
    ax.set_title(f"PGA – Güvenlik Faktörü İlişkisi — Heyelan ID {slope_id}", fontsize=14)
    ax.legend(fontsize=10, loc="upper right")
    ax.grid(True, alpha=0.35)
    ax.set_xlim(0, pga_values[-1])
    ax.set_ylim(0, max(fs_values) * 1.1)
    ax.yaxis.set_minor_locator(ticker.AutoMinorLocator())

    plt.tight_layout()

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")

    return fig


def plot_pga_fs_multi(
    slope_params: list[dict],
    save_path: str | Path | None = None,
    figsize: tuple[float, float] = (11, 6),
) -> plt.Figure:
    """
    Birden fazla heyelan ID'si için PGA–FS eğrilerini karşılaştırır.

    slope_params: [{"slope_id": 1327, "c": 15, "phi_deg": 30, ...}, ...]
    """
    fig, ax = plt.subplots(figsize=figsize)
    colors = plt.cm.tab10(np.linspace(0, 1, len(slope_params)))

    for sp, color in zip(slope_params, colors):
        sid = sp["slope_id"]
        pga_values, fs_values, ky = compute_fs_vs_pga(
            sp["c"], sp["phi_deg"], sp["gamma"], sp["z"], sp["theta_deg"],
            sp.get("u", 0.0),
        )
        ax.plot(pga_values, fs_values, lw=2, color=color,
                label=f"ID {sid} (Ky={ky:.2f}g)")
        if 0 < ky < pga_values[-1]:
            ax.axvline(x=ky, color=color, linestyle=":", lw=1.2, alpha=0.7)

    ax.axhline(y=1.0, color="red", linestyle="--", lw=2, label="FS = 1 Eşiği", zorder=5)
    ax.set_xlabel("PGA (g)", fontsize=13)
    ax.set_ylabel("Güvenlik Faktörü (FS)", fontsize=13)
    ax.set_title("PGA – FS Karşılaştırması — Pilot Noktalar", fontsize=14)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.35)
    plt.tight_layout()

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")

    return fig
