"""
Dinamik Risk Haritaları.

'X g'de nereler kayar?' sorusunu piksel bazlı görselleştirir.
PGA senaryoları: 0.2g, 0.4g, 0.6g, 0.8g.

11 il bölgesi için upscaling çıktısı.
"""
from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from pathlib import Path


SCENARIO_PGAS = [0.2, 0.4, 0.6, 0.8]


def plot_dynamic_risk_map(
    ky_map: np.ndarray,
    pga_scenario: float,
    title_suffix: str = "",
    ax: plt.Axes | None = None,
    cmap: str = "RdYlGn_r",
    probability_map: np.ndarray | None = None,
) -> plt.Axes:
    """
    Tek bir PGA senaryosu için risk haritası çizer.

    Parametreler
    -----------
    ky_map         : 2D array — kritik ivme haritası (g)
    pga_scenario   : Senaryo PGA değeri (g)
    title_suffix   : Grafik başlığı eki
    ax             : Varolan matplotlib Axes (None → yeni oluştur)
    cmap           : Renk paleti
    probability_map: PINN model olasılık haritası (varsa, ky_map yerine kullanılır)
    """
    if ax is None:
        _, ax = plt.subplots(figsize=(8, 6))

    if probability_map is not None:
        risk = probability_map
        label = "P(Heyelan)"
        vmin, vmax = 0.0, 1.0
    else:
        risk = (pga_scenario > ky_map).astype(float)
        label = "Risk (1=Heyelan)"
        vmin, vmax = 0.0, 1.0

    im = ax.imshow(risk, cmap=cmap, vmin=vmin, vmax=vmax, origin="upper")
    plt.colorbar(im, ax=ax, label=label, fraction=0.046, pad=0.04)
    ax.set_title(f"PGA = {pga_scenario}g{' — ' + title_suffix if title_suffix else ''}")
    ax.set_xlabel("Doğu (piksel)")
    ax.set_ylabel("Kuzey (piksel)")
    ax.axis("on")

    # Risk piksel yüzdesi
    if probability_map is None:
        risk_pct = 100.0 * risk.mean()
        ax.text(0.02, 0.02, f"Risk alanı: {risk_pct:.1f}%",
                transform=ax.transAxes, fontsize=9,
                bbox={"boxstyle": "round,pad=0.3", "facecolor": "white", "alpha": 0.7})

    return ax


def plot_scenario_grid(
    ky_map: np.ndarray,
    scenarios: list[float] | None = None,
    save_path: str | Path | None = None,
    figsize: tuple[float, float] = (20, 5),
    title: str = "Farklı PGA Senaryolarında Heyelan Risk Haritası — Kahramanmaraş 11 İl",
) -> plt.Figure:
    """
    4 PGA senaryosu için 1×4 panel risk haritası.

    Parametreler
    -----------
    ky_map    : 2D array — kritik ivme haritası
    scenarios : PGA değerleri listesi, varsayılan [0.2, 0.4, 0.6, 0.8]
    save_path : Kaydedilecek dosya yolu
    """
    if scenarios is None:
        scenarios = SCENARIO_PGAS

    fig, axes = plt.subplots(1, len(scenarios), figsize=figsize)
    if len(scenarios) == 1:
        axes = [axes]

    for ax, pga_s in zip(axes, scenarios):
        plot_dynamic_risk_map(ky_map, pga_s, ax=ax)

    fig.suptitle(title, fontsize=14, fontweight="bold", y=1.02)
    plt.tight_layout()

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=200, bbox_inches="tight")
        print(f"Risk haritası kaydedildi: {save_path}")

    return fig


def plot_risk_comparison(
    ky_map_analytical: np.ndarray,
    pinn_prob_map: np.ndarray,
    pga_scenario: float = 0.4,
    save_path: str | Path | None = None,
) -> plt.Figure:
    """
    Analitik Ky haritası ile PINN olasılık haritasını yan yana karşılaştırır.
    """
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    plot_dynamic_risk_map(ky_map_analytical, pga_scenario,
                          title_suffix="Analitik Ky Modeli", ax=axes[0])
    plot_dynamic_risk_map(ky_map_analytical, pga_scenario,
                          title_suffix="PINN Modeli",
                          ax=axes[1], probability_map=pinn_prob_map)

    fig.suptitle(f"Heyelan Risk Tahmini Karşılaştırması — PGA={pga_scenario}g",
                 fontsize=14, fontweight="bold")
    plt.tight_layout()

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")

    return fig
