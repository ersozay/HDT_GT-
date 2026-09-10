"""
PINN Eğitim Grafikleri ve SHAP Önem Analizi.
"""
from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path


def plot_training_history(
    history,
    save_path: str | Path | None = None,
    figsize: tuple[float, float] = (14, 8),
) -> plt.Figure:
    """
    Eğitim tarihçesi — Loss, F1, AUC ve bileşen kayıpları.

    Parametreler
    -----------
    history  : TrainHistory dataclass (trainer.py'dan)
    save_path: Kaydedilecek dosya yolu
    """
    fig, axes = plt.subplots(2, 2, figsize=figsize)

    epochs = range(1, len(history.train_loss) + 1)

    # 1. Train vs Val Loss
    ax = axes[0, 0]
    ax.plot(epochs, history.train_loss, label="Train", color="steelblue", lw=1.5)
    ax.plot(epochs, history.val_loss,   label="Val",   color="darkorange",  lw=1.5)
    ax.set_title("Total Loss")
    ax.set_xlabel("Epoch")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 2. Val F1 ve AUC
    ax = axes[0, 1]
    ax.plot(epochs, history.val_f1,  label="Val F1",  color="forestgreen", lw=1.5)
    ax.plot(epochs, history.val_auc, label="Val AUC", color="crimson",      lw=1.5, linestyle="--")
    ax.axhline(y=0.8, color="gray", linestyle=":", lw=1.0, label="0.8 Hedef")
    ax.set_ylim(0, 1)
    ax.set_title("Val F1 & ROC-AUC")
    ax.set_xlabel("Epoch")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 3. Focal Loss bileşeni
    ax = axes[1, 0]
    ax.plot(epochs, history.focal_loss, color="mediumpurple", lw=1.5)
    ax.set_title("Focal Loss Bileşeni")
    ax.set_xlabel("Epoch")
    ax.grid(True, alpha=0.3)

    # 4. Physics Penalty bileşeni
    ax = axes[1, 1]
    ax.plot(epochs, history.phys_loss, color="coral", lw=1.5)
    ax.set_title("Physics Penalty Bileşeni")
    ax.set_xlabel("Epoch")
    ax.grid(True, alpha=0.3)

    fig.suptitle("LandslidePINN Eğitim Tarihçesi", fontsize=14, fontweight="bold")
    plt.tight_layout()

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")

    return fig


def plot_shap_summary(
    shap_values: np.ndarray,
    X_explain: np.ndarray,
    feature_names: list[str],
    top_n: int = 15,
    save_path: str | Path | None = None,
) -> plt.Figure:
    """
    SHAP önem grafiği — bar plot ve beeswarm.

    Parametreler
    -----------
    shap_values   : (n_samples, n_features) ndarray
    X_explain     : Açıklanan örnekler
    feature_names : Öznitelik isimleri
    top_n         : Gösterilecek en önemli öznitelik sayısı
    """
    shap_values = np.asarray(shap_values)
    # Sol panel: ortalama |SHAP| bar grafiği (manuel — her zaman güvenilir)
    mean_abs = np.abs(shap_values).mean(axis=0)
    top_idx  = np.argsort(mean_abs)[::-1][:top_n]

    fig, axes = plt.subplots(1, 2, figsize=(16, 7))
    axes[0].barh(
        [feature_names[i] for i in top_idx[::-1]],
        mean_abs[top_idx[::-1]],
        color="steelblue",
    )
    axes[0].set_xlabel("|SHAP Değeri| Ortalaması")
    axes[0].set_title(f"SHAP — Ortalama Önem (Top {top_n})")
    axes[0].grid(True, alpha=0.3, axis="x")

    # Sağ panel: beeswarm. shap kendi figürünü yönetmek istediğinden aktif
    # ekseni axes[1] olarak ayarlayıp öyle çiziyoruz.
    try:
        import shap
        plt.sca(axes[1])
        shap.summary_plot(shap_values, X_explain,
                          feature_names=feature_names,
                          max_display=top_n,
                          show=False, plot_size=None)
        axes[1].set_title("SHAP — Beeswarm")
    except Exception as e:
        axes[1].text(0.5, 0.5, f"Beeswarm çizilemedi:\n{e}",
                     ha="center", va="center", transform=axes[1].transAxes)

    plt.tight_layout()

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")

    return fig


def plot_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    save_path: str | Path | None = None,
) -> plt.Figure:
    """Karmaşıklık matrisi görselleştirmesi."""
    from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

    cm   = confusion_matrix(y_true, y_pred)
    disp = ConfusionMatrixDisplay(cm, display_labels=["Kararlı (0)", "Heyelan (1)"])

    fig, ax = plt.subplots(figsize=(6, 5))
    disp.plot(ax=ax, cmap="Blues", colorbar=False)
    ax.set_title("Karmaşıklık Matrisi — LandslidePINN")
    plt.tight_layout()

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")

    return fig
