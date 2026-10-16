"""Eğitilmiş PINN checkpoint'lerini yükleme (v1 ve v2 uyumlu)."""
from __future__ import annotations

import json
import pickle
from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass
class LoadedModel:
    model: "torch.nn.Module"
    feat_cols: list[str]
    mean: np.ndarray
    std: np.ndarray
    threshold: float
    meta: dict

    def predict(self, X_raw: np.ndarray, batch: int = 65536) -> np.ndarray:
        import torch
        Xn = ((np.asarray(X_raw, dtype=np.float32) - self.mean) / self.std).astype(np.float32)
        out = []
        with torch.no_grad():
            for i in range(0, len(Xn), batch):
                out.append(self.model(torch.from_numpy(Xn[i:i + batch])).numpy().reshape(-1))
        return np.concatenate(out) if out else np.zeros(0, dtype=np.float32)


def load_checkpoint(ckpt_path: str | Path, scaler_path: str | Path | None = None,
                    threshold_path: str | Path | None = None) -> LoadedModel:
    """
    Checkpoint'i yükler. v2 checkpoint'leri feat_cols + scaler + eşik içerir;
    v1 için öznitelik listesi ALL_FEATURE_COLUMNS, scaler/eşik ayrı dosyalardır.
    """
    import torch
    from .model import LandslidePINN
    from ..data_engineering.feature_matrix import ALL_FEATURE_COLUMNS

    ckpt = torch.load(Path(ckpt_path), map_location="cpu", weights_only=False)
    n_in = int(ckpt["model_state"]["layer1.0.weight"].shape[1])
    feat_cols = list(ckpt.get("feat_cols") or ALL_FEATURE_COLUMNS)[:n_in]

    sc = ckpt.get("scaler")
    if sc is None and scaler_path is not None:
        sc = pickle.loads(Path(scaler_path).read_bytes())
    mean = np.asarray(sc["mean"], dtype=np.float32)
    std = np.asarray(sc["std"], dtype=np.float32)

    thr = ckpt.get("threshold")
    if thr is None and threshold_path is not None and Path(threshold_path).exists():
        thr = json.loads(Path(threshold_path).read_text()).get("decision_threshold")
    thr = float(thr if thr is not None else 0.5)

    model = LandslidePINN(in_features=n_in)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    meta = {k: v for k, v in ckpt.items() if k not in ("model_state", "optimizer", "scaler")}
    return LoadedModel(model, feat_cols, mean, std, thr, meta)
