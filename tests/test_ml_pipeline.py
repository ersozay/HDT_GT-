"""Mekânsal CV, fizik-yolu örnekleyicisi, metrikler ve kayıp fonksiyonu."""
import numpy as np
import pandas as pd
import pytest
import torch

from src.data_engineering.feature_matrix import build_feature_matrix, MODEL_FEATURES
from src.evaluation.spatial_cv import assign_spatial_blocks, spatial_folds, inner_validation_split
from src.evaluation.metrics import set_pga, pga_monotonicity, critical_pga_from_curve
from src.pinn.loss import PhysicsInformedLoss
from src.pinn.physics_paths import PhysicsPathSampler


def _toy_table(n=3000, seed=0):
    rng = np.random.RandomState(seed)
    df, _ = build_feature_matrix(
        elevation_m=rng.uniform(200, 2000, n), slope_deg=rng.uniform(0, 50, n),
        aspect_deg=rng.uniform(0, 360, n), litho_array=rng.choice([7, 10, 11, 15, 27], n),
        pga_array=rng.uniform(0.05, 1.0, n))
    df["lon"] = rng.uniform(35.7, 39.2, n); df["lat"] = rng.uniform(35.9, 38.6, n)
    df["label"] = (rng.rand(n) < 0.3).astype(int)
    df["polygon_id"] = np.where(df.label == 1, rng.randint(0, 200, n), 10_000 + np.arange(n))
    return df


def test_spatial_folds_have_disjoint_blocks_and_whole_polygons():
    df = _toy_table()
    blocks = assign_spatial_blocks(df, 0.25)
    y = df.label.to_numpy()
    for tr, te in spatial_folds(y, blocks, 5):
        assert not set(blocks[tr]) & set(blocks[te])
        pid = df.polygon_id.to_numpy()
        pos = df.label.to_numpy() == 1
        assert not set(pid[tr][pos[tr]]) & set(pid[te][pos[te]])
        tr2, va = inner_validation_split(tr, y, blocks)
        assert not set(blocks[tr2]) & set(blocks[va])


def test_set_pga_recomputes_dn_consistently():
    df = _toy_table(200)
    X = df[MODEL_FEATURES].to_numpy(np.float64)
    X2 = set_pga(X, MODEL_FEATURES, 0.8)
    ref, _ = build_feature_matrix(
        elevation_m=df.elevation_m.values, slope_deg=df.slope_deg.values,
        aspect_deg=np.rad2deg(np.arctan2(df.aspect_sin, df.aspect_cos)).values % 360,
        litho_array=df.litho_id.values, pga_array=np.full(len(df), 0.8))
    i = MODEL_FEATURES.index("log_dn_cm")
    assert np.allclose(X2[:, i], ref["log_dn_cm"].values)


def test_physics_path_pairs_follow_physics():
    df = _toy_table(500)
    X = df[MODEL_FEATURES].to_numpy(np.float64)
    mean, std = X.mean(0), np.where(X.std(0) < 1e-8, 1, X.std(0))
    s = PhysicsPathSampler(X, MODEL_FEATURES, mean, std)
    Xa, Xb = s.sample(400)
    A, B = Xa * std + mean, Xb * std + mean
    ip, isl, iky, idn = (MODEL_FEATURES.index(c) for c in ("pga_g", "slope_deg", "ky_g", "log_dn_cm"))
    assert np.all(B[:200, ip] > A[:200, ip])                       # PGA yolu
    assert np.all(B[200:, isl] >= A[200:, isl])                    # eğim yolu
    assert np.all(B[200:, iky] <= A[200:, iky] + 1e-6)             # θ↑ ⇒ Ky↓
    assert np.all(B[:, idn] >= A[:, idn] - 1e-6)                   # her iki yolda Dn↑


def test_path_penalty_zero_when_monotone_and_positive_when_not():
    loss = PhysicsInformedLoss(lambda_mono=1.0)
    a = torch.tensor([0.1, 0.2]); b = torch.tensor([0.3, 0.4])
    assert loss.path_penalty(a, b).item() == 0.0
    assert loss.path_penalty(b, a).item() == pytest.approx(0.2)


def test_lambda_annealing_is_linear_from_initial_value():
    loss = PhysicsInformedLoss(lambda_phys=1.0)
    for ep in range(1, 11):
        loss.anneal_lambda(ep, 10, max_lambda=5.0)
        assert loss.lambda_phys == pytest.approx(1.0 + 4.0 * ep / 10)


def test_monotonicity_metric_and_crossing():
    P = np.array([[0.1, 0.5], [0.2, 0.4], [0.3, 0.6]])
    m = pga_monotonicity(P, tol=0.01)
    assert m["mono_violation_rate"] == 0.5
    grid = np.array([0.1, 0.2, 0.3])
    assert critical_pga_from_curve(grid, np.array([0.2, 0.4, 0.6]), 0.5) == pytest.approx(0.25)
    assert critical_pga_from_curve(grid, np.array([0.1, 0.2, 0.3]), 0.5) is None
