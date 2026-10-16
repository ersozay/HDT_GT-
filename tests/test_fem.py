"""FEM-SSR çözücüsü: Mohr-Coulomb ölçütü ve literatür kıyas problemleri."""
import numpy as np
import pytest

from src.fem.ssr_fem import SlopeGeometry, Material, SlopeSSR, mohr_coulomb_f, build_mesh


def test_mohr_coulomb_triaxial_compression():
    # σ3 = 100 kPa, c = 10, φ = 30° → q_f = 2c cosφ/(1−sinφ) + 2σ3 sinφ/(1−sinφ)
    c, phi, p3 = 10.0, np.deg2rad(30), 100.0
    qf = 2 * c * np.cos(phi) / (1 - np.sin(phi)) + 2 * p3 * np.sin(phi) / (1 - np.sin(phi))
    s = lambda q: np.array([-p3, -p3 - q, 0.0, -p3])     # çekme pozitif
    assert mohr_coulomb_f(s(qf * 0.99), c, phi) < 0
    assert mohr_coulomb_f(s(qf * 1.01), c, phi) > 0


def test_mesh_jacobians_positive():
    for D in (1.0, 1.5):
        g = SlopeGeometry(height=10, slope_deg=30, depth_factor=D)
        m = build_mesh(g)
        assert len(m.elems) > 0 and m.nodes[:, 1].min() == pytest.approx(0.0)
        SlopeSSR(g, Material(c=10, phi_deg=25, gamma=20, E=1e5))   # det(J) ≤ 0 → ValueError


@pytest.mark.slow
def test_griffiths_lane_example1():
    # Griffiths & Lane (1999) Örnek 1: β = 26.57°, φ' = 20°, c'/(γH) = 0.05, D = 1.
    # Bishop & Morgenstern FS = 1.380; G&L FEM ≈ 1.4.
    g = SlopeGeometry(height=10, slope_deg=26.565, depth_factor=1.0, nx_crest=10, nx_slope=20, ny=10)
    s = SlopeSSR(g, Material(c=0.05 * 20 * 10, phi_deg=20, gamma=20, E=1e5, nu=0.3))
    fs, _, _ = s.factor_of_safety(lo=1.0, hi=2.0, tol=0.02)
    assert 1.30 <= fs <= 1.45


@pytest.mark.slow
def test_cohesionless_slope_matches_infinite_slope():
    # c' = 0 → kritik mekanizma yüzeye paralel; FS → tanφ/tanθ
    g = SlopeGeometry(height=20, slope_deg=28, depth_factor=1.5, nx_crest=8, nx_slope=12, nx_toe=8, ny=8)
    s = SlopeSSR(g, Material(c=0.0, phi_deg=32, gamma=16.5, E=3e3, nu=0.35), max_iter=600)
    fs, _, _ = s.factor_of_safety(lo=0.8, hi=2.0, tol=0.02)
    assert fs == pytest.approx(np.tan(np.deg2rad(32)) / np.tan(np.deg2rad(28)), abs=0.06)
