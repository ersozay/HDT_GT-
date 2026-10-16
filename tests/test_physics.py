"""Fizik çekirdeği: sonsuz şev, kritik ivme, Newmark — bilinen değerlerle."""
import numpy as np
import pytest

from src.geomechanics.mohr_coulomb import StaticFS, CriticalAcceleration, DynamicFS
from src.geomechanics.newmark import NewmarkDisplacement, LandslideLabel


def test_cohesionless_infinite_slope_equals_tan_ratio():
    # c' = 0, kuru: FS = tanφ / tanθ (ders kitabı sonucu)
    fs = StaticFS(0.0, 32.0, 18.0, 5.0, 28.0)
    assert fs == pytest.approx(np.tan(np.deg2rad(32)) / np.tan(np.deg2rad(28)), rel=1e-9)


def test_static_fs_hand_calculation():
    # c'=15, φ'=28, γ=19, z=8, θ=22 → elle: (15 + 152·cos²22·tan28) / (152·sin22·cos22)
    t, p = np.deg2rad(22), np.deg2rad(28)
    expected = (15 + 152 * np.cos(t) ** 2 * np.tan(p)) / (152 * np.sin(t) * np.cos(t))
    assert StaticFS(15.0, 28.0, 19.0, 8.0, 22.0) == pytest.approx(expected, rel=1e-9)


def test_pore_pressure_reduces_fs():
    assert StaticFS(10, 30, 19, 5, 30, u=20) < StaticFS(10, 30, 19, 5, 30, u=0)


def test_critical_acceleration_nonnegative_and_formula():
    assert CriticalAcceleration(0.8, 30.0) == 0.0
    assert CriticalAcceleration(1.5, 30.0) == pytest.approx(0.5 * 0.5)


def test_dynamic_fs_equals_static_at_zero_pga():
    assert DynamicFS(10, 30, 19, 5, 30, 0.0) == pytest.approx(StaticFS(10, 30, 19, 5, 30))


def test_jibson2007_known_value():
    # Jibson (2007) Denk. 6, Ky/amax = 0.5: log Dn = 0.215 + log[(0.5)^2.341 (0.5)^-1.438]
    expected_cm = 10 ** (0.215 + np.log10(0.5 ** 2.341 * 0.5 ** -1.438))
    assert NewmarkDisplacement(0.2, 0.4) * 100 == pytest.approx(expected_cm, rel=1e-9)


def test_newmark_zero_below_ky_and_monotone_in_pga():
    assert NewmarkDisplacement(0.3, 0.25) == 0.0
    pga = np.linspace(0.05, 1.5, 60)
    dn = NewmarkDisplacement(np.full_like(pga, 0.15), pga)
    assert np.all(np.diff(dn) >= -1e-12)


def test_newmark_monotone_decreasing_in_ky():
    ky = np.linspace(0.02, 0.6, 40)
    dn = NewmarkDisplacement(ky, np.full_like(ky, 0.6))
    assert np.all(np.diff(dn) <= 1e-12)


def test_label_threshold_10cm():
    assert list(LandslideLabel(np.array([0.05, 0.10, 0.2]))) == [0, 1, 1]
