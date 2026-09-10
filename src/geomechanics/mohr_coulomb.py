"""
Mohr-Coulomb yenilme kriteri tabanlı şev stabilitesi hesaplamaları.

Teorik temel: Bishop (1955), Newmark (1965), Infinite Slope modeli.
"""
import numpy as np


def StaticFS(
    c_prime: float | np.ndarray,
    phi_prime_deg: float | np.ndarray,
    gamma: float | np.ndarray,
    z: float | np.ndarray,
    theta_deg: float | np.ndarray,
    u: float | np.ndarray = 0.0,
) -> float | np.ndarray:
    """
    Sonsuz şev (infinite slope) modelinde statik güvenlik faktörü.

    FS = [ c' + (γz·cos²θ – u)·tanφ' ] / [ γz·sinθ·cosθ ]

    Parametreler
    -----------
    c_prime      : Efektif kohezyon (kPa)
    phi_prime_deg: Efektif içsel sürtünme açısı (derece)
    gamma        : Birim hacim ağırlık (kN/m³)
    z            : Hesap derinliği (m)
    theta_deg    : Şev eğimi (derece)
    u            : Boşluk suyu basıncı (kPa), varsayılan 0 (kuru)

    Döndürür
    --------
    FS : float veya ndarray  — Güvenlik faktörü
    """
    phi = np.deg2rad(phi_prime_deg)
    theta = np.deg2rad(theta_deg)

    normal_stress = gamma * z * np.cos(theta) ** 2
    shear_driving = gamma * z * np.sin(theta) * np.cos(theta)

    shear_resist = c_prime + (normal_stress - u) * np.tan(phi)

    # Sıfıra bölünmeyi önle
    denom = np.where(np.abs(shear_driving) < 1e-12, 1e-12, shear_driving)
    return shear_resist / denom


def DynamicFS(
    c_prime: float | np.ndarray,
    phi_prime_deg: float | np.ndarray,
    gamma: float | np.ndarray,
    z: float | np.ndarray,
    theta_deg: float | np.ndarray,
    pga_g: float | np.ndarray,
    u: float | np.ndarray = 0.0,
) -> float | np.ndarray:
    """
    Sismik kuvvet dahil dinamik güvenlik faktörü.

    FS_din = [c + (γz·cos²θ – u)·tanφ] / [γz·sinθ·cosθ + kh·γz·cos²θ]
    kh = PGA / g  (g = 9.81 m/s², burada normalize g=1 kullanılır)

    Parametreler
    -----------
    pga_g : Tepe yer ivmesi (g cinsinden, örn. 0.4)
    Diğerleri: StaticFS ile aynı

    Döndürür
    --------
    FS_din : float veya ndarray
    """
    phi = np.deg2rad(phi_prime_deg)
    theta = np.deg2rad(theta_deg)
    kh = pga_g  # normalize: PGA/g

    normal_stress = gamma * z * np.cos(theta) ** 2
    shear_resist = c_prime + (normal_stress - u) * np.tan(phi)

    static_drive = gamma * z * np.sin(theta) * np.cos(theta)
    seismic_drive = kh * gamma * z * np.cos(theta) ** 2

    denom = static_drive + seismic_drive
    denom = np.where(np.abs(denom) < 1e-12, 1e-12, denom)
    return shear_resist / denom


def CriticalAcceleration(
    fs_static: float | np.ndarray,
    theta_deg: float | np.ndarray,
) -> float | np.ndarray:
    """
    Newmark (1965) kritik ivme (Ky) hesabı.

    Ky = (FS_statik – 1) · sin(θ)

    Ky değeri ≥ 0 olmalıdır; negatif değerler (FS < 1, zaten göçmüş) 0'a kırpılır.

    Parametreler
    -----------
    fs_static  : Statik güvenlik faktörü
    theta_deg  : Şev eğimi (derece)

    Döndürür
    --------
    Ky : float veya ndarray  — Kritik ivme (g cinsinden)
    """
    theta = np.deg2rad(theta_deg)
    ky = (fs_static - 1.0) * np.sin(theta)
    return np.maximum(0.0, ky)
