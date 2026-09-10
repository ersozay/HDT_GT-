"""
Newmark (1965) rijit blok kalıcı yer değiştirme analizi.

Referanslar
-----------
- Jibson, R.W. (2007). Regression models for estimating coseismic landslide
  displacement. Engineering Geology 91(2–4), 209–218.
    · Denklem 6  — yalnız Ky/amax oranı (jibson2007).
    · Denklem 9  — Arias Yoğunluğu (Ia) + Ky tabanlı (jibson2007_ia).
- Ambraseys, N.N. & Menu, J.M. (1988). Earthquake-induced ground displacements.
  Earthquake Engineering & Structural Dynamics 16(7), 985–1006.
    · Oran (Ky/amax) tabanlı ampirik regresyon (ambraseys1988).
"""
import numpy as np


def NewmarkDisplacement(
    ky: float | np.ndarray,
    pga_g: float | np.ndarray,
    method: str = "jibson2007",
) -> float | np.ndarray:
    """
    Kalıcı yer değiştirme (Dn) tahmini — oran (Ky/amax) tabanlı bağıntılar.

    Yöntemler
    ---------
    "jibson2007"    : Jibson (2007) Denklem 6 — yalnızca Ky ve PGA (amax) kullanır:
        log10(Dn) = 0.215 + log10[(1 – Ky/amax)^2.341 · (Ky/amax)^–1.438]

    "ambraseys1988" : Ambraseys & Menu (1988) oran-tabanlı regresyon:
        log10(Dn) = 0.90 + log10[(1 – Ky/amax)^2.53 · (Ky/amax)^–1.09]

    "simple"        : Basit lineer orantı (hızlı kalibrasyon):
        Dn = (PGA – Ky) · 0.5  (m)

    Parametreler
    -----------
    ky    : Kritik ivme (g cinsinden, Ky ≥ 0)
    pga_g : Tepe yer ivmesi (g cinsinden)
    method: "jibson2007", "ambraseys1988" veya "simple"

    Döndürür
    --------
    Dn : float veya ndarray — Kalıcı yer değiştirme (METRE)
    """
    ky    = np.asarray(ky,    dtype=float)
    pga_g = np.asarray(pga_g, dtype=float)

    if method in ("jibson2007", "ambraseys1988"):
        # PGA ≤ Ky → hareket yok
        active = pga_g > ky
        # Oran r = Ky / amax. Bu regresyonlar yalnızca r ≳ 0.01 için ampirik
        # olarak geçerlidir; r→0 (neredeyse düz arazi, Ky→0) durumunda rijit-blok
        # modeli fiziksel değildir ve Dn patlar. Oran tabanı 0.01'e sabitlenir,
        # Dn üstten 1000 cm (10 m) ile kırpılır.
        amax_safe = np.where(pga_g < 1e-9, 1e-9, pga_g)
        r         = np.clip(ky / amax_safe, 0.01, 1.0 - 1e-9)

        if method == "jibson2007":
            intercept, p_hi, p_lo = 0.215, 2.341, -1.438
        else:  # ambraseys1988
            intercept, p_hi, p_lo = 0.90, 2.53, -1.09

        factor    = (1.0 - r) ** p_hi * r ** p_lo
        log_dn_cm = intercept + np.log10(np.maximum(factor, 1e-12))
        dn_cm     = np.where(active, 10.0 ** log_dn_cm, 0.0)
        dn_cm     = np.clip(dn_cm, 0.0, 1000.0)   # fiziksel üst sınır
        return dn_cm / 100.0  # cm → m

    elif method == "simple":
        excess = np.maximum(pga_g - ky, 0.0)
        return excess * 0.5  # kalibre katsayı (FEM ile güncellenecek)

    else:
        raise ValueError(
            f"Bilinmeyen method: {method!r}. "
            "'jibson2007', 'ambraseys1988' veya 'simple' kullanın."
        )


def NewmarkDisplacementIa(
    ky: float | np.ndarray,
    ia_ms: float | np.ndarray,
) -> float | np.ndarray:
    """
    Arias Yoğunluğu (Ia) tabanlı kalıcı yer değiştirme — Jibson (2007) Denklem 9.

        log10(Dn) = 0.561·log10(Ia) − 3.833·log10(Ky) − 1.474

    Burada Dn cm, Ia birimi m/s, Ky birimi g'dir. Arias Yoğunluğu, deprem
    sarsıntısının toplam enerji içeriğini temsil eder ve zaman-serisi kaydının
    integraliyle tanımlanır:  Ia = (π / 2g) ∫ a(t)² dt.

    Parametreler
    -----------
    ky    : Kritik (akma) ivmesi (g, Ky > 0)
    ia_ms : Arias Yoğunluğu (m/s)

    Döndürür
    --------
    Dn : float veya ndarray — Kalıcı yer değiştirme (METRE)

    Not
    ---
    Ky → 0 için log terimi ıraksar; Ky tabanı 1e-3 g'ye sabitlenir. Dn üstten
    1000 cm ile kırpılır (rijit-blok fiziksel üst sınırı).
    """
    ky    = np.asarray(ky,    dtype=float)
    ia_ms = np.asarray(ia_ms, dtype=float)

    ky_safe = np.clip(ky, 1e-3, None)
    ia_safe = np.clip(ia_ms, 1e-6, None)

    log_dn_cm = 0.561 * np.log10(ia_safe) - 3.833 * np.log10(ky_safe) - 1.474
    dn_cm     = 10.0 ** log_dn_cm
    dn_cm     = np.clip(dn_cm, 0.0, 1000.0)
    return dn_cm / 100.0  # cm → m


def LandslideLabel(
    dn_m: float | np.ndarray,
    threshold_m: float = 0.10,
) -> np.ndarray:
    """
    Kalıcı yer değiştirme eşiğine göre binary etiket oluşturur.

    y = 1 → Dn ≥ eşik (Heyelan)
    y = 0 → Dn <  eşik (Kararlı)

    Parametreler
    -----------
    dn_m        : Kalıcı yer değiştirme (metre)
    threshold_m : Eşik (varsayılan 0.10 m = 10 cm, Jibson 2007)
    """
    return (np.asarray(dn_m, dtype=float) >= threshold_m).astype(int)
