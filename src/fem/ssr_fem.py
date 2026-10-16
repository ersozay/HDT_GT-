"""
2B Düzlem-Şekil Değiştirme Şev FEM'i — Kayma Dayanımı Azaltma (SSR) Yöntemi.

Griffiths & Lane (1999) elasto-viskoplastik algoritması (Smith & Griffiths,
"Programming the Finite Element Method", Program 6.3 esas alınmıştır):

  · 8 düğümlü serendipity dörtgen elemanlar, 2×2 indirgenmiş Gauss integrasyonu
  · Elastik–tam plastik Mohr-Coulomb; dilatasyonsuz (ψ = 0) akış kuralı
  · Yerçekimi tek adımda uygulanır ("gravity turn-on"); FS bu yaklaşımda
    başlangıç gerilme yolundan pratik olarak bağımsızdır (Griffiths & Lane 1999)
  · Güvenlik faktörü: c_f = c'/SRF, tanφ_f = tanφ'/SRF. Viskoplastik iterasyon
    `max_iter` içinde yakınsamıyorsa yamaç göçmüştür → FS = yakınsamanın
    kaybolduğu en küçük SRF (ikiye bölme ile)
  · Psödo-statik sismik yük: yatay gövde kuvveti k_h·γ (yamaç aşağı yönde).
    FS(k_h) = 1 olan k_h, FEM'den elde edilen kritik ivmedir: K_y,FEM

Birimler: kPa, kN/m³, m.

Doğrulama: Griffiths & Lane (1999) Örnek 1 — β = 26.57° (2:1), φ' = 20°,
c'/(γH) = 0.05, D = 1 → Bishop & Morgenstern FS ≈ 1.38, FEM ≈ 1.4.
`tests/test_fem.py` bu kıyası otomatik yapar.

Referanslar:
  Griffiths, D.V. & Lane, P.A. (1999). Slope stability analysis by finite
    elements. Géotechnique 49(3), 387–403.
  Smith, I.M., Griffiths, D.V. & Margetts, L. (2014). Programming the Finite
    Element Method, 5th ed. Wiley.
  Zienkiewicz, O.C., Humpheson, C. & Lewis, R.W. (1975). Géotechnique 25(4).
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla

# 8 düğümlü eleman yerel koordinatları (saat yönü tersine: köşeler, sonra kenar ortaları)
_XI = np.array([-1, 1, 1, -1, 0, 1, 0, -1], float)
_ETA = np.array([-1, -1, 1, 1, -1, 0, 1, 0], float)
_G = 1.0 / np.sqrt(3.0)
_GAUSS = [(-_G, -_G), (_G, -_G), (_G, _G), (-_G, _G)]


def _shape(xi, eta):
    """N (8,) ve dN/d(ξ,η) (2, 8)."""
    N = np.zeros(8); dN = np.zeros((2, 8))
    for i in range(4):
        a, b = _XI[i], _ETA[i]
        N[i] = 0.25 * (1 + a * xi) * (1 + b * eta) * (a * xi + b * eta - 1)
        dN[0, i] = 0.25 * a * (1 + b * eta) * (2 * a * xi + b * eta)
        dN[1, i] = 0.25 * b * (1 + a * xi) * (a * xi + 2 * b * eta)
    for i in range(4, 8):
        a, b = _XI[i], _ETA[i]
        if a == 0:
            N[i] = 0.5 * (1 - xi ** 2) * (1 + b * eta)
            dN[0, i] = -xi * (1 + b * eta)
            dN[1, i] = 0.5 * b * (1 - xi ** 2)
        else:
            N[i] = 0.5 * (1 + a * xi) * (1 - eta ** 2)
            dN[0, i] = 0.5 * a * (1 - eta ** 2)
            dN[1, i] = -eta * (1 + a * xi)
    return N, dN


# --------------------------------------------------------------------------- #
# Ağ (mesh)
# --------------------------------------------------------------------------- #
@dataclass
class SlopeGeometry:
    """
    Basit şev geometrisi (Griffiths & Lane 1999, Şekil 1 ile aynı düzen):

        y
        │  ─────────────┐ kret (y = D·H + H)
        │               ╲  şev açısı β
        │                ╲________________ topuk (y = D·H)
        │__________________________________ sabit taban (y = 0)
         0      x_kret   x_topuk          L
    """
    height: float = 10.0          # H (m)
    slope_deg: float = 26.57      # β
    depth_factor: float = 1.0     # D (toplam derinlik = D·H, topuk seviyesi = (D−1)·H)
    crest_len: float | None = None
    toe_len: float | None = None
    nx_crest: int = 8
    nx_slope: int = 10
    nx_toe: int = 10
    ny: int = 10

    def __post_init__(self):
        H = self.height
        self.crest_len = self.crest_len if self.crest_len is not None else 1.2 * H
        self.toe_len = self.toe_len if self.toe_len is not None else 1.5 * H

    @property
    def run(self) -> float:
        return self.height / np.tan(np.deg2rad(self.slope_deg))

    def y_top(self, x):
        H, D = self.height, self.depth_factor
        x1 = self.crest_len; x2 = x1 + self.run
        toe = (D - 1.0) * H; crest = D * H
        return np.where(x <= x1, crest,
                        np.where(x >= x2, toe, crest - (x - x1) / self.run * H))


@dataclass
class Mesh:
    nodes: np.ndarray        # (nn, 2)
    elems: np.ndarray        # (ne, 8)
    fixed_x: np.ndarray      # düğüm indeksleri
    fixed_y: np.ndarray
    surface: np.ndarray      # üst yüzey düğümleri (x sıralı)


def build_mesh(g: SlopeGeometry) -> Mesh:
    """
    Yapılandırılmış 8 düğümlü dörtgen ağ, iki blok:
      · Üst blok (topuk seviyesi → kret): her satır x = 0 … şev yüzü aralığına
        yatay eşlenir (trapez) — şev yüzü ve kret köşesi düğümlerle tam izlenir.
      · Alt blok (taban → topuk seviyesi, D > 1 ise): dikdörtgen temel.
    Arayüz satırı iki blokça paylaşılır.
    """
    H = g.height
    yt = (g.depth_factor - 1.0) * H           # topuk seviyesi
    x1 = g.crest_len; x2 = x1 + g.run
    has_low = yt > 1e-9
    nx_up = g.nx_crest + g.nx_slope
    nx_toe = g.nx_toe if has_low else 0
    L = x2 + (g.toe_len if has_low else 0.0)
    ny_up = g.ny
    ny_lo = max(1, int(round(g.ny * (g.depth_factor - 1.0)))) if has_low else 0

    nodes: list[tuple[float, float]] = []

    def add(x, y):
        nodes.append((float(x), float(y)))
        return len(nodes) - 1

    # Alt blok sütun konumları (üst bloğun alt satırıyla eşleşir)
    xs_lo = np.concatenate([np.linspace(0, x2, 2 * nx_up + 1),
                            np.linspace(x2, L, 2 * nx_toe + 1)[1:]])
    elems = []
    if has_low:
        nxl = nx_up + nx_toe
        idx_lo = -np.ones((2 * nxl + 1, 2 * ny_lo + 1), int)
        for i in range(2 * nxl + 1):
            for j in range(2 * ny_lo + 1):
                if i % 2 == 1 and j % 2 == 1:
                    continue
                idx_lo[i, j] = add(xs_lo[i], yt * j / (2 * ny_lo))
        for ex in range(nxl):
            for ey in range(ny_lo):
                i, j = 2 * ex, 2 * ey
                elems.append([idx_lo[i, j], idx_lo[i + 2, j], idx_lo[i + 2, j + 2], idx_lo[i, j + 2],
                              idx_lo[i + 1, j], idx_lo[i + 2, j + 1], idx_lo[i + 1, j + 2], idx_lo[i, j + 1]])

    idx_up = -np.ones((2 * nx_up + 1, 2 * ny_up + 1), int)
    for j in range(2 * ny_up + 1):
        y = yt + H * j / (2 * ny_up)
        x_face = x1 + (yt + H - y) / np.tan(np.deg2rad(g.slope_deg))
        xr = np.linspace(0, x_face, 2 * nx_up + 1)
        for i in range(2 * nx_up + 1):
            if i % 2 == 1 and j % 2 == 1:
                continue
            if j == 0 and has_low:
                idx_up[i, j] = idx_lo[i, -1]
            else:
                idx_up[i, j] = add(xr[i], y)
    for ex in range(nx_up):
        for ey in range(ny_up):
            i, j = 2 * ex, 2 * ey
            elems.append([idx_up[i, j], idx_up[i + 2, j], idx_up[i + 2, j + 2], idx_up[i, j + 2],
                          idx_up[i + 1, j], idx_up[i + 2, j + 1], idx_up[i + 1, j + 2], idx_up[i, j + 1]])

    nodes = np.array(nodes)
    elems = np.array(elems)
    on = lambda v, t: np.where(np.abs(v - t) < 1e-9)[0]
    bottom = on(nodes[:, 1], 0.0)
    left = on(nodes[:, 0], 0.0)
    right = on(nodes[:, 0], L) if has_low else np.array([], int)
    fixed_x = np.unique(np.concatenate([bottom, left, right]))
    fixed_y = np.unique(bottom)
    surface = idx_up[:, -1]
    return Mesh(nodes, elems, fixed_x, fixed_y, surface)


# --------------------------------------------------------------------------- #
# Mohr-Coulomb (Smith & Griffiths gerilme invaryantları; çekme pozitif)
# --------------------------------------------------------------------------- #
def _invariants(s):
    sx, sy, txy, sz = s[..., 0], s[..., 1], s[..., 2], s[..., 3]
    sm = (sx + sy + sz) / 3.0
    dsbar = np.sqrt(((sx - sy) ** 2 + (sy - sz) ** 2 + (sz - sx) ** 2 + 6 * txy ** 2) / 2.0)
    dx, dy, dz = sx - sm, sy - sm, sz - sm
    j3 = dx * dy * dz - dz * txy ** 2
    with np.errstate(divide="ignore", invalid="ignore"):
        sine = np.where(dsbar > 1e-10, -13.5 * j3 / dsbar ** 3, 0.0)
    theta = np.arcsin(np.clip(sine, -1.0, 1.0)) / 3.0
    return sm, dsbar, theta


def mohr_coulomb_f(s, c, phi_rad):
    """F = σm sinφ + σ̄ (cosθ/√3 − sinθ sinφ/3) − c cosφ   (F > 0 → akma)."""
    sm, dsbar, th = _invariants(s)
    sp_ = np.sin(phi_rad)
    return sm * sp_ + dsbar * (np.cos(th) / np.sqrt(3.0) - np.sin(th) * sp_ / 3.0) - c * np.cos(phi_rad)


def _dQ(s, psi_rad, h=1e-4):
    """Plastik potansiyel türevi ∂Q/∂σ — merkezi sonlu farklarla (vektörel)."""
    scale = np.maximum(np.abs(s).max(axis=-1, keepdims=True), 1.0)
    out = np.zeros_like(s)
    for k in range(4):
        e = np.zeros(4); e[k] = 1.0
        d = h * scale
        out[..., k] = (mohr_coulomb_f(s + d * e, 0.0, psi_rad) -
                       mohr_coulomb_f(s - d * e, 0.0, psi_rad)) / (2 * d[..., 0])
    return out


# --------------------------------------------------------------------------- #
# Çözücü
# --------------------------------------------------------------------------- #
@dataclass
class Material:
    c: float          # c' (kPa)
    phi_deg: float    # φ'
    gamma: float      # γ (kN/m³)
    E: float          # kPa
    nu: float = 0.3
    psi_deg: float = 0.0


@dataclass
class SSRResult:
    srf: float
    kh: float
    converged: bool
    iterations: int
    u: np.ndarray
    evp: np.ndarray                 # (ne, 4, 4) birikmiş viskoplastik gerinme
    stress: np.ndarray              # (ne, 4, 4) Gauss noktası gerilmeleri
    gp_xy: np.ndarray               # (ne, 4, 2)
    max_disp: float
    meta: dict = field(default_factory=dict)

    @property
    def von_mises(self) -> np.ndarray:
        _, dsbar, _ = _invariants(self.stress)
        return dsbar

    @property
    def plastic_shear_strain(self) -> np.ndarray:
        e = self.evp
        ex, ey, gxy, ez = e[..., 0], e[..., 1], e[..., 2], e[..., 3]
        return np.sqrt(2.0 / 3.0 * ((ex - ey) ** 2 + (ey - ez) ** 2 + (ez - ex) ** 2) / 2.0
                       + gxy ** 2 / 3.0)


class SlopeSSR:
    """Şev için elasto-viskoplastik SSR çözücüsü."""

    def __init__(self, geom: SlopeGeometry, mat: Material, max_iter: int = 1000, tol: float = 1e-4):
        self.geom, self.mat = geom, mat
        self.mesh = build_mesh(geom)
        self.max_iter, self.tol = max_iter, tol
        self._setup()

    def _D(self):
        E, nu = self.mat.E, self.mat.nu
        f = E / ((1 + nu) * (1 - 2 * nu))
        return f * np.array([[1 - nu, nu, 0, nu],
                             [nu, 1 - nu, 0, nu],
                             [0, 0, (1 - 2 * nu) / 2, 0],
                             [nu, nu, 0, 1 - nu]])

    def _setup(self):
        m = self.mesh
        nn, ne = len(m.nodes), len(m.elems)
        self.ndof = 2 * nn
        D = self._D(); self.D = D
        B_all = np.zeros((ne, 4, 4, 16))
        w_all = np.zeros((ne, 4))
        xy_all = np.zeros((ne, 4, 2))
        Nv_all = np.zeros((ne, 4, 8))
        rows, cols, vals = [], [], []
        self.edofs = np.zeros((ne, 16), int)
        for e, conn in enumerate(m.elems):
            X = m.nodes[conn]
            edof = np.empty(16, int); edof[0::2] = 2 * conn; edof[1::2] = 2 * conn + 1
            self.edofs[e] = edof
            ke = np.zeros((16, 16))
            for g, (xi, eta) in enumerate(_GAUSS):
                N, dN = _shape(xi, eta)
                J = dN @ X
                detJ = np.linalg.det(J)
                if detJ <= 0:
                    raise ValueError(f"Eleman {e}: det(J) ≤ 0 — ağ bozuk")
                dNx = np.linalg.solve(J, dN)
                B = np.zeros((4, 16))
                B[0, 0::2] = dNx[0]; B[1, 1::2] = dNx[1]
                B[2, 0::2] = dNx[1]; B[2, 1::2] = dNx[0]
                ke += B.T @ D @ B * detJ
                B_all[e, g] = B; w_all[e, g] = detJ
                xy_all[e, g] = N @ X; Nv_all[e, g] = N
            r, c_ = np.meshgrid(edof, edof, indexing="ij")
            rows.append(r.ravel()); cols.append(c_.ravel()); vals.append(ke.ravel())
        K = sp.coo_matrix((np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))),
                          shape=(self.ndof, self.ndof)).tocsr()

        fixed = np.unique(np.concatenate([2 * m.fixed_x, 2 * m.fixed_y + 1]))
        self.free = np.setdiff1d(np.arange(self.ndof), fixed)
        self.lu = spla.splu(K[self.free][:, self.free].tocsc())
        self.B, self.w, self.gp_xy, self.Nv = B_all, w_all, xy_all, Nv_all

    def _body_load(self, kh: float):
        """Gövde kuvvetleri: yerçekimi (−γ, y) + psödo-statik (+k_h·γ, x yamaç aşağı)."""
        g = self.mat.gamma
        F = np.zeros(self.ndof)
        fe = np.zeros((len(self.mesh.elems), 16))
        for gp in range(4):
            N = self.Nv[:, gp, :]
            fe[:, 1::2] += -g * N * self.w[:, gp, None]
            fe[:, 0::2] += kh * g * N * self.w[:, gp, None]
        np.add.at(F, self.edofs, fe)
        return F

    def solve(self, srf: float = 1.0, kh: float = 0.0) -> SSRResult:
        mat = self.mat
        c_f = mat.c / srf
        phi_f = np.arctan(np.tan(np.deg2rad(mat.phi_deg)) / srf)
        psi_f = np.arctan(np.tan(np.deg2rad(mat.psi_deg)) / srf)
        E, nu = mat.E, mat.nu
        dt = 4 * (1 + nu) * (1 - 2 * nu) / (E * (1 - 2 * nu + np.sin(phi_f) ** 2))

        F_ext = self._body_load(kh)
        ne = len(self.mesh.elems)
        evp = np.zeros((ne, 4, 4))
        u = np.zeros(self.ndof); u_old = np.zeros(self.ndof)
        converged = False
        BtD = np.einsum("egsi,st->egit", self.B, self.D)   # (ne,4,16,4)
        for it in range(1, self.max_iter + 1):
            # K u = F + ∫ Bᵀ D ε_vp (toplam viskoplastik gerinmeden gelen gövde yükü)
            fvp = np.einsum("egit,egt,eg->ei", BtD, evp, self.w)
            F = F_ext.copy(); np.add.at(F, self.edofs, fvp)
            u = np.zeros(self.ndof); u[self.free] = self.lu.solve(F[self.free])

            ue = u[self.edofs]
            eps = np.einsum("egsi,ei->egs", self.B, ue)
            sig = np.einsum("st,egt->egs", self.D, eps - evp)
            f = mohr_coulomb_f(sig, c_f, phi_f)
            yld = f > 0
            if yld.any():
                dq = _dQ(sig[yld], psi_f)
                evp[yld] += dt * f[yld, None] * dq

            du = np.abs(u - u_old).max(); umax = np.abs(u).max()
            u_old = u
            if it > 1 and du / max(umax, 1e-12) < self.tol:
                converged = True
                break
        sig = np.einsum("st,egt->egs", self.D, np.einsum("egsi,ei->egs", self.B, u[self.edofs]) - evp)
        return SSRResult(srf, kh, converged, it, u, evp, sig, self.gp_xy,
                         float(np.abs(u).max()),
                         meta={"c_f": c_f, "phi_f_deg": float(np.rad2deg(phi_f))})

    def factor_of_safety(self, kh: float = 0.0, lo: float = 0.3, hi: float = 5.0,
                         tol: float = 0.01) -> tuple[float, SSRResult, SSRResult]:
        """
        FS = yakınsamanın kaybolduğu SRF (ikiye bölme). Döndürür:
        (FS, son yakınsayan çözüm [göçme öncesi], ilk yakınsamayan çözüm [göçme mekanizması]).
        """
        r_lo = self.solve(lo, kh)
        if not r_lo.converged:
            return lo, r_lo, r_lo
        r_hi = self.solve(hi, kh)
        if r_hi.converged:
            return hi, r_hi, r_hi
        while hi - lo > tol:
            mid = 0.5 * (lo + hi)
            r = self.solve(mid, kh)
            if r.converged:
                lo, r_lo = mid, r
            else:
                hi, r_hi = mid, r
        return 0.5 * (lo + hi), r_lo, r_hi

    def yield_acceleration(self, kh_hi: float = 1.0, tol: float = 0.005) -> float:
        """K_y,FEM: SRF = 1'de yakınsamanın kaybolduğu k_h (ikiye bölme)."""
        if not self.solve(1.0, 0.0).converged:
            return 0.0
        lo, hi = 0.0, kh_hi
        if self.solve(1.0, hi).converged:
            return hi
        while hi - lo > tol:
            mid = 0.5 * (lo + hi)
            if self.solve(1.0, mid).converged:
                lo = mid
            else:
                hi = mid
        return 0.5 * (lo + hi)
