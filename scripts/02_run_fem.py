"""
Script 2: FEM Pilot Analizi — Kayma Dayanımı Azaltma (SSR), ID 1327 ve ID 1353

v1'de SfePy kurulu olmadığından FEM konturları analitik yedekti. v2'de
src/fem/ssr_fem.py içindeki elasto-viskoplastik SSR çözücüsü (Griffiths &
Lane 1999) kullanılır; harici kütüphane gerekmez.

Her pilot yamaç için:
  1. Statik güvenlik faktörü FS_FEM (SSR)
  2. Psödo-statik kritik ivme K_y,FEM: SRF = 1'de göçmeye yol açan k_h
  3. FS–k_h eğrisi (psödo-statik yatay katsayı taraması)
  4. Göçme mekanizması: yakınsamayan ilk SRF'de yer değiştirme alanı ve
     birikmiş plastik kayma gerinmesi; SRF = 1'de von Mises gerilmesi
  5. Sonsuz şev (Mohr-Coulomb) Ky ve Newmark Dn ile karşılaştırma
     → FEM'in Newmark girdisini (Ky) nasıl değiştirdiği

Geometri: şev yüksekliği H = 20 m, temel derinlik faktörü D = 1.5 (Griffiths &
Lane tanımı). Eğim ve litoloji pilot tanımlarından alınır. Bu bir 2B kesit
idealizasyonudur; gerçek pilot topoğrafyası değildir.

Kullanım:
  python3 scripts/02_run_fem.py [--slope-id 1327] [--fast]
"""
import sys
import json
import time
import argparse
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from config import FIGURES_DIR, OUTPUTS_DIR, DN_THRESHOLD_M, ensure_dirs
from src.geomechanics.lithology import LITHOLOGY_TABLE
from src.geomechanics.mohr_coulomb import StaticFS, CriticalAcceleration
from src.geomechanics.newmark import NewmarkDisplacement
from src.fem.ssr_fem import SlopeGeometry, Material, SlopeSSR

PILOT_PARAMS = {
    1327: {"litho_id": 11, "theta_deg": 28.0, "z_m": 5.0, "actual_pga": 0.58},
    1353: {"litho_id": 7,  "theta_deg": 22.0, "z_m": 8.0, "actual_pga": 0.45},
}
H_SLOPE, D_FACTOR = 20.0, 1.5
INK, MUTED, BLUE, ORANGE = "#0b0b0b", "#52514e", "#2a78d6", "#eb6834"


def run_pilot(sid: int, p: dict, fast: bool = False) -> dict:
    lith = LITHOLOGY_TABLE[p["litho_id"]]
    mat = Material(c=lith["c_prime"], phi_deg=lith["phi_prime"], gamma=lith["gamma"],
                   E=lith["E_mpa"] * 1e3, nu=lith["nu"])
    n = 1 if not fast else 0
    geom = SlopeGeometry(height=H_SLOPE, slope_deg=p["theta_deg"], depth_factor=D_FACTOR,
                         nx_crest=8 + 4 * n, nx_slope=12 + 6 * n, nx_toe=8 + 4 * n, ny=8 + 4 * n)
    t0 = time.time()
    solver = SlopeSSR(geom, mat, max_iter=600 if fast else 1000)
    print(f"\n[ID {sid}] {lith['description']}  θ={p['theta_deg']}°  c'={mat.c} kPa  φ'={mat.phi_deg}°  "
          f"| {len(solver.mesh.elems)} eleman, {solver.ndof} SD")

    fs, pre, post = solver.factor_of_safety(kh=0.0, lo=0.5, hi=4.0, tol=0.01)
    print(f"  FS_FEM (statik SSR)      = {fs:.3f}")
    ky_fem = solver.yield_acceleration(kh_hi=1.0, tol=0.005)
    print(f"  K_y,FEM (psödo-statik)   = {ky_fem:.3f} g")

    khs = np.round(np.linspace(0.0, max(0.05, 1.6 * ky_fem), 7), 3)
    fs_kh = []
    for kh in khs:
        f, _, _ = solver.factor_of_safety(kh=float(kh), lo=0.3, hi=max(fs, 1.0) + 0.2, tol=0.02)
        fs_kh.append(float(f))
    static = solver.solve(1.0, 0.0)

    # Sonsuz şev karşılaştırması
    fs_inf = float(StaticFS(mat.c, mat.phi_deg, mat.gamma, p["z_m"], p["theta_deg"]))
    ky_inf = float(CriticalAcceleration(fs_inf, p["theta_deg"]))
    pga = p["actual_pga"]
    dn_inf = float(NewmarkDisplacement(ky_inf, pga)) * 100
    dn_fem = float(NewmarkDisplacement(ky_fem, pga)) * 100
    res = {
        "slope_id": sid, "lithology": lith["description"], "theta_deg": p["theta_deg"],
        "c_kpa": mat.c, "phi_deg": mat.phi_deg, "gamma": mat.gamma, "E_kpa": mat.E, "nu": mat.nu,
        "geometry": {"H_m": H_SLOPE, "D": D_FACTOR, "elements": int(len(solver.mesh.elems))},
        "fs_fem": round(fs, 3), "ky_fem_g": round(ky_fem, 3),
        "fs_infinite_slope": round(fs_inf, 3), "ky_infinite_slope_g": round(ky_inf, 3),
        "actual_pga_g": pga,
        "dn_newmark_ky_infinite_cm": round(dn_inf, 2), "dn_newmark_ky_fem_cm": round(dn_fem, 2),
        "triggered_ky_infinite": bool(dn_inf >= DN_THRESHOLD_M * 100),
        "triggered_ky_fem": bool(dn_fem >= DN_THRESHOLD_M * 100),
        "fs_vs_kh": [{"kh": float(k), "fs": round(f, 3)} for k, f in zip(khs, fs_kh)],
        "runtime_s": round(time.time() - t0, 1),
    }
    print(f"  Sonsuz şev: FS={fs_inf:.3f}, Ky={ky_inf:.3f} g | PGA={pga} g → "
          f"Dn(Ky_inf)={dn_inf:.1f} cm, Dn(Ky_FEM)={dn_fem:.1f} cm")

    _plot_fields(sid, solver, static, pre, post, fs, lith["description"], p["theta_deg"])
    _plot_fs_kh(sid, khs, fs_kh, ky_fem, ky_inf, fs)
    return res


# ------------------------------------------------------------------ grafikler
def _tri(solver):
    """8 düğümlü elemanları köşe düğümlerinden 2 üçgene böler (çizim için)."""
    import matplotlib.tri as mtri
    e = solver.mesh.elems
    tris = np.vstack([e[:, [0, 1, 2]], e[:, [0, 2, 3]]])
    return mtri.Triangulation(solver.mesh.nodes[:, 0], solver.mesh.nodes[:, 1], tris)


def _gp_to_nodes(solver, vals):
    """Gauss noktası değerlerini düğümlere ortalama ile taşır."""
    nn = len(solver.mesh.nodes)
    acc = np.zeros(nn); cnt = np.zeros(nn)
    ev = vals.mean(axis=1)
    for k in range(8):
        np.add.at(acc, solver.mesh.elems[:, k], ev)
        np.add.at(cnt, solver.mesh.elems[:, k], 1)
    return acc / np.maximum(cnt, 1)


def _plot_fields(sid, solver, static, pre, post, fs, lith, theta):
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    tri = _tri(solver)
    # Mekanizma: göçme sonrası (yakınsamayan) ile son kararlı çözüm arasındaki fark —
    # elastik oturmayı ayıklar, kayan kütleyi gösterir.
    u = (post.u - pre.u).reshape(-1, 2)
    umag = np.hypot(u[:, 0], u[:, 1])
    fields = [
        ("displacement", umag / max(umag.max(), 1e-12), "Göçme mekanizması — artımsal yer değiştirme (normalize)",
         "Blues", f"SRF {pre.srf:.2f} → {post.srf:.2f} (FS ≈ {fs:.2f})"),
        ("plastic_strain", _gp_to_nodes(solver, post.plastic_shear_strain),
         "Birikmiş plastik kayma gerinmesi (kayma yüzeyi)", "Oranges", f"SRF = {post.srf:.2f}"),
        ("von_mises", _gp_to_nodes(solver, static.von_mises), "Von Mises (σ̄) gerilmesi, kPa",
         "Purples", "SRF = 1 (statik)"),
    ]
    for key, val, title, cmap, sub in fields:
        fig, ax = plt.subplots(figsize=(10, 4.6))
        cs = ax.tricontourf(tri, val, levels=14, cmap=cmap)
        ax.triplot(tri, color="white", lw=0.15, alpha=0.5)
        if key == "displacement":
            sc = 0.12 * np.ptp(solver.mesh.nodes[:, 0]) / max(np.abs(u).max(), 1e-12)
            corner = np.unique(solver.mesh.elems[::2, 0])
            st = corner[umag[corner] > 0.15 * umag.max()]
            ax.quiver(solver.mesh.nodes[st, 0], solver.mesh.nodes[st, 1], u[st, 0] * sc, u[st, 1] * sc,
                      angles="xy", scale_units="xy", scale=1, color=INK, width=0.0016, alpha=0.6)
        cb = fig.colorbar(cs, ax=ax, shrink=0.6); cb.ax.tick_params(colors=MUTED)
        ax.set_aspect("equal"); ax.set_xlabel("x (m)", color=MUTED); ax.set_ylabel("y (m)", color=MUTED)
        ax.tick_params(colors=MUTED)
        for s in ax.spines.values():
            s.set_color("#c9c8c2")
        ax.set_title(f"ID {sid} · {lith}, θ = {theta:.0f}° — {title}\n{sub} · FEM-SSR (Griffiths & Lane 1999)",
                     loc="left", fontsize=10.5, color=INK)
        fig.tight_layout()
        fig.savefig(FIGURES_DIR / f"fem_{sid}_{key}.png", dpi=160); plt.close(fig)


def _plot_fs_kh(sid, khs, fs_kh, ky_fem, ky_inf, fs0):
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(khs, fs_kh, "o-", color=BLUE, lw=2, ms=7, mec="white", mew=1.5)
    ax.text(khs[-1] + 0.005, fs_kh[-1], "FEM-SSR", color=INK, fontsize=9, va="center")
    ax.axhline(1.0, color=MUTED, ls=":", lw=1)
    ax.axvline(ky_fem, color=BLUE, lw=1, alpha=0.6)
    ax.text(ky_fem, ax.get_ylim()[1] if False else max(fs_kh) * 0.98, f" K_y,FEM = {ky_fem:.3f} g",
            color=INK, fontsize=9, va="top")
    ax.axvline(ky_inf, color=ORANGE, lw=1, ls="--")
    ax.text(ky_inf, max(fs_kh) * 0.88, f" K_y,sonsuz şev = {ky_inf:.3f} g", color=INK, fontsize=9, va="top")
    ax.set_xlabel("Psödo-statik yatay katsayı k_h (g)", color=MUTED)
    ax.set_ylabel("Güvenlik faktörü FS", color=MUTED)
    ax.set_title(f"ID {sid} — FS vs k_h (FS₀ = {fs0:.2f})", loc="left", fontsize=11.5, color=INK)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.grid(color="#ecebe7", lw=0.8); ax.tick_params(colors=MUTED)
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / f"pga_fs_curve_ID{sid}.png", dpi=160); plt.close(fig)


def run(slope_ids=None, fast=False):
    ensure_dirs()
    out = []
    for sid in slope_ids or list(PILOT_PARAMS):
        out.append(run_pilot(sid, PILOT_PARAMS[sid], fast))
    d = OUTPUTS_DIR / "results"; d.mkdir(parents=True, exist_ok=True)
    path = d / "fem_pilots.json"
    if path.exists():   # tek pilot koşulursa diğerinin sonucu korunur
        done = {r["slope_id"] for r in out}
        out = [r for r in json.loads(path.read_text()) if r["slope_id"] not in done] + out
    out.sort(key=lambda r: r["slope_id"])
    path.write_text(json.dumps(out, indent=2, ensure_ascii=False))
    print(f"\nFEM sonuçları: {d / 'fem_pilots.json'}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="FEM-SSR pilot analizi")
    ap.add_argument("--slope-id", type=int, default=None)
    ap.add_argument("--fast", action="store_true", help="Kaba ağ (hızlı deneme)")
    a = ap.parse_args()
    run([a.slope_id] if a.slope_id else None, a.fast)
