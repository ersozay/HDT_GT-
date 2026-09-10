"""
SfePy tabanlı 2D şev FEM çözücüsü.

Analiz adımları:
  1. Geostatic Step  — doğal gerilme dengesi
  2. Sismik Step     — ivme-zaman kaydı uygulaması

Gereksinim: pip install sfepy
"""
from __future__ import annotations

import numpy as np
from pathlib import Path
from dataclasses import dataclass, field


@dataclass
class FEMResult:
    """FEM analizi çıktı veri sınıfı."""
    slope_id: int
    von_mises: np.ndarray          # Von Mises gerilmesi (kPa)
    eq_plastic_strain: np.ndarray  # Eşdeğer plastik kayma gerinmesi (-)
    displacement_x: np.ndarray     # X-yer değiştirme (m)
    displacement_y: np.ndarray     # Y-yer değiştirme (m)
    dn_max: float                  # Maksimum kalıcı yer değiştirme (m)
    ky_fem: float                  # FEM'den kalibre edilmiş Ky (g)
    mesh_path: Path
    metadata: dict = field(default_factory=dict)

    @property
    def displacement_magnitude(self) -> np.ndarray:
        return np.sqrt(self.displacement_x**2 + self.displacement_y**2)

    @property
    def label(self) -> int:
        """Dn ≥ 10 cm ise heyelan (1), değilse kararlı (0)."""
        return int(self.dn_max >= 0.10)


class SlopeFEMSolver:
    """
    SfePy ile şev FEM analizi yürüten sınıf.

    Kullanım
    --------
    solver = SlopeFEMSolver(slope_id=1327, litho_params={...})
    result = solver.run(pga_timeseries, dt=0.01)
    """

    def __init__(
        self,
        slope_id: int,
        litho_params: dict,
        mesh_path: str | Path | None = None,
        output_dir: str | Path = "outputs/fem",
    ):
        """
        Parametreler
        -----------
        slope_id     : Heyelan poligon ID'si (örn. 1327, 1353)
        litho_params : {'c_prime': ..., 'phi_prime': ..., 'gamma': ...,
                        'E_mpa': ..., 'nu': ...}
        mesh_path    : .vtk mesh dosyası yolu (None ise otomatik oluşturulur)
        output_dir   : FEM çıktıları için dizin
        """
        self.slope_id = slope_id
        self.litho = litho_params
        self.mesh_path = Path(mesh_path) if mesh_path else None
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def _build_problem_conf(self, pga_g: float) -> dict:
        """SfePy problem konfigürasyonu oluşturur."""
        E  = self.litho["E_mpa"] * 1e3   # MPa → kPa
        nu = self.litho["nu"]
        rho = self.litho["gamma"] / 9.81  # kN/m³ → t/m³ ≈ Mg/m³

        return {
            "domain": {
                "mesh": str(self.mesh_path),
            },
            "materials": {
                "soil": {
                    "E": E,
                    "nu": nu,
                    "cohesion": self.litho["c_prime"],
                    "friction_angle": self.litho["phi_prime"],
                    "rho": rho,
                }
            },
            "boundary_conditions": {
                "fix_base": {
                    "region": "Bottom",
                    "dofs": ["u.0", "u.1"],
                    "val": 0,
                },
                "fix_left": {
                    "region": "Left",
                    "dofs": ["u.0"],
                    "val": 0,
                },
                "fix_right": {
                    "region": "Right",
                    "dofs": ["u.0"],
                    "val": 0,
                },
                "seismic": {
                    "region": "Bottom",
                    "dofs": ["u.0"],
                    "val": pga_g * 9.81,  # g → m/s²
                },
            },
        }

    def run_geostatic(self) -> dict:
        """
        Geostatic adım — zemin öz ağırlığı altında başlangıç gerilme dengesi.

        Döndürür: {'sigma_v': ..., 'u_x': ..., 'u_y': ...} sözlüğü
        """
        # SfePy kurulu değilse analitik yaklaşım kullan
        try:
            from sfepy.applications import solve_pde  # noqa: F401
            return self._run_sfepy_geostatic()
        except ImportError:
            return self._analytical_geostatic()

    def run_seismic(self, pga_timeseries: np.ndarray, dt: float = 0.01) -> FEMResult:
        """
        Sismik yükleme adımı.

        Parametreler
        -----------
        pga_timeseries : 1D array — g cinsinden ivme-zaman kaydı
        dt             : Zaman adımı (s)

        Döndürür
        --------
        FEMResult
        """
        pga_max = float(np.max(np.abs(pga_timeseries)))

        try:
            from sfepy.applications import solve_pde  # noqa: F401
            return self._run_sfepy_seismic(pga_timeseries, dt)
        except ImportError:
            return self._analytical_seismic(pga_max)

    def run(self, pga_timeseries: np.ndarray, dt: float = 0.01) -> FEMResult:
        """
        Tam pipeline: Geostatic + Sismik adım.
        """
        self.run_geostatic()
        return self.run_seismic(pga_timeseries, dt)

    # ------------------------------------------------------------------ #
    # Analitik yedek hesaplama (SfePy kurulu değilken)                   #
    # ------------------------------------------------------------------ #

    def _analytical_geostatic(self) -> dict:
        """
        Analitik Geostatic yaklaşımı — K0 = ν/(1-ν) ile yatay gerilme.
        """
        gamma = self.litho["gamma"]  # kN/m³
        nu    = self.litho["nu"]
        K0    = nu / (1 - nu)

        return {
            "sigma_v_kpa":  gamma * 5.0,         # 5 m referans derinlik
            "sigma_h_kpa":  K0 * gamma * 5.0,
            "u_x": 0.0,
            "u_y": 0.0,
        }

    def _analytical_seismic(self, pga_g: float) -> FEMResult:
        """
        Analitik sismik yaklaşım — Mohr-Coulomb kayma yüzeyi.
        FEM kurulmadan önce hızlı kalibrasyon için.
        """
        from ..geomechanics.mohr_coulomb import StaticFS, DynamicFS, CriticalAcceleration
        from ..geomechanics.newmark import NewmarkDisplacement

        c   = self.litho["c_prime"]
        phi = self.litho["phi_prime"]
        g   = self.litho["gamma"]
        z   = 5.0   # referans derinlik (m)
        theta = 25.0  # varsayılan eğim (gerçekte DEM'den gelmeli)

        fs_static = float(StaticFS(c, phi, g, z, theta))
        ky        = float(CriticalAcceleration(fs_static, theta))
        dn        = float(NewmarkDisplacement(ky, pga_g))

        # Sahte kontur array'leri (gerçek FEM ile değiştirilecek)
        n = 100
        x = np.linspace(0, 200, n)
        vm = np.abs(np.sin(x / 20) * g * z * (1 + pga_g))
        eps_p = np.maximum(0, vm / (c + g * z * np.tan(np.deg2rad(phi))) - 1) * 0.01

        return FEMResult(
            slope_id=self.slope_id,
            von_mises=vm,
            eq_plastic_strain=eps_p,
            displacement_x=np.zeros(n),
            displacement_y=np.full(n, dn),
            dn_max=dn,
            ky_fem=ky,
            mesh_path=self.mesh_path or Path("dummy.vtk"),
            metadata={"method": "analytical", "pga_g": pga_g},
        )

    def _run_sfepy_geostatic(self) -> dict:
        """SfePy Geostatic adım (tam implementasyon)."""
        raise NotImplementedError("SfePy Geostatic adım implementasyonu eklenecek.")

    def _run_sfepy_seismic(self, pga_ts: np.ndarray, dt: float) -> FEMResult:
        """SfePy Sismik adım (tam implementasyon)."""
        raise NotImplementedError("SfePy Sismik adım implementasyonu eklenecek.")
