"""
Physics-Informed Custom Loss Fonksiyonu.

Total Loss = Focal Loss + λ_s · Stabilite Cezası + λ_m · Monotonluk Cezası

Bileşenler:
  1. Focal Loss  — sınıf dengesizliğini giderir (heyelan azınlık sınıfı)
  2. Stabilite cezası — limit denge fiziğine aykırı tahminleri cezalandırır:
       Eğim < 10° VE PGA < 0.1g → heyelan fiziksel olarak neredeyse imkânsız;
       model yine de yüksek P verirse ceza.
  3. Fizik-yolu monotonluk cezası (v2) — Mohr-Coulomb + Newmark zincirinin
     işaret kısıtları, kolokasyon çiftleri (x_a → x_b) üzerinde uygulanır:
       PGA yolu : PGA ↑ (Dn yeniden hesaplanır)          ⇒ P düşmemeli
       Eğim yolu: θ ↑  (FS, Ky, Dn yeniden hesaplanır)   ⇒ P düşmemeli
     L_path = mean( ReLU(P(x_a) − P(x_b)) ). Çiftler `physics_paths.py` ile
     gözlem aralığından geniş bir PGA bandında üretilir (PINN kolokasyon
     noktaları). Bu kısıt Kritik PGA taramasının fiziksel olarak tutarlı
     olmasını sağlar.

Referans: Raissi et al. (2019) PINN · Lin et al. (2017) Focal Loss ·
          Karpatne et al. (2017) Physics-guided neural networks ·
          Sill (1998) / Gupta et al. (2016) monotonic neural networks.
"""
import torch
import torch.nn as nn


class PhysicsInformedLoss(nn.Module):
    """
    PINN kayıp fonksiyonu: Focal + λ_s · Stabilite + λ_m · Monotonluk.

    Parametreler
    -----------
    lambda_phys : Stabilite cezası ağırlığı (başlangıç; annealing ile artar)
    alpha       : Focal Loss heyelan sınıfı ağırlığı
    gamma       : Focal Loss odak parametresi
    slope_threshold_deg: Stabilite cezası eğim eşiği (varsayılan 10°)
    pga_threshold_g    : Stabilite cezası PGA eşiği (varsayılan 0.1g)
    epsilon     : Stabilite cezası tahmin toleransı
    lambda_mono : Fizik-yolu cezası ağırlığı (0 → kapalı, v1 davranışı)
    """

    def __init__(
        self,
        lambda_phys: float = 1.0,
        alpha: float = 0.25,
        gamma: float = 2.0,
        slope_threshold_deg: float = 10.0,
        pga_threshold_g: float = 0.10,
        epsilon: float = 0.05,
        lambda_mono: float = 0.0,
    ):
        super().__init__()
        self.lambda_phys         = lambda_phys
        self.lambda_phys_0       = lambda_phys
        self.alpha               = alpha
        self.gamma               = gamma
        self.slope_threshold_deg = slope_threshold_deg
        self.pga_threshold_g     = pga_threshold_g
        self.epsilon             = epsilon
        self.lambda_mono         = lambda_mono

    @property
    def uses_paths(self) -> bool:
        return self.lambda_mono > 0

    def focal_loss(
        self,
        p_hat: torch.Tensor,
        y: torch.Tensor,
    ) -> torch.Tensor:
        """
        Binary Focal Loss.

        FL = –α_t · (1 – p_t)^γ · log(p_t)

        Heyelan (y=1) sınıfına α ağırlığı, kararlı (y=0) sınıfına (1-α) ağırlığı.
        """
        p_hat = p_hat.reshape(-1)
        y     = y.float().reshape(-1)

        pt      = torch.where(y == 1, p_hat, 1.0 - p_hat)
        alpha_t = torch.where(y == 1,
                              torch.full_like(p_hat, self.alpha),
                              torch.full_like(p_hat, 1.0 - self.alpha))

        log_pt = torch.log(pt + 1e-8)
        loss   = -alpha_t * (1.0 - pt) ** self.gamma * log_pt
        return loss.mean()

    def physics_penalty(
        self,
        p_hat: torch.Tensor,
        slope_deg: torch.Tensor,
        pga_g: torch.Tensor,
    ) -> torch.Tensor:
        """
        Limit denge fiziğine aykırı tahminlere ceza.

        Kural: Eğim < 10° VE PGA < 0.1g → P(heyelan) ≈ 0 olmalı.
        penalty = max(P̂ – ε, 0)  for physically stable zones
        """
        p_hat     = p_hat.reshape(-1)
        slope_deg = slope_deg.reshape(-1)
        pga_g     = pga_g.reshape(-1)

        physically_stable = (slope_deg < self.slope_threshold_deg) & \
                            (pga_g < self.pga_threshold_g)

        penalty = torch.where(
            physically_stable,
            torch.clamp(p_hat - self.epsilon, min=0.0),
            torch.zeros_like(p_hat),
        )
        return penalty.mean()

    @staticmethod
    def path_penalty(p_a: torch.Tensor, p_b: torch.Tensor) -> torch.Tensor:
        """Fizik yolunda x_a → x_b adımında P'nin düşüşüne ceza."""
        return torch.relu(p_a.reshape(-1) - p_b.reshape(-1)).mean()

    def forward(
        self,
        p_hat: torch.Tensor,
        y: torch.Tensor,
        slope_deg: torch.Tensor,
        pga_g: torch.Tensor,
        p_a: torch.Tensor | None = None,
        p_b: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, dict]:
        """
        Total loss hesabı.

        Parametreler
        -----------
        p_hat     : (batch, 1) — model çıkışı P(Heyelan) [0,1]
        y         : (batch,) — binary hedef
        slope_deg : (batch,) — eğim (derece, ham)
        pga_g     : (batch,) — PGA (g, ham)
        p_a, p_b  : kolokasyon çiftlerinde model çıkışları (None → ceza atlanır)

        Döndürür
        --------
        total_loss : scalar tensor
        components : {'focal', 'physics', 'mono', 'total'}
        """
        fl   = self.focal_loss(p_hat, y)
        phys = self.physics_penalty(p_hat, slope_deg, pga_g)
        total = fl + self.lambda_phys * phys

        mono_val = 0.0
        if p_a is not None and self.uses_paths:
            mono = self.path_penalty(p_a, p_b)
            total = total + self.lambda_mono * mono
            mono_val = mono.item()

        return total, {
            "focal":   fl.item(),
            "physics": phys.item(),
            "mono":    mono_val,
            "total":   total.item(),
        }

    def anneal_lambda(self, epoch: int, max_epochs: int, max_lambda: float = 5.0):
        """
        Doğrusal λ annealing: eğitim ilerledikçe stabilite ceza ağırlığını artır.

            λ(t) = λ_0 + (λ_max − λ_0) · (epoch / max_epochs)

        Not: v1'de λ kendi üzerine biriktiriliyordu (λ ← λ + (λ_max−λ)·t) ve
        ilk ~10 epoch'ta λ_max'a doyuyordu; artık λ_0'dan hesaplanır.
        """
        progress = min(epoch / max(max_epochs, 1), 1.0)
        self.lambda_phys = self.lambda_phys_0 + (max_lambda - self.lambda_phys_0) * progress
