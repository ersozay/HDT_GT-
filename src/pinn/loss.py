"""
Physics-Informed Custom Loss Fonksiyonu.

Total Loss = Focal Loss + λ · Physics Penalty

Bileşenler:
  1. Focal Loss  — sınıf dengesizliğini giderir (heyelan azınlık sınıfı)
  2. Physics Penalty — limit denge fiziğine aykırı tahminleri cezalandırır

Fizik ceza terimi kuralı:
  Eğim < 10° VE PGA < 0.1g → heyelan fiziksel olarak neredeyse imkânsız
  Model yine de '1' tahmin ederse büyük ceza

Referans: Raissi et al. (2019) PINN · Lin et al. Focal Loss (2017).
"""
import torch
import torch.nn as nn


class PhysicsInformedLoss(nn.Module):
    """
    PINN kayıp fonksiyonu: Focal Loss + λ · Physics Penalty.

    Parametreler
    -----------
    lambda_phys : Fizik ceza ağırlığı (varsayılan 1.0)
    alpha       : Focal Loss heyelan sınıfı ağırlığı (varsayılan 0.25)
    gamma       : Focal Loss odak parametresi (varsayılan 2.0)
    slope_threshold_deg: Fizik ceza eğim eşiği (varsayılan 10.0°)
    pga_threshold_g    : Fizik ceza PGA eşiği (varsayılan 0.1g)
    epsilon     : Fizik ceza ε (tahmin toleransı, varsayılan 0.05)
    """

    def __init__(
        self,
        lambda_phys: float = 1.0,
        alpha: float = 0.25,
        gamma: float = 2.0,
        slope_threshold_deg: float = 10.0,
        pga_threshold_g: float = 0.10,
        epsilon: float = 0.05,
    ):
        super().__init__()
        self.lambda_phys         = lambda_phys
        self.alpha               = alpha
        self.gamma               = gamma
        self.slope_threshold_deg = slope_threshold_deg
        self.pga_threshold_g     = pga_threshold_g
        self.epsilon             = epsilon

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
        p_hat = p_hat.squeeze()
        y     = y.float().squeeze()

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
        Model bu koşullarda yüksek olasılık tahmin ederse penaltı uygulanır.

        penalty = max(P̂ – ε, 0)  for physically stable zones
        """
        p_hat     = p_hat.squeeze()
        slope_deg = slope_deg.squeeze()
        pga_g     = pga_g.squeeze()

        physically_stable = (slope_deg < self.slope_threshold_deg) & \
                            (pga_g < self.pga_threshold_g)

        penalty = torch.where(
            physically_stable,
            torch.clamp(p_hat - self.epsilon, min=0.0),
            torch.zeros_like(p_hat),
        )
        return penalty.mean()

    def forward(
        self,
        p_hat: torch.Tensor,
        y: torch.Tensor,
        slope_deg: torch.Tensor,
        pga_g: torch.Tensor,
    ) -> tuple[torch.Tensor, dict]:
        """
        Total loss hesabı.

        Parametreler
        -----------
        p_hat     : (batch, 1) — model çıkışı P(Heyelan) [0,1]
        y         : (batch,) veya (batch,1) — binary hedef
        slope_deg : (batch,) — eğim değerleri (derece)
        pga_g     : (batch,) — PGA değerleri (g)

        Döndürür
        --------
        total_loss : scalar tensor
        components : {'focal': ..., 'physics': ..., 'total': ...}
        """
        fl   = self.focal_loss(p_hat, y)
        phys = self.physics_penalty(p_hat, slope_deg, pga_g)

        total = fl + self.lambda_phys * phys

        return total, {
            "focal":   fl.item(),
            "physics": phys.item(),
            "total":   total.item(),
        }

    def anneal_lambda(self, epoch: int, max_epochs: int, max_lambda: float = 5.0):
        """
        Adaptive λ annealing: eğitim ilerledikçe fizik ceza ağırlığını artır.

        λ(t) = λ_0 + (λ_max - λ_0) · (epoch / max_epochs)

        Parametreler
        -----------
        epoch      : Mevcut epoch numarası
        max_epochs : Toplam epoch sayısı
        max_lambda : Maksimum λ değeri
        """
        progress = min(epoch / max(max_epochs, 1), 1.0)
        self.lambda_phys = self.lambda_phys + (max_lambda - self.lambda_phys) * progress
