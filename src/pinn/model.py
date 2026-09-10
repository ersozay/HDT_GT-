"""
LandslidePINN — Fizik Tabanlı Yapay Sinir Ağı Modeli.

Mimari:
  Girdi(n) → Dense(256)+BN+Swish+Drop(0.3)
           → Dense(128)+BN+Swish+Drop(0.25)
           → Dense(64)+BN+Swish+Drop(0.2)
           → Dense(32)+Swish
           → Dense(1)+Sigmoid → P(Heyelan)

Swish aktivasyonu: f(x) = x · σ(βx), β öğrenilebilir parametre.
Öğrenilebilir β, tabular jeomekanik veri için sabit ReLU'ya göre
dying gradient sorununu önler.

Referans: Ramachandran et al. 2017, Raissi et al. 2019.
"""
import torch
import torch.nn as nn


class Swish(nn.Module):
    """
    Swish aktivasyon fonksiyonu: f(x) = x · σ(βx).

    β, öğrenilebilir nn.Parameter olarak başlatılır.
    Başlangıç β=1 → standart Swish'e eşdeğer.
    """

    def __init__(self):
        super().__init__()
        self.beta = nn.Parameter(torch.tensor(1.0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * torch.sigmoid(self.beta * x)

    def extra_repr(self) -> str:
        return f"beta={self.beta.item():.4f}"


class LandslidePINN(nn.Module):
    """
    Fizik bilgisi gömülü heyelan olasılığı tahmincisi.

    Parametreler
    -----------
    in_features : Girdi öznitelik sayısı
    dropout_rates: Her katman için dropout oranları (3 değer)
    """

    def __init__(
        self,
        in_features: int,
        dropout_rates: tuple[float, float, float] = (0.30, 0.25, 0.20),
    ):
        super().__init__()
        d1, d2, d3 = dropout_rates

        self.layer1 = nn.Sequential(
            nn.Linear(in_features, 256),
            nn.BatchNorm1d(256),
            Swish(),
            nn.Dropout(d1),
        )
        self.layer2 = nn.Sequential(
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            Swish(),
            nn.Dropout(d2),
        )
        self.layer3 = nn.Sequential(
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            Swish(),
            nn.Dropout(d3),
        )
        self.layer4 = nn.Sequential(
            nn.Linear(64, 32),
            Swish(),
        )
        self.output = nn.Sequential(
            nn.Linear(32, 1),
            nn.Sigmoid(),
        )

        self._init_weights()

    def _init_weights(self):
        """He (Kaiming) uniform başlatma — Swish ile uyumlu."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.kaiming_uniform_(module.weight, nonlinearity="relu")
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        İleri geçiş.

        Parametreler
        -----------
        x : (batch_size, in_features) tensor

        Döndürür
        --------
        p : (batch_size, 1) — P(Heyelan) olasılığı [0, 1]
        """
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        return self.output(x)

    def predict_proba(self, x: torch.Tensor) -> torch.Tensor:
        """Eval modunda olasılık tahmini."""
        self.eval()
        with torch.no_grad():
            return self(x)

    def predict(self, x: torch.Tensor, threshold: float = 0.5) -> torch.Tensor:
        """Binary sınıf tahmini."""
        p = self.predict_proba(x)
        return (p >= threshold).long()

    def count_parameters(self) -> int:
        """Eğitilebilir parametre sayısı."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def __repr__(self) -> str:
        n = self.count_parameters()
        return (
            f"LandslidePINN("
            f"params={n:,}, "
            f"architecture=256→128→64→32→1"
            f")"
        )
