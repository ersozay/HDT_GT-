"""AI çıkarım servisi katmanı — eğitilmiş PINN modelini saf Python API olarak sunar."""
from .inference import LandslideModel, get_model

__all__ = ["LandslideModel", "get_model"]
