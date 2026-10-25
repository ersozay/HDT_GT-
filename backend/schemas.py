"""Pydantic istek/yanıt modelleri."""
from __future__ import annotations

from pydantic import BaseModel, Field


class PointRequest(BaseModel):
    slope: float = Field(..., ge=0, le=89, description="Eğim (derece)")
    elevation: float = Field(1200, ge=0, le=6000, description="Yükseklik (m)")
    aspect: float = Field(180, ge=0, le=360, description="Bakı (derece)")
    litho_id: int = Field(..., ge=1, le=27, description="Litoloji sınıfı (1-27)")
    pga: float = Field(..., ge=0.001, le=3.0, description="PGA (g)")
    saturation: float = Field(0.0, ge=0.0, le=1.0, description="Su tablası oranı m (yağış senaryosu)")


class PointResponse(BaseModel):
    p: float
    ky: float
    dn_cm: float
    label: int
    threshold: float
    static_marginal: bool


class CriticalRequest(BaseModel):
    slope: float = Field(..., ge=0, le=89)
    elevation: float = Field(1200, ge=0, le=6000)
    aspect: float = Field(180, ge=0, le=360)
    litho_id: int = Field(..., ge=1, le=27)
    pga_min: float = Field(0.05, ge=0.001, le=3.0)
    pga_max: float = Field(1.50, ge=0.05, le=3.0)
    step: float = Field(0.05, ge=0.01, le=0.5)
    saturation: float = Field(0.0, ge=0.0, le=1.0)


class CriticalResponse(BaseModel):
    pga: list[float]
    prob: list[float]
    dn_cm: list[float]
    critical_pga: float | None
    threshold: float


class LithologyItem(BaseModel):
    id: int
    description: str
    c_prime: float
    phi_prime: float
    gamma: float


class LithologyLegendItem(BaseModel):
    id: int
    description: str
    color: str


class InventoryItem(BaseModel):
    id: str
    lat: float
    lon: float


class NewmarkCurvesRequest(BaseModel):
    slope: float = Field(..., ge=0, le=89)
    elevation: float = Field(1000, ge=0, le=6000)
    aspect: float = Field(180, ge=0, le=360)
    litho_id: int = Field(..., ge=1, le=27)
    pga_min: float = Field(0.05, ge=0.001, le=3.0)
    pga_max: float = Field(1.50, ge=0.05, le=3.0)
    step: float = Field(0.05, ge=0.01, le=0.5)
    saturation: float = Field(0.0, ge=0.0, le=1.0)



class NewmarkCurvesResponse(BaseModel):
    ky: float
    fs_static: float
    pga: list[float]
    dn_jibson_cm: list[float]
    dn_ambraseys_cm: list[float]
    critical_pga_jibson: float | None
    critical_pga_ambraseys: float | None
    dn_threshold_cm: float


class LatLonResponse(BaseModel):
    lat: float
    lon: float
    in_bounds: bool
    slope: float
    elevation: float
    aspect: float
    pga: float
    litho_id: int
    litho_name: str
    p: float
    ky: float
    dn_cm: float
    label: int
    threshold: float
    static_marginal: bool
