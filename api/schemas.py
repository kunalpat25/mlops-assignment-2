"""
Pydantic schemas for API request/response models.
"""

from pydantic import BaseModel
from typing import Dict


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    version: str


class PredictionResponse(BaseModel):
    label: str
    confidence: float
    probabilities: Dict[str, float]


class ErrorResponse(BaseModel):
    error: str
    detail: str
