from datetime import datetime
from typing import List, Optional

from pydantic import BaseModel, Field, field_validator


SIGNAL_LENGTH = 187


class ManualPredictionRequest(BaseModel):
    signal: List[float] = Field(..., min_length=SIGNAL_LENGTH, max_length=SIGNAL_LENGTH)
    true_class: Optional[int] = Field(default=None, ge=0, le=4)

    @field_validator("signal")
    @classmethod
    def validate_signal_length(cls, value: List[float]) -> List[float]:
        if len(value) != SIGNAL_LENGTH:
            raise ValueError(f"La muestra debe tener exactamente {SIGNAL_LENGTH} valores.")
        return value


class PredictionUpdateRequest(BaseModel):
    signal: Optional[List[float]] = Field(default=None, min_length=SIGNAL_LENGTH, max_length=SIGNAL_LENGTH)
    true_class: Optional[int] = Field(default=None, ge=0, le=4)
    source: Optional[str] = Field(default=None, max_length=50)
    filename: Optional[str] = Field(default=None, max_length=255)
    row_number: Optional[int] = Field(default=None, ge=1)

    @field_validator("signal")
    @classmethod
    def validate_signal_length(cls, value: Optional[List[float]]) -> Optional[List[float]]:
        if value is not None and len(value) != SIGNAL_LENGTH:
            raise ValueError(f"La muestra debe tener exactamente {SIGNAL_LENGTH} valores.")
        return value


class PredictionResponse(BaseModel):
    id: int
    source: str
    filename: Optional[str]
    row_number: Optional[int]
    signal: List[float]
    true_class: Optional[int]
    true_class_name: Optional[str]
    predicted_class: int
    predicted_class_name: str
    confidence: float
    is_correct: Optional[bool]
    created_at: datetime

    class Config:
        from_attributes = True


class BatchPredictionResponse(BaseModel):
    total_rows: int
    saved_rows: int
    predictions: List[PredictionResponse]


class DeleteResponse(BaseModel):
    message: str
    deleted_id: int


class PredictionStatsResponse(BaseModel):
    total_predictions: int
    demo_seed_rows: int
    manual_rows: int
    file_upload_rows: int
    other_rows: int
