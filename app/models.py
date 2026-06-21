from sqlalchemy import Boolean, Column, DateTime, Float, Integer, JSON, String, func

from app.database import Base


class EcgPrediction(Base):
    __tablename__ = "ecg_predictions"

    id = Column(Integer, primary_key=True, index=True)
    source = Column(String(50), nullable=False, default="manual")
    filename = Column(String(255), nullable=True)
    row_number = Column(Integer, nullable=True)
    signal = Column(JSON, nullable=False)
    true_class = Column(Integer, nullable=True)
    true_class_name = Column(String(50), nullable=True)
    predicted_class = Column(Integer, nullable=False)
    predicted_class_name = Column(String(50), nullable=False)
    confidence = Column(Float, nullable=False)
    is_correct = Column(Boolean, nullable=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now(), nullable=False)
