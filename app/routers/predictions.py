from typing import List

import pandas as pd
from fastapi import APIRouter, Depends, File, HTTPException, UploadFile
from sqlalchemy.orm import Session

from app.database import get_db
from app.ml import POINT_COLUMNS, class_name, dataframe_from_upload, predict_signals
from app.models import EcgPrediction
from app.schemas import BatchPredictionResponse, ManualPredictionRequest, PredictionResponse


router = APIRouter(prefix="/predictions", tags=["Predictions"])


def save_prediction(
    db: Session,
    *,
    signal: List[float],
    predicted_class: int,
    confidence: float,
    source: str,
    true_class: int | None = None,
    filename: str | None = None,
    row_number: int | None = None,
) -> EcgPrediction:
    prediction = EcgPrediction(
        source=source,
        filename=filename,
        row_number=row_number,
        signal=signal,
        true_class=true_class,
        true_class_name=class_name(true_class),
        predicted_class=predicted_class,
        predicted_class_name=class_name(predicted_class),
        confidence=round(float(confidence), 4),
        is_correct=(true_class == predicted_class) if true_class is not None else None,
    )
    db.add(prediction)
    db.commit()
    db.refresh(prediction)
    return prediction


@router.post("/manual", response_model=PredictionResponse)
def predict_manual(payload: ManualPredictionRequest, db: Session = Depends(get_db)):
    try:
        predicted_classes, confidences = predict_signals([payload.signal])
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc

    return save_prediction(
        db,
        signal=payload.signal,
        true_class=payload.true_class,
        predicted_class=int(predicted_classes[0]),
        confidence=float(confidences[0]),
        source="manual",
    )


@router.post("/file", response_model=BatchPredictionResponse)
def predict_file(file: UploadFile = File(...), db: Session = Depends(get_db)):
    if not file.filename or not file.filename.lower().endswith(".csv"):
        raise HTTPException(status_code=400, detail="Sube un archivo CSV.")

    try:
        dataframe = dataframe_from_upload(file.file)
        signals = dataframe[POINT_COLUMNS].values.astype("float32").tolist()
        predicted_classes, confidences = predict_signals(signals)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"No se pudo procesar el archivo: {exc}") from exc

    saved_predictions: List[EcgPrediction] = []
    for index, signal in enumerate(signals):
        true_class = None
        if "clase" in dataframe.columns and pd.notna(dataframe.iloc[index]["clase"]):
            true_class = int(dataframe.iloc[index]["clase"])

        saved_predictions.append(
            save_prediction(
                db,
                signal=[float(value) for value in signal],
                true_class=true_class,
                predicted_class=int(predicted_classes[index]),
                confidence=float(confidences[index]),
                source="file_upload",
                filename=file.filename,
                row_number=index + 1,
            )
        )

    return BatchPredictionResponse(
        total_rows=len(signals),
        saved_rows=len(saved_predictions),
        predictions=saved_predictions,
    )


@router.get("/", response_model=List[PredictionResponse])
def list_predictions(skip: int = 0, limit: int = 100, db: Session = Depends(get_db)):
    return (
        db.query(EcgPrediction)
        .order_by(EcgPrediction.created_at.desc(), EcgPrediction.id.desc())
        .offset(skip)
        .limit(limit)
        .all()
    )


@router.get("/{prediction_id}", response_model=PredictionResponse)
def get_prediction(prediction_id: int, db: Session = Depends(get_db)):
    prediction = db.query(EcgPrediction).filter(EcgPrediction.id == prediction_id).first()
    if prediction is None:
        raise HTTPException(status_code=404, detail="Predicción no encontrada.")
    return prediction
