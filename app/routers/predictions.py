from typing import List

import pandas as pd
from fastapi import APIRouter, Depends, File, HTTPException, UploadFile
from sqlalchemy.orm import Session

from app.database import get_db
from app.ml import POINT_COLUMNS, class_name, dataframe_from_upload, predict_signals
from app.models import EcgPrediction
from app.schemas import (
    BatchPredictionResponse,
    DeleteResponse,
    ManualPredictionRequest,
    PredictionResponse,
    PredictionStatsResponse,
    PredictionUpdateRequest,
)


router = APIRouter(prefix="/predictions", tags=["Predictions"])


def get_prediction_or_404(db: Session, prediction_id: int) -> EcgPrediction:
    prediction = db.query(EcgPrediction).filter(EcgPrediction.id == prediction_id).first()
    if prediction is None:
        raise HTTPException(status_code=404, detail="Predicción no encontrada.")
    return prediction


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


@router.get("/stats/", response_model=PredictionStatsResponse)
def get_prediction_stats(db: Session = Depends(get_db)):
    total_predictions = db.query(EcgPrediction).count()
    demo_seed_rows = db.query(EcgPrediction).filter(EcgPrediction.source == "demo_seed").count()
    manual_rows = db.query(EcgPrediction).filter(EcgPrediction.source == "manual").count()
    file_upload_rows = db.query(EcgPrediction).filter(EcgPrediction.source == "file_upload").count()

    return PredictionStatsResponse(
        total_predictions=total_predictions,
        demo_seed_rows=demo_seed_rows,
        manual_rows=manual_rows,
        file_upload_rows=file_upload_rows,
        other_rows=total_predictions - demo_seed_rows - manual_rows - file_upload_rows,
    )


@router.get("/{prediction_id}", response_model=PredictionResponse)
def get_prediction(prediction_id: int, db: Session = Depends(get_db)):
    prediction = db.query(EcgPrediction).filter(EcgPrediction.id == prediction_id).first()
    if prediction is None:
        raise HTTPException(status_code=404, detail="Predicción no encontrada.")
    return prediction
@router.put("/{prediction_id}", response_model=PredictionResponse)
def replace_prediction(
    prediction_id: int,
    payload: ManualPredictionRequest,
    db: Session = Depends(get_db),
):
    prediction = get_prediction_or_404(db, prediction_id)
    predicted_classes, confidences = predict_signals([payload.signal])
    predicted_class = int(predicted_classes[0])

    prediction.signal = [float(value) for value in payload.signal]
    prediction.true_class = payload.true_class
    prediction.true_class_name = class_name(payload.true_class)
    prediction.predicted_class = predicted_class
    prediction.predicted_class_name = class_name(predicted_class)
    prediction.confidence = round(float(confidences[0]), 4)
    prediction.is_correct = (
        payload.true_class == predicted_class if payload.true_class is not None else None
    )
    prediction.source = "manual_edit"
    prediction.filename = None
    prediction.row_number = None

    db.commit()
    db.refresh(prediction)
    return prediction


@router.patch("/{prediction_id}", response_model=PredictionResponse)
def update_prediction(
    prediction_id: int,
    payload: PredictionUpdateRequest,
    db: Session = Depends(get_db),
):
    prediction = get_prediction_or_404(db, prediction_id)
    update_data = payload.model_dump(exclude_unset=True)

    if "signal" in update_data:
        signal = [float(value) for value in update_data["signal"]]
        predicted_classes, confidences = predict_signals([signal])
        predicted_class = int(predicted_classes[0])
        prediction.signal = signal
        prediction.predicted_class = predicted_class
        prediction.predicted_class_name = class_name(predicted_class)
        prediction.confidence = round(float(confidences[0]), 4)

    if "true_class" in update_data:
        prediction.true_class = update_data["true_class"]
        prediction.true_class_name = class_name(update_data["true_class"])

    if "source" in update_data and update_data["source"]:
        prediction.source = update_data["source"]

    if "filename" in update_data:
        prediction.filename = update_data["filename"]

    if "row_number" in update_data:
        prediction.row_number = update_data["row_number"]

    prediction.is_correct = (
        prediction.true_class == prediction.predicted_class
        if prediction.true_class is not None
        else None
    )

    db.commit()
    db.refresh(prediction)
    return prediction


@router.delete("/{prediction_id}", response_model=DeleteResponse)
def delete_prediction(prediction_id: int, db: Session = Depends(get_db)):
    prediction = get_prediction_or_404(db, prediction_id)
    db.delete(prediction)
    db.commit()
    return DeleteResponse(message="Predicción eliminada correctamente.", deleted_id=prediction_id)
