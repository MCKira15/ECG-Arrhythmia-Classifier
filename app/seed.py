import os

import pandas as pd
from sqlalchemy.orm import Session

from app.ml import POINT_COLUMNS, class_name, predict_signals
from app.models import EcgPrediction


BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEMO_WITH_CLASS_PATH = os.path.join(BASE_DIR, "test", "demo_ecg_con_clase.csv")


def seed_demo_sample(db: Session) -> None:
    existing = db.query(EcgPrediction).first()
    if existing:
        return

    dataframe = pd.read_csv(DEMO_WITH_CLASS_PATH)
    if dataframe.empty:
        return

    first_row = dataframe.iloc[0]
    signal = [float(first_row[column]) for column in POINT_COLUMNS]
    true_class = int(first_row["clase"]) if "clase" in dataframe.columns else None
    predicted_classes, confidences = predict_signals([signal])
    predicted_class = int(predicted_classes[0])

    prediction = EcgPrediction(
        source="demo_seed",
        filename=os.path.basename(DEMO_WITH_CLASS_PATH),
        row_number=1,
        signal=signal,
        true_class=true_class,
        true_class_name=class_name(true_class),
        predicted_class=predicted_class,
        predicted_class_name=class_name(predicted_class),
        confidence=round(float(confidences[0]), 4),
        is_correct=(true_class == predicted_class) if true_class is not None else None,
    )
    db.add(prediction)
    db.commit()
