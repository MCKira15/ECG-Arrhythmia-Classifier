import os
from functools import lru_cache
from typing import Dict, Iterable, List, Optional

import numpy as np
import pandas as pd
from tensorflow import keras

from app.schemas import SIGNAL_LENGTH


BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(BASE_DIR, "model", "modelo_ecg_cnn1d_final.keras")
POINT_COLUMNS = [f"punto_{index}" for index in range(SIGNAL_LENGTH)]

CLASS_NAMES: Dict[int, str] = {
    0: "Normal",
    1: "Supraventricular",
    2: "Ventricular",
    3: "Fusion",
    4: "Unknown",
}


@lru_cache(maxsize=1)
def get_model():
    return keras.models.load_model(MODEL_PATH)


def class_name(class_id: Optional[int]) -> Optional[str]:
    if class_id is None:
        return None
    return CLASS_NAMES.get(int(class_id), "Desconocida")


def predict_signals(signals: Iterable[List[float]]):
    x_values = np.array(list(signals), dtype="float32")
    if x_values.ndim != 2 or x_values.shape[1] != SIGNAL_LENGTH:
        raise ValueError(f"Cada muestra debe tener exactamente {SIGNAL_LENGTH} valores.")

    x_values = x_values.reshape(x_values.shape[0], SIGNAL_LENGTH, 1)
    probabilities = get_model().predict(x_values, verbose=0)
    predicted_classes = np.argmax(probabilities, axis=1)
    confidences = np.max(probabilities, axis=1)
    return predicted_classes, confidences


def dataframe_from_upload(file_obj) -> pd.DataFrame:
    dataframe = pd.read_csv(file_obj)
    if "clase" in dataframe.columns:
        expected = POINT_COLUMNS + ["clase"]
        if list(dataframe.columns) != expected:
            missing = [column for column in POINT_COLUMNS if column not in dataframe.columns]
            if missing:
                raise ValueError(f"El CSV no contiene todas las columnas punto_0 a punto_186.")
            dataframe = dataframe[expected]
    else:
        if list(dataframe.columns) != POINT_COLUMNS:
            missing = [column for column in POINT_COLUMNS if column not in dataframe.columns]
            if missing:
                raise ValueError(f"El CSV debe tener 187 columnas punto_0 a punto_186.")
            dataframe = dataframe[POINT_COLUMNS]

    dataframe[POINT_COLUMNS] = dataframe[POINT_COLUMNS].astype("float32")
    if "clase" in dataframe.columns:
        dataframe["clase"] = dataframe["clase"].astype("Int64")
    return dataframe
