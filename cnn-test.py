import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import tensorflow as tf
from tensorflow import keras
import numpy as np
import pandas as pd

# 1. Rutas base
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

model_path = os.path.join(BASE_DIR, "model", "modelo_ecg_cnn1d_final.keras")
demo_api_path = os.path.join(BASE_DIR, "test", "demo_ecg_api.csv")
demo_real_path = os.path.join(BASE_DIR, "test", "demo_ecg_con_clase.csv")

# 2. Nombres de clases
class_names = {
    0: "Normal",
    1: "Supraventricular",
    2: "Ventricular",
    3: "Fusion",
    4: "Unknown"
}

# 3. Cargar modelo
loaded_model = keras.models.load_model(model_path)
print("Modelo cargado exitosamente.")

# 4. Cargar demos
demo_api = pd.read_csv(demo_api_path)
demo_real = pd.read_csv(demo_real_path)

print("Demo API shape:", demo_api.shape)
print("Demo real shape:", demo_real.shape)

# 5. Validar formato
if demo_api.shape[1] != 187:
    raise ValueError(f"demo_ecg_api.csv debe tener 187 columnas. Tiene {demo_api.shape[1]}.")

if "clase" not in demo_real.columns:
    raise ValueError("demo_ecg_con_clase.csv debe tener una columna llamada 'clase'.")

# 6. Preparar datos para CNN 1D
X_demo = demo_api.values.astype("float32").reshape(demo_api.shape[0], 187, 1)

# 7. Predecir
demo_probs = loaded_model.predict(X_demo)
demo_pred = np.argmax(demo_probs, axis=1)

# 8. Crear resultados
clases_reales = demo_real["clase"].astype(int).values

resultados_demo = pd.DataFrame({
    "latido": range(1, len(demo_pred) + 1),
    "clase_real": clases_reales,
    "nombre_real": [class_names[i] for i in clases_reales],
    "clase_predicha": demo_pred,
    "nombre_predicho": [class_names[i] for i in demo_pred],
    "confianza": np.max(demo_probs, axis=1).round(4),
    "acierto": clases_reales == demo_pred
})

# 9. Mostrar resultados
print("\nResultados demo:")
print(resultados_demo.to_string(index=False))
