
"""
Actividad 2: API de inferencia Credit Default.

Sirve el modelo con el alias `champion` de MLflow.

Ejecución desde la raíz de actividad-2:
    poetry run uvicorn infraestructure.api.main:app --port 8000

Documentación interactiva:
    http://localhost:8000/docs

El modelo se carga al arrancar la API.
Si el pipeline de entrenamiento asigna un nuevo champion,
es necesario reiniciar la API.
"""

import os

import mlflow
import mlflow.sklearn
import pandas as pd
import yaml

from fastapi import FastAPI, HTTPException
from mlflow.tracking import MlflowClient
from pydantic import BaseModel

# ---------------------------------------------------------
# 1. Cargar configuración
# ---------------------------------------------------------

with open("configs/config.yaml", encoding="utf-8") as f:
    cfg = yaml.safe_load(f)

NOMBRE_MODELO = cfg["registro"]["nombre_modelo"]

# ---------------------------------------------------------
# 2. Configurar MLflow
# ---------------------------------------------------------

tracking_uri = os.environ.get(
    "MLFLOW_TRACKING_URI",
    cfg["mlflow"]["tracking_uri"],
)

mlflow.set_tracking_uri(tracking_uri)

client = MlflowClient()

# ---------------------------------------------------------
# 3. Cargar modelo champion
# ---------------------------------------------------------

modelo = mlflow.sklearn.load_model(
    f"models:/{NOMBRE_MODELO}@champion"
)

version_info = client.get_model_version_by_alias(
    NOMBRE_MODELO,
    "champion",
)

VERSION = str(version_info.version)

# ---------------------------------------------------------
# 4. Crear API
# ---------------------------------------------------------

app = FastAPI(
    title="API de Credit Default",
    description=(
        "API de predicción de incumplimiento de crédito "
        "utilizando el modelo champion de MLflow."
    ),
    version="1.0.0",
)

# ---------------------------------------------------------
# 5. Esquema de entrada
# ---------------------------------------------------------


class Cliente(BaseModel):
    """
    Datos de un cliente del dataset Credit Default.

    Los nombres corresponden a las columnas del CSV.
    """

    LIMIT_BAL: float

    SEX: int
    EDUCATION: int
    MARRIAGE: int
    AGE: int

    PAY_0: int
    PAY_2: int
    PAY_3: int
    PAY_4: int
    PAY_5: int
    PAY_6: int

    BILL_AMT1: float
    BILL_AMT2: float
    BILL_AMT3: float
    BILL_AMT4: float
    BILL_AMT5: float
    BILL_AMT6: float

    PAY_AMT1: float
    PAY_AMT2: float
    PAY_AMT3: float
    PAY_AMT4: float
    PAY_AMT5: float
    PAY_AMT6: float


# ---------------------------------------------------------
# 6. Health check
# ---------------------------------------------------------


@app.get("/health")
def health():
    """
    Verifica el estado de la API y del modelo.
    """

    return {
        "estado": "ok",
        "modelo": NOMBRE_MODELO,
        "version": VERSION,
        "alias": "champion",
    }


# ---------------------------------------------------------
# 7. Endpoint de predicción
# ---------------------------------------------------------


@app.post("/predict")
def predict(cliente: Cliente):
    """
    Calcula la probabilidad de incumplimiento
    de un cliente.
    """

    try:
        # Convertir entrada a DataFrame
        datos = pd.DataFrame(
            [cliente.model_dump()]
        )

        # Mantener exactamente las variables del modelo
        datos = datos[
            list(modelo.feature_names_in_)
        ]

        # Calcular probabilidad de default
        proba = float(
            modelo.predict_proba(datos)[0, 1]
        )

        # Umbral configurado
        umbral = cfg["metricas"]["umbral"]

        prediccion = int(proba >= umbral)

        return {
            "probabilidad_default": round(proba, 4),
            "prediccion_default": prediccion,
            "umbral": umbral,
            "version_modelo": VERSION,
        }

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error al realizar la predicción: {str(e)}",
        )


# ---------------------------------------------------------
# 8. Endpoint raíz
# ---------------------------------------------------------


@app.get("/")
def root():
    return {
        "mensaje": "API Credit Default funcionando",
        "documentacion": "/docs",
        "health": "/health",
        "predict": "/predict",
    }
