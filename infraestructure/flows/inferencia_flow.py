
"""
Actividad 2: pipeline de inferencia Credit Default.

Pasos:
cargar modelo champion -> ingesta del lote -> predicción
-> guardar CSV -> registrar en MLflow.

El modelo se busca por el alias `champion`.
Cuando el pipeline de entrenamiento asigna un nuevo champion,
la inferencia utiliza ese modelo automáticamente.

El run de inferencia guarda la versión del modelo utilizada
y el run de entrenamiento para mantener la trazabilidad.

Requisitos:
    PREFECT_API_URL=http://localhost:4200/api
    MLFLOW_TRACKING_URI=http://localhost:5000

Ejecución:
    poetry run python infraestructure/flows/inferencia_flow.py
"""

import os
from datetime import date
from pathlib import Path

import mlflow
import mlflow.sklearn
import yaml

from mlflow.tracking import MlflowClient
from prefect import flow, task

from scripts.load_data import cargar_datos as leer_datos


CONFIG = "configs/config.yaml"


@task
def cargar_champion(cfg):
    """
    Carga el modelo champion desde MLflow.

    Devuelve:
        modelo, version_modelo, run_id_entrenamiento
    """

    print("Buscando modelo champion en MLflow...")

    nombre = cfg["registro"]["nombre_modelo"]

    client = MlflowClient()

    # Obtener versión asociada al alias champion
    version = client.get_model_version_by_alias(
        nombre,
        "champion",
    )

    # Cargar modelo champion
    modelo = mlflow.sklearn.load_model(
        f"models:/{nombre}@champion"
    )

    print(
        f"Modelo cargado: {nombre} | "
        f"Versión={version.version} | "
        f"Run entrenamiento={version.run_id}"
    )

    return modelo, str(version.version), version.run_id


@task
def ingesta(cfg):
    """
    Simula la ingesta de un lote de clientes
    utilizando una muestra del dataset.
    """

    tamano_lote = cfg["inferencia"]["tamano_lote"]

    print(f"Ingestando lote de {tamano_lote} clientes...")

    df = leer_datos(cfg["datos"]["ruta"])

    # Evitar solicitar más filas de las disponibles
    n = min(tamano_lote, len(df))

    lote = df.sample(
        n=n,
        random_state=42,
    ).copy()

    print(f"Lote cargado: {len(lote)} registros")

    return lote


@task
def predecir(modelo, lote, cfg):
    """
    Calcula la probabilidad de incumplimiento de crédito
    y guarda las predicciones en un CSV.
    """

    print("Preparando variables para inferencia...")

    # Utilizar las mismas variables del modelo entrenado
    variables = list(modelo.feature_names_in_)

    X = lote[variables]

    print(
        f"Prediciendo con modelo "
        f"{cfg['registro']['nombre_modelo']}..."
    )

    # Probabilidad de incumplimiento (clase 1)
    proba = modelo.predict_proba(X)[:, 1]

    resultado = lote.copy()

    resultado["probabilidad_default"] = proba

    resultado["prediccion_default"] = (
        proba >= cfg["metricas"]["umbral"]
    ).astype(int)

    # Guardar CSV
    print("Guardando predicciones en CSV...")

    carpeta = Path(
        cfg["inferencia"]["carpeta_salida"]
    )

    carpeta.mkdir(
        parents=True,
        exist_ok=True,
    )

    ruta = carpeta / (
        f"predicciones_credit_default_{date.today().isoformat()}.csv"
    )

    resultado.to_csv(
        ruta,
        index=False,
        encoding="utf-8",
    )

    print(f"Predicciones guardadas en {ruta}")

    return (
        str(ruta),
        len(proba),
        float(proba.mean()),
    )


@task
def registrar_inferencia(
    version,
    run_id_entrenamiento,
    ruta,
    filas,
    proba_media,
    cfg,
):
    """
    Registra el resultado de inferencia en MLflow
    con trazabilidad del modelo utilizado.
    """

    print(
        f"Registrando inferencia para modelo "
        f"versión {version}..."
    )

    with mlflow.start_run(
        run_name=f"inferencia-credit-default-{date.today().isoformat()}"
    ) as run:

        # Tags de trazabilidad
        mlflow.set_tag("fase", "inferencia")
        mlflow.set_tag("modelo_version", version)
        mlflow.set_tag(
            "modelo_run_id",
            run_id_entrenamiento,
        )
        mlflow.set_tag(
            "modelo_nombre",
            cfg["registro"]["nombre_modelo"],
        )
        mlflow.set_tag(
            "dataset",
            "credit-default",
        )

        # Métricas de inferencia
        mlflow.log_metric(
            "filas_predichas",
            filas,
        )

        mlflow.log_metric(
            "probabilidad_default_media",
            proba_media,
        )

        # Guardar archivo de predicciones como artefacto
        mlflow.log_artifact(ruta)

        print(
            f"Inferencia registrada en MLflow. "
            f"Run ID={run.info.run_id}"
        )


@flow(
    name="inferencia-credit-default",
    log_prints=True,
)
def inferencia_flow():

    # 1. Leer configuración
    with open(
        CONFIG,
        encoding="utf-8",
    ) as f:
        cfg = yaml.safe_load(f)

    # 2. Configurar MLflow
    tracking_uri = os.environ.get(
        "MLFLOW_TRACKING_URI",
        cfg["mlflow"]["tracking_uri"],
    )

    mlflow.set_tracking_uri(tracking_uri)

    mlflow.set_experiment(
        cfg["mlflow"]["experimento"]
    )

    print(f"MLflow Tracking URI: {tracking_uri}")

    # 3. Cargar modelo champion
    modelo, version, run_id_entrenamiento = cargar_champion(
        cfg
    )

    # 4. Ingestar lote
    lote = ingesta(cfg)

    # 5. Generar predicciones
    ruta, filas, proba_media = predecir(
        modelo,
        lote,
        cfg,
    )

    # 6. Registrar inferencia en MLflow
    registrar_inferencia(
        version,
        run_id_entrenamiento,
        ruta,
        filas,
        proba_media,
        cfg,
    )

    print("\nPipeline de inferencia finalizado correctamente.")

    return ruta


if __name__ == "__main__":
    inferencia_flow()
