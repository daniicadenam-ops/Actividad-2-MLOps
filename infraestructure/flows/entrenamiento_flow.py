
"""
Actividad 2: pipeline de entrenamiento Credit Default.

Pasos:
cargar datos -> entrenar y registrar candidatos en MLflow
-> promover a champion el mejor si mejora.

El modelo recibe el alias `champion` solo si su AUC supera
registro.umbral_auc y el AUC del champion actual.

La lógica del modelo está en domain/modelo.py.
Este archivo solo orquesta y registra en MLflow.

Requisitos:
    PREFECT_API_URL=http://localhost:4200/api
    MLFLOW_TRACKING_URI=http://localhost:5000

Ejecución:
    poetry run python infraestructure/flows/entrenamiento_flow.py
"""

import os

import mlflow
import mlflow.sklearn
import yaml

from mlflow.exceptions import MlflowException
from mlflow.models import infer_signature
from mlflow.tracking import MlflowClient

from prefect import flow, task
from sklearn.model_selection import train_test_split

from domain.modelo import (
    OBJETIVO,
    calcular_metricas,
    construir_modelo,
    resolver_features,
)

from scripts.load_data import cargar_datos as leer_datos


CONFIG = "configs/config.yaml"


@task
def cargar_datos(ruta):
    """Carga el CSV y valida sus columnas."""
    print(f"Cargando datos desde {ruta}...")
    return leer_datos(ruta)


@task
def entrenar_y_registrar(df, cfg, features):
    """
    Entrena el modelo, calcula métricas y registra el run
    en MLflow.

    Devuelve:
        run_id, model_uri, métricas
    """

    tipo = cfg["modelo"]["tipo"]
    parametros = cfg["modelo"].get("parametros") or {}

    print(
        f"Entrenando {tipo} "
        f"con features={features['nombre']}..."
    )

    # 1. Resolver variables
    numericas, categoricas = resolver_features(
        features["columnas"]
    )

    variables = numericas + categoricas

    # 2. Separar entrenamiento y prueba
    division = cfg["division"]

    train_df, test_df = train_test_split(
        df,
        test_size=division["test_size"],
        random_state=division["random_state"],
        stratify=df[OBJETIVO],
    )

    X_train = train_df[variables]
    y_train = train_df[OBJETIVO]

    X_test = test_df[variables]
    y_test = test_df[OBJETIVO]

    print(f"Train: {len(X_train):,} filas")
    print(f"Test: {len(X_test):,} filas")
    print(f"Features: {len(variables)}")
    print(f"Incumplimiento en test: {y_test.mean():.2%}")

    # 3. Construir y entrenar modelo
    modelo_entrenado = construir_modelo(
        tipo,
        parametros,
        numericas,
        categoricas,
    )

    print("Entrenando modelo...")
    modelo_entrenado.fit(X_train, y_train)

    # 4. Calcular métricas
    umbral = cfg["metricas"]["umbral"]

    proba = modelo_entrenado.predict_proba(X_test)[:, 1]

    metricas = calcular_metricas(
        y_test,
        proba,
        umbral,
    )

    print(f"Métricas: {metricas}")

    # 5. Configurar MLflow
    tracking_uri = os.environ.get(
        "MLFLOW_TRACKING_URI",
        cfg["mlflow"]["tracking_uri"],
    )

    mlflow.set_tracking_uri(tracking_uri)

    mlflow.set_experiment(
        cfg["mlflow"]["experimento"]
    )

    # 6. Registrar modelo y resultados
    with mlflow.start_run(
        run_name=cfg["mlflow"]["nombre_run"]
    ) as run:

        mlflow.set_tag("fase", "entrenamiento")
        mlflow.set_tag("dataset", "credit-default")

        mlflow.log_params({
            "modelo": tipo,
            **parametros,
            "semilla": division["random_state"],
            "umbral": umbral,
            "features": features["nombre"],
            "n_features": len(variables),
        })

        mlflow.log_metrics(metricas)

        # Crear signature del modelo
        signature = infer_signature(
            X_test,
            modelo_entrenado.predict(X_test),
        )

        # Registrar modelo en MLflow
        info = mlflow.sklearn.log_model(
            modelo_entrenado,
            name="model",
            signature=signature,
            input_example=X_test.head(3),
            skops_trusted_types=[
                "sklearn.tree._tree.Tree"
            ],
        )

        run_id = run.info.run_id
        model_uri = info.model_uri

        print(
            f"Run registrado: {run_id}\n"
            f"URI modelo: {model_uri}"
        )

    return run_id, model_uri, metricas


@task
def promover_si_mejora(model_uri, metricas, cfg):
    """
    Promueve el modelo a champion si supera el AUC mínimo
    y el AUC del champion actual.
    """

    tracking_uri = os.environ.get(
        "MLFLOW_TRACKING_URI",
        cfg["mlflow"]["tracking_uri"],
    )

    mlflow.set_tracking_uri(tracking_uri)

    client = MlflowClient()

    nombre = cfg["registro"]["nombre_modelo"]
    auc = metricas["auc"]
    auc_minimo = cfg["registro"]["umbral_auc"]

    print(
        f"AUC candidato={auc:.4f} | "
        f"AUC mínimo={auc_minimo:.4f}"
    )

    # Validar AUC mínimo
    if auc < auc_minimo:
        print("No cumple el AUC mínimo. No se promueve.")
        return None

    # Consultar champion actual
    try:
        champion = client.get_model_version_by_alias(
            nombre,
            "champion",
        )
    except MlflowException:
        champion = None

    # Comparar contra champion
    if champion is not None:

        auc_champion = client.get_run(
            champion.run_id
        ).data.metrics["auc"]

        print(f"AUC champion actual: {auc_champion:.4f}")

        if auc <= auc_champion:
            print(
                "El candidato no supera al champion. "
                "Se mantiene el modelo actual."
            )
            return None

    # Registrar nuevo modelo
    version = mlflow.register_model(
        model_uri,
        nombre,
    )

    client.set_registered_model_alias(
        nombre,
        "champion",
        version.version,
    )

    client.set_model_version_tag(
        nombre,
        version.version,
        "estado",
        "aprobado",
    )

    print(
        f"Modelo promovido a champion. "
        f"Versión={version.version}, AUC={auc:.4f}"
    )

    return version.version


@flow(name="entrenamiento-credit-default", log_prints=True)
def entrenamiento_flow():

    # 1. Leer configuración
    with open(CONFIG, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    # 2. Cargar dataset
    df = cargar_datos(cfg["datos"]["ruta"])

    resultados = []

    candidatos = cfg["candidatos"]
    conjuntos_features = cfg["features"]

    print(
        f"Candidatos: {[c['nombre_run'] for c in candidatos]}"
    )

    print(
        f"Features: {[f['nombre'] for f in conjuntos_features]}"
    )

    # 3. Entrenar cada combinación
    for candidato in candidatos:

        for features in conjuntos_features:

            cfg_run = {
                **cfg,
                "modelo": {
                    "tipo": candidato["tipo"],
                    "parametros": candidato["parametros"],
                },
                "mlflow": {
                    **cfg["mlflow"],
                    "nombre_run": (
                        f"{candidato['nombre_run']}-"
                        f"{features['nombre']}"
                    ),
                },
            }

            run_id, model_uri, metricas = entrenar_y_registrar(
                df,
                cfg_run,
                features,
            )

            print(
                f"{candidato['nombre_run']} | "
                f"Features={features['nombre']} | "
                f"AUC={metricas['auc']:.4f}"
            )

            resultados.append({
                "run_id": run_id,
                "model_uri": model_uri,
                "metricas": metricas,
                "candidato": candidato["nombre_run"],
                "features": features["nombre"],
            })

    # 4. Seleccionar el mejor modelo
    mejor = max(
        resultados,
        key=lambda r: r["metricas"]["auc"],
    )

    print("\nMejor modelo:")
    print(f"Candidato: {mejor['candidato']}")
    print(f"Features: {mejor['features']}")
    print(f"AUC: {mejor['metricas']['auc']:.4f}")

    # 5. Promover a champion si cumple las condiciones
    promover_si_mejora(
        mejor["model_uri"],
        mejor["metricas"],
        cfg,
    )

    return mejor["run_id"]


if __name__ == "__main__":
    entrenamiento_flow()
