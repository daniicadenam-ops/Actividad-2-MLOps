
"""Lógica de negocio del modelo Credit Default: variables,
pipeline y métricas.
"""

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import (
    GradientBoostingClassifier,
    RandomForestClassifier,
)
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


# Variables del modelo.
# ID es identificador y no se utiliza como predictor.
NUMERICAS = [
    "LIMIT_BAL",
    "AGE",
    "PAY_0",
    "PAY_2",
    "PAY_3",
    "PAY_4",
    "PAY_5",
    "PAY_6",
    "BILL_AMT1",
    "BILL_AMT2",
    "BILL_AMT3",
    "BILL_AMT4",
    "BILL_AMT5",
    "BILL_AMT6",
    "PAY_AMT1",
    "PAY_AMT2",
    "PAY_AMT3",
    "PAY_AMT4",
    "PAY_AMT5",
    "PAY_AMT6"
]

CATEGORICAS = [
    "SEX",
    "EDUCATION",
    "MARRIAGE",
]

VARIABLES = NUMERICAS + CATEGORICAS

OBJETIVO = "default_payment_next_month"


def resolver_features(columnas):
    """
    Convierte `columnas` de config.yaml ('ALL' o lista)
    en (numericas, categoricas).
    """

    if isinstance(columnas, str) and columnas.upper() == "ALL":
        elegidas = VARIABLES
    else:
        desconocidas = [
            v for v in columnas
            if v not in VARIABLES
        ]

        if desconocidas:
            raise ValueError(
                f"variables desconocidas en 'columnas': {desconocidas}"
            )

        elegidas = columnas

    if not elegidas:
        raise ValueError(
            "la selección de features no tiene ninguna variable"
        )

    numericas = [
        v for v in NUMERICAS
        if v in elegidas
    ]

    categoricas = [
        v for v in CATEGORICAS
        if v in elegidas
    ]

    return numericas, categoricas


def construir_modelo(
    tipo,
    parametros,
    numericas,
    categoricas
):
    """
    Preprocesado más el estimador elegido en configuración.

    Los modelos utilizan exclusivamente las variables seleccionadas.
    """

    if tipo == "logistic_regression":
        estimador = LogisticRegression(**parametros)

    elif tipo == "gradient_boosting":
        estimador = GradientBoostingClassifier(**parametros)

    elif tipo == "random_forest":
        estimador = RandomForestClassifier(**parametros)

    else:
        raise ValueError(
            f"modelo '{tipo}' no soportado"
        )

    transformadores = []

    if numericas:
        transformadores.append(
            ("num", StandardScaler(), numericas)
        )

    if categoricas:
        transformadores.append(
            ("cat", "passthrough", categoricas)
        )

    preprocesado = ColumnTransformer(
        transformadores,
        remainder="drop"
    )

    return Pipeline([
        ("preprocesado", preprocesado),
        ("modelo", estimador),
    ])


def calcular_metricas(y_true, proba, umbral):
    """
    Métricas de clasificación binaria a partir
    de la probabilidad de incumplimiento.
    """

    pred = (proba >= umbral).astype(int)

    return {
        "auc": float(
            roc_auc_score(y_true, proba)
        ),
        "accuracy": float(
            accuracy_score(y_true, pred)
        ),
        "precision": float(
            precision_score(
                y_true,
                pred,
                zero_division=0
            )
        ),
        "recall": float(
            recall_score(
                y_true,
                pred,
                zero_division=0
            )
        ),
        "f1": float(
            f1_score(
                y_true,
                pred,
                zero_division=0
            )
        ),
    }
