
"""
Carga el dataset Default of Credit Card Clients,
valida sus columnas y muestra su tamaño.

Por defecto lee `datos.ruta` de configs/config.yaml.
Las rutas se resuelven desde el directorio de trabajo.

Ejecución (desde la raíz del proyecto):
    poetry run python scripts/load_data.py

Para usarla desde un flow:
    from scripts.load_data import cargar_datos
"""

from pathlib import Path
import pandas as pd
import yaml


RUTA_CONFIG = "configs/config.yaml"

# Columnas del dataset Default of Credit Card Clients
COLUMNAS = [
    "ID",
    "LIMIT_BAL",
    "SEX",
    "EDUCATION",
    "MARRIAGE",
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
    "PAY_AMT6",
    "default_payment_next_month"
]

TARGET = "default_payment_next_month"

FEATURES = [
    col for col in COLUMNAS
    if col not in ["ID", TARGET]
]


def cargar_datos(ruta=None):
    """
    Devuelve el DataFrame Default of Credit Card Clients.

    Acepta:
    - Ruta directa al CSV.
    - None: utiliza datos.ruta de configs/config.yaml.
    """

    if ruta is None:
        with open(RUTA_CONFIG, encoding="utf-8") as f:
            ruta = yaml.safe_load(f)["datos"]["ruta"]

    ruta = Path(ruta)

    if not ruta.exists():
        raise FileNotFoundError(
            f"No se encontró el dataset: {ruta.resolve()}"
        )

    # Leer CSV separado por punto y coma
    df = pd.read_csv(
        ruta,
        sep=";",
        encoding="utf-8",
        low_memory=False
    )

    # Normalizar nombre de la variable objetivo
    df.columns = [
        col.strip().replace(" ", "_")
        for col in df.columns
    ]

    # Validar estructura
    if df.shape[1] != len(COLUMNAS):
        raise ValueError(
            f"El archivo tiene {df.shape[1]} columnas, "
            f"pero se esperaban {len(COLUMNAS)}."
        )

    # Validar nombres
    if set(df.columns) != set(COLUMNAS):
        raise ValueError(
            "Las columnas del dataset no coinciden con las esperadas.\n"
            f"Esperadas: {COLUMNAS}\n"
            f"Encontradas: {list(df.columns)}"
        )

    # Validar target
    valores_target = set(df[TARGET].dropna().unique())

    if not valores_target.issubset({0, 1}):
        raise ValueError(
            f"El target contiene valores inesperados: {valores_target}"
        )

    return df.copy()


def main():

    ruta = None  # None = datos.ruta de configs/config.yaml

    df = cargar_datos(ruta)

    print("Dataset Default of Credit Card Clients")
    print("-" * 50)
    print(f"Filas: {len(df):,}")
    print(f"Columnas: {df.shape[1]}")
    print(f"Nulos: {int(df.isna().sum().sum()):,}")
    print(f"Clientes únicos: {df['ID'].nunique():,}")
    print(f"Target: {TARGET}")
    print(f"Features: {len(FEATURES)}")
    print(f"Columnas: {list(df.columns)}")

    print("\nDistribución del target:")
    print(df[TARGET].value_counts())
    print("\nPrimeras observaciones:")
    print(df.head())


if __name__ == "__main__":
    main()
