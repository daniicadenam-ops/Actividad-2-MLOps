
"""
Actividad 2: despliegue de pipelines en Prefect.

- entrenamiento-credit-default:
    Sin horario. Se ejecuta manualmente desde la UI de Prefect.

- inferencia-credit-default:
    Se ejecuta diariamente a las 06:00 UTC.
    Cron: 0 6 * * *

Ejecución desde la raíz de actividad-2:
    poetry run python infraestructure/flows/desplegar.py

Mantener esta terminal abierta mientras Prefect ejecuta
los deployments mediante serve().
"""

from prefect import serve

from entrenamiento_flow import entrenamiento_flow
from inferencia_flow import inferencia_flow


if __name__ == "__main__":

    # Deployment de entrenamiento (manual)
    despliegue_entrenamiento = (
        entrenamiento_flow.to_deployment(
            name="entrenamiento-credit-default",
        )
    )

    # Deployment de inferencia diaria
    despliegue_inferencia = (
        inferencia_flow.to_deployment(
            name="inferencia-credit-default-diaria",
            cron="0 6 * * *",
        )
    )

    print("Iniciando deployments en Prefect...")
    print("Entrenamiento: ejecución manual")
    print("Inferencia: diariamente a las 06:00 UTC")

    serve(
        despliegue_entrenamiento,
        despliegue_inferencia,
    )
