# Actividad 2 – MLOps: Predicción de incumplimiento de pago (Credit Default)

## 1. Descripción del proyecto

Este proyecto implementa un flujo MLOps para entrenar, registrar, ejecutar inferencias y exponer un modelo de Machine Learning que estima la probabilidad de incumplimiento de pago de clientes de tarjeta de crédito.

Se utiliza el dataset **Default of Credit Card Clients**, con una variable objetivo binaria:

- `default_payment_next_month = 0`: no presenta incumplimiento el siguiente mes.
- `default_payment_next_month = 1`: presenta incumplimiento el siguiente mes.

El proyecto busca integrar las etapas de análisis de datos, entrenamiento, seguimiento de experimentos, inferencia y despliegue, aplicando herramientas de automatización y gestión del ciclo de vida del modelo.

> **Nota:** La predicción representa una estimación del modelo y no debe interpretarse como una decisión crediticia definitiva.

## 2. Dataset

Archivo esperado:

```text
data/default of credit card clients.csv
```

El CSV se carga usando separador `;`. La variable `ID` se excluye de las variables predictoras y la variable objetivo se separa de `X`.

### Variables principales

- Información del cliente: `LIMIT_BAL`, `SEX`, `EDUCATION`, `MARRIAGE`, `AGE`.
- Historial de pagos: `PAY_0`, `PAY_2`, `PAY_3`, `PAY_4`, `PAY_5`, `PAY_6`.
- Saldos facturados: `BILL_AMT1` a `BILL_AMT6`.
- Pagos realizados: `PAY_AMT1` a `PAY_AMT6`.
- Variable objetivo: `default_payment_next_month`.

## 3. Tecnologías

- **Python**: lenguaje principal.
- **Pandas / NumPy**: preparación y manipulación de datos.
- **Scikit-learn**: transformación de variables, entrenamiento y evaluación.
- **MLflow**: tracking de experimentos, métricas, parámetros y registro de modelos.
- **Prefect**: orquestación de flujos y programación de ejecuciones.
- **FastAPI**: exposición del modelo mediante API REST.
- **Docker Compose**: levantamiento de servicios definidos en `services/docker-compose.yml`.
- **Jupyter / VS Code**: análisis exploratorio y experimentación.

## 4. Estructura del repositorio

```text
Actividad-2-MLOps/
├── configs/
│   └── config.yaml                 # Configuración general del proyecto
├── data/
│   └── default of credit card clients.csv
├── domain/
│   └── modelo.py                   # Construcción/preprocesamiento del modelo
├── infraestructure/
│   ├── api/
│   │   └── main.py                 # API REST con FastAPI
│   └── flows/
│       ├── entrenamiento_flow.py    # Flujo de entrenamiento y registro
│       ├── inferencia_flow.py      # Flujo de inferencia
│       └── desplegar.py             # Despliegues/serving de Prefect
├── notebooks/
│   ├── eda_credit_default.ipynb    # Análisis exploratorio
│   └── train_credit_default.ipynb  # Entrenamiento experimental
├── scripts/
│   └── load_data.py                # Carga de datos reutilizable
├── services/
│   ├── mlflow_store/               # Persistencia local de MLflow
│   ├── CLAUDE.md                   # Notas/instrucciones del proyecto
│   ├── docker-compose.yml           # Servicios de infraestructura
│   ├── README.md                    # Documentación de servicios
│   └── requirements.txt              # Dependencias asociadas a servicios
├── poetry.lock
├── pyproject.toml                    # Dependencias y configuración Poetry
└── README.md
```

## 5. Requisitos previos

- Python compatible con la versión definida en `pyproject.toml`.
- Poetry instalado.
- Docker Desktop (si se levantan los servicios mediante Docker Compose).
- Dataset ubicado en `data/`.
- Acceso a los puertos configurados en `services/docker-compose.yml` (por ejemplo, MLflow y Prefect).

Instalar dependencias del proyecto:

```powershell
poetry install
```

Activar el entorno virtual:

```powershell
poetry shell
```

> Si la versión de Poetry no incluye `poetry shell`, ejecutar los comandos con `poetry run`.

## 6. Configuración

Revisar `configs/config.yaml` antes de ejecutar los flujos. Allí se centralizan, según la configuración existente:

- Ruta del dataset.
- Parámetros de división entrenamiento/prueba.
- Parámetros y candidatos de modelos.
- Umbral de clasificación.
- Configuración de MLflow (tracking URI y nombre del experimento).
- Parámetros de inferencia y registro.

No se deben guardar credenciales ni secretos directamente en el repositorio. Usar variables de entorno para valores sensibles.

## 7. Orden recomendado de ejecución

### Paso 1. Levantar los servicios

Desde la raíz del proyecto:

```powershell
cd services
docker compose up -d
```

Verificar el estado:

```powershell
docker compose ps
```

Para detenerlos:

```powershell
docker compose down
```

Consultar `services/README.md` y `services/docker-compose.yml` para confirmar puertos, servicios y URLs configuradas.

### Paso 2. Análisis exploratorio (EDA)

Abrir en VS Code/Jupyter:

```text
notebooks/eda_credit_default.ipynb
```

Ejecutar las celdas en orden. Este notebook permite revisar:

- Dimensiones y estructura del dataset.
- Calidad de datos y valores faltantes.
- Distribución de la variable objetivo.
- Estadísticas descriptivas y distribución de variables.
- Relaciones entre variables y default.
- Hallazgos para orientar el entrenamiento.

### Paso 3. Entrenamiento experimental en notebook

Abrir:

```text
notebooks/train_credit_default.ipynb
```

Ejecutar las celdas en orden para:

1. Cargar y preparar el dataset.
2. Separar variables predictoras y target.
3. Dividir train y test de forma estratificada.
4. Construir pipelines de preprocesamiento y modelos.
5. Entrenar candidatos.
6. Evaluar métricas.
7. Registrar parámetros, métricas y modelos en MLflow.

Este notebook sirve para experimentación y análisis. El flujo productivo/orquestado se ejecuta en el paso siguiente.

### Paso 4. Entrenamiento orquestado con Prefect

Desde la raíz del proyecto:

```powershell
poetry run python -m infraestructure.flows.entrenamiento_flow
```

Si el archivo no tiene bloque de ejecución directa, ejecutar el flujo desde una sesión Python/Jupyter:

```python
from infraestructure.flows.entrenamiento_flow import entrenamiento_flow

entrenamiento_flow()
```

El flujo realiza la carga de datos, entrenamiento de candidatos, evaluación y registro/promoción del modelo según las reglas definidas en `configs/config.yaml`.

### Paso 5. Ejecutar inferencia por lote

```powershell
poetry run python -m infraestructure.flows.inferencia_flow
```

O desde Python:

```python
from infraestructure.flows.inferencia_flow import inferencia_flow

inferencia_flow()
```

La inferencia carga el modelo registrado con alias `champion`, procesa un lote y genera un archivo de predicciones. Revisar la ruta de salida que imprime el flujo y la configuración de inferencia.

### Paso 6. Desplegar los flows en Prefect

```powershell
poetry run python infraestructure/flows/desplegar.py
```

Este script registra/ejecuta los deployments configurados para entrenamiento e inferencia. Mantener la terminal abierta mientras se ejecuta `serve` y verificar los deployments en la interfaz de Prefect.

### Paso 7. Levantar la API REST

Desde la raíz del proyecto:

```powershell
poetry run uvicorn infraestructure.api.main:app --host 0.0.0.0 --port 8000
```

Abrir documentación interactiva:

- Swagger: http://localhost:8000/docs
- ReDoc: http://localhost:8000/redoc
- Health check: http://localhost:8000/health

La API utiliza el modelo disponible bajo el alias `champion`. Si se promociona una nueva versión, reiniciar la API para que cargue la versión actualizada.

## 8. Orden recomendado resumido

1. Levantar los servicios desde `services/`.
2. Ejecutar EDA: `notebooks/eda_credit_default.ipynb`.
3. Entrenar experimentalmente: `notebooks/train_credit_default.ipynb`.
4. Entrenar y registrar: `infraestructure/flows/entrenamiento_flow.py`.
5. Inferir por lote: `infraestructure/flows/inferencia_flow.py`.
6. Desplegar en Prefect: `infraestructure/flows/desplegar.py`.
7. Levantar la API: `infraestructure.api.main:app`.

## 9. MLflow: seguimiento y registro

MLflow permite consultar:

- Experimentos y ejecuciones.
- Parámetros de entrenamiento.
- Métricas de evaluación.
- Artefactos y modelos.
- Versiones del modelo registrado y alias `champion`.

La URL de MLflow depende de `configs/config.yaml` y/o `services/docker-compose.yml`. Consultar la configuración local para abrir la interfaz.

El modelo debe registrarse y, cuando cumpla el criterio de selección definido en la configuración, promocionarse al alias `champion`. La API y el flujo de inferencia esperan ese alias.

## 10. Métricas de evaluación

Para un problema de clasificación binaria, se recomienda analizar conjuntamente:

- **ROC-AUC**: capacidad de ordenar positivos por encima de negativos.
- **PR-AUC / Average Precision**: útil cuando la clase de incumplimiento está desbalanceada.
- **Precision**: proporción de predicciones positivas que realmente son positivas.
- **Recall**: proporción de incumplimientos detectados.
- **F1-score**: equilibrio entre precision y recall.
- **Matriz de confusión**: errores de clasificación por clase.

No seleccionar un modelo únicamente por accuracy, ya que puede ser engañosa cuando existe desbalance de clases. El umbral de clasificación debe justificarse según el objetivo del caso de uso.

## 11. API: ejemplo de consumo

La API recibe las variables predictoras de un cliente (sin `ID` ni target). Ejemplo conceptual:

```json
{
  "LIMIT_BAL": 50000,
  "SEX": 2,
  "EDUCATION": 2,
  "MARRIAGE": 1,
  "AGE": 30,
  "PAY_0": 0,
  "PAY_2": 0,
  "PAY_3": 0,
  "PAY_4": 0,
  "PAY_5": 0,
  "PAY_6": 0,
  "BILL_AMT1": 10000,
  "BILL_AMT2": 9000,
  "BILL_AMT3": 8000,
  "BILL_AMT4": 7000,
  "BILL_AMT5": 6000,
  "BILL_AMT6": 5000,
  "PAY_AMT1": 2000,
  "PAY_AMT2": 2000,
  "PAY_AMT3": 2000,
  "PAY_AMT4": 2000,
  "PAY_AMT5": 2000,
  "PAY_AMT6": 2000
}
```

Enviar el objeto mediante `POST /predict` desde Swagger (`/docs`). Los campos deben coincidir con el esquema definido en `infraestructure/api/main.py`.
