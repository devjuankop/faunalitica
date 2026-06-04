# Guía de instalación

Este documento describe los pasos necesarios para configurar el entorno de desarrollo y ejecutar el proyecto Faunalítica localmente.

---

## Requisitos previos

| Herramienta | Versión recomendada | Notas |
|---|---|---|
| Python | 3.10 o superior | Se recomienda usar un entorno virtual |
| pip | ≥ 23.x | Incluido con Python |
| Git | Cualquier versión reciente | Para clonar el repositorio |
| Docker + Docker Compose | Docker ≥ 24, Compose ≥ 2.x | Requerido para Prometheus y Grafana |

> **Nota:** El proyecto fue desarrollado y probado en sistemas Linux/macOS. En Windows se recomienda usar WSL2.

---

## 1. Clonar el repositorio

```bash
git clone https://github.com/devjuankop/faunalitica.git
cd faunalitica
```

---

## 2. Crear y activar un entorno virtual

```bash
# Crear el entorno virtual
python -m venv .venv

# Activar en Linux/macOS
source .venv/bin/activate

# Activar en Windows (PowerShell)
.venv\Scripts\Activate.ps1
```

---

## 3. Instalar dependencias

Todas las dependencias del proyecto están listadas en `requirements.txt`:

```bash
pip install -r requirements.txt
```

### Dependencias principales

| Paquete | Propósito |
|---|---|
| `mlflow` | Rastreo de experimentos y registro de modelos |
| `pytorch` / `torchvision` | Entrenamiento e inferencia del clasificador de imágenes |
| `fastapi` + `uvicorn` | API REST para servir el modelo |
| `streamlit` | Interfaz de usuario de la demo |
| `scikit-learn` | Métricas de evaluación |
| `pandas` / `numpy` | Manipulación de datos |
| `matplotlib` / `seaborn` | Visualización de resultados |
| `prometheus_client` | Exportación de métricas de rendimiento |
| `python-dotenv` | Gestión de variables de entorno |
| `joblib` | Serialización de artefactos |
| `pytest` + `httpx` | Pruebas unitarias y de integración |
| `requests` | Llamadas HTTP entre componentes |

---

## 4. Variables de entorno

Crea un archivo `.env` en la raíz del proyecto basándote en el siguiente ejemplo:

```dotenv
# URL del servidor de MLflow
MLFLOW_TRACKING_URI=http://localhost:5000

# Nombre del experimento registrado en MLflow
MLFLOW_EXPERIMENT_NAME=faunalitica_classifier

# Puerto de la API FastAPI
API_PORT=8000

# Puerto de la aplicación Streamlit
STREAMLIT_PORT=8501
```

> El archivo `.env` está incluido en `.gitignore` y **nunca debe subirse al repositorio**.

---

## 5. Configuración de MLflow

El servidor de MLflow gestiona el rastreo de experimentos y el registro de modelos. Para iniciarlo localmente:

```bash
mlflow server \
  --backend-store-uri sqlite:///mlflow.db \
  --default-artifact-root ./mlartifacts \
  --host 0.0.0.0 \
  --port 5000
```

La interfaz web de MLflow estará disponible en: [http://localhost:5000](http://localhost:5000)

---

## 6. Configuración de monitoreo (Prometheus + Grafana)

El archivo `docker-compose.yml` en la raíz del proyecto levanta los servicios de monitoreo:

```bash
docker compose up -d
```

| Servicio | URL local | Puerto |
|---|---|---|
| Prometheus | [http://localhost:9090](http://localhost:9090) | 9090 |
| Grafana | [http://localhost:3000](http://localhost:3000) | 3000 |

La configuración de scraping de métricas se encuentra en `docker/prometheus.yml`.

---

## Resumen de servicios y puertos

| Componente | Puerto | Comando de inicio |
|---|---|---|
| MLflow Tracking Server | 5000 | `mlflow server ...` |
| FastAPI (modelo API) | 8000 | `uvicorn api.model_api:app ...` |
| Streamlit (demo UI) | 8501 | `streamlit run app/streamlit_app.py` |
| Prometheus | 9090 | `docker compose up -d` |
| Grafana | 3000 | `docker compose up -d` |

Para instrucciones de ejecución detalladas, consulta [`instructions.md`](instructions.md).
