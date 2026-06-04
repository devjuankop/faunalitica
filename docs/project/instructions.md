# Instrucciones de ejecución

Este documento describe cómo ejecutar cada componente del proyecto Faunalítica una vez completada la instalación. Para la configuración del entorno, consulta primero [`installations.md`](installations.md).

---

## Orden de ejecución recomendado

Para una ejecución completa de la demo se recomienda levantar los componentes en el siguiente orden:

1. Servidor de MLflow
2. Pipeline de datos y entrenamiento (si aún no hay modelo registrado)
3. API de inferencia (FastAPI)
4. Interfaz de usuario (Streamlit)
5. Monitoreo (Prometheus + Grafana) — opcional

---

## 1. Iniciar el servidor de MLflow

```bash
mlflow server \
  --backend-store-uri sqlite:///mlflow.db \
  --default-artifact-root ./mlartifacts \
  --host 0.0.0.0 \
  --port 5000
```

Deja esta terminal activa. La UI de MLflow estará en [http://localhost:5000](http://localhost:5000).

---

## 2. Ejecutar el pipeline de datos y entrenamiento

> Omite este paso si ya existe un modelo registrado en MLflow en estado `Production`.

### 2.1 Preprocesamiento y particionado de datos

```bash
python src/data_engineering/data_engineering.py
```

Este script carga las imágenes desde `data/raw/`, aplica las transformaciones necesarias (redimensionado, normalización, aumentación de datos) y genera las particiones `train`, `val` y `test` en `data/splits/`.

### 2.2 Entrenamiento del modelo

```bash
python src/model_engineering/model_engineering.py
```

Entrena el clasificador de imágenes y registra el experimento en MLflow, incluyendo hiperparámetros, métricas (accuracy, loss, F1-score por clase) y el artefacto del modelo.

### 2.3 Registro del modelo en MLflow

```bash
python src/model_registry/model_registry.py
```

Evalúa el modelo entrenado frente al umbral de calidad definido y, si lo supera, lo promueve al stage `Production` en el MLflow Model Registry.

---

## 3. Iniciar la API de inferencia (FastAPI)

```bash
uvicorn api.model_api:app --host 0.0.0.0 --port 8000 --reload
```

La API cargará automáticamente la última versión del modelo en stage `Production` desde MLflow. Una vez activa:

- **Documentación interactiva (Swagger):** [http://localhost:8000/docs](http://localhost:8000/docs)
- **Endpoint de predicción:** `POST http://localhost:8000/predict`

### Ejemplo de llamada a la API

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: multipart/form-data" \
  -F "file=@ruta/a/imagen.jpg"
```

**Respuesta esperada:**

```json
{
  "species": "Nombre de la especie",
  "confidence": 0.95,
  "top_predictions": [
    {"species": "Especie A", "confidence": 0.95},
    {"species": "Especie B", "confidence": 0.04},
    {"species": "Especie C", "confidence": 0.01}
  ]
}
```

---

## 4. Iniciar la interfaz Streamlit

En una nueva terminal (con el entorno virtual activo):

```bash
streamlit run app/streamlit_app.py
```

La demo de Streamlit estará disponible en [http://localhost:8501](http://localhost:8501).

### Uso de la demo

1. Abre la aplicación en el navegador.
2. Carga una imagen de fauna (formatos admitidos: JPG, PNG).
3. La aplicación enviará la imagen a la API FastAPI y mostrará el resultado de clasificación: especie predicha, nivel de confianza y distribución de probabilidades entre las clases.

---

## 5. Iniciar el monitoreo (Prometheus + Grafana)

```bash
docker compose up -d
```

- **Prometheus:** [http://localhost:9090](http://localhost:9090) — recopila métricas expuestas por la API (latencia, número de peticiones, etc.).
- **Grafana:** [http://localhost:3000](http://localhost:3000) — visualización de dashboards. Credenciales por defecto: `admin` / `admin`.

Para detener los servicios de monitoreo:

```bash
docker compose down
```

---

## 6. Ejecutar pruebas

```bash
pytest tests/ -v
```

Los tests cubren la validación del pipeline de datos, la lógica de inferencia y los endpoints de la API.

---

## Solución de problemas frecuentes

| Problema | Causa probable | Solución |
|---|---|---|
| `Connection refused` en la API | La API o MLflow no está activo | Verifica que ambos procesos estén corriendo |
| `No model in Production stage` | No hay modelo promovido en MLflow | Ejecuta el pipeline de entrenamiento y registro |
| Error de importación de `torch` | PyTorch no instalado correctamente | Reinstala con `pip install torch torchvision` |
| Prometheus no recopila métricas | La API no expone el endpoint `/metrics` | Asegúrate de que `prometheus_client` esté inicializado en `model_api.py` |
| Puerto ocupado | Otro proceso usa el mismo puerto | Cambia el puerto en el comando de inicio o en `.env` |
