# Plan de despliegue

Este documento describe la arquitectura de despliegue de la demo Faunalítica, los componentes involucrados y las decisiones técnicas que guían la estrategia de puesta en producción.

---

## 1. Visión general

La demo Faunalítica adopta una arquitectura de microservicios desacoplada, en la que cada componente cumple una responsabilidad específica y se comunica con los demás mediante interfaces bien definidas. Esta arquitectura facilita la mantenibilidad, el reemplazo independiente de componentes y la trazabilidad del ciclo de vida del modelo.

```
  [Imágenes de           [MLflow]
   cámara trampa]    Registro y versión
         │            del modelo
         │                │
         ▼                ▼
   [Streamlit App]  →  [FastAPI]  →  [Modelo PyTorch]
    Interfaz demo      API REST       (cargado desde
                                      MLflow Registry)
         │
         ▼
   [Prometheus]  →  [Grafana]
   Recolección       Dashboards
   de métricas       de monitoreo
```

---

## 2. Componentes del despliegue

### 2.1 MLflow Tracking Server y Model Registry

**Responsabilidad:** Gestionar el ciclo de vida de los experimentos y los modelos.

- Almacena el historial de experimentos de entrenamiento (hiperparámetros, métricas, artefactos).
- Mantiene el registro versionado de modelos y su estado (`Staging`, `Production`, `Archived`).
- La API FastAPI consulta automáticamente el Model Registry para cargar la versión en stage `Production` al iniciarse.

**Configuración local:**

```bash
mlflow server \
  --backend-store-uri sqlite:///mlflow.db \
  --default-artifact-root ./mlartifacts \
  --host 0.0.0.0 \
  --port 5000
```

### 2.2 API de inferencia (FastAPI)

**Responsabilidad:** Exponer el modelo de clasificación como servicio REST.

- Carga el modelo en `Production` desde MLflow al arrancar.
- Expone el endpoint `POST /predict` que recibe una imagen y devuelve la especie predicha, el nivel de confianza y el top de predicciones.
- Expone el endpoint `GET /metrics` para que Prometheus recolecte métricas operativas (latencia, número de peticiones, distribución de confianza).
- Documenta la API automáticamente vía Swagger en `/docs`.

**Inicio:**

```bash
uvicorn api.model_api:app --host 0.0.0.0 --port 8000
```

### 2.3 Interfaz de usuario (Streamlit)

**Responsabilidad:** Proveer una demo interactiva y accesible del clasificador.

- Permite cargar una imagen de fauna (JPG, PNG) desde el navegador.
- Consume la API FastAPI para obtener la predicción.
- Muestra la especie identificada, el porcentaje de confianza y la distribución de probabilidades entre las clases.

**Inicio:**

```bash
streamlit run app/streamlit_app.py
```

### 2.4 Monitoreo (Prometheus + Grafana)

**Responsabilidad:** Observabilidad operativa del sistema en tiempo real.

- **Prometheus** realiza scraping periódico del endpoint `/metrics` de la API para recolectar métricas de rendimiento.
- **Grafana** presenta dashboards configurables sobre latencia de predicción, tasa de peticiones y distribución de las clases predichas.

**Configuración de scraping** (`docker/prometheus.yml`):

```yaml
scrape_configs:
  - job_name: 'faunalitica_api'
    static_configs:
      - targets: ['host.docker.internal:8000']
```

**Inicio con Docker Compose:**

```bash
docker compose up -d
```

---

## 3. Flujo de despliegue de una nueva versión del modelo

Cuando se entrena y evalúa un nuevo modelo que supera al actual en producción, el proceso de actualización es el siguiente:

1. El script `src/model_registry/model_registry.py` compara el nuevo modelo frente al modelo en stage `Production` usando las métricas de evaluación definidas.
2. Si el nuevo modelo supera los umbrales, se registra en MLflow y se promueve al stage `Production`.
3. La API FastAPI, al reiniciarse (o en el próximo ciclo de carga), detecta automáticamente la nueva versión en `Production` y la carga.
4. No se requiere modificar el código de la API ni de la aplicación Streamlit para actualizar el modelo.

---

## 4. CI/CD

El repositorio incluye workflows de GitHub Actions en `.github/workflows/` que automatizan las siguientes tareas:

- Ejecución del conjunto de pruebas (`pytest`) en cada push o pull request.
- Verificación de la calidad del código (linting).

---

## 5. Consideraciones para escalar hacia Edge AI

La arquitectura actual de la demo está diseñada para correr en un servidor convencional. En el marco del proyecto de investigación, el siguiente paso contempla la migración del modelo hacia dispositivos de borde (Raspberry Pi, NVIDIA Jetson Nano), lo que implicará:

- **Cuantización del modelo:** Reducción de precisión (FP32 → INT8) para disminuir el tamaño del modelo y acelerar la inferencia en hardware con recursos limitados.
- **Exportación a formatos optimizados:** TorchScript, ONNX o TensorFlow Lite según el hardware objetivo.
- **Adaptación de la API:** Versión ligera del servicio de inferencia adecuada para ejecución local sin conectividad permanente.
- **Protocolo de comunicación:** Definición del mecanismo de transmisión de resultados desde el dispositivo de campo hacia los actores ambientales y la comunidad (objetivo 3 del proyecto).
