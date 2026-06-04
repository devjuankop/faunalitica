# Metodología CRISP-DM

Este documento describe cómo se aplicó la metodología **CRISP-DM** (*Cross-Industry Standard Process for Data Mining*) en el desarrollo del clasificador de fauna del proyecto Faunalítica.

---

## Visión general

CRISP-DM es un proceso iterativo compuesto por seis fases. En este proyecto, cada fase tiene un correlato directo con los objetivos específicos definidos en el anteproyecto de investigación sobre el Humedal Siracusa.

```
  Comprensión      Comprensión     Preparación
  del negocio  →   de los datos  →  de los datos
       ↑                                 ↓
   Despliegue   ←  Evaluación  ←   Modelado
```

---

## Fase 1: Comprensión del negocio

**Objetivo del proyecto:** Desarrollar un sistema de monitoreo automatizado de fauna basado en Edge AI que reduzca los costos operativos y la dependencia de personal especializado para el seguimiento ecológico del Humedal Siracusa (Sevilla, Valle del Cauca).

**Problema central:** Los métodos tradicionales de monitoreo requieren recorridos de campo con personal especializado, lo que limita la frecuencia y cobertura del monitoreo y eleva los costos operativos.

**Solución propuesta:** Un clasificador de imágenes capaz de identificar automáticamente las especies de fauna más representativas del humedal a partir de fotografías de cámaras trampa, como componente de demostración de un sistema mayor orientado a Edge AI.

**Criterio de éxito:** Un modelo con suficiente precisión para distinguir las especies objetivo de manera confiable, integrado en una arquitectura desplegable (FastAPI + Streamlit) con trazabilidad de experimentos vía MLflow.

---

## Fase 2: Comprensión de los datos

Las imágenes utilizadas provienen de **cámaras trampa** instaladas en o cerca del Humedal Siracusa, complementadas con datos de fuentes abiertas de repositorios de biodiversidad.

**Actividades realizadas:**

- Inventario de especies de fauna más relevantes del humedal (principalmente aves y fauna asociada a humedales urbanos).
- Revisión de la calidad y condiciones de las imágenes: variaciones de iluminación, oclusión parcial, fondo natural no controlado.
- Estimación de la distribución de clases y detección de posible desbalance entre especies.
- Análisis de resolución, formato y metadatos de las imágenes disponibles.

**Hallazgos clave:**

- Las imágenes presentan condiciones ambientales variables (iluminación diurna/nocturna, vegetación variable), típicas de entornos de campo.
- Algunas clases de especies cuentan con menos muestras, lo que requiere estrategias de aumentación de datos y/o transfer learning.

Para detalles sobre la estructura del dataset, consulta [`data_definition.md`](../data_report/data_definition.md).

---

## Fase 3: Preparación de los datos

Implementada en `src/data_engineering/data_engineering.py`.

**Transformaciones aplicadas:**

- Redimensionado de imágenes a una resolución uniforme compatible con la arquitectura del modelo.
- Normalización de valores de píxeles según estadísticas del dataset de preentrenamiento (ImageNet).
- Aumentación de datos (rotación, volteo horizontal, variación de brillo y contraste) para mejorar la robustez del modelo y compensar el desbalance de clases.
- Particionado estratificado en conjuntos de entrenamiento, validación y prueba.

**Estructura de salida:**

```
data/
├── raw/            # Imágenes originales sin modificar
├── processed/      # Imágenes transformadas listas para entrenamiento
└── splits/
    ├── train/
    ├── val/
    └── test/
```

---

## Fase 4: Modelado

Implementada en `src/model_engineering/model_engineering.py`.

**Arquitectura:** Se evaluaron modelos de redes neuronales convolucionales (CNN) preentrenados en ImageNet, aplicando **transfer learning** con fine-tuning para adaptarlos al conjunto de especies del Humedal Siracusa. Esta estrategia es especialmente adecuada cuando el volumen de datos anotados disponibles es limitado, como es el caso en ecosistemas específicos.

**Estrategia de entrenamiento:**

- Se congela la parte convolucional base del modelo preentrenado en una primera etapa.
- Se entrena únicamente el clasificador final adaptado al número de clases del proyecto.
- Opcionalmente, se realiza un fine-tuning de las últimas capas convolucionales.

**Rastreo de experimentos con MLflow:**

Cada experimento queda registrado en MLflow con:

- Hiperparámetros: tasa de aprendizaje, tamaño de batch, número de épocas, arquitectura seleccionada.
- Métricas: accuracy, loss de entrenamiento y validación, F1-score por clase, AUC-ROC.
- Artefactos: pesos del modelo, curvas de aprendizaje, matriz de confusión.

```bash
# Visualizar experimentos en la UI de MLflow
mlflow ui --port 5000
```

---

## Fase 5: Evaluación

Implementada en `src/model_registry/model_registry.py`.

**Criterios de evaluación:**

El modelo es promovido al stage `Production` en el MLflow Model Registry únicamente si supera los umbrales de calidad definidos:

- **Accuracy** en el conjunto de prueba: umbral mínimo definido en la configuración.
- **F1-score macro** (relevante ante desbalance de clases): criterio secundario de selección.
- Comparación automática frente al modelo actualmente en producción.

**Validación adicional:**

- Revisión visual de la matriz de confusión para identificar clases problemáticas.
- Análisis de las predicciones incorrectas más frecuentes.
- Verificación del comportamiento del modelo en imágenes de condiciones adversas (baja iluminación, oclusión parcial).

Los reportes de evaluación se exportan en `reports/metrics/` y las figuras en `reports/figures/`.

---

## Fase 6: Despliegue

El despliegue de la demo comprende tres capas:

| Capa | Tecnología | Responsabilidad |
|---|---|---|
| Inferencia | FastAPI + Uvicorn | Expone el modelo como endpoint REST |
| Interfaz | Streamlit | Demo interactiva para carga y clasificación de imágenes |
| Observabilidad | Prometheus + Grafana | Monitoreo de métricas de la API en tiempo real |

El proceso de despliegue completo está documentado en [`deployment_plan.md`](../deployment/deployment_plan.md).

**Naturaleza iterativa:** Los resultados de la evaluación en producción y el uso de la demo retroalimentan las fases anteriores. Nuevas imágenes de campo, incorporación de nuevas especies o cambios en las condiciones ambientales del humedal pueden desencadenar un nuevo ciclo de preparación de datos, re-entrenamiento y re-evaluación.
