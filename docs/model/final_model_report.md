# Reporte del modelo final

Este documento describe la arquitectura, el proceso de entrenamiento, las métricas de evaluación y las decisiones técnicas del clasificador de imágenes de fauna utilizado en la demo Faunalítica.

---

## 1. Contexto y objetivo del modelo

El modelo tiene como propósito identificar automáticamente las especies de fauna presentes en fotografías tomadas por cámaras trampa en el Humedal Siracusa (Sevilla, Valle del Cauca). La clasificación automatizada reduce la carga de trabajo de personal especializado y habilita un monitoreo ecológico más frecuente y sostenible.

**Tipo de tarea:** Clasificación multiclase de imágenes  
**Clases objetivo:** Especies de fauna relevantes del Humedal Siracusa (aves y fauna asociada a humedales urbanos)  
**Framework:** PyTorch + torchvision  
**Rastreo de experimentos:** MLflow

---

## 2. Arquitectura

El clasificador está basado en **transfer learning** a partir de una red neuronal convolucional (CNN) preentrenada en ImageNet. Esta estrategia permite obtener representaciones visuales ricas a partir de un volumen de datos anotados limitado, situación habitual en proyectos de monitoreo de biodiversidad en ecosistemas específicos.

**Etapas del modelo:**

1. **Base convolucional preentrenada:** Extrae características visuales jerárquicas de las imágenes. Las capas de esta sección son inicialmente congeladas durante la primera fase de entrenamiento.
2. **Cabeza de clasificación:** Capas densas añadidas sobre la base convolucional, adaptadas al número de especies objetivo del proyecto. Esta sección se entrena desde el inicio.
3. **Fine-tuning (opcional):** En una segunda fase, se descongelan las últimas capas de la base convolucional para ajustar las representaciones aprendidas a las características visuales específicas de las imágenes de cámaras trampa del humedal.

---

## 3. Preprocesamiento de entrada

Las imágenes son transformadas antes de alimentar el modelo:

| Transformación | Descripción |
|---|---|
| Redimensionado | A la resolución de entrada de la arquitectura base |
| Normalización | Media y desviación estándar de ImageNet |
| Aumentación (entrenamiento) | Rotación aleatoria, volteo horizontal, variación de brillo y contraste |
| Sin aumentación (inferencia) | Solo redimensionado y normalización |

---

## 4. Hiperparámetros de entrenamiento

Los hiperparámetros utilizados en el experimento registrado como modelo final en MLflow son los siguientes (valores de referencia; los definitivos se registran en el experimento de MLflow):

| Hiperparámetro | Valor de referencia |
|---|---|
| Optimizador | Adam |
| Tasa de aprendizaje inicial | 1e-3 (fase 1) / 1e-4 (fine-tuning) |
| Tamaño de batch | 32 |
| Épocas (fase 1) | 15–20 |
| Épocas (fine-tuning) | 10–15 |
| Función de pérdida | CrossEntropyLoss |
| Scheduler | ReduceLROnPlateau |

> Los valores exactos del experimento final se pueden consultar en la UI de MLflow bajo el experimento `faunalitica_classifier`.

---

## 5. Métricas de evaluación

La evaluación se realiza sobre el conjunto de prueba, separado antes del entrenamiento y nunca utilizado durante el mismo.

| Métrica | Descripción |
|---|---|
| **Accuracy** | Proporción de predicciones correctas sobre el total |
| **F1-score macro** | Media del F1 por clase, sin ponderar por frecuencia (relevante con clases desbalanceadas) |
| **Matriz de confusión** | Distribución de predicciones correctas e incorrectas por clase |
| **AUC-ROC (por clase)** | Capacidad discriminativa del modelo para cada especie |

Los reportes de métricas se exportan en `reports/metrics/` y las figuras (curvas de aprendizaje, matriz de confusión) en `reports/figures/`.

---

## 6. Registro en MLflow Model Registry

El modelo final es promovido al stage `Production` en el MLflow Model Registry tras superar los umbrales de evaluación definidos. El proceso de promoción está implementado en `src/model_registry/model_registry.py`.

**Información registrada en MLflow:**

- Parámetros del experimento (hiperparámetros)
- Métricas de entrenamiento y validación por época
- Métricas finales en el conjunto de prueba
- Artefactos: pesos del modelo (`.pt`), clases del clasificador, transformaciones de entrada

La API FastAPI carga automáticamente la versión en stage `Production` al iniciarse.

---

## 7. Consideraciones y limitaciones

- **Tamaño del dataset:** El volumen de imágenes anotadas del Humedal Siracusa es limitado, lo que hace que el transfer learning sea una estrategia crítica. A medida que se recopilen más imágenes de campo, el modelo puede re-entrenarse con mejores garantías de generalización.
- **Condiciones de imagen:** Las cámaras trampa generan imágenes con iluminación variable (diurna, nocturna con IR), fondo natural complejo y posibles oclusiones parciales. Las técnicas de aumentación de datos buscan mitigar estas variaciones.
- **Desbalance de clases:** Algunas especies pueden estar subrepresentadas en el dataset. El F1-score macro es la métrica principal de referencia para evitar que este desbalance enmascare el rendimiento real del modelo.
- **Escalabilidad a Edge AI:** La demo actual corre en un servidor convencional. El siguiente paso del proyecto contempla la optimización y cuantización del modelo para despliegue en dispositivos embebidos (Raspberry Pi, NVIDIA Jetson) como parte del sistema de monitoreo en campo.
