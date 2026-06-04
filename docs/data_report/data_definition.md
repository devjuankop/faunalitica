# Definición de datos

Este documento describe las fuentes de datos, las clases de especies, la estructura del dataset y las consideraciones de preprocesamiento utilizadas en el proyecto Faunalítica.

---

## 1. Contexto del dataset

El clasificador de imágenes de Faunalítica está orientado a identificar fauna silvestre a partir de fotografías tomadas por **cámaras trampa** (*camera traps*) instaladas en o en las proximidades del **Humedal Siracusa** (Sevilla, Valle del Cauca, Colombia). Las cámaras trampa son dispositivos de captura automática que se activan por movimiento y permiten el monitoreo no invasivo de fauna sin presencia humana continua.

---

## 2. Fuentes de datos

| Fuente | Descripción |
|---|---|
| **Dataset Orinoquia Camera Traps** | Dataset de imágenes de cámaras trampa de la región de la Orinoquia colombiana, en formato COCO. Almacenado en `data/original_metadata/orinoquia_camera_traps.json`. Utilizado como base de entrenamiento por su similitud ecológica con el contexto del Humedal Siracusa. |
| **Imágenes de prueba locales** | Imágenes de referencia almacenadas en `data/test/` (e.g., `ave.png`, `tapir-directory-2.jpg`) para validación manual y pruebas de la demo. |

> El dataset de Orinoquia incluye metadatos de anotación en formato COCO, que son procesados por `src/data.py` para generar el subconjunto (`subset_coco.json`, `subset_manifest.csv`) y las particiones de entrenamiento.

---

## 3. Clases del dataset

El dataset contempla las **especies de fauna más relevantes del Humedal Siracusa**, con énfasis en aves y fauna característica de humedales urbanos del Valle del Cauca. La selección de especies objetivo se realizó en la Fase 1 del diseño metodológico del proyecto (identificación de especies relevantes y definición de requerimientos técnicos).

> La lista definitiva de especies y el número de muestras por clase se documenta en el experimento de MLflow correspondiente al entrenamiento del modelo final, accesible en la UI de MLflow bajo el experimento `faunalitica_classifier`.

---

## 4. Estructura del dataset

Las imágenes se organizan en la carpeta `data/` con la siguiente estructura:

```
data/
├── original_metadata/
│   └── orinoquia_camera_traps.json  # Metadatos originales del dataset (formato COCO)
│
├── processed/
│   ├── splits/                      # Particiones train / val / test
│   ├── class_map.json               # Mapeo de índices a nombres de especies
│   ├── dataset.py                   # Clase Dataset de PyTorch para carga de imágenes
│   ├── subset_coco.json             # Subconjunto del dataset en formato COCO
│   └── subset_manifest.csv         # Manifiesto CSV del subconjunto procesado
│
└── test/
    ├── ave.png                      # Imagen de prueba (ave)
    └── tapir-directory-2.jpg        # Imagen de prueba (tapir)
```

Las particiones train / val / test dentro de `processed/splits/` se generan con una división estratificada por clase para mantener la distribución original del dataset en cada subconjunto.

---

## 5. Características de las imágenes

| Característica | Descripción |
|---|---|
| **Formato** | JPG / PNG |
| **Condiciones de captura** | Diurnas e infrarrojo nocturno (cámaras trampa) |
| **Variabilidad** | Iluminación variable, fondo natural no controlado, posibles oclusiones parciales por vegetación |
| **Resolución de entrada al modelo** | Uniforme tras preprocesamiento (definida por la arquitectura CNN base) |

Las condiciones propias de las cámaras trampa (imágenes en blanco y negro de noche, fondos con vegetación densa, animales parcialmente visibles) representan el principal desafío técnico del dataset y motivan el uso de aumentación de datos durante el entrenamiento.

---

## 6. Preprocesamiento aplicado

El preprocesamiento está implementado en `src/data.py` y contempla:

| Paso | Descripción | Aplicado en |
|---|---|---|
| Redimensionado | Ajuste a la resolución de entrada del modelo base (EfficientNet-B0) | Entrenamiento e inferencia |
| Normalización | Media y desviación estándar de ImageNet | Entrenamiento e inferencia |
| Rotación aleatoria | Rotación ±15° | Solo entrenamiento |
| Volteo horizontal | Espejado aleatorio | Solo entrenamiento |
| Variación de brillo/contraste | Simula condiciones de iluminación variable | Solo entrenamiento |
| Partición estratificada | División train/val/test a partir de `subset_manifest.csv`, manteniendo proporción de clases | Una vez, al procesar el dataset |

---

## 7. Consideraciones éticas y legales

- La recolección de imágenes de fauna en el Humedal Siracusa se enmarca en los protocolos éticos y metodológicos para la toma de datos en ecosistemas sensibles definidos en el diseño metodológico del proyecto.
- El dataset no contiene imágenes de personas; en caso de que alguna cámara trampa capture accidentalmente personas, dichas imágenes serán excluidas del dataset en cumplimiento de la **Ley 1581 de 2012** sobre protección de datos personales.
- Las imágenes de fuentes externas se usan únicamente bajo licencias que permiten su uso en investigación científica.
