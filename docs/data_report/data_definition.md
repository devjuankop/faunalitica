# Definición de datos

Este documento describe las fuentes de datos, las clases de especies, la estructura del dataset y las consideraciones de preprocesamiento utilizadas en el proyecto Faunalítica.

---

## 1. Contexto del dataset

El clasificador de imágenes de Faunalítica está orientado a identificar fauna silvestre a partir de fotografías tomadas por **cámaras trampa** (*camera traps*) instaladas en o en las proximidades del **Humedal Siracusa** (Sevilla, Valle del Cauca, Colombia). Las cámaras trampa son dispositivos de captura automática que se activan por movimiento y permiten el monitoreo no invasivo de fauna sin presencia humana continua.

---

## 2. Fuentes de datos

| Fuente | Descripción |
|---|---|
| **Cámaras trampa en campo** | Fotografías capturadas directamente en el Humedal Siracusa durante las fases de recolección de datos del proyecto. |
| **Repositorios abiertos de biodiversidad** | Imágenes complementarias de especies presentes en el humedal, obtenidas de fuentes de acceso abierto para aumentar el volumen de datos en clases con pocas muestras de campo. |

> La incorporación de datos de repositorios externos busca mitigar el desbalance de clases inherente a las observaciones de campo, donde algunas especies son considerablemente más difíciles de capturar en imagen.

---

## 3. Clases del dataset

El dataset contempla las **especies de fauna más relevantes del Humedal Siracusa**, con énfasis en aves y fauna característica de humedales urbanos del Valle del Cauca. La selección de especies objetivo se realizó en la Fase 1 del diseño metodológico del proyecto (identificación de especies relevantes y definición de requerimientos técnicos).

> La lista definitiva de especies y el número de muestras por clase se documenta en el experimento de MLflow correspondiente al entrenamiento del modelo final, accesible en la UI de MLflow bajo el experimento `faunalitica_classifier`.

---

## 4. Estructura del dataset

Las imágenes se organizan en la carpeta `data/` con la siguiente estructura:

```
data/
├── raw/                    # Imágenes originales sin modificar
│   ├── clase_especie_1/
│   ├── clase_especie_2/
│   └── ...
│
├── processed/              # Imágenes preprocesadas (redimensionadas y normalizadas)
│   ├── clase_especie_1/
│   ├── clase_especie_2/
│   └── ...
│
└── splits/                 # Particiones del dataset
    ├── train/              # ~70% de los datos
    │   ├── clase_especie_1/
    │   └── ...
    ├── val/                # ~15% de los datos
    │   ├── clase_especie_1/
    │   └── ...
    └── test/               # ~15% de los datos (nunca visto durante entrenamiento)
        ├── clase_especie_1/
        └── ...
```

La partición es **estratificada** por clase para mantener la distribución original en cada subconjunto.

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

El preprocesamiento está implementado en `src/data_engineering/data_engineering.py` y contempla:

| Paso | Descripción | Aplicado en |
|---|---|---|
| Redimensionado | Ajuste a la resolución de entrada del modelo base | Entrenamiento e inferencia |
| Normalización | Media y desviación estándar de ImageNet | Entrenamiento e inferencia |
| Rotación aleatoria | Rotación ±15° | Solo entrenamiento |
| Volteo horizontal | Espejado aleatorio | Solo entrenamiento |
| Variación de brillo/contraste | Simula condiciones de iluminación variable | Solo entrenamiento |
| Partición estratificada | División train/val/test manteniendo proporción de clases | Una vez, sobre `raw/` |

---

## 7. Consideraciones éticas y legales

- La recolección de imágenes de fauna en el Humedal Siracusa se enmarca en los protocolos éticos y metodológicos para la toma de datos en ecosistemas sensibles definidos en el diseño metodológico del proyecto.
- El dataset no contiene imágenes de personas; en caso de que alguna cámara trampa capture accidentalmente personas, dichas imágenes serán excluidas del dataset en cumplimiento de la **Ley 1581 de 2012** sobre protección de datos personales.
- Las imágenes de fuentes externas se usan únicamente bajo licencias que permiten su uso en investigación científica.
