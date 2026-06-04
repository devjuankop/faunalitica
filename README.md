# Proyecto Faunalítica

## Índice

- [Descripción del proyecto](#descripción-del-proyecto)
- [Estructura del proyecto](#estructura-del-proyecto)
- [Documentación](#documentación)

---

## Descripción del proyecto

**Faunalítica** es una demo de clasificación de imágenes orientada a la identificación automatizada de fauna silvestre a partir de fotografías tomadas por cámaras trampa. El proyecto forma parte de la investigación *"Inteligencia Artificial de Borde aplicado al Monitoreo y Apropiación Social de Fauna susceptible de Protección Ambiental en el Humedal Siracusa"*, desarrollada en la **Maestría en Inteligencia Artificial y Ciencia de Datos** de la **Universidad Autónoma de Occidente** (Santiago de Cali, 2026).

El Humedal Siracusa, ubicado en el municipio de Sevilla (Valle del Cauca), es un ecosistema urbano en proceso de restauración ecológica. Esta demo demuestra la viabilidad de utilizar modelos de visión por computador — entrenados, rastreados y servidos mediante una arquitectura MLOps — para identificar las especies de fauna más representativas del humedal, reduciendo la dependencia de inspecciones presenciales especializadas y sentando las bases para un sistema de monitoreo continuo basado en Edge AI.

**Autores:** Juan José Bonilla Pinzón · Ricardo Muñoz Bocanegra  
**Director:** Diego Armando Burgos Salamanca  
**Codirector:** Juan Manuel Núñez Velasco  
**Repositorio:** [https://github.com/devjuankop/faunalitica](https://github.com/devjuankop/faunalitica)

---

## Estructura del proyecto

```
faunalitica/
│
├── .github/
│   └── workflows/                      # Pipelines CI/CD (GitHub Actions)
│
├── api/
│   └── main.py                         # API REST (FastAPI) para inferencia del modelo
│
├── app/
│   └── streamlit_app.py                # Interfaz de usuario (Streamlit)
│
├── data/
│   ├── original_metadata/
│   │   └── orinoquia_camera_traps.json # Metadatos originales del dataset (formato COCO)
│   ├── processed/
│   │   ├── splits/                     # Particiones train / val / test
│   │   ├── class_map.json              # Mapeo de índices a nombres de especies
│   │   ├── dataset.py                  # Clase Dataset de PyTorch para carga de imágenes
│   │   ├── subset_coco.json            # Subconjunto del dataset en formato COCO
│   │   └── subset_manifest.csv        # Manifiesto CSV del subconjunto procesado
│   └── test/
│       ├── ave.png                     # Imagen de prueba (ave)
│       └── tapir-directory-2.jpg       # Imagen de prueba (tapir)
│
├── docker/
│   └── prometheus.yml                  # Configuración de Prometheus para monitoreo
│
├── docs/
│   ├── data_report/
│   │   └── data_definition.md
│   ├── deployment/
│   │   └── deployment_plan.md
│   ├── model/
│   │   └── final_model_report.md
│   └── project/
│       ├── CRISP-DM.md
│       ├── installations.md
│       └── instructions.md
│
├── reports/
│   ├── efficientnet_b0_lr1e3.pt        # Pesos del modelo entrenado (EfficientNet-B0)
│   └── resultados.json                 # Métricas y resultados del entrenamiento
│
├── src/
│   ├── __init__.py
│   ├── config.py                       # Configuración global (rutas, hiperparámetros)
│   ├── data.py                         # Carga y preprocesamiento de datos
│   ├── experimentos.py                 # Rastreo de experimentos con MLflow
│   ├── predict.py                      # Lógica de inferencia
│   ├── register_model.py               # Registro y promoción del modelo en MLflow
│   └── train.py                        # Entrenamiento del modelo
│
├── tests/
│   └── test_api.py                     # Pruebas de integración de la API
│
├── .gitignore
├── docker-compose.yml                  # Orquestación de Prometheus y Grafana
├── README.md
└── requirements.txt                    # Dependencias del proyecto
```

---

## Documentación

La documentación del proyecto está organizada dentro de la carpeta `docs/`, estructurada así:

### Carpeta `project/`

| Documento | Descripción |
|---|---|
| [`installations.md`](docs/project/installations.md) | Guía de instalación y configuración del entorno, dependencias y versiones requeridas para ejecutar el proyecto. |
| [`instructions.md`](docs/project/instructions.md) | Instrucciones paso a paso para ejecutar cada componente del proyecto (entrenamiento, API, Streamlit, monitoreo). |
| [`CRISP-DM.md`](docs/project/CRISP-DM.md) | Descripción de la metodología CRISP-DM aplicada al desarrollo del clasificador de fauna. |

### Carpeta `model/`

| Documento | Descripción |
|---|---|
| [`final_model_report.md`](docs/model/final_model_report.md) | Reporte técnico del modelo de clasificación de imágenes entrenado, con arquitectura, métricas y análisis de resultados. |

### Carpeta `deployment/`

| Documento | Descripción |
|---|---|
| [`deployment_plan.md`](docs/deployment/deployment_plan.md) | Descripción de la estrategia de despliegue implementada para la demo (FastAPI + Streamlit + monitoreo). |

### Carpeta `data_report/`

| Documento | Descripción |
|---|---|
| [`data_definition.md`](docs/data_report/data_definition.md) | Descripción de las fuentes de datos, clases de especies, estructura del dataset y consideraciones de preprocesamiento. |
