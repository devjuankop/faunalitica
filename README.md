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
│   └── workflows/          # Pipelines CI/CD (GitHub Actions)
│
├── api/
│   └── model_api.py        # API REST (FastAPI) para inferencia del modelo
│
├── app/
│   └── streamlit_app.py    # Interfaz de usuario (Streamlit)
│
├── data/
│   ├── raw/                # Imágenes originales de cámaras trampa
│   ├── processed/          # Imágenes preprocesadas
│   └── splits/             # Particiones train / val / test
│
├── docker/
│   └── prometheus.yml      # Configuración de Prometheus para monitoreo
│
├── docs/
│   ├── data_report/
│   │   └── data_definition.md
│   ├── deployment/
│   │   └── deployment_plan.md
│   ├── model/
│   │   └── final_model_report.md
│   └── project/
│       ├── installations.md
│       ├── instructions.md
│       └── CRISP-DM.md
│
├── reports/
│   ├── figures/            # Gráficas y visualizaciones exportadas
│   └── metrics/            # Métricas registradas durante el entrenamiento
│
├── src/
│   ├── data_engineering/   # Carga, limpieza y particionado de datos
│   ├── model_engineering/  # Entrenamiento y evaluación del modelo
│   ├── model_registry/     # Registro y promoción de modelos en MLflow
│   ├── performance_monitoring/ # Instrumentación con Prometheus
│   └── prediction_service/ # Lógica de inferencia expuesta por la API
│
├── tests/                  # Pruebas unitarias e integración
│
├── .gitignore
├── docker-compose.yml      # Orquestación de Prometheus y Grafana
├── README.md
└── requirements.txt        # Dependencias del proyecto
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
