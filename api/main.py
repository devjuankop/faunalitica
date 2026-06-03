import json
import subprocess
import sys
import io
from pathlib import Path
from typing import Any, Dict, Optional

import torch
from PIL import Image
from fastapi import FastAPI, HTTPException, UploadFile, File, Depends
from pydantic import BaseModel, Field

from src.predict import (
    DEFAULT_CLASS_MAP_PATH,
    DEFAULT_TRACKING_URI,
    DEFAULT_REGISTERED_MODEL_NAME,
    DEFAULT_ALIAS,
    load_model,
    get_idx_to_class_map,
    predict_image_from_pil,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
EXPERIMENT_SCRIPT = PROJECT_ROOT / "src" / "experimentos.py"
DEFAULT_IMGS = PROJECT_ROOT / "data" / "raw"
DEFAULT_SUBSET = PROJECT_ROOT / "data" / "processed"
DEFAULT_OUT = PROJECT_ROOT / "reports"

# Global variables for the model and class map
GLOBAL_MODEL: Optional[torch.nn.Module] = None
GLOBAL_IDX_TO_CLASS: Optional[Dict[int, str]] = None


app = FastAPI(title="Orinoquia Experimentos API")


@app.on_event("startup")
async def load_mlflow_model():
    global GLOBAL_MODEL, GLOBAL_IDX_TO_CLASS
    try:
        GLOBAL_MODEL, _ = load_model(
            tracking_uri=DEFAULT_TRACKING_URI,
            registered_model_name=DEFAULT_REGISTERED_MODEL_NAME,
            alias=DEFAULT_ALIAS,
        )
        GLOBAL_IDX_TO_CLASS = get_idx_to_class_map(DEFAULT_CLASS_MAP_PATH)
        print("MLflow model and class map loaded successfully on startup.")
    except Exception as e:
        print(f"Error loading MLflow model on startup: {e}")
        # Optionally re-raise or set a flag to indicate model loading failure


class ExperimentRequest(BaseModel):
    imgs: str = Field(default=str(DEFAULT_IMGS), description="Carpeta raíz de imágenes")
    subset: str = Field(default=str(DEFAULT_SUBSET), description="Carpeta output_subset")
    epochs: int = Field(default=15, ge=1, description="Número de épocas")
    out: str = Field(default=str(DEFAULT_OUT), description="Carpeta de salida")


def _resolve_user_path(path_value: str) -> Path:
    path = Path(path_value)
    if path.is_absolute():
        return path
    return (PROJECT_ROOT / path).resolve()


def _validate_paths(imgs: Path, subset: Path) -> None:
    if not EXPERIMENT_SCRIPT.exists():
        raise HTTPException(
            status_code=500,
            detail=f"No se encontró el script de experimentos en {EXPERIMENT_SCRIPT}",
        )

    if not imgs.exists():
        raise HTTPException(
            status_code=400,
            detail=f"La ruta de imágenes no existe: {imgs}",
        )

    if not subset.exists():
        raise HTTPException(
            status_code=400,
            detail=f"La ruta de subset no existe: {subset}",
        )


def _tail_lines(text: str, limit: int = 40) -> list[str]:
    lines = [line for line in text.splitlines() if line.strip()]
    return lines[-limit:]


@app.get("/")
def root() -> Dict[str, str]:
    return {"message": "API de experimentos activa"}


@app.get("/health")
def health() -> Dict[str, Any]:
    return {
        "status": "ok",
        "experimentos_script": str(EXPERIMENT_SCRIPT),
        "script_exists": EXPERIMENT_SCRIPT.exists(),
    }


@app.post("/experimentos/run")
def run_experimentos(request: ExperimentRequest) -> Dict[str, Any]:
    imgs_path = _resolve_user_path(request.imgs)
    subset_path = _resolve_user_path(request.subset)
    out_path = _resolve_user_path(request.out)

    _validate_paths(imgs_path, subset_path)
    out_path.mkdir(parents=True, exist_ok=True)

    command = [
        sys.executable,
        str(EXPERIMENT_SCRIPT),
        "--imgs",
        str(imgs_path),
        "--subset",
        str(subset_path),
        "--epochs",
        str(request.epochs),
        "--out",
        str(out_path),
    ]

    result = subprocess.run(
        command,
        cwd=str(PROJECT_ROOT),
        capture_output=True,
        text=True,
    )

    if result.returncode != 0:
        raise HTTPException(
            status_code=500,
            detail={
                "message": "La ejecución de experimentos.py falló",
                "return_code": result.returncode,
                "stderr_tail": _tail_lines(result.stderr),
                "stdout_tail": _tail_lines(result.stdout),
            },
        )

    resultados_path = out_path / "resultados.json"
    resultados: Optional[Any] = None

    if resultados_path.exists():
        with open(resultados_path, "r", encoding="utf-8") as f:
            resultados = json.load(f)

    return {
        "status": "completed",
        "command": command,
        "output_dir": str(out_path),
        "resultados_path": str(resultados_path),
        "resultados": resultados,
        "stdout_tail": _tail_lines(result.stdout),
    }


@app.post("/predict/image")
async def predict_image_api(file: UploadFile = File(...), top_k: int = 3):
    if GLOBAL_MODEL is None or GLOBAL_IDX_TO_CLASS is None:
        raise HTTPException(
            status_code=503, detail="Model not loaded yet. Please try again in a moment."
        )

    try:
        # Read image
        contents = await file.read()
        image = Image.open(io.BytesIO(contents)).convert("RGB")

        # Perform prediction
        predictions = predict_image_from_pil(
            model=GLOBAL_MODEL,
            image=image,
            idx_to_class=GLOBAL_IDX_TO_CLASS,
            top_k=top_k,
        )
        return predictions
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {e}")
