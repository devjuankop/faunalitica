import argparse
import json
import os
from pathlib import Path
from typing import Any, Dict, Optional

import mlflow
import mlflow.pytorch
import torch
import torch.nn as nn
import torchvision.models as tv_models
from mlflow.tracking import MlflowClient


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RESULTS_JSON = PROJECT_ROOT / "reports" / "resultados.json"
DEFAULT_CHECKPOINTS_DIR = PROJECT_ROOT / "reports"
DEFAULT_SUBSET_DIR = PROJECT_ROOT / "data" / "processed"

DEFAULT_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://127.0.0.1:5000")
DEFAULT_EXPERIMENT_NAME = os.getenv("MLFLOW_EXPERIMENT_NAME", "orinoquia-species-classification")
DEFAULT_REGISTERED_MODEL_NAME = os.getenv("REGISTERED_MODEL_NAME", "orinoquia-species-classifier")
DEFAULT_ALIAS = os.getenv("REGISTERED_MODEL_ALIAS", "champion")


def _resolve_path(path_value: str) -> Path:
    path = Path(path_value)
    if path.is_absolute():
        return path
    return (PROJECT_ROOT / path).resolve()


def _load_winner(results_json_path: Path, winner_name: Optional[str] = None) -> Dict[str, Any]:
    if not results_json_path.exists():
        raise FileNotFoundError(f"No existe resultados.json en: {results_json_path}")

    with open(results_json_path, "r", encoding="utf-8") as f:
        resultados = json.load(f)

    if not isinstance(resultados, list) or not resultados:
        raise ValueError("resultados.json no contiene experimentos válidos")

    if winner_name is None:
        return resultados[0]

    for item in resultados:
        if item.get("nombre") == winner_name:
            return item

    raise ValueError(f"No se encontró el experimento '{winner_name}' en resultados.json")


def _build_model(model_name: str, num_classes: int) -> nn.Module:
    if model_name == "efficientnet_b0":
        model = tv_models.efficientnet_b0(weights=None)
        in_features = model.classifier[1].in_features
        model.classifier[1] = nn.Linear(in_features, num_classes)
        return model

    if model_name == "mobilenet_v3_small":
        model = tv_models.mobilenet_v3_small(weights=None)
        in_features = model.classifier[-1].in_features
        model.classifier[-1] = nn.Linear(in_features, num_classes)
        return model

    if model_name == "resnet18":
        model = tv_models.resnet18(weights=None)
        in_features = model.fc.in_features
        model.fc = nn.Linear(in_features, num_classes)
        return model

    raise ValueError(
        f"Modelo no soportado: {model_name}. "
        "Opciones: efficientnet_b0, mobilenet_v3_small, resnet18"
    )


def register_best_model(
    results_json_path: Path = DEFAULT_RESULTS_JSON,
    checkpoints_dir: Path = DEFAULT_CHECKPOINTS_DIR,
    subset_dir: Path = DEFAULT_SUBSET_DIR,
    tracking_uri: str = DEFAULT_TRACKING_URI,
    experiment_name: str = DEFAULT_EXPERIMENT_NAME,
    registered_model_name: str = DEFAULT_REGISTERED_MODEL_NAME,
    alias: str = DEFAULT_ALIAS,
    winner_name: Optional[str] = None,
) -> Dict[str, Any]:
    winner = _load_winner(results_json_path=results_json_path, winner_name=winner_name)

    run_name = winner.get("nombre")
    model_name = winner.get("modelo")
    if not run_name or not model_name:
        raise ValueError("El experimento ganador no contiene 'nombre' y 'modelo'")

    checkpoint_path = checkpoints_dir / f"{run_name}.pt"
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"No existe checkpoint para el ganador: {checkpoint_path}")

    class_map_path = subset_dir / "class_map.json"
    if not class_map_path.exists():
        raise FileNotFoundError(f"No existe class_map.json en: {class_map_path}")

    with open(class_map_path, "r", encoding="utf-8") as f:
        class_map = json.load(f)

    num_classes = len(class_map)
    model = _build_model(model_name=model_name, num_classes=num_classes)

    state_dict = torch.load(checkpoint_path, map_location="cpu")
    model.load_state_dict(state_dict)
    model.eval()

    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(experiment_name)
    client = MlflowClient(tracking_uri=tracking_uri)

    with mlflow.start_run(run_name=f"register_{run_name}") as run:
        mlflow.log_param("source_experimento_nombre", run_name)
        mlflow.log_param("source_model_architecture", model_name)
        mlflow.log_param("num_classes", num_classes)
        mlflow.log_param("checkpoint_path", str(checkpoint_path))
        mlflow.log_param("class_map_path", str(class_map_path))

        if isinstance(winner.get("mejor_val_acc"), (int, float)):
            mlflow.log_metric("mejor_val_acc", float(winner["mejor_val_acc"]))
        if isinstance(winner.get("test_accuracy"), (int, float)):
            mlflow.log_metric("test_accuracy", float(winner["test_accuracy"]))

        mlflow.log_text(
            json.dumps(winner, ensure_ascii=False, indent=2),
            artifact_file="metadata/winner_result.json",
        )
        mlflow.log_artifact(str(class_map_path), artifact_path="metadata")

        logged_model = mlflow.pytorch.log_model(
            pytorch_model=model,
            artifact_path="model",
        )

    registration = mlflow.register_model(
        model_uri=logged_model.model_uri,
        name=registered_model_name,
    )

    client.set_registered_model_alias(
        name=registered_model_name,
        alias=alias,
        version=registration.version,
    )

    client.set_model_version_tag(
        name=registered_model_name,
        version=registration.version,
        key="source_experimento",
        value=run_name,
    )
    client.set_model_version_tag(
        name=registered_model_name,
        version=registration.version,
        key="source_model_architecture",
        value=model_name,
    )

    result = {
        "mlflow_run_id": run.info.run_id,
        "registered_model_name": registered_model_name,
        "version": registration.version,
        "alias": alias,
        "winner": run_name,
        "model_architecture": model_name,
        "checkpoint_path": str(checkpoint_path),
    }

    print(json.dumps(result, ensure_ascii=False, indent=2))
    return result


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Registra en MLflow un modelo ganador entrenado por src/experimentos.py"
    )
    parser.add_argument("--results-json", default=str(DEFAULT_RESULTS_JSON))
    parser.add_argument("--checkpoints-dir", default=str(DEFAULT_CHECKPOINTS_DIR))
    parser.add_argument("--subset-dir", default=str(DEFAULT_SUBSET_DIR))
    parser.add_argument("--tracking-uri", default=DEFAULT_TRACKING_URI)
    parser.add_argument("--experiment-name", default=DEFAULT_EXPERIMENT_NAME)
    parser.add_argument("--registered-model-name", default=DEFAULT_REGISTERED_MODEL_NAME)
    parser.add_argument("--alias", default=DEFAULT_ALIAS)
    parser.add_argument(
        "--winner-name",
        default=None,
        help="Nombre del experimento a registrar. Si no se define, usa el primero de resultados.json",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    register_best_model(
        results_json_path=_resolve_path(args.results_json),
        checkpoints_dir=_resolve_path(args.checkpoints_dir),
        subset_dir=_resolve_path(args.subset_dir),
        tracking_uri=args.tracking_uri,
        experiment_name=args.experiment_name,
        registered_model_name=args.registered_model_name,
        alias=args.alias,
        winner_name=args.winner_name,
    )
