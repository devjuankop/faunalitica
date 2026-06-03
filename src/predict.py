import argparse
import json
import os
from pathlib import Path
from typing import Any, Dict, Optional

import mlflow
import mlflow.pytorch
import torch
from PIL import Image
from torchvision import transforms


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CLASS_MAP_PATH = PROJECT_ROOT / "data" / "processed" / "class_map.json"
DEFAULT_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://127.0.0.1:5000")
DEFAULT_REGISTERED_MODEL_NAME = os.getenv("REGISTERED_MODEL_NAME", "orinoquia-species-classifier")
DEFAULT_ALIAS = os.getenv("REGISTERED_MODEL_ALIAS", "champion")

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def _resolve_path(path_value: str) -> Path:
    path = Path(path_value)
    if path.is_absolute():
        return path
    return (PROJECT_ROOT / path).resolve()


def get_idx_to_class_map(class_map_path: Path) -> Dict[int, str]:
    return _load_idx_to_class(class_map_path)


def get_image_transform() -> transforms.Compose:
    return _build_transform()


def predict_image_from_pil(
    model: torch.nn.Module,
    image: Image.Image,
    idx_to_class: Dict[int, str],
    top_k: int,
) -> Dict[str, Any]:
    transform = get_image_transform()
    batch = transform(image.convert("RGB")).unsqueeze(0)

    with torch.no_grad():
        logits = model(batch)
        probabilities = torch.softmax(logits, dim=1).squeeze(0)

    k = min(max(top_k, 1), probabilities.numel())
    top_probs, top_indices = torch.topk(probabilities, k=k)

    predictions = []
    for prob, idx in zip(top_probs.tolist(), top_indices.tolist()):
        predictions.append(
            {
                "class_index": idx,
                "class_name": idx_to_class.get(idx, f"class_{idx}"),
                "probability": round(float(prob), 6),
            }
        )

    return {
        "predicted_class": predictions[0]["class_name"],
        "predictions": predictions,
    }


def _load_idx_to_class(class_map_path: Path) -> Dict[int, str]:
    if not class_map_path.exists():
        raise FileNotFoundError(f"No existe class_map.json en: {class_map_path}")

    with open(class_map_path, "r", encoding="utf-8") as f:
        class_map = json.load(f)

    return {int(idx): label for label, idx in class_map.items()}


def _build_transform() -> transforms.Compose:
    return transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]
    )


def load_model(
    tracking_uri: str = DEFAULT_TRACKING_URI,
    registered_model_name: str = DEFAULT_REGISTERED_MODEL_NAME,
    alias: str = DEFAULT_ALIAS,
    model_uri: Optional[str] = None,
) -> tuple[torch.nn.Module, str]:
    mlflow.set_tracking_uri(tracking_uri)
    resolved_uri = model_uri or f"models:/{registered_model_name}@{alias}"
    model = mlflow.pytorch.load_model(resolved_uri)
    model.eval()
    return model, resolved_uri


def predict_image(
    image_path: Path,
    class_map_path: Path = DEFAULT_CLASS_MAP_PATH,
    tracking_uri: str = DEFAULT_TRACKING_URI,
    registered_model_name: str = DEFAULT_REGISTERED_MODEL_NAME,
    alias: str = DEFAULT_ALIAS,
    model_uri: Optional[str] = None,
    top_k: int = 3,
) -> Dict[str, Any]:
    if not image_path.exists():
        raise FileNotFoundError(f"No existe la imagen: {image_path}")

    idx_to_class = _load_idx_to_class(class_map_path)
    model, resolved_uri = load_model(
        tracking_uri=tracking_uri,
        registered_model_name=registered_model_name,
        alias=alias,
        model_uri=model_uri,
    )

    transform = _build_transform()
    image = Image.open(image_path).convert("RGB")
    batch = transform(image).unsqueeze(0)

    with torch.no_grad():
        logits = model(batch)
        probabilities = torch.softmax(logits, dim=1).squeeze(0)

    k = min(max(top_k, 1), probabilities.numel())
    top_probs, top_indices = torch.topk(probabilities, k=k)

    predictions = []
    for prob, idx in zip(top_probs.tolist(), top_indices.tolist()):
        predictions.append(
            {
                "class_index": idx,
                "class_name": idx_to_class.get(idx, f"class_{idx}"),
                "probability": round(float(prob), 6),
            }
        )

    result = {
        "image_path": str(image_path),
        "model_uri": resolved_uri,
        "top_k": k,
        "predicted_class": predictions[0]["class_name"],
        "predictions": predictions,
    }

    print(json.dumps(result, ensure_ascii=False, indent=2))
    return result


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Predicción de especie usando un modelo registrado en MLflow"
    )
    parser.add_argument("--image", required=True, help="Ruta de imagen a clasificar")
    parser.add_argument("--class-map", default=str(DEFAULT_CLASS_MAP_PATH))
    parser.add_argument("--tracking-uri", default=DEFAULT_TRACKING_URI)
    parser.add_argument("--registered-model-name", default=DEFAULT_REGISTERED_MODEL_NAME)
    parser.add_argument("--alias", default=DEFAULT_ALIAS)
    parser.add_argument("--model-uri", default=None, help="URI MLflow opcional. Ej: models:/modelo/3")
    parser.add_argument("--top-k", type=int, default=3)
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    predict_image(
        image_path=_resolve_path(args.image),
        class_map_path=_resolve_path(args.class_map),
        tracking_uri=args.tracking_uri,
        registered_model_name=args.registered_model_name,
        alias=args.alias,
        model_uri=args.model_uri,
        top_k=args.top_k,
    )
