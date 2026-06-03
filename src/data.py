"""
orinoquia_subset.py
====================
Filtra el dataset Orinoquia Camera Traps (formato COCO) para quedarse
con las carpetas A06 y N27, excluye etiquetas ruidosas, aplica un cap
por especie y genera un Dataset de PyTorch listo para entrenar.

Uso:
    python orinoquia_subset.py \
        --json  path/to/orinoquia_camera_traps.json \
        --imgs  path/to/images_root/ \
        --out   path/to/output_subset/

Salidas:
    output_subset/
        subset_coco.json        ← JSON COCO filtrado y balanceado
        subset_manifest.csv     ← tabla file_name, species, location, seq_id
        class_map.json          ← {"collared_peccary": 0, ...}
        splits/
            train.txt           ← file_names para entrenamiento (70%)
            val.txt             ← file_names para validación  (15%)
            test.txt            ← file_names para test        (15%)
"""

import json
import csv
import random
import argparse
from pathlib import Path
from collections import defaultdict


# ─────────────────────────── configuración ───────────────────────────

TARGET_LOCATIONS = {"A06", "N27"}

CAP_PER_CLASS = 150          # máximo de imágenes por especie
MIN_PER_CLASS = 30           # descartar clases con menos de esto (tras filtro)

SPLIT_RATIOS = (0.70, 0.15, 0.15)   # train / val / test
RANDOM_SEED  = 42

# Etiquetas ruidosas que se excluyen por completo
NOISE_LABELS = {
    "empty", "human",
    "cattle", "domestic_horse", "domestic_dog",
    "unknown",
    "unknown_bird", "unknown_armadillo", "unknown_possum",
    "unknown_cervid", "unknown_reptile", "unknown_squirrel_monkey",
    "unknown_capuchin_monkey", "unknown_peccary", "unknown_nightjar",
    "unknown_turtle", "unknown_mammal", "unknown_howler_monkey",
    "unknown_tayra", "unknown_weasel",
    "insect", "ants", "rodent",
}


# ─────────────────────────── funciones ───────────────────────────────

def load_json(path: str) -> dict:
    print(f"[1/6] Cargando JSON: {path}")
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def build_lookups(data: dict):
    """Construye diccionarios de acceso rápido."""
    cat_by_id   = {c["id"]: c["name"] for c in data["categories"]}
    cat_name_to_id = {v: k for k, v in cat_by_id.items()}
    img_by_id   = {img["id"]: img for img in data["images"]}
    return cat_by_id, cat_name_to_id, img_by_id


def filter_annotations(data, cat_by_id, img_by_id):
    """
    Devuelve sólo las anotaciones donde:
      - la imagen pertenece a TARGET_LOCATIONS
      - la categoría NO está en NOISE_LABELS
    """
    
    print(f"[2/6] Filtrando por ubicación {TARGET_LOCATIONS} y excluyendo ruido...")
    kept = []
    for ann in data["annotations"]:
        img   = img_by_id.get(ann["image_id"])
        if img is None:
            continue
        if img["location"] not in TARGET_LOCATIONS:
            continue
        label = cat_by_id.get(ann["category_id"], "unknown")
        if label in NOISE_LABELS:
            continue
        kept.append({
            "ann_id"    : ann["id"],
            "image_id"  : ann["image_id"],
            "category_id": ann["category_id"],
            "label"     : label,
            "file_name" : img["file_name"],
            "location"  : img["location"],
            "seq_id"    : img["seq_id"],
            "datetime"  : img.get("datetime", ""),
        })
    print(f"    → {len(kept):,} anotaciones tras filtro de ubicación/ruido")
    return kept


def group_by_species(annotations: list) -> dict:
    """Agrupa registros por etiqueta de especie."""
    groups = defaultdict(list)
    for ann in annotations:
        groups[ann["label"]].append(ann)
    return groups


def apply_cap(groups: dict, cap: int, min_count: int) -> dict:
    """
    - Descarta especies con < min_count muestras.
    - Submuestrea las que superan el cap.
    - El submuestreo respeta la diversidad de secuencias:
      primero selecciona secuencias únicas al azar y toma
      una imagen por secuencia hasta alcanzar el cap.
    """
    print(f"[3/6] Aplicando cap={cap} y mínimo={min_count} imágenes por especie...")
    rng = random.Random(RANDOM_SEED)
    balanced = {}

    for label, records in sorted(groups.items()):
        if len(records) < min_count:
            print(f"    ✗ {label:38s} {len(records):4d} imgs → descartada (< {min_count})")
            continue

        if len(records) <= cap:
            balanced[label] = records
            print(f"    ✓ {label:38s} {len(records):4d} imgs → se usan todas")
            continue

        # Agrupar por secuencia y tomar 1 frame por secuencia (menos leakage)
        seq_map = defaultdict(list)
        for r in records:
            seq_map[r["seq_id"]].append(r)

        seqs = list(seq_map.values())
        rng.shuffle(seqs)

        selected = []
        for seq in seqs:
            selected.append(rng.choice(seq))  # 1 frame por secuencia
            if len(selected) >= cap:
                break

        # Si hay pocas secuencias, completar con frames extra
        if len(selected) < cap:
            remaining = [r for r in records if r not in selected]
            rng.shuffle(remaining)
            selected += remaining[:cap - len(selected)]

        balanced[label] = selected[:cap]
        print(f"    ✓ {label:38s} {len(records):4d} imgs → cap a {len(balanced[label])}")

    return balanced


def split_by_sequence(balanced: dict, ratios: tuple) -> tuple:
    """
    Divide en train/val/test a nivel de seq_id para evitar data leakage.
    Cada secuencia va COMPLETA a un solo split.
    """
    print("[4/6] Generando splits por seq_id (sin data leakage)...")
    rng = random.Random(RANDOM_SEED)

    train_ids, val_ids, test_ids = set(), set(), set()

    for label, records in balanced.items():
        # Agrupar file_names por secuencia
        seq_to_files = defaultdict(list)
        for r in records:
            seq_to_files[r["seq_id"]].append(r["file_name"])

        seqs = list(seq_to_files.keys())
        rng.shuffle(seqs)

        n      = len(seqs)
        n_val  = max(1, round(n * ratios[1]))
        n_test = max(1, round(n * ratios[2]))

        val_seqs  = set(seqs[:n_val])
        test_seqs = set(seqs[n_val:n_val + n_test])
        train_seqs = set(seqs[n_val + n_test:])

        for seq_id, files in seq_to_files.items():
            if seq_id in val_seqs:
                val_ids.update(files)
            elif seq_id in test_seqs:
                test_ids.update(files)
            else:
                train_ids.update(files)

    total = len(train_ids) + len(val_ids) + len(test_ids)
    print(f"    train={len(train_ids)} | val={len(val_ids)} | test={len(test_ids)} | total={total}")
    return train_ids, val_ids, test_ids


def build_coco_output(balanced: dict, data: dict, img_by_id: dict) -> dict:
    """Construye un JSON COCO limpio con el subset balanceado."""
    used_labels = sorted(balanced.keys())
    label_to_new_id = {label: i for i, label in enumerate(used_labels)}

    new_categories = [
        {"id": i, "name": label, "supercategory": "animal"}
        for i, label in enumerate(used_labels)
    ]

    seen_img_ids = set()
    new_images, new_annotations = [], []
    ann_id_counter = 0

    for label, records in balanced.items():
        new_cat_id = label_to_new_id[label]
        for r in records:
            if r["image_id"] not in seen_img_ids:
                seen_img_ids.add(r["image_id"])
                new_images.append(img_by_id[r["image_id"]])
            new_annotations.append({
                "id"        : ann_id_counter,
                "image_id"  : r["image_id"],
                "category_id": new_cat_id,
                "sequence_level_annotation": False,
            })
            ann_id_counter += 1

    return {
        "info"       : data.get("info", {}),
        "images"     : new_images,
        "annotations": new_annotations,
        "categories" : new_categories,
    }


def save_outputs(out_dir: Path, balanced: dict, coco_out: dict,
                 train_ids, val_ids, test_ids):
    """Escribe todos los archivos de salida."""
    print("[5/6] Guardando archivos de salida...")
    out_dir.mkdir(parents=True, exist_ok=True)
    splits_dir = out_dir / "splits"
    splits_dir.mkdir(exist_ok=True)

    # subset_coco.json
    coco_path = out_dir / "subset_coco.json"
    with open(coco_path, "w", encoding="utf-8") as f:
        json.dump(coco_out, f, indent=2, ensure_ascii=False)
    print(f"    ✓ {coco_path}")

    # class_map.json
    used_labels = sorted(balanced.keys())
    class_map = {label: i for i, label in enumerate(used_labels)}
    cm_path = out_dir / "class_map.json"
    with open(cm_path, "w", encoding="utf-8") as f:
        json.dump(class_map, f, indent=2)
    print(f"    ✓ {cm_path}")

    # subset_manifest.csv
    csv_path = out_dir / "subset_manifest.csv"
    all_records = [r for recs in balanced.values() for r in recs]
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f, fieldnames=["file_name","label","location","seq_id","datetime","split"]
        )
        writer.writeheader()
        for r in all_records:
            split = ("train" if r["file_name"] in train_ids
                     else "val" if r["file_name"] in val_ids
                     else "test")
            writer.writerow({
                "file_name": r["file_name"],
                "label"    : r["label"],
                "location" : r["location"],
                "seq_id"   : r["seq_id"],
                "datetime" : r["datetime"],
                "split"    : split,
            })
    print(f"    ✓ {csv_path}")

    # splits/*.txt
    for name, ids in [("train", train_ids), ("val", val_ids), ("test", test_ids)]:
        p = splits_dir / f"{name}.txt"
        p.write_text("\n".join(sorted(ids)), encoding="utf-8")
        print(f"    ✓ {p}  ({len(ids)} imágenes)")


# ─────────────────────── Dataset PyTorch ──────────────────────────────

PYTORCH_DATASET_CODE = '''
# ──────────────────────────────────────────────────────────────────────
# dataset.py  –  PyTorch Dataset para el subset de Orinoquia
# ──────────────────────────────────────────────────────────────────────
#
# Uso rápido:
#   from dataset import OrinoquisSubsetDataset, get_loaders
#   train_loader, val_loader, test_loader, class_map = get_loaders(
#       imgs_root = "path/to/images/",
#       subset_dir= "path/to/output_subset/",
#   )
# ──────────────────────────────────────────────────────────────────────

import json
from pathlib import Path
from PIL import Image

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms


class OrinoquisSubsetDataset(Dataset):
    """
    Carga imágenes del subset balanceado de Orinoquia.

    Args:
        imgs_root  : carpeta raíz donde están las imágenes del dataset original
        subset_dir : carpeta generada por orinoquia_subset.py
        split      : "train" | "val" | "test"
        transform  : torchvision transform (None → usa el default según split)
    """

    # Estadísticas de ImageNet (adecuadas para fine-tuning de modelos preentrenados)
    MEAN = [0.485, 0.456, 0.406]
    STD  = [0.229, 0.224, 0.225]

    def __init__(self, imgs_root: str, subset_dir: str,
                 split: str = "train", transform=None):
        self.imgs_root  = Path(imgs_root)
        self.subset_dir = Path(subset_dir)
        self.split      = split

        # Cargar class_map
        with open(self.subset_dir / "class_map.json") as f:
            self.class_map = json.load(f)            # {"black_agouti": 0, ...}
        self.idx_to_class = {v: k for k, v in self.class_map.items()}
        self.num_classes  = len(self.class_map)

        # Cargar manifest y filtrar por split
        import csv
        self.samples = []   # lista de (file_name, label_idx)
        with open(self.subset_dir / "subset_manifest.csv", newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                if row["split"] == split:
                    label_idx = self.class_map[row["label"]]
                    self.samples.append((row["file_name"], label_idx))

        # Transform por defecto si no se pasa uno
        self.transform = transform or self._default_transform(split)

    def _default_transform(self, split: str):
        if split == "train":
            return transforms.Compose([
                transforms.Resize((256, 256)),
                transforms.RandomCrop(224),
                transforms.RandomHorizontalFlip(),
                transforms.ColorJitter(brightness=0.3, contrast=0.3,
                                       saturation=0.2, hue=0.05),
                transforms.ToTensor(),
                transforms.Normalize(self.MEAN, self.STD),
            ])
        else:   # val / test
            return transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(self.MEAN, self.STD),
            ])

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        file_name, label_idx = self.samples[idx]
        img_path = self.imgs_root / file_name

        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)

        return image, label_idx

    def class_weights(self) -> torch.Tensor:
        """
        Devuelve un tensor con los pesos inversos por clase,
        útil para nn.CrossEntropyLoss(weight=...) cuando hay desbalance.
        """
        counts = torch.zeros(self.num_classes)
        for _, label_idx in self.samples:
            counts[label_idx] += 1
        # peso = total / (num_clases * count_clase)
        weights = len(self.samples) / (self.num_classes * counts.clamp(min=1))
        return weights


def get_loaders(imgs_root: str, subset_dir: str,
                batch_size: int = 8,
                num_workers: int = 2) -> tuple:
    """
    Construye y devuelve (train_loader, val_loader, test_loader, class_map).

    batch_size=8  recomendado para tu PC (Ryzen 5 7520U, 16 GB RAM).
    """
    train_ds = OrinoquisSubsetDataset(imgs_root, subset_dir, split="train")
    val_ds   = OrinoquisSubsetDataset(imgs_root, subset_dir, split="val")
    test_ds  = OrinoquisSubsetDataset(imgs_root, subset_dir, split="test")

    common = dict(num_workers=num_workers, pin_memory=False)

    train_loader = DataLoader(train_ds, batch_size=batch_size,
                              shuffle=True,  **common)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size,
                              shuffle=False, **common)
    test_loader  = DataLoader(test_ds,  batch_size=batch_size,
                              shuffle=False, **common)

    print(f"Dataset listo — train={len(train_ds)} | val={len(val_ds)} | test={len(test_ds)}")
    print(f"Clases ({train_ds.num_classes}): {list(train_ds.class_map.keys())}")

    return train_loader, val_loader, test_loader, train_ds.class_map
'''


# ─────────────────────────── main ────────────────────────────────────

def main():
    global RANDOM_SEED
    # parser = argparse.ArgumentParser(description="Genera subset balanceado de Orinoquia")
    # parser.add_argument("C:\\Users\\devju\\Documents\\Maestria IA\\Gestión de Proyecto en IA y CD\\Trabajo Final\\data\\original_metadata\\", required=True, help="Ruta al JSON COCO original")
    # parser.add_argument("C:\\Users\\devju\\Documents\\Maestria IA\\Gestión de Proyecto en IA y CD\\Trabajo Final\\data\\raw", required=True, help="Carpeta raíz de imágenes")
    # parser.add_argument("C:\\Users\\devju\\Documents\\Maestria IA\\Gestión de Proyecto en IA y CD\\Trabajo Final\\data\\processed",  required=True, help="Carpeta de salida")
    # parser.add_argument("150",  type=int, default=CAP_PER_CLASS,
    #                     help=f"Cap por especie (default={CAP_PER_CLASS})")
    # parser.add_argument("30",  type=int, default=MIN_PER_CLASS,
    #                     help=f"Mínimo por especie (default={MIN_PER_CLASS})")
    # parser.add_argument("42", type=int, default=RANDOM_SEED)

    parser = argparse.ArgumentParser(description="Genera subset balanceado de Orinoquia")
    parser.add_argument("--json", default = "C:\\Users\\devju\\Documents\\Maestria IA\\Gestión de Proyecto en IA y CD\\Trabajo Final\\data\\original_metadata\\orinoquia_camera_traps.json", help="Ruta al JSON COCO original")
    parser.add_argument("--imgs", default = "C:\\Users\\devju\\Documents\\Maestria IA\\Gestión de Proyecto en IA y CD\\Trabajo Final\\data\\raw", help="Carpeta raíz de imágenes")
    parser.add_argument("--out",  default = "C:\\Users\\devju\\Documents\\Maestria IA\\Gestión de Proyecto en IA y CD\\Trabajo Final\\data\\processed", help="Carpeta de salida")
    parser.add_argument("--cap",  type=int, default=CAP_PER_CLASS,
                        help=f"Cap por especie (default={CAP_PER_CLASS})")
    parser.add_argument("--min",  type=int, default=MIN_PER_CLASS,
                        help=f"Mínimo por especie (default={MIN_PER_CLASS})")
    parser.add_argument("--seed", type=int, default=RANDOM_SEED)
    args = parser.parse_args()


    out_dir = Path(args.out)
    RANDOM_SEED = args.seed

    # Pipeline
    data                        = load_json(args.json)
    cat_by_id, _, img_by_id     = build_lookups(data)
    annotations                 = filter_annotations(data, cat_by_id, img_by_id)
    groups                      = group_by_species(annotations)
    balanced                    = apply_cap(groups, args.cap, args.min)
    train_ids, val_ids, test_ids = split_by_sequence(balanced, SPLIT_RATIOS)
    coco_out                    = build_coco_output(balanced, data, img_by_id)
    save_outputs(out_dir, balanced, coco_out, train_ids, val_ids, test_ids)

    # Guardar dataset.py junto al subset
    dataset_py = out_dir / "dataset.py"
    dataset_py.write_text(PYTORCH_DATASET_CODE, encoding="utf-8")
    print(f"    ✓ {dataset_py}")

    # Resumen final
    total = sum(len(v) for v in balanced.values())
    print(f"\n[6/6] ✅ Subset listo en: {out_dir}")
    print(f"      Especies: {len(balanced)}  |  Imágenes totales: {total}")
    print(f"\nPróximo paso:")
    print(f"  from dataset import get_loaders")
    print(f'  train_loader, val_loader, test_loader, class_map = get_loaders(')
    print(f'      imgs_root = "{args.imgs}",')
    print(f'      subset_dir= "{args.out}",')
    print(f'  )')


if __name__ == "__main__":
    main()