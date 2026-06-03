
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
