"""
experimentos.py
================
Script completo de entrenamiento y experimentación para el subset
de Orinoquia Camera Traps (A06 + N27, 11 especies).

Modelos soportados:
  - EfficientNet-B0   (~5 M params, recomendado)
  - MobileNet V3 Small (~2 M params, el más rápido)
  - ResNet-18          (~11 M params, línea base clásica)

Optimizado para CPU (Ryzen 5 7520U, 16 GB RAM):
  - batch_size = 8
  - Solo se afina la cabeza clasificadora (freeze del backbone)
  - Checkpoint automático por mejor val_accuracy

Uso:
    python experimentos.py \
        --imgs    ruta/a/imagenes/ \
        --subset  output_subset/ \
        --epochs  15 \
        --out     resultados/

Salidas en --out/:
    resultados.json            ← tabla resumen de todos los experimentos
    <nombre_experimento>.pt    ← checkpoint del mejor modelo de cada run
    <nombre_experimento>.csv   ← curvas loss/acc por época
"""

import json
import csv
import time
import argparse
from pathlib import Path
from copy import deepcopy

import torch
import torch.nn as nn
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR
import torchvision.models as tv_models

# dataset.py debe estar en output_subset/ o en el mismo directorio
import sys


# ═══════════════════════════════════════════════════════
#  CONFIGURACIÓN DE EXPERIMENTOS
#  Edita esta lista para añadir/quitar combinaciones
# ═══════════════════════════════════════════════════════

EXPERIMENTOS = [
    {
        "nombre":  "efficientnet_b0_lr1e3",
        "modelo":  "efficientnet_b0",
        "lr":      1e-3,
        "freeze":  True,    # True = solo afinar cabeza (más rápido)
    },
    # {
    #     "nombre":  "efficientnet_b0_lr1e4",
    #     "modelo":  "efficientnet_b0",
    #     "lr":      1e-4,
    #     "freeze":  True,
    # },
    # {
    #     "nombre":  "mobilenet_v3_small_lr1e3",
    #     "modelo":  "mobilenet_v3_small",
    #     "lr":      1e-3,
    #     "freeze":  True,
    # },
    # {
    #     "nombre":  "mobilenet_v3_small_lr1e4",
    #     "modelo":  "mobilenet_v3_small",
    #     "lr":      1e-4,
    #     "freeze":  True,
    # },
    # {
    #     "nombre":  "resnet18_lr1e3",
    #     "modelo":  "resnet18",
    #     "lr":      1e-3,
    #     "freeze":  True,
    # },
    # {
    #     "nombre":  "resnet18_full_lr1e4",  # sin freeze: afina todo
    #     "modelo":  "resnet18",
    #     "lr":      1e-4,
    #     "freeze":  False,
    # },
]


# ═══════════════════════════════════════════════════════
#  CREAR MODELO
# ═══════════════════════════════════════════════════════

def crear_modelo(nombre: str, num_clases: int, freeze_backbone: bool) -> nn.Module:
    """
    Instancia un modelo preentrenado en ImageNet y reemplaza su
    cabeza clasificadora por una de `num_clases` salidas.

    Recibe:
        nombre          : "efficientnet_b0" | "mobilenet_v3_small" | "resnet18"
        num_clases      : número de especies (11 en el subset A06+N27)
        freeze_backbone : si True, congela todos los pesos excepto la cabeza.
                          Hace el entrenamiento ~3x más rápido en CPU.

    Devuelve:
        nn.Module listo para entrenar con torch.optim
    """

    # ── EfficientNet-B0 ──────────────────────────────────────────────
    if nombre == "efficientnet_b0":
        model = tv_models.efficientnet_b0(weights=tv_models.EfficientNet_B0_Weights.DEFAULT)

        if freeze_backbone:
            for param in model.parameters():
                param.requires_grad = False

        # La cabeza de EfficientNet está en model.classifier
        # classifier[1] es el Linear original → lo reemplazamos
        in_features = model.classifier[1].in_features  # 1280
        model.classifier[1] = nn.Linear(in_features, num_clases)
        # La nueva capa tiene requires_grad=True por defecto

    # ── MobileNet V3 Small ───────────────────────────────────────────
    elif nombre == "mobilenet_v3_small":
        model = tv_models.mobilenet_v3_small(weights=tv_models.MobileNet_V3_Small_Weights.DEFAULT)

        if freeze_backbone:
            for param in model.parameters():
                param.requires_grad = False

        # La cabeza de MobileNetV3 está en model.classifier
        # classifier[-1] es el Linear original
        in_features = model.classifier[-1].in_features  # 1024
        model.classifier[-1] = nn.Linear(in_features, num_clases)

    # ── ResNet-18 ────────────────────────────────────────────────────
    elif nombre == "resnet18":
        model = tv_models.resnet18(weights=tv_models.ResNet18_Weights.DEFAULT)

        if freeze_backbone:
            for param in model.parameters():
                param.requires_grad = False

        # La cabeza de ResNet está en model.fc
        in_features = model.fc.in_features  # 512
        model.fc = nn.Linear(in_features, num_clases)

    else:
        raise ValueError(
            f"Modelo '{nombre}' no soportado. "
            f"Opciones: efficientnet_b0, mobilenet_v3_small, resnet18"
        )

    # Resumen de parámetros entrenables
    total   = sum(p.numel() for p in model.parameters())
    activos = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"   Parámetros totales:      {total:>10,}")
    print(f"   Parámetros entrenables:  {activos:>10,}  "
          f"({'solo cabeza' if freeze_backbone else 'todos'})")

    return model


# ═══════════════════════════════════════════════════════
#  FUNCIÓN DE ENTRENAMIENTO
# ═══════════════════════════════════════════════════════

def entrenar(
    model:        nn.Module,
    train_loader,
    val_loader,
    class_weights: torch.Tensor,
    lr:            float,
    epochs:        int,
    out_dir:       Path,
    nombre:        str,
) -> dict:
    """
    Entrena un modelo por `epochs` épocas y devuelve el resumen
    del mejor checkpoint (por val_accuracy).

    Recibe:
        model          : modelo creado por crear_modelo()
        train_loader   : DataLoader de entrenamiento
        val_loader     : DataLoader de validación
        class_weights  : tensor [num_clases] de dataset.class_weights()
        lr             : learning rate inicial
        epochs         : número de épocas
        out_dir        : carpeta donde guardar .pt y .csv
        nombre         : prefijo para los archivos de salida

    Devuelve:
        dict con claves: mejor_val_acc, mejor_epoca, tiempo_total_min,
                         historia (lista de dicts por época)
    """

    device = torch.device("cpu")   # tu PC no tiene GPU dedicada
    model = model.to(device)

    # Loss con pesos por clase para compensar desbalance residual
    criterion = nn.CrossEntropyLoss(weight=class_weights.to(device))

    # Solo optimizamos los parámetros con requires_grad=True
    optimizer = Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=lr,
    )

    # Learning rate se reduce suavemente hacia 0 al final
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs, eta_min=lr * 0.01)

    mejor_val_acc  = 0.0
    mejor_epoca    = 0
    mejor_pesos    = None
    historia       = []
    t0             = time.time()

    print(f"\n   {'Época':>5}  {'Loss train':>10}  "
          f"{'Acc train':>9}  {'Loss val':>8}  {'Acc val':>8}  {'Tiempo':>6}")
    print("   " + "─" * 58)

    for epoch in range(1, epochs + 1):
        t_ep = time.time()

        # ── Entrenamiento ──────────────────────────────────────────
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0

        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)            # [batch, num_clases]
            loss    = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            train_loss    += loss.item() * images.size(0)
            preds          = outputs.argmax(dim=1)
            train_correct += (preds == labels).sum().item()
            train_total   += images.size(0)

        scheduler.step()

        avg_train_loss = train_loss / train_total
        avg_train_acc  = train_correct / train_total

        # ── Validación ─────────────────────────────────────────────
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0

        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs  = model(images)
                loss     = criterion(outputs, labels)

                val_loss    += loss.item() * images.size(0)
                preds        = outputs.argmax(dim=1)
                val_correct += (preds == labels).sum().item()
                val_total   += images.size(0)

        avg_val_loss = val_loss / val_total
        avg_val_acc  = val_correct / val_total
        elapsed      = time.time() - t_ep

        # Guardar mejor modelo
        if avg_val_acc > mejor_val_acc:
            mejor_val_acc  = avg_val_acc
            mejor_epoca    = epoch
            mejor_pesos    = deepcopy(model.state_dict())
            marca = " ← mejor"
        else:
            marca = ""

        print(f"   {epoch:>5}  {avg_train_loss:>10.4f}  "
              f"{avg_train_acc:>8.1%}  {avg_val_loss:>8.4f}  "
              f"{avg_val_acc:>7.1%}{marca}  {elapsed:>4.0f}s")

        historia.append({
            "epoca":      epoch,
            "train_loss": round(avg_train_loss, 4),
            "train_acc":  round(avg_train_acc,  4),
            "val_loss":   round(avg_val_loss,   4),
            "val_acc":    round(avg_val_acc,    4),
        })

    # ── Guardar checkpoint del mejor modelo ────────────────────────
    ckpt_path = out_dir / f"{nombre}.pt"
    torch.save(mejor_pesos, ckpt_path)
    print(f"\n   Checkpoint guardado → {ckpt_path}  "
          f"(época {mejor_epoca}, val_acc={mejor_val_acc:.1%})")

    # ── Guardar curvas por época en CSV ────────────────────────────
    csv_path = out_dir / f"{nombre}.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(
            f, fieldnames=["epoca","train_loss","train_acc","val_loss","val_acc"]
        )
        writer.writeheader()
        writer.writerows(historia)

    tiempo_total = (time.time() - t0) / 60

    return {
        "mejor_val_acc":   round(mejor_val_acc,  4),
        "mejor_epoca":     mejor_epoca,
        "tiempo_total_min": round(tiempo_total,  1),
        "historia":         historia,
    }


# ═══════════════════════════════════════════════════════
#  EVALUACIÓN EN TEST (después de seleccionar el ganador)
# ═══════════════════════════════════════════════════════

def evaluar_test(
    model:       nn.Module,
    ckpt_path:   Path,
    test_loader,
    class_map:   dict,
) -> dict:
    """
    Carga el mejor checkpoint y evalúa en el split de test.

    Devuelve:
        dict con test_accuracy y accuracy por especie
    """
    model.load_state_dict(torch.load(ckpt_path, map_location="cpu"))
    model.eval()

    idx_to_class = {v: k for k, v in class_map.items()}
    correct_per_class = {k: 0 for k in class_map}
    total_per_class   = {k: 0 for k in class_map}
    total_correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in test_loader:
            outputs = model(images)
            preds   = outputs.argmax(dim=1)

            for pred, label in zip(preds.tolist(), labels.tolist()):
                clase = idx_to_class[label]
                total_per_class[clase]   += 1
                if pred == label:
                    correct_per_class[clase] += 1
                    total_correct += 1
                total += 1

    acc_por_clase = {
        clase: round(correct_per_class[clase] / max(total_per_class[clase], 1), 4)
        for clase in class_map
    }

    print(f"\n   Test accuracy: {total_correct/total:.1%}")
    print("   Por especie:")
    for clase, acc in sorted(acc_por_clase.items(), key=lambda x: -x[1]):
        bar = "█" * int(acc * 20)
        print(f"     {clase:35s} {acc:5.1%}  {bar}")

    return {
        "test_accuracy":  round(total_correct / total, 4),
        "acc_por_clase":  acc_por_clase,
    }


# ═══════════════════════════════════════════════════════
#  MAIN — orquesta todos los experimentos
# ═══════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="Experimentos de clasificación de especies — Orinoquia"
    )
    # parser.add_argument("--imgs",   required=True,
    #                     help="Carpeta raíz de imágenes del dataset")
    # parser.add_argument("--subset", required=True,
    #                     help="Carpeta output_subset/ generada por orinoquia_subset.py")
    # parser.add_argument("--epochs", type=int, default=15,
    #                     help="Número de épocas por experimento (default: 15)")
    # parser.add_argument("--out",    default="resultados",
    #                     help="Carpeta de salida para checkpoints y CSVs")
    parser.add_argument("--imgs",   default="C:\\Users\\devju\\Documents\\Maestria IA\\Gestión de Proyecto en IA y CD\\Trabajo Final\\data\\raw", help="Carpeta raíz de imágenes del dataset")
    parser.add_argument("--subset", default="C:\\Users\\devju\\Documents\\Maestria IA\\Gestión de Proyecto en IA y CD\\Trabajo Final\\data\\processed", help="Carpeta output_subset/ generada por orinoquia_subset.py")
    parser.add_argument("--epochs", type=int, default=15, help="Número de épocas por experimento (default: 15)")
    parser.add_argument("--out",    default="C:\\Users\\devju\\Documents\\Maestria IA\\Gestión de Proyecto en IA y CD\\Trabajo Final\\reports", help="Carpeta de salida para checkpoints y CSVs")
    args = parser.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Importar dataset.py desde la carpeta del subset
    sys.path.insert(0, args.subset)
    from dataset import get_loaders, OrinoquisSubsetDataset

    # ── Cargar datos UNA sola vez ──────────────────────────────────
    print("Cargando dataset...")
    train_loader, val_loader, test_loader, class_map = get_loaders(
        imgs_root  = args.imgs,
        subset_dir = args.subset,
        batch_size = 8,
        num_workers= 2,
    )
    num_clases = len(class_map)

    # Calcular class_weights desde el split de entrenamiento
    train_ds = OrinoquisSubsetDataset(args.imgs, args.subset, split="train")
    pesos    = train_ds.class_weights()
    print(f"Class weights: {pesos.tolist()}")

    # ── Loop de experimentos ───────────────────────────────────────
    resumen = []

    for i, cfg in enumerate(EXPERIMENTOS, 1):
        print(f"\n{'═'*60}")
        print(f"Experimento {i}/{len(EXPERIMENTOS)}: {cfg['nombre']}")
        print(f"  modelo={cfg['modelo']}  lr={cfg['lr']}  freeze={cfg['freeze']}")
        print(f"{'═'*60}")

        model = crear_modelo(cfg["modelo"], num_clases, cfg["freeze"])

        resultado = entrenar(
            model        = model,
            train_loader = train_loader,
            val_loader   = val_loader,
            class_weights= pesos,
            lr           = cfg["lr"],
            epochs       = args.epochs,
            out_dir      = out_dir,
            nombre       = cfg["nombre"],
        )

        resumen.append({
            "nombre":          cfg["nombre"],
            "modelo":          cfg["modelo"],
            "lr":              cfg["lr"],
            "freeze":          cfg["freeze"],
            "mejor_val_acc":   resultado["mejor_val_acc"],
            "mejor_epoca":     resultado["mejor_epoca"],
            "tiempo_min":      resultado["tiempo_total_min"],
        })

    # ── Tabla resumen ──────────────────────────────────────────────
    print(f"\n{'═'*60}")
    print("RESUMEN DE EXPERIMENTOS")
    print(f"{'═'*60}")
    resumen_sorted = sorted(resumen, key=lambda x: -x["mejor_val_acc"])

    print(f"\n  {'Experimento':<35} {'Val acc':>7}  {'Época':>5}  {'Tiempo':>7}")
    print("  " + "─" * 58)
    for r in resumen_sorted:
        marca = " ← ganador" if r == resumen_sorted[0] else ""
        print(f"  {r['nombre']:<35} {r['mejor_val_acc']:>6.1%}  "
              f"{r['mejor_epoca']:>5}  {r['tiempo_min']:>5.1f} min{marca}")

    # Guardar JSON completo
    json_path = out_dir / "resultados.json"
    with open(json_path, "w") as f:
        json.dump(resumen_sorted, f, indent=2)
    print(f"\nResultados guardados en {json_path}")

    # ── Evaluación final en test con el modelo ganador ─────────────
    ganador = resumen_sorted[0]
    print(f"\n{'═'*60}")
    print(f"EVALUACIÓN EN TEST — {ganador['nombre']}")
    print(f"{'═'*60}")

    model_final = crear_modelo(ganador["modelo"], num_clases, freeze_backbone=False)
    ckpt_final  = out_dir / f"{ganador['nombre']}.pt"

    test_resultado = evaluar_test(
        model_final, ckpt_final, test_loader, class_map
    )

    # Añadir resultados de test al JSON
    resumen_sorted[0]["test_accuracy"]  = test_resultado["test_accuracy"]
    resumen_sorted[0]["acc_por_clase"]  = test_resultado["acc_por_clase"]
    with open(json_path, "w") as f:
        json.dump(resumen_sorted, f, indent=2)

    print(f"\nListo. Todos los resultados en: {out_dir}/")


if __name__ == "__main__":
    main()
