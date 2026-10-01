#!/usr/bin/env python3
"""
evaluar_mask2former.py
Mask2Former sobre imagenes propias, con distintos juegos de pesos.

Misma arquitectura en todos los casos (Cheng et al., CVPR 2022); lo unico
que cambia son los pesos. Sirve para el experimento de transferencia de
dominio del informe.

Uso:
    python evaluar_mask2former.py img1.jpg img2.jpg
    python evaluar_mask2former.py img1.jpg --modelo facebook/mask2former-swin-large-mapillary-vistas-semantic

Checkpoints utiles (todos cargan igual, sin detectron2):
    facebook/mask2former-swin-tiny-ade-semantic                 150 clases de escena, 47 M
    facebook/mask2former-swin-large-ade-semantic                las mismas clases, ~200 M
    facebook/mask2former-swin-large-mapillary-vistas-semantic    65 clases de exteriores
    facebook/mask2former-swin-tiny-cityscapes-semantic           19 clases urbanas
    facebook/mask2former-swin-tiny-coco-instance                 80 clases COCO
"""

import sys
from pathlib import Path

import numpy as np
import torch
import matplotlib.pyplot as plt
from PIL import Image
from transformers import AutoImageProcessor, Mask2FormerForUniversalSegmentation


POR_DEFECTO = "facebook/mask2former-swin-tiny-ade-semantic"
LADO_MAX = 1024        # las fotos de dron son enormes; en CPU hay que reducir
MIN_PCT = 0.5          # ignorar clases residuales en la tabla


def leer_argumentos():
    """Separa la opcion --modelo de la lista de imagenes."""
    args = sys.argv[1:]
    modelo = POR_DEFECTO

    if "--modelo" in args:
        i = args.index("--modelo")
        if i + 1 >= len(args):
            sys.exit("Falta el nombre del modelo despues de --modelo")
        modelo = args[i + 1]
        del args[i:i + 2]

    if not args:
        sys.exit("Uso: python evaluar_mask2former.py <imagen> [...] [--modelo <id>]")

    return modelo, args


def segmentar(proc, model, ruta):
    img = Image.open(ruta).convert("RGB")
    original = img.size
    img.thumbnail((LADO_MAX, LADO_MAX))

    inputs = proc(images=img, return_tensors="pt")
    with torch.no_grad():
        out = model(**inputs)

    sem = proc.post_process_semantic_segmentation(
        out, target_sizes=[img.size[::-1]]
    )[0].numpy()

    return img, sem, original


def tabla(sem, id2label):
    filas = []
    for cid in np.unique(sem):
        pct = (sem == cid).sum() / sem.size * 100
        if pct >= MIN_PCT:
            filas.append((id2label[cid], pct))
    return sorted(filas, key=lambda x: -x[1])


def figura(img, sem, filas, ruta, destino, modelo):
    clases = np.unique(sem)
    mapa = plt.get_cmap("tab20")
    capa = np.zeros((*sem.shape, 3))
    for i, c in enumerate(clases):
        capa[sem == c] = mapa(i % 20)[:3]

    fig, ax = plt.subplots(1, 3, figsize=(21, 7))
    ax[0].imshow(img);  ax[0].set_title("Original")
    ax[1].imshow(capa); ax[1].set_title("Segmentacion semantica")
    ax[2].imshow(img);  ax[2].imshow(capa, alpha=0.55); ax[2].set_title("Superpuesto")
    for a in ax:
        a.axis("off")

    etiquetas = "   ".join(f"{n} {p:.0f}%" for n, p in filas[:6])
    fig.suptitle(f"{ruta.name}  |  {modelo.split('/')[-1]}\n{etiquetas}", fontsize=11)
    plt.tight_layout()
    plt.savefig(destino, dpi=120, bbox_inches="tight")
    plt.close()


def main():
    modelo, imagenes = leer_argumentos()
    sufijo = modelo.split("/")[-1].replace("mask2former-", "")

    print(f"Cargando {modelo} ...", flush=True)
    proc = AutoImageProcessor.from_pretrained(modelo)
    model = Mask2FormerForUniversalSegmentation.from_pretrained(modelo).eval()
    id2label = model.config.id2label

    print(f"\nClases   : {len(id2label)}")
    print(f"Queries  : {model.config.num_queries}")
    print(f"Params   : {sum(p.numel() for p in model.parameters())/1e6:.1f} M")

    resumen = {}

    for arg in imagenes:
        ruta = Path(arg)
        if not ruta.exists():
            print(f"\n[omitida] no existe: {ruta}")
            continue

        print(f"\n{'=' * 60}\n{ruta.name}")
        img, sem, original = segmentar(proc, model, ruta)
        print(f"  {original[0]}x{original[1]} px -> {img.size[0]}x{img.size[1]} px")

        filas = tabla(sem, id2label)
        for nombre, pct in filas:
            print(f"    {nombre:<25} {pct:5.1f} %")
        resumen[ruta.name] = filas

        destino = ruta.with_name(f"{ruta.stem}_{sufijo}.png")
        figura(img, sem, filas, ruta, destino, modelo)
        print(f"  figura: {destino.name}")

    if len(resumen) > 1:
        todas = sorted({n for filas in resumen.values() for n, _ in filas})
        ancho = max(len(n) for n in todas) + 2

        print(f"\n{'=' * 60}\nCOMPARATIVA con {sufijo} (% de pixeles)\n")
        cab = "clase".ljust(ancho) + "".join(f"{k[:14]:>16}" for k in resumen)
        print(cab)
        print("-" * len(cab))
        for clase in todas:
            fila = clase.ljust(ancho)
            for filas in resumen.values():
                v = dict(filas).get(clase)
                fila += f"{v:>15.1f}%" if v else f"{'-':>16}"
            print(fila)


if __name__ == "__main__":
    main()
