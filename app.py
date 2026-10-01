#!/usr/bin/env python3
"""
app.py — Demostracion interactiva de Mask2Former
Trabajo final, Procesamiento de Datos Secuenciales

Arquitectura: Cheng et al., "Masked-attention Mask Transformer for Universal
Image Segmentation", CVPR 2022 (arXiv 2112.01527)

Ejecutar:
    streamlit run app.py
"""

import io
import json
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import streamlit as st
import torch
from PIL import Image
from transformers import AutoImageProcessor, Mask2FormerForUniversalSegmentation


# --------------------------------------------------------------------------
# Juegos de pesos disponibles. Todos comparten la MISMA arquitectura:
# lo unico que cambia son los datos con los que se afinaron.
# --------------------------------------------------------------------------
MODELOS = {
    "ADE20K tiny (150 clases de escena, 47 M)":
        "facebook/mask2former-swin-tiny-ade-semantic",
    "ADE20K large (150 clases, 215 M)":
        "facebook/mask2former-swin-large-ade-semantic",
    "Mapillary Vistas large (65 clases de exteriores, 215 M)":
        "facebook/mask2former-swin-large-mapillary-vistas-semantic",
    "Cityscapes tiny (19 clases urbanas, 47 M)":
        "facebook/mask2former-swin-tiny-cityscapes-semantic",
    "COCO tiny (80 clases cotidianas, 47 M)":
        "facebook/mask2former-swin-tiny-coco-instance",
}

LADO_MAX = 1024

# Resultados de PhenoBench exportados desde Colab (seccion 11 del notebook).
# Los pesos de PhenoBench no corren localmente: dependen de Detectron2 y de
# un kernel CUDA compilado. Por eso se ejecutan en Colab y aqui solo se leen.
CARPETA_PB = Path(__file__).parent / "resultados" / "phenobench"

# Reemplazar <usuario> por el usuario de GitHub para que el boton abra el notebook.
COLAB_URL = ("https://colab.research.google.com/github/<usuario>/"
             "mask2former-agricola/blob/main/notebooks/phenobench_mask2former_colab.ipynb")


st.set_page_config(page_title="Mask2Former", layout="wide")


def pagina_phenobench():
    """Muestra los resultados de PhenoBench calculados en Colab."""
    st.title("PhenoBench: resultados precalculados en Colab")
    st.caption(
        "Mask2Former con backbone ResNet-50 y pesos de la Universidad de Bonn, "
        "entrenados en campos de remolacha azucarera. 3 clases: suelo, cultivo y maleza."
    )
    st.info(
        "Estos resultados NO se calculan en este equipo. El checkpoint de PhenoBench "
        "requiere Detectron2 y un kernel CUDA compilado, que no se pueden instalar sin "
        "permisos de administrador. La inferencia se hizo en Google Colab (GPU T4) y "
        "aqui se muestran sus salidas tal cual."
    )

    if "<usuario>" not in COLAB_URL:
        st.link_button("Abrir la inferencia en vivo en Colab", COLAB_URL)

    resumen_json = CARPETA_PB / "resumen.json"
    if not resumen_json.exists():
        st.warning(
            f"No se encontro {resumen_json}. Ejecuta la seccion 11 del notebook de Colab, "
            "descarga el zip y descomprimelo en resultados/phenobench/."
        )
        return

    resumen = json.loads(resumen_json.read_text(encoding="utf-8"))
    nombre = st.selectbox("Imagen", list(resumen))
    datos = resumen[nombre]

    figura = CARPETA_PB / datos["archivo_figura"]
    if figura.exists():
        st.image(str(figura), use_container_width=True,
                 caption="Original | Segmentacion semantica | Instancias sobre el umbral")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Instancias sobre el umbral", datos["instancias_sobre_umbral"])
    c2.metric("Queries descartadas", 100 - datos["instancias_sobre_umbral"])
    c3.metric("Score medio",
              "-" if datos["score_medio"] is None else f'{datos["score_medio"]:.3f}')
    c4.metric("Tiempo en GPU T4", f'{datos["tiempo_s"]} s')

    st.subheader("Cobertura por clase")
    st.dataframe(
        [{"clase": k, "porcentaje": v} for k, v in datos["cobertura_pct"].items()],
        use_container_width=True, hide_index=True,
    )
    if datos["por_clase"]:
        st.subheader("Instancias por clase")
        st.dataframe(
            [{"clase": k, "instancias": v} for k, v in datos["por_clase"].items()],
            use_container_width=True, hide_index=True,
        )

    st.subheader("Comparacion de todas las imagenes")
    st.dataframe(
        [{"imagen": n,
          "suelo %": d["cobertura_pct"].get("suelo"),
          "cultivo %": d["cobertura_pct"].get("cultivo"),
          "maleza %": d["cobertura_pct"].get("maleza"),
          "instancias": d["instancias_sobre_umbral"],
          "score medio": d["score_medio"]}
         for n, d in resumen.items()],
        use_container_width=True, hide_index=True,
    )


@st.cache_resource(show_spinner=False)
def cargar_modelo(nombre):
    proc = AutoImageProcessor.from_pretrained(nombre)
    model = Mask2FormerForUniversalSegmentation.from_pretrained(nombre).eval()
    return proc, model


def colorear(mapa):
    """Un color distinto por clase presente."""
    clases = np.unique(mapa)
    cmap = plt.get_cmap("tab20")
    capa = np.zeros((*mapa.shape, 3))
    for i, c in enumerate(clases):
        if c >= 0:
            capa[mapa == c] = cmap(i % 20)[:3]
    return capa


def superponer(img, capa, alpha=0.55):
    base = np.asarray(img, dtype=float) / 255.0
    return np.clip(base * (1 - alpha) + capa * alpha, 0, 1)


def figura_atencion(pesos, titulo):
    """pesos: [num_queries, H, W] ya promediados sobre cabezas."""
    fig, ax = plt.subplots(figsize=(5, 4))
    im = ax.imshow(pesos, cmap="inferno")
    ax.set_title(titulo, fontsize=10)
    ax.axis("off")
    fig.colorbar(im, ax=ax, fraction=0.045)
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=110, bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)
    return buf


# --------------------------------------------------------------------------
# Barra lateral
# --------------------------------------------------------------------------
st.sidebar.title("Configuracion")

modo = st.sidebar.radio(
    "Modo",
    ["Inferencia local", "PhenoBench (precalculado en Colab)"],
)
if modo.startswith("PhenoBench"):
    pagina_phenobench()
    st.stop()

etiqueta = st.sidebar.selectbox("Juego de pesos", list(MODELOS))
nombre_modelo = MODELOS[etiqueta]

archivo = st.sidebar.file_uploader(
    "Imagen", type=["jpg", "jpeg", "png", "tif", "tiff"]
)

lado = st.sidebar.slider("Lado maximo (px)", 384, 1536, LADO_MAX, 128)
alpha = st.sidebar.slider("Opacidad de la mascara", 0.0, 1.0, 0.55, 0.05)
umbral_pct = st.sidebar.slider("Ocultar clases bajo (%)", 0.0, 5.0, 0.5, 0.1)

st.sidebar.markdown("---")
st.sidebar.caption(
    "Todos los juegos de pesos usan la misma arquitectura Mask2Former. "
    "Cambiar de uno a otro permite observar el efecto del dominio de "
    "entrenamiento sobre el resultado."
)


# --------------------------------------------------------------------------
# Cuerpo
# --------------------------------------------------------------------------
st.title("Mask2Former: segmentacion universal por queries")
st.caption(
    "Cheng et al., *Masked-attention Mask Transformer for Universal Image "
    "Segmentation*, CVPR 2022"
)

if archivo is None:
    st.info("Carga una imagen en el panel de la izquierda para empezar.")
    st.stop()

with st.spinner(f"Cargando {etiqueta}..."):
    proc, model = cargar_modelo(nombre_modelo)

cfg = model.config
id2label = cfg.id2label
es_instancia = nombre_modelo.endswith(("-instance", "-panoptic"))

img = Image.open(archivo).convert("RGB")
tam_original = img.size
img.thumbnail((lado, lado))

inputs = proc(images=img, return_tensors="pt")

with st.spinner("Ejecutando inferencia..."):
    with torch.no_grad():
        try:
            out = model(**inputs, output_attentions=True)
        except TypeError:
            out = model(**inputs)

tab1, tab2, tab3, tab4 = st.tabs(
    ["Segmentacion", "Queries", "Atencion cruzada", "Arquitectura"]
)


# --------------------------------------------------------------------------
with tab1:
    if es_instancia:
        res = proc.post_process_instance_segmentation(
            out, target_sizes=[img.size[::-1]], threshold=0.8
        )[0]
        mapa = res["segmentation"].numpy()
        segmentos = res.get("segments_info", [])
    else:
        mapa = proc.post_process_semantic_segmentation(
            out, target_sizes=[img.size[::-1]]
        )[0].numpy()
        segmentos = None

    capa = colorear(mapa)

    c1, c2, c3 = st.columns(3)
    c1.image(img, caption=f"Original ({tam_original[0]}x{tam_original[1]} px)",
             use_container_width=True)
    c2.image(capa, caption="Segmentacion", use_container_width=True)
    c3.image(superponer(img, capa, alpha), caption="Superpuesto",
             use_container_width=True)

    st.subheader("Cobertura por clase")

    if es_instancia:
        if segmentos:
            filas = [
                {"instancia": s["id"],
                 "clase": id2label.get(s["label_id"], str(s["label_id"])),
                 "score": round(float(s["score"]), 3)}
                for s in segmentos
            ]
            st.dataframe(filas, use_container_width=True, hide_index=True)
        else:
            st.warning(
                "Ninguna instancia supero el umbral. Con imagenes fuera del "
                "dominio de entrenamiento es un resultado esperable, no un error."
            )
    else:
        filas = []
        for cid in np.unique(mapa):
            pct = (mapa == cid).sum() / mapa.size * 100
            if pct >= umbral_pct:
                filas.append({"clase": id2label.get(int(cid), str(cid)),
                              "porcentaje": round(float(pct), 1)})
        filas.sort(key=lambda r: -r["porcentaje"])
        st.dataframe(filas, use_container_width=True, hide_index=True)


# --------------------------------------------------------------------------
with tab2:
    st.markdown(
        "El decoder emite **siempre** el mismo numero de predicciones, haya "
        "tres objetos o cincuenta. Cada query produce su propia mascara y su "
        "propia distribucion de clases; las que no encuentran nada votan por "
        "la clase \"sin objeto\". Por eso el modelo no necesita NMS."
    )

    masks = out.masks_queries_logits[0]          # [num_queries, h, w]
    clases = out.class_queries_logits[0]         # [num_queries, num_clases + 1]
    probs = clases.softmax(-1)
    sin_objeto = probs[:, -1]

    n_activas = int((sin_objeto < 0.5).sum())
    c1, c2, c3 = st.columns(3)
    c1.metric("Queries totales", masks.shape[0])
    c2.metric("Con objeto asignado", n_activas)
    c3.metric("Vacias", masks.shape[0] - n_activas)

    # Las queries mas seguras de haber encontrado algo
    orden = torch.argsort(sin_objeto)[:6]
    st.markdown("**Las seis queries mas activas**")
    cols = st.columns(6)
    for col, q in zip(cols, orden):
        m = masks[q].sigmoid().numpy()
        etq = id2label.get(int(probs[q, :-1].argmax()), "?")
        col.image(m, clamp=True, use_container_width=True,
                  caption=f"q{int(q)} · {etq}")


# --------------------------------------------------------------------------
with tab3:
    st.markdown(
        "En la atencion cruzada del decoder, las **queries** aportan Q y las "
        "caracteristicas de pixel del encoder aportan K y V. La atencion "
        "enmascarada restringe cada query a mirar solo la region de su "
        "prediccion anterior, en lugar de toda la imagen."
    )

    atenciones = getattr(out, "attentions", None) or \
        getattr(out, "transformer_decoder_cross_attentions", None)

    if not atenciones:
        st.warning(
            "Esta version de transformers no expuso los pesos de atencion. "
            "Se pueden capturar con un hook sobre el modulo de cross-attention "
            "del decoder."
        )
    else:
        capa_idx = st.slider("Capa del decoder", 0, len(atenciones) - 1, 0)
        a = atenciones[capa_idx]
        if a is None:
            st.warning("La capa seleccionada no tiene pesos disponibles.")
        else:
            # [batch, cabezas, queries, posiciones]
            a = a[0].mean(0)                     # promedio sobre cabezas
            st.caption(
                f"Forma del tensor: {tuple(atenciones[capa_idx].shape)} "
                "(lote, cabezas, queries, posiciones del encoder)"
            )
            query = st.slider("Query", 0, a.shape[0] - 1, int(orden[0]))
            vec = a[query].numpy()

            lado_g = int(np.sqrt(vec.size))
            if lado_g * lado_g == vec.size:
                st.image(figura_atencion(vec.reshape(lado_g, lado_g),
                                         f"Query {query}, capa {capa_idx}"))
            else:
                st.line_chart(vec)
                st.caption(
                    "Las posiciones del encoder son multiescala y no forman "
                    "una cuadricula cuadrada, asi que se muestran como perfil."
                )


# --------------------------------------------------------------------------
with tab4:
    mf = getattr(cfg, "num_queries", None)
    datos = {
        "Queries de objeto": cfg.num_queries,
        "Capas del decoder": cfg.decoder_layers,
        "Dimension oculta": cfg.hidden_dim,
        "Cabezas de atencion": cfg.num_attention_heads,
        "Dimension por cabeza": cfg.hidden_dim // cfg.num_attention_heads,
        "Clases del checkpoint": len(id2label),
        "Parametros (M)": round(sum(p.numel() for p in model.parameters()) / 1e6, 1),
    }
    st.table([{"parametro": k, "valor": v} for k, v in datos.items()])

    st.markdown("**Formas de los tensores de salida**")
    st.code(
        f"masks_queries_logits : {tuple(out.masks_queries_logits.shape)}\n"
        f"class_queries_logits : {tuple(out.class_queries_logits.shape)}",
        language="text",
    )
    st.caption(
        "La ultima dimension de class_queries_logits es el numero de clases "
        "mas uno: esa columna extra es la clase \"sin objeto\"."
    )

    st.markdown("**Las tres innovaciones del paper**")
    st.markdown(
        "1. Pixel decoder con atencion deformable multiescala, que sustituye "
        "al encoder Transformer convencional.\n"
        "2. Decoder con atencion enmascarada: cada query atiende solo a la "
        "region de su prediccion previa.\n"
        "3. Perdida calculada sobre puntos submuestreados en lugar de mascaras "
        "completas, lo que reduce el costo de entrenamiento."
    )