
import io
from pathlib import Path

import streamlit as st
import numpy as np
import cv2

from PIL import Image, ImageOps
from streamlit_drawable_canvas import st_canvas


# ==================================================
# EINSTELLUNGEN
# ==================================================

st.set_page_config(
    page_title="Meine Stadt. Meine Idee.",
    page_icon="🏙️",
    layout="wide"
)

BASE_DIR = Path(__file__).resolve().parent

MODEL_PATH = BASE_DIR / "model" / "keras_model.h5"
LABEL_PATH = BASE_DIR / "model" / "labels.txt"

st.title("🏙️ Meine Stadt. Meine Idee.")

st.write(
    "Lade ein Foto hoch und gestalte deinen eigenen Stadtplatz."
)


# ==================================================
# KI-MODELL
# ==================================================

@st.cache_resource
def load_model():

    import tensorflow as tf

    model = tf.keras.models.load_model(
        str(MODEL_PATH),
        compile=False
    )

    with open(LABEL_PATH, encoding="utf-8") as f:
        labels = []

        for line in f:
            line = line.strip()

            if not line:
                continue

            parts = line.split(maxsplit=1)

            if len(parts) == 2 and parts[0].isdigit():
                labels.append(parts[1])
            else:
                labels.append(line)

    return model, labels


# ==================================================
# BILDER VORBEREITEN
# ==================================================

def prepare_image(uploaded_file):

    image = Image.open(
        io.BytesIO(uploaded_file.getvalue())
    )

    image = ImageOps.exif_transpose(image)
    image = image.convert("RGB")

    image.thumbnail(
        (850, 600),
        Image.Resampling.LANCZOS
    )

    return image


# ==================================================
# FARBPALETTE
# ==================================================

colors = {
    "Grün": "#00AA00",
    "Dunkelgrün": "#006400",
    "Hellgrün": "#90EE90",
    "Braun": "#8B4513",
    "Rot": "#FF0000",
    "Blau": "#0000FF",
    "Hellblau": "#00BFFF",
    "Gelb": "#FFD700",
    "Orange": "#FF8C00",
    "Pink": "#FF69B4",
    "Lila": "#800080",
    "Türkis": "#40E0D0",
    "Schwarz": "#000000",
    "Weiß": "#FFFFFF"
}


def hex_to_rgba(hex_color, alpha=0.3):

    color = hex_color.lstrip("#")

    r = int(color[0:2], 16)
    g = int(color[2:4], 16)
    b = int(color[4:6], 16)

    return f"rgba({r},{g},{b},{alpha})"


# ==================================================
# ZEICHNUNG MIT FOTO KOMBINIEREN
# ==================================================

def combine_images(background, drawing_data):

    background = background.convert("RGBA")

    drawing = Image.fromarray(
        np.asarray(drawing_data, dtype=np.uint8)
    ).convert("RGBA")

    if drawing.size != background.size:
        drawing = drawing.resize(
            background.size,
            Image.Resampling.LANCZOS
        )

    return Image.alpha_composite(
        background,
        drawing
    )


# ==================================================
# VERÄNDERTE BEREICHE FINDEN
# ==================================================

def detect_regions(drawing_data):

    array = np.asarray(
        drawing_data,
        dtype=np.uint8
    )

    if array.shape[2] == 4:
        mask = array[:, :, 3]
    else:
        mask = cv2.cvtColor(
            array,
            cv2.COLOR_RGB2GRAY
        )

    _, mask = cv2.threshold(
        mask,
        5,
        255,
        cv2.THRESH_BINARY
    )

    kernel = np.ones((7, 7), np.uint8)

    mask = cv2.dilate(
        mask,
        kernel,
        iterations=1
    )

    count, labels, stats, centroids = (
        cv2.connectedComponentsWithStats(
            mask,
            connectivity=8
        )
    )

    regions = []

    for i in range(1, count):

        x = stats[i, cv2.CC_STAT_LEFT]
        y = stats[i, cv2.CC_STAT_TOP]

        w = stats[i, cv2.CC_STAT_WIDTH]
        h = stats[i, cv2.CC_STAT_HEIGHT]

        area = stats[i, cv2.CC_STAT_AREA]

        if area < 8:
            continue

        regions.append({
            "x": int(x),
            "y": int(y),
            "w": int(w),
            "h": int(h)
        })

    return regions


# ==================================================
# BILDAUSSCHNITTE ERSTELLEN
# ==================================================

def make_crops(image, regions):

    crops = []

    width, height = image.size

    for region in regions:

        x = region["x"]
        y = region["y"]

        w = region["w"]
        h = region["h"]

        padding = 20

        left = max(0, x - padding)
        top = max(0, y - padding)

        right = min(width, x + w + padding)
        bottom = min(height, y + h + padding)

        crop = image.crop(
            (left, top, right, bottom)
        )

        crop = ImageOps.pad(
            crop,
            (224, 224),
            method=Image.Resampling.LANCZOS,
            color=(255, 255, 255)
        )

        crops.append(crop)

    return crops


# ==================================================
# KLASSIFIZIERUNG
# ==================================================

def classify(crop, model, labels):

    image = crop.convert("RGB").resize(
        (224, 224),
        Image.Resampling.LANCZOS
    )

    array = np.asarray(
        image,
        dtype=np.float32
    )

    array = array / 127.5 - 1.0

    array = np.expand_dims(
        array,
        axis=0
    )

    prediction = model.predict(
        array,
        verbose=0
    )[0]

    index = int(np.argmax(prediction))

    category = (
        labels[index]
        if index < len(labels)
        else f"Kategorie {index}"
    )

    confidence = float(prediction[index])

    return category, confidence


# ==================================================
# FOTO HOCHLADEN
# ==================================================

st.subheader("1. Foto hochladen")

uploaded = st.file_uploader(
    "Wähle ein Stadtfoto aus",
    type=["jpg", "jpeg", "png"]
)

if uploaded is None:

    st.info("Bitte lade zuerst ein Foto hoch.")

    st.stop()


background = prepare_image(uploaded)

width, height = background.size

st.success("Foto erfolgreich geladen!")


# ==================================================
# FARBEN AUSWÄHLEN
# ==================================================

st.subheader("2. Zeichenwerkzeuge")

col1, col2 = st.columns(2)

with col1:

    selected_color = st.selectbox(
        "Zeichenfarbe",
        list(colors.keys())
    )

    drawing_color = colors[selected_color]

    custom_color = st.color_picker(
        "Eigene Farbe auswählen",
        drawing_color
    )

with col2:

    stroke_width = st.slider(
        "Stiftgröße",
        2,
        40,
        8
    )

    tool = st.selectbox(
        "Zeichenwerkzeug",
        [
            "Freihand",
            "Linie",
            "Rechteck",
            "Kreis",
            "Verschieben"
        ]
    )


drawing_modes = {
    "Freihand": "freedraw",
    "Linie": "line",
    "Rechteck": "rect",
    "Kreis": "circle",
    "Verschieben": "transform"
}

drawing_mode = drawing_modes[tool]

fill_color = hex_to_rgba(
    custom_color,
    0.30
)


# ==================================================
# ZEICHENFLÄCHE
# ==================================================

st.subheader("3. Zeichne direkt auf deinem Foto")

canvas_result = st_canvas(

    fill_color=fill_color,

    stroke_width=stroke_width,

    stroke_color=custom_color,

    background_color="",

    background_image=background,

    width=width,

    height=height,

    drawing_mode=drawing_mode,

    update_streamlit=True,

    key="stadt_zeichenflaeche",

)


# ==================================================
# ZEICHNUNG PRÜFEN
# ==================================================

if canvas_result.json_data is None:

    st.info("Zeichne zuerst etwas auf dein Foto.")

    st.stop()


objects = canvas_result.json_data.get(
    "objects",
    []
)

if not objects:

    st.info("Noch keine Zeichnung vorhanden.")

    st.stop()


if canvas_result.image_data is None:

    st.warning(
        "Die Zeichenbilddaten sind noch nicht verfügbar."
    )

    st.stop()


drawing_data = np.asarray(
    canvas_result.image_data,
    dtype=np.uint8
)


# ==================================================
# FOTO UND ZEICHNUNG ZUSAMMENSETZEN
# ==================================================

combined = combine_images(
    background,
    drawing_data
)


# ==================================================
# BEREICHE ERKENNEN
# ==================================================

regions = detect_regions(
    drawing_data
)

if not regions:

    st.warning(
        "Keine sichtbare Zeichnung erkannt."
    )

    st.stop()


st.divider()

st.subheader("4. Dein gestalteter Stadtplatz")

st.image(
    combined.convert("RGB"),
    caption="Dein neues Stadtbild",
    width=width
)

st.success(
    f"{len(regions)} veränderte Bereiche erkannt."
)


# ==================================================
# AUSSCHNITTE
# ==================================================

crops = make_crops(
    combined,
    regions
)

st.subheader("5. Erkannte Bildbereiche")

columns = st.columns(3)

for i, crop in enumerate(crops):

    with columns[i % 3]:

        st.image(
            crop,
            caption=f"Ausschnitt {i + 1}"
        )


# ==================================================
# KI-ANALYSE
# ==================================================

st.divider()

if st.button(
    "🤖 Meine Zeichnung mit KI analysieren",
    type="primary"
):

    try:

        with st.spinner("KI analysiert deine Zeichnung ..."):

            model, labels = load_model()

            results = []

            for crop in crops:

                category, confidence = classify(
                    crop,
                    model,
                    labels
                )

                results.append(
                    (crop, category, confidence)
                )

        st.success("Analyse abgeschlossen!")

        st.subheader("6. Ergebnisse")

        for i, (crop, category, confidence) in enumerate(results):

            col1, col2 = st.columns(2)

            with col1:
                st.image(crop)

            with col2:

                st.markdown(
                    f"### Ausschnitt {i + 1}"
                )

                st.write(
                    f"**Kategorie:** {category}"
                )

                st.write(
                    f"**Konfidenz:** {confidence * 100:.2f} %"
                )

                st.progress(
                    min(1.0, max(0.0, confidence))
                )

    except Exception as error:

        st.error(
            f"Fehler bei der KI-Analyse: {error}"
        )
