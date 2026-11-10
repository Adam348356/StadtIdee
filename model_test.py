
import streamlit as st
import tensorflow as tf
import numpy as np
import cv2
import hashlib
from io import BytesIO
from PIL import Image

from streamlit_drawable_canvas import st_canvas


# =========================================
# 1. SEITE
# =========================================

st.set_page_config(
    page_title="Meine Stadt. Meine Idee.",
    page_icon="🌳",
    layout="wide"
)

st.title("🌳 Meine Stadt. Meine Idee.")

st.write(
    "Lade ein Foto hoch, zeichne neue Dinge darauf "
    "und lasse dein KI-Modell die Zeichnung erkennen."
)


# =========================================
# 2. KERAS-MODELL LADEN
# =========================================

class CompatibleDepthwiseConv2D(
    tf.keras.layers.DepthwiseConv2D
):
    @classmethod
    def from_config(cls, config):
        config.pop("groups", None)
        return cls(**config)


@st.cache_resource
def load_model():
    return tf.keras.models.load_model(
        "model/keras_model.h5",
        compile=False,
        custom_objects={
            "DepthwiseConv2D": CompatibleDepthwiseConv2D
        }
    )


@st.cache_data
def load_labels():
    with open(
        "model/labels.txt",
        "r",
        encoding="utf-8"
    ) as file:
        return [
            line.strip().split(" ", 1)[-1]
            for line in file.readlines()
        ]


try:
    model = load_model()
    labels = load_labels()

except Exception as error:
    st.error(f"Modellfehler: {error}")
    st.stop()


# =========================================
# 3. FOTO HOCHLADEN
# =========================================

uploaded_file = st.file_uploader(
    "Lade dein Stadtfoto hoch",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is None:
    st.info("Bitte lade zuerst ein Foto hoch.")
    st.stop()


original = Image.open(uploaded_file).convert("RGB")

st.subheader("Dein Originalfoto")
st.image(original, width="stretch")


# =========================================
# 4. ZEICHENFLÄCHE VORBEREITEN
# =========================================

st.subheader("✏️ Zeichne deine Stadtidee")

st.write(
    "Das Foto soll direkt unter deiner Zeichnung liegen. "
    "Male beispielsweise einen Baum, eine Bank "
    "oder ein Blumenbeet."
)

color = st.color_picker(
    "Zeichenfarbe",
    "#228B22"
)

brush_size = st.slider(
    "Stiftdicke",
    min_value=2,
    max_value=40,
    value=12
)

# Bild proportional verkleinern
scale = min(
    1.0,
    900 / original.width,
    700 / original.height
)

canvas_width = max(1, int(original.width * scale))
canvas_height = max(1, int(original.height * scale))

background = original.resize(
    (canvas_width, canvas_height),
    Image.Resampling.LANCZOS
)

# Bild als PNG-Bytes vorbereiten.
# Diese Übergabe vermeidet Probleme bei der
# Übertragung von Pillow-Bildern an die Canvas.

buffer = BytesIO()
background.save(buffer, format="PNG")
background_bytes = buffer.getvalue()

photo_key = hashlib.md5(
    uploaded_file.getvalue()
).hexdigest()


# =========================================
# 5. ZEICHENFLÄCHE ANZEIGEN
# =========================================

st.info("Hier kannst du direkt auf deinem Foto malen.")

canvas_result = st_canvas(
    fill_color="rgba(0, 0, 0, 0)",
    stroke_width=brush_size,
    stroke_color=color,

    background_image=background_bytes,
    background_image_fit="stretch",

    width=canvas_width,
    height=canvas_height,

    drawing_mode="freedraw",
    update_streamlit=True,

    return_image_data=True,

    key=f"canvas_{photo_key}"
)


# =========================================
# 6. PRÜFEN, OB WIRKLICH GEMALT WURDE
# =========================================

if canvas_result.json_data is None:
    st.info("Male etwas auf das Foto, um fortzufahren.")
    st.stop()

objects = canvas_result.json_data.get("objects", [])

if len(objects) == 0:
    st.info(
        "Du hast noch nichts eingezeichnet. "
        "Male beispielsweise einen Baum auf das Foto."
    )
    st.stop()

if canvas_result.image_data is None:
    st.warning("Die Zeichnung konnte nicht geladen werden.")
    st.stop()


# =========================================
# 7. ZEICHNUNG MIT FOTO ZUSAMMENFÜHREN
# =========================================

drawing = Image.fromarray(
    canvas_result.image_data.astype(np.uint8)
).convert("RGBA")

background_rgba = background.convert("RGBA")

# Transparente Zeichnung auf das Originalfoto legen
combined = Image.alpha_composite(
    background_rgba,
    drawing
)

after_image = combined.convert("RGB").resize(
    original.size,
    Image.Resampling.LANCZOS
)

st.subheader("Deine neue Stadtidee")
st.image(after_image, width="stretch")


# =========================================
# 8. ERKENNUNG STARTEN
# =========================================

if st.button(
    "🔍 Zeichnung mit KI erkennen",
    type="primary"
):

    before_array = np.array(original)
    after_array = np.array(after_image)

    # Bildvergleich
    before_gray = cv2.cvtColor(
        before_array,
        cv2.COLOR_RGB2GRAY
    )

    after_gray = cv2.cvtColor(
        after_array,
        cv2.COLOR_RGB2GRAY
    )

    difference = cv2.absdiff(
        before_gray,
        after_gray
    )

    _, mask = cv2.threshold(
        difference,
        25,
        255,
        cv2.THRESH_BINARY
    )

    # Kleine Lücken schließen
    kernel = np.ones((15, 15), np.uint8)

    mask = cv2.morphologyEx(
        mask,
        cv2.MORPH_CLOSE,
        kernel
    )

    mask = cv2.dilate(
        mask,
        np.ones((9, 9), np.uint8),
        iterations=2
    )


    # =====================================
    # 9. GEÄNDERTEN BEREICH FINDEN
    # =====================================

    contours, _ = cv2.findContours(
        mask,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    contours = [
        c for c in contours
        if cv2.contourArea(c) > 100
    ]

    if not contours:
        st.warning(
            "Es konnte kein ausreichend großer "
            "veränderter Bereich gefunden werden."
        )
        st.stop()

    boxes = [
        cv2.boundingRect(c)
        for c in contours
    ]

    x1 = min(x for x, y, w, h in boxes)
    y1 = min(y for x, y, w, h in boxes)

    x2 = max(x + w for x, y, w, h in boxes)
    y2 = max(y + h for x, y, w, h in boxes)

    padding = 40

    x1 = max(0, x1 - padding)
    y1 = max(0, y1 - padding)

    x2 = min(after_image.width, x2 + padding)
    y2 = min(after_image.height, y2 + padding)

    crop = after_image.crop(
        (x1, y1, x2, y2)
    )

    st.subheader("Dein erkannter Bildausschnitt")
    st.image(crop)


    # =====================================
    # 10. KERAS-KLASSIFIZIERUNG
    # =====================================

    model_image = crop.resize(
        (224, 224),
        Image.Resampling.LANCZOS
    )

    image_array = np.asarray(
        model_image,
        dtype=np.float32
    )

    image_array = image_array / 127.5 - 1

    image_array = np.expand_dims(
        image_array,
        axis=0
    )

    prediction = model.predict(
        image_array,
        verbose=0
    )

    class_index = int(
        np.argmax(prediction[0])
    )

    confidence = float(
        prediction[0][class_index]
    )

    detected_class = labels[class_index]


    # =====================================
    # 11. ERGEBNIS
    # =====================================

    st.subheader("🎯 KI-Ergebnis")

    st.success(
        f"Erkannt: {detected_class}"
    )

    st.metric(
        "Sicherheit des Modells",
        f"{confidence * 100:.2f} %"
    )

    if confidence < 0.60:
        st.warning(
            "Das Modell ist sich bei dieser Erkennung unsicher."
        )

    # Position markieren
    result_image = after_array.copy()

    cv2.rectangle(
        result_image,
        (x1, y1),
        (x2, y2),
        (0, 255, 0),
        4
    )

    st.subheader("Position der Zeichnung")

    st.image(
        result_image,
        width="stretch"
    )
