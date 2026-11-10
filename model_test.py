
import streamlit as st
import tensorflow as tf
import numpy as np
import cv2

from PIL import Image
from streamlit_drawable_canvas import st_canvas


# ==========================================
# SEITENEINSTELLUNGEN
# ==========================================

st.set_page_config(
    page_title="Meine Stadt. Meine Idee.",
    page_icon="🌳",
    layout="wide"
)

st.title("🌳 Meine Stadt. Meine Idee.")
st.write(
    "Lade ein Foto hoch und gestalte deine eigene Stadt."
)


# ==========================================
# 1. KERAS-MODELL LADEN
# ==========================================

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
    st.error(f"Fehler beim Laden des Modells: {error}")
    st.stop()


# ==========================================
# 2. FOTO HOCHLADEN
# ==========================================

uploaded_file = st.file_uploader(
    "Wähle ein Stadtfoto aus",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is None:
    st.info("Bitte lade zuerst ein Foto hoch.")
    st.stop()


original = Image.open(uploaded_file).convert("RGB")

st.subheader("Dein Originalfoto")
st.image(original)


# ==========================================
# 3. ZEICHENFLÄCHE VORBEREITEN
# ==========================================

st.subheader("✏️ Gestalte deine Stadt")

st.write(
    "Male direkt auf das Foto. "
    "Du kannst zum Beispiel Bäume, Bänke "
    "oder Blumenbeete ergänzen."
)

stroke_color = st.color_picker(
    "Farbe deiner Zeichnung",
    "#228B22"
)

stroke_width = st.slider(
    "Stiftdicke",
    min_value=2,
    max_value=40,
    value=12
)

# Foto proportional verkleinern,
# damit die Zeichenfläche nicht zu groß wird.

scale = min(
    1.0,
    900 / original.width,
    700 / original.height
)

canvas_width = max(
    1,
    int(original.width * scale)
)

canvas_height = max(
    1,
    int(original.height * scale)
)

# Foto exakt auf Zeichenflächengröße bringen.

background = original.resize(
    (canvas_width, canvas_height),
    Image.Resampling.LANCZOS
)


# ==========================================
# 4. ZEICHENFLÄCHE MIT FOTO
# ==========================================

canvas_result = st_canvas(

    # Transparente Zeichenfarbe
    fill_color="rgba(0, 0, 0, 0)",

    stroke_color=stroke_color,
    stroke_width=stroke_width,

    # DAS HOCHGELADENE FOTO ALS HINTERGRUND
    background_image=background,

    # Zeichenfläche
    width=canvas_width,
    height=canvas_height,

    drawing_mode="freedraw",

    update_streamlit=True,

    # Bilddaten zurück an Python senden
    return_image_data=True,

    key="stadt_zeichenflaeche"

)


# ==========================================
# 5. ZEICHNUNG MIT ORIGINALFOTO VERBINDEN
# ==========================================

if canvas_result.image_data is None:

    st.warning("Die Zeichenfläche wurde noch nicht geladen.")

    st.stop()


# Zurückgegebenes Bild als RGBA öffnen.
overlay = Image.fromarray(
    canvas_result.image_data.astype(np.uint8)
).convert("RGBA")

# Hintergrund ebenfalls in RGBA umwandeln.
background_rgba = background.convert("RGBA")

# Zeichnung über das Foto legen.
combined = Image.alpha_composite(
    background_rgba,
    overlay
)

# Wieder auf ursprüngliche Bildgröße bringen.
after_image = combined.convert("RGB").resize(
    original.size,
    Image.Resampling.LANCZOS
)

st.subheader("Deine gestaltete Stadt")

st.image(after_image)


# ==========================================
# 6. ZEICHNUNG ERKENNEN
# ==========================================

if st.button(
    "🔍 Meine Zeichnung erkennen",
    type="primary"
):

    before_array = np.array(original)
    after_array = np.array(after_image)

    # Bilder in Graustufen umwandeln.
    before_gray = cv2.cvtColor(
        before_array,
        cv2.COLOR_RGB2GRAY
    )

    after_gray = cv2.cvtColor(
        after_array,
        cv2.COLOR_RGB2GRAY
    )

    # Unterschiede zwischen beiden Bildern finden.
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

    # Kleine Lücken zwischen Zeichnungsstrichen schließen.
    kernel = np.ones(
        (15, 15),
        np.uint8
    )

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

    # ======================================
    # 7. VERÄNDERTEN BEREICH FINDEN
    # ======================================

    contours, _ = cv2.findContours(
        mask,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    valid_contours = [
        contour
        for contour in contours
        if cv2.contourArea(contour) > 100
    ]

    if not valid_contours:

        st.warning(
            "Es wurde keine Veränderung gefunden. "
            "Bitte male etwas auf das Foto."
        )

        st.stop()

    # Alle Zeichnungsbereiche zusammenfassen.
    boxes = [
        cv2.boundingRect(contour)
        for contour in valid_contours
    ]

    x1 = min(x for x, y, w, h in boxes)
    y1 = min(y for x, y, w, h in boxes)

    x2 = max(x + w for x, y, w, h in boxes)
    y2 = max(y + h for x, y, w, h in boxes)

    # Etwas Umgebung behalten.
    padding = 40

    x1 = max(0, x1 - padding)
    y1 = max(0, y1 - padding)

    x2 = min(after_image.width, x2 + padding)
    y2 = min(after_image.height, y2 + padding)

    # Ausschnitt erstellen.
    crop = after_image.crop(
        (x1, y1, x2, y2)
    )

    st.subheader("Erkannter Bildausschnitt")
    st.image(crop)


    # ======================================
    # 8. KERAS-MODELL KLASSIFIZIERT
    # ======================================

    # Teachable Machine erwartet 224 x 224.
    model_image = crop.resize(
        (224, 224),
        Image.Resampling.LANCZOS
    )

    image_array = np.asarray(
        model_image,
        dtype=np.float32
    )

    # Normalisierung wie bei Teachable Machine.
    image_array = (
        image_array / 127.5
    ) - 1

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


    # ======================================
    # 9. ERGEBNIS ANZEIGEN
    # ======================================

    st.subheader("🎯 Ergebnis der Erkennung")

    st.success(
        f"Erkannte Kategorie: {detected_class}"
    )

    st.write(
        f"Modell-Sicherheit: {confidence * 100:.1f} %"
    )

    if confidence < 0.60:

        st.warning(
            "Das Modell ist sich nicht sicher. "
            "Bitte überprüfe das Ergebnis."
        )

    # Position der Veränderung markieren.
    result_image = after_array.copy()

    cv2.rectangle(
        result_image,
        (x1, y1),
        (x2, y2),
        (0, 255, 0),
        4
    )

    st.subheader("Position deiner Zeichnung")

    st.image(result_image)

