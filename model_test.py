
import io
import hashlib
from pathlib import Path

import cv2
import numpy as np
import streamlit as st

from PIL import Image, ImageOps
from streamlit_drawable_canvas import st_canvas


# ============================================================
# 1. GRUNDEINSTELLUNGEN
# ============================================================

st.set_page_config(
    page_title="Meine Stadt. Meine Idee.",
    page_icon="🏙️",
    layout="wide"
)

BASE_DIR = Path(__file__).resolve().parent
MODEL_DIR = BASE_DIR / "model"

MODEL_PATH = MODEL_DIR / "keras_model.h5"
LABEL_PATH = MODEL_DIR / "labels.txt"

MODEL_SIZE = (224, 224)

MAX_CANVAS_WIDTH = 900
MAX_CANVAS_HEIGHT = 650

st.title("🏙️ Meine Stadt. Meine Idee.")

st.write(
    "Lade ein Foto deines Stadtplatzes oder deiner Straße hoch "
    "und gestalte deine eigene Stadt der Zukunft."
)


# ============================================================
# 2. KI-MODELL LADEN
# ============================================================

@st.cache_resource
def load_model():

    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Modell nicht gefunden: {MODEL_PATH}"
        )

    if not LABEL_PATH.exists():
        raise FileNotFoundError(
            f"Labels nicht gefunden: {LABEL_PATH}"
        )

    import tensorflow as tf

    model = tf.keras.models.load_model(
        str(MODEL_PATH),
        compile=False
    )

    labels = []

    with open(LABEL_PATH, "r", encoding="utf-8") as file:

        for line in file:

            line = line.strip()

            if not line:
                continue

            parts = line.split(maxsplit=1)

            if len(parts) == 2 and parts[0].isdigit():
                labels.append(parts[1].strip())
            else:
                labels.append(line)

    return model, labels


# ============================================================
# 3. BILD VORBEREITEN
# ============================================================

def prepare_background(uploaded_file):

    image = Image.open(
        io.BytesIO(uploaded_file.getvalue())
    )

    # EXIF-Drehung berücksichtigen
    image = ImageOps.exif_transpose(image)

    # RGB-Farben sicherstellen
    image = image.convert("RGB")

    original_size = image.size

    # Bild proportional auf die Canvas-Größe verkleinern
    scale = min(
        MAX_CANVAS_WIDTH / image.width,
        MAX_CANVAS_HEIGHT / image.height,
        1.0
    )

    width = max(1, int(image.width * scale))
    height = max(1, int(image.height * scale))

    canvas_image = image.resize(
        (width, height),
        Image.Resampling.LANCZOS
    )

    return image, canvas_image, original_size


# ============================================================
# 4. ZEICHNUNG MIT ORIGINALFOTO ZUSAMMENSETZEN
# ============================================================

def combine_images(original_image, drawing_data):

    original = original_image.convert("RGBA")

    if drawing_data is None:
        return original

    drawing_array = np.asarray(
        drawing_data,
        dtype=np.uint8
    )

    if drawing_array.ndim != 3:
        raise ValueError(
            "Die zurückgegebenen Zeichenbilddaten sind ungültig."
        )

    drawing_layer = Image.fromarray(
        drawing_array
    ).convert("RGBA")

    # Canvas und Original müssen dieselbe Größe haben
    if drawing_layer.size != original.size:

        drawing_layer = drawing_layer.resize(
            original.size,
            Image.Resampling.LANCZOS
        )

    # Transparente Zeichenebene über das Original legen
    result = Image.alpha_composite(
        original,
        drawing_layer
    )

    return result


# ============================================================
# 5. TATSÄCHLICHE VERÄNDERUNGEN ERKENNEN
# ============================================================

def detect_changes(original_image, combined_image):

    original = np.asarray(
        original_image.convert("RGB")
    )

    combined = np.asarray(
        combined_image.convert("RGB")
    )

    # Pixelweiser Vergleich
    difference = cv2.absdiff(
        original,
        combined
    )

    # Nur sichtbare Unterschiede berücksichtigen
    gray_difference = cv2.cvtColor(
        difference,
        cv2.COLOR_RGB2GRAY
    )

    _, mask = cv2.threshold(
        gray_difference,
        12,
        255,
        cv2.THRESH_BINARY
    )

    # Kleine Lücken in zusammenhängenden Zeichnungen schließen
    kernel = np.ones(
        (9, 9),
        np.uint8
    )

    mask = cv2.morphologyEx(
        mask,
        cv2.MORPH_CLOSE,
        kernel
    )

    # Zusammenhängende Bereiche erkennen
    count, labels, stats, centroids = (
        cv2.connectedComponentsWithStats(
            mask,
            connectivity=8
        )
    )

    regions = []

    for index in range(1, count):

        x = stats[index, cv2.CC_STAT_LEFT]
        y = stats[index, cv2.CC_STAT_TOP]

        width = stats[index, cv2.CC_STAT_WIDTH]
        height = stats[index, cv2.CC_STAT_HEIGHT]

        area = stats[index, cv2.CC_STAT_AREA]

        # Sehr kleine Bildveränderungen ignorieren
        if area < 8:
            continue

        regions.append({
            "x": int(x),
            "y": int(y),
            "width": int(width),
            "height": int(height),
            "area": int(area)
        })

    return regions, mask


# ============================================================
# 6. EINZELNE BILDAUSSCHNITTE ERSTELLEN
# ============================================================

def extract_regions(combined_image, regions):

    image = combined_image.convert("RGB")

    image_width, image_height = image.size

    crops = []

    for region in regions:

        x = region["x"]
        y = region["y"]

        width = region["width"]
        height = region["height"]

        # Etwas zusätzlichen Bildbereich einbeziehen
        padding = max(
            12,
            int(max(width, height) * 0.15)
        )

        left = max(0, x - padding)
        top = max(0, y - padding)

        right = min(
            image_width,
            x + width + padding
        )

        bottom = min(
            image_height,
            y + height + padding
        )

        crop = image.crop(
            (left, top, right, bottom)
        )

        # Seitenverhältnis beibehalten und auf 224x224 bringen
        crop = ImageOps.pad(
            crop,
            MODEL_SIZE,
            method=Image.Resampling.LANCZOS,
            color=(255, 255, 255)
        )

        crops.append(crop)

    return crops


# ============================================================
# 7. TEACHABLE-MACHINE-KLASSIFIZIERUNG
# ============================================================

def classify_image(image, model, labels):

    image = image.convert("RGB")

    image = image.resize(
        MODEL_SIZE,
        Image.Resampling.LANCZOS
    )

    # In ein NumPy-Array umwandeln
    image_array = np.asarray(
        image,
        dtype=np.float32
    )

    # Teachable-Machine-Normalisierung
    image_array = image_array / 127.5 - 1.0

    # Batch-Dimension hinzufügen
    image_array = np.expand_dims(
        image_array,
        axis=0
    )

    # Vorhersage
    prediction = model.predict(
        image_array,
        verbose=0
    )

    probabilities = prediction[0]

    predicted_index = int(
        np.argmax(probabilities)
    )

    if predicted_index < len(labels):
        predicted_label = labels[predicted_index]
    else:
        predicted_label = f"Kategorie {predicted_index}"

    confidence = float(
        probabilities[predicted_index]
    )

    return predicted_label, confidence, probabilities


# ============================================================
# 8. FOTO HOCHLADEN
# ============================================================

st.subheader("1. Foto hochladen")

uploaded_file = st.file_uploader(
    "Wähle ein Foto deiner Stadt aus.",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is None:

    st.info(
        "Bitte lade zuerst ein Foto hoch, "
        "damit du es anschließend bearbeiten kannst."
    )

    st.stop()


# ============================================================
# 9. FOTO VORBEREITEN
# ============================================================

original_image, canvas_background, original_size = (
    prepare_background(uploaded_file)
)

canvas_width, canvas_height = canvas_background.size

# Eindeutiger Schlüssel pro hochgeladenem Foto
image_hash = hashlib.sha256(
    uploaded_file.getvalue()
).hexdigest()[:16]


# ============================================================
# 10. ZEICHENWERKZEUGE
# ============================================================

st.subheader("2. Gestalte deinen Stadtplatz")

st.write(
    "Zeichne direkt auf das Foto. "
    "Du kannst beispielsweise Bäume, Bänke, Blumenbeete "
    "oder Fahrradständer hinzufügen."
)

col1, col2 = st.columns(2)

with col1:

    drawing_color = st.color_picker(
        "Zeichenfarbe",
        "#00AA55"
    )

with col2:

    stroke_width = st.slider(
        "Stiftgröße",
        min_value=2,
        max_value=30,
        value=8
    )


# ============================================================
# 11. INTERAKTIVE ZEICHENFLÄCHE
# ============================================================

canvas_result = st_canvas(

    fill_color="rgba(0, 170, 85, 0.25)",

    stroke_width=stroke_width,

    stroke_color=drawing_color,

    background_color="",

    # Das hochgeladene Foto ist der echte Canvas-Hintergrund
    background_image=canvas_background,

    # Bildproportionen bleiben erhalten
    background_image_fit="stretch",

    width=canvas_width,

    height=canvas_height,

    drawing_mode="freedraw",

    update_streamlit=True,

    return_image_data=True,

    key=f"city_canvas_{image_hash}"
)


# ============================================================
# 12. PRÜFEN, OB TATSÄCHLICH GEZEICHNET WURDE
# ============================================================

drawing_objects = []

if canvas_result.json_data is not None:

    drawing_objects = canvas_result.json_data.get(
        "objects",
        []
    )

if not drawing_objects:

    st.info(
        "Noch keine Zeichnung vorhanden. "
        "Zeichne zuerst etwas auf das Foto."
    )

    st.stop()


# ============================================================
# 13. ZEICHNUNG UND ORIGINALFOTO ZUSAMMENFÜGEN
# ============================================================

if canvas_result.image_data is None:

    st.warning(
        "Die Zeichenbilddaten wurden noch nicht übertragen. "
        "Bitte zeichne erneut oder bewege den Stift auf der Fläche."
    )

    st.stop()


combined_image = combine_images(
    canvas_background,
    canvas_result.image_data
)


# ============================================================
# 14. VERÄNDERTE BEREICHE ERKENNEN
# ============================================================

regions, change_mask = detect_changes(
    canvas_background,
    combined_image
)

if not regions:

    st.warning(
        "Es wurde keine tatsächliche Bildveränderung erkannt. "
        "Bitte zeichne erneut."
    )

    st.stop()


# ============================================================
# 15. VERÄNDERTES BILD ANZEIGEN
# ============================================================

st.divider()

st.subheader("3. Dein gestalteter Stadtplatz")

st.image(
    combined_image.convert("RGB"),
    width=canvas_width
)

st.success(
    f"{len(regions)} veränderte Bildbereiche erkannt."
)


# ============================================================
# 16. EINZELNE BEREICHE AUSSCHNEIDEN
# ============================================================

crops = extract_regions(
    combined_image,
    regions
)

st.subheader("4. Erkannte Bildbereiche")

columns = st.columns(3)

for index, crop in enumerate(crops):

    with columns[index % 3]:

        st.image(
            crop,
            caption=f"Ausschnitt {index + 1}",
            width=180
        )


# ============================================================
# 17. KLASSIFIZIERUNG STARTEN
# ============================================================

st.divider()

st.subheader("5. KI-Klassifizierung")

if st.button(
    "Zeichnung mit meiner KI analysieren",
    type="primary"
):

    try:

        with st.spinner(
            "Deine Zeichnung wird analysiert ..."
        ):

            model, labels = load_model()

            results = []

            for index, crop in enumerate(crops):

                category, confidence, probabilities = (
                    classify_image(
                        crop,
                        model,
                        labels
                    )
                )

                results.append({
                    "index": index + 1,
                    "image": crop,
                    "category": category,
                    "confidence": confidence
                })

        st.success(
            "Die Analyse wurde abgeschlossen!"
        )

        st.subheader("Ergebnisse")

        for result in results:

            st.markdown(
                f"### Ausschnitt {result['index']}"
            )

            result_col1, result_col2 = st.columns(2)

            with result_col1:

                st.image(
                    result["image"],
                    width=224
                )

            with result_col2:

                st.write(
                    f"**Erkannte Kategorie:** "
                    f"{result['category']}"
                )

                st.progress(
                    min(
                        1.0,
                        max(0.0, result["confidence"])
                    )
                )

                st.write(
                    f"**Modell-Konfidenz:** "
                    f"{result['confidence'] * 100:.2f} %"
                )

    except Exception as error:

        st.error(
            f"Bei der KI-Analyse ist ein Fehler aufgetreten: {error}"
        )

