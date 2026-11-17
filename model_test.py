
import streamlit as st
import tensorflow as tf
import numpy as np
import cv2
import io

from PIL import Image
from streamlit_drawable_canvas import st_canvas
from collections import Counter


# ==========================================
# 1. SEITENEINSTELLUNGEN
# ==========================================

st.set_page_config(
    page_title="Meine Stadt. Meine Idee.",
    page_icon="🏙️",
    layout="wide"
)

st.title("🏙️ Meine Stadt. Meine Idee.")
st.write(
    "Lade ein Stadtfoto hoch, zeichne deine Ideen direkt darauf "
    "und lasse dein trainiertes KI-Modell die Veränderungen erkennen."
)


# ==========================================
# 2. MODELL LADEN
# ==========================================

class CompatibleDepthwiseConv2D(tf.keras.layers.DepthwiseConv2D):

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
    with open("model/labels.txt", "r", encoding="utf-8") as file:
        return [
            line.strip().split(" ", 1)[-1]
            for line in file.readlines()
        ]


try:
    model = load_model()
    labels = load_labels()
except Exception as error:
    st.error(f"Das KI-Modell konnte nicht geladen werden: {error}")
    st.stop()


EXPECTED_CLASSES = {
    "Bäume",
    "Beete",
    "Fahrradständer",
    "Bänke"
}

if len(labels) != 4 or set(labels) != EXPECTED_CLASSES:
    st.error(
        "Bitte kontrolliere model/labels.txt. "
        "Es müssen genau diese vier Klassen enthalten sein: "
        "Bäume, Beete, Fahrradständer und Bänke."
    )
    st.stop()


# ==========================================
# 3. FOTO HOCHLADEN
# ==========================================

st.header("1. Stadtfoto hochladen")

uploaded_file = st.file_uploader(
    "Wähle dein Foto aus",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is None:
    st.info("Bitte lade zuerst ein Foto hoch.")
    st.stop()

before_image = Image.open(uploaded_file).convert("RGB")


# ==========================================
# 4. ZEICHENWERKZEUGE
# ==========================================

st.header("2. Deine Stadtidee zeichnen")

col1, col2 = st.columns(2)

with col1:
    stroke_color = st.color_picker(
        "Zeichenfarbe",
        "#228B22"
    )

with col2:
    stroke_width = st.slider(
        "Stiftgröße",
        min_value=2,
        max_value=40,
        value=12
    )


# ==========================================
# 5. FOTO FÜR DIE ZEICHENFLÄCHE
# ==========================================

max_width = 900
max_height = 650

scale = min(
    max_width / before_image.width,
    max_height / before_image.height,
    1.0
)

canvas_width = max(1, int(before_image.width * scale))
canvas_height = max(1, int(before_image.height * scale))

background = before_image.resize(
    (canvas_width, canvas_height),
    Image.Resampling.LANCZOS
).convert("RGBA")

# Foto als PNG-Bytes übergeben
buffer = io.BytesIO()
background.save(buffer, format="PNG")
background_bytes = buffer.getvalue()


# ==========================================
# 6. ZEICHENFLÄCHE
# ==========================================

st.subheader("Zeichne direkt auf deinem Originalfoto")

canvas_result = st_canvas(
    fill_color="rgba(0, 0, 0, 0)",
    stroke_width=stroke_width,
    stroke_color=stroke_color,

    # Das hochgeladene Foto ist der Canvas-Hintergrund
    background_image=background_bytes,
    background_image_fit="stretch",

    background_color="rgba(0,0,0,0)",

    update_streamlit=True,
    height=canvas_height,
    width=canvas_width,
    drawing_mode="freedraw",

    return_image_data=True,

    key="stadt_zeichenflaeche"
)


# ==========================================
# 7. ZEICHNUNG ÜBER DAS FOTO LEGEN
# ==========================================

if canvas_result.image_data is None:
    st.info("Die Zeichenfläche wird vorbereitet.")
    st.stop()

drawing_array = canvas_result.image_data.astype(np.uint8)

drawing_layer = Image.fromarray(
    drawing_array,
    mode="RGBA"
)

# Das Originalfoto bleibt erhalten.
# Die transparente Zeichnung wird darübergelegt.
after_small = Image.alpha_composite(
    background,
    drawing_layer
).convert("RGB")

after_image = after_small.resize(
    before_image.size,
    Image.Resampling.LANCZOS
)


# ==========================================
# 8. ORIGINAL UND PROJEKT ANZEIGEN
# ==========================================

st.header("3. Dein Stadtprojekt")

col1, col2 = st.columns(2)

with col1:
    st.image(
        before_image,
        caption="Originalfoto",
        use_container_width=True
    )

with col2:
    st.image(
        after_image,
        caption="Dein gezeichnetes Stadtprojekt",
        use_container_width=True
    )


# ==========================================
# 9. ZEICHNUNG ALS DIFFERENZMASKE
# ==========================================

# Die Alpha-Ebene enthält die gezeichneten Pixel.
# So werden nicht versehentlich Schatten oder
# Helligkeitsunterschiede des Fotos erkannt.

drawing_alpha = drawing_array[:, :, 3]

_, drawing_mask = cv2.threshold(
    drawing_alpha,
    10,
    255,
    cv2.THRESH_BINARY
)

# Kleine Lücken zwischen zusammengehörigen Strichen schließen
kernel = np.ones((21, 21), np.uint8)

drawing_mask = cv2.morphologyEx(
    drawing_mask,
    cv2.MORPH_CLOSE,
    kernel
)

drawing_mask = cv2.dilate(
    drawing_mask,
    np.ones((5, 5), np.uint8),
    iterations=1
)


# ==========================================
# 10. KI-ANALYSE STARTEN
# ==========================================

if st.button(
    "🔍 Meine Zeichnung mit KI klassifizieren",
    type="primary",
    use_container_width=True
):

    contours, _ = cv2.findContours(
        drawing_mask,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    valid_contours = [
        contour for contour in contours
        if cv2.contourArea(contour) > 30
    ]

    if not valid_contours:
        st.warning(
            "Es wurde keine Zeichnung erkannt. "
            "Zeichne zuerst ein Objekt auf das Foto."
        )
        st.stop()

    results = []
    marked_image = np.array(after_image).copy()

    # Umrechnung von Canvas-Koordinaten auf Originalbild
    scale_x = before_image.width / canvas_width
    scale_y = before_image.height / canvas_height

    for index, contour in enumerate(valid_contours, start=1):

        x, y, w, h = cv2.boundingRect(contour)

        # Canvas-Koordinaten in Originalbild-Koordinaten umrechnen
        x1 = max(0, int(x * scale_x))
        y1 = max(0, int(y * scale_y))
        x2 = min(
            before_image.width,
            int((x + w) * scale_x)
        )
        y2 = min(
            before_image.height,
            int((y + h) * scale_y)
        )

        # Etwas Umgebung zum Objekt hinzufügen
        padding = 35

        x1 = max(0, x1 - padding)
        y1 = max(0, y1 - padding)
        x2 = min(before_image.width, x2 + padding)
        y2 = min(before_image.height, y2 + padding)

        if x2 <= x1 or y2 <= y1:
            continue

        crop = after_image.crop((x1, y1, x2, y2))

        # Teachable Machine erwartet 224 x 224 RGB
        model_image = crop.resize(
            (224, 224),
            Image.Resampling.LANCZOS
        )

        image_array = np.asarray(
            model_image,
            dtype=np.float32
        )

        image_array = image_array / 127.5 - 1.0
        image_array = np.expand_dims(image_array, axis=0)

        prediction = model.predict(
            image_array,
            verbose=0
        )[0]

        class_index = int(np.argmax(prediction))
        category = labels[class_index]
        confidence = float(prediction[class_index])

        results.append({
            "Nummer": index,
            "Kategorie": category,
            "Sicherheit": confidence,
            "Bild": crop,
            "Position": (x1, y1, x2, y2)
        })

        # Erkannte Stelle im Bild markieren
        cv2.rectangle(
            marked_image,
            (x1, y1),
            (x2, y2),
            (0, 200, 0),
            4
        )

        cv2.putText(
            marked_image,
            str(index),
            (x1, max(25, y1 - 10)),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 200, 0),
            3
        )


    # ==========================================
    # 11. ERGEBNISSE ANZEIGEN
    # ==========================================

    if not results:
        st.warning("Es konnten keine Bildbereiche analysiert werden.")
        st.stop()

    st.header("4. Ergebnisse der KI")

    counts = Counter(
        result["Kategorie"]
        for result in results
    )

    cols = st.columns(4)

    for col, category in zip(cols, labels):
        with col:
            st.metric(
                label=category,
                value=counts.get(category, 0)
            )

    st.divider()

    for result in results:

        st.subheader(
            f"Erkanntes Objekt {result['Nummer']}"
        )

        col1, col2 = st.columns(2)

        with col1:
            st.image(
                result["Bild"],
                caption="Analysierter Bildbereich",
                use_container_width=True
            )

        with col2:
            st.write(
                f"**Kategorie:** {result['Kategorie']}"
            )

            st.write(
                f"**Modell-Sicherheit:** "
                f"{result['Sicherheit'] * 100:.1f} %"
            )

            st.write(
                f"**Bildposition:** {result['Position']}"
            )

            if result["Sicherheit"] < 0.60:
                st.warning(
                    "Das Modell ist bei dieser Zuordnung unsicher."
                )
            else:
                st.success("Klassifizierung abgeschlossen.")


    # ==========================================
    # 12. MARKIERTES GESAMTBILD
    # ==========================================

    st.header("5. Dein Stadtprojekt mit KI-Markierungen")

    st.image(
        marked_image,
        caption="Erkannte und klassifizierte Veränderungen",
        use_container_width=True
    )

    # Projekt herunterladen
    output = io.BytesIO()

    Image.fromarray(marked_image).save(
        output,
        format="PNG"
    )

    st.download_button(
        "📥 Markiertes Projekt herunterladen",
        data=output.getvalue(),
        file_name="mein_stadtprojekt.png",
        mime="image/png"
    )
