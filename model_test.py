
import streamlit as st
import tensorflow as tf
import numpy as np
import cv2
import hashlib

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
    "Lade ein Foto hoch, zeichne deine Ideen direkt darauf "
    "und lasse die KI deine Veränderungen erkennen."
)


# ==========================================
# 2. KI-MODELL LADEN
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


# ==========================================
# 3. VIER MODELLKLASSEN
# ==========================================

EXPECTED_CLASSES = [
    "Bäume",
    "Bänke",
    "Fahrradständer",
    "Beete"
]

if len(labels) != 4:

    st.error(
        "Das Modell muss genau vier Klassen enthalten."
    )

    st.stop()


# ==========================================
# 4. FOTO HOCHLADEN
# ==========================================

st.header("1. Stadtfoto hochladen")

uploaded_file = st.file_uploader(
    "Wähle dein Stadtfoto aus",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is None:

    st.info("Bitte lade zuerst ein Foto hoch.")
    st.stop()


image_bytes = uploaded_file.getvalue()

before_image = Image.open(
    uploaded_file
).convert("RGB")


# Eindeutiger Schlüssel für jedes hochgeladene Foto
image_hash = hashlib.md5(image_bytes).hexdigest()


# ==========================================
# 5. ZEICHENWERKZEUGE
# ==========================================

st.header("2. Deine Stadtidee zeichnen")

st.write(
    "Das Foto soll direkt im Zeichenfenster sichtbar sein. "
    "Zeichne beispielsweise Bäume, Bänke, "
    "Fahrradständer oder Beete darauf."
)

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
# 6. HINTERGRUNDBILD VORBEREITEN
# ==========================================

scale = min(
    1.0,
    850 / before_image.width,
    650 / before_image.height
)

canvas_width = max(
    1,
    int(before_image.width * scale)
)

canvas_height = max(
    1,
    int(before_image.height * scale)
)

background = before_image.resize(
    (canvas_width, canvas_height),
    Image.Resampling.LANCZOS
)

background = background.convert("RGBA")


# ==========================================
# 7. ZEICHENFENSTER MIT FOTO
# ==========================================

st.subheader("Zeichne direkt auf deinem Foto")

canvas_result = st_canvas(

    fill_color="rgba(0, 0, 0, 0)",

    stroke_width=stroke_width,

    stroke_color=stroke_color,

    background_image=background,

    background_image_fit="contain",

    update_streamlit=True,

    height=canvas_height,

    width=canvas_width,

    drawing_mode="freedraw",

    return_image_data=True,

    key=f"stadt_canvas_{image_hash}"

)


# ==========================================
# 8. ZEICHNUNG MIT FOTO VERBINDEN
# ==========================================

if canvas_result.image_data is None:

    st.warning(
        "Die Zeichenfläche wird vorbereitet. "
        "Bitte warte kurz oder zeichne einmal auf das Bild."
    )

    st.stop()


drawing_layer = Image.fromarray(
    canvas_result.image_data.astype(np.uint8)
).convert("RGBA")


background_rgba = background.convert("RGBA")

composed_image = Image.alpha_composite(
    background_rgba,
    drawing_layer
).convert("RGB")


after_image = composed_image.resize(
    before_image.size,
    Image.Resampling.LANCZOS
)


# ==========================================
# 9. ORIGINAL UND ZEICHNUNG ANZEIGEN
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
        caption="Deine Zeichnung auf dem Foto",
        use_container_width=True
    )


# ==========================================
# 10. DIFFERENZBILD ERSTELLEN
# ==========================================

if st.button(
    "🔍 Unterschiede erkennen und klassifizieren",
    type="primary",
    use_container_width=True
):

    before_array = np.array(before_image)
    after_array = np.array(after_image)

    difference = cv2.absdiff(
        before_array,
        after_array
    )

    difference_gray = cv2.cvtColor(
        difference,
        cv2.COLOR_RGB2GRAY
    )

    _, mask = cv2.threshold(
        difference_gray,
        25,
        255,
        cv2.THRESH_BINARY
    )

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


    # ==========================================
    # 11. DIFFERENZBILD ANZEIGEN
    # ==========================================

    st.header("4. Erkannte Veränderungen")

    col1, col2 = st.columns(2)

    with col1:

        st.image(
            difference,
            caption="Differenzbild",
            use_container_width=True
        )

    with col2:

        st.image(
            mask,
            caption="Veränderte Bereiche",
            use_container_width=True
        )


    # ==========================================
    # 12. VERÄNDERTE BEREICHE FINDEN
    # ==========================================

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
            "Es wurden keine Veränderungen gefunden."
        )

    else:

        results = []
        result_image = after_array.copy()


        # ==========================================
        # 13. BEREICHE KLASSIFIZIEREN
        # ==========================================

        for index, contour in enumerate(
            valid_contours,
            start=1
        ):

            x, y, w, h = cv2.boundingRect(contour)

            padding = 30

            x1 = max(0, x - padding)
            y1 = max(0, y - padding)

            x2 = min(
                after_image.width,
                x + w + padding
            )

            y2 = min(
                after_image.height,
                y + h + padding
            )

            crop = after_image.crop(
                (x1, y1, x2, y2)
            )

            crop = crop.resize((224, 224))

            image_array = np.asarray(
                crop,
                dtype=np.float32
            )

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
            )[0]

            class_index = int(np.argmax(prediction))
            confidence = float(prediction[class_index])
            detected_class = labels[class_index]

            results.append({

                "Bereich": index,
                "Kategorie": detected_class,
                "Sicherheit": confidence,
                "Ausschnitt": crop,
                "Position": (x1, y1, x2, y2)

            })


            cv2.rectangle(
                result_image,
                (x1, y1),
                (x2, y2),
                (0, 255, 0),
                4
            )

            cv2.putText(
                result_image,
                str(index),
                (x1, max(25, y1 - 10)),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 255, 0),
                3
            )


        # ==========================================
        # 14. ERGEBNISÜBERSICHT
        # ==========================================

        st.header("5. Ergebnisse der KI")

        categories = Counter(
            result["Kategorie"]
            for result in results
        )

        for category in EXPECTED_CLASSES:

            st.write(
                f"**{category}:** "
                f"{categories.get(category, 0)}"
            )

        st.divider()


        # ==========================================
        # 15. EINZELNE ERGEBNISSE
        # ==========================================

        for result in results:

            st.subheader(
                f"Veränderter Bereich {result['Bereich']}"
            )

            col1, col2 = st.columns(2)

            with col1:

                st.image(
                    result["Ausschnitt"],
                    caption="Analysierter Ausschnitt",
                    use_container_width=True
                )

            with col2:

                st.write(
                    f"**Kategorie:** {result['Kategorie']}"
                )

                st.write(
                    f"**Sicherheit:** "
                    f"{result['Sicherheit'] * 100:.1f} %"
                )

                st.write(
                    f"**Position:** {result['Position']}"
                )

                if result["Sicherheit"] < 0.60:

                    st.warning("Das Modell ist unsicher.")

                else:

                    st.success("Kategorie erkannt.")


        # ==========================================
        # 16. MARKIERTES GESAMTBILD
        # ==========================================

        st.header("6. Alle Veränderungen im Stadtbild")

        st.image(
            result_image,
            caption="Markierte Veränderungen",
            use_container_width=True
        )
