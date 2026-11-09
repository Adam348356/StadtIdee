
import streamlit as st
import tensorflow as tf
import numpy as np
import cv2
import hashlib

from PIL import Image
from streamlit_drawable_canvas import st_canvas


st.set_page_config(
    page_title="Meine Stadt. Meine Idee.",
    layout="wide"
)

st.title("Meine Stadt. Meine Idee.")
st.write("Lade ein Stadtfoto hoch und male deine Ideen direkt darauf.")


# --------------------------------
# 1. Modell laden
# --------------------------------

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
    st.error(f"Modell konnte nicht geladen werden: {error}")
    st.stop()


# --------------------------------
# 2. Foto hochladen
# --------------------------------

uploaded_file = st.file_uploader(
    "Lade dein Stadtfoto hoch",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:

    before_image = Image.open(uploaded_file).convert("RGB")

    st.subheader("Dein Originalfoto")
    st.image(before_image, use_container_width=True)

    # Eindeutiger Schlüssel für jedes Foto
    image_key = hashlib.md5(
        uploaded_file.getvalue()
    ).hexdigest()

    # --------------------------------
    # 3. Zeichenfläche vorbereiten
    # --------------------------------

    st.subheader("Male deine Idee auf das Foto")

    stroke_color = st.color_picker(
        "Farbe deiner Zeichnung",
        "#228B22"
    )

    stroke_width = st.slider(
        "Dicke des Stifts",
        2,
        40,
        12
    )

    scale = min(
        1.0,
        900 / before_image.width,
        700 / before_image.height
    )

    canvas_width = max(1, int(before_image.width * scale))
    canvas_height = max(1, int(before_image.height * scale))

    canvas_background = before_image.resize(
        (canvas_width, canvas_height),
        Image.Resampling.LANCZOS
    )

    # --------------------------------
    # 4. Zeichenfläche mit Foto
    # --------------------------------

    canvas_result = st_canvas(
        fill_color="rgba(0, 0, 0, 0)",
        stroke_width=stroke_width,
        stroke_color=stroke_color,

        background_image=canvas_background,

        update_streamlit=True,
        height=canvas_height,
        width=canvas_width,

        drawing_mode="freedraw",

        return_image_data=True,
        key=f"stadt_canvas_{image_key}"
    )

    # --------------------------------
    # 5. Gezeichnetes Bild erstellen
    # --------------------------------

    if canvas_result.image_data is not None:

        after_image = Image.fromarray(
            canvas_result.image_data.astype("uint8")
        ).convert("RGB")

        after_image = after_image.resize(
            before_image.size,
            Image.Resampling.LANCZOS
        )

        st.subheader("Deine neue Stadtidee")
        st.image(after_image, use_container_width=True)

        # --------------------------------
        # 6. Zeichnung erkennen
        # --------------------------------

        if st.button("Meine Zeichnung erkennen"):

            before_array = np.array(before_image)
            after_array = np.array(after_image)

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

            # --------------------------------
            # 7. Veränderung finden
            # --------------------------------

            contours, _ = cv2.findContours(
                mask,
                cv2.RETR_EXTERNAL,
                cv2.CHAIN_APPROX_SIMPLE
            )

            valid_contours = [
                contour for contour in contours
                if cv2.contourArea(contour) > 100
            ]

            if not valid_contours:

                st.warning("Keine Zeichnung erkannt.")

            else:

                boxes = [
                    cv2.boundingRect(contour)
                    for contour in valid_contours
                ]

                x1 = min(x for x, y, w, h in boxes)
                y1 = min(y for x, y, w, h in boxes)
                x2 = max(x + w for x, y, w, h in boxes)
                y2 = max(y + h for x, y, w, h in boxes)

                padding = 30

                x1 = max(0, x1 - padding)
                y1 = max(0, y1 - padding)
                x2 = min(after_image.width, x2 + padding)
                y2 = min(after_image.height, y2 + padding)

                crop = after_image.crop(
                    (x1, y1, x2, y2)
                )

                st.subheader("Erkannte Veränderung")
                st.image(crop)

                # --------------------------------
                # 8. Modell klassifiziert Ausschnitt
                # --------------------------------

                crop = crop.resize((224, 224))

                image_array = np.asarray(
                    crop,
                    dtype=np.float32
                )

                image_array = image_array / 127.5 - 1
                image_array = np.expand_dims(image_array, axis=0)

                prediction = model.predict(
                    image_array,
                    verbose=0
                )

                class_index = int(np.argmax(prediction[0]))
                confidence = float(prediction[0][class_index])

                detected_class = labels[class_index]

                st.subheader("Ergebnis")

                st.success(f"Erkannt: {detected_class}")

                st.write(
                    f"Sicherheit: {confidence * 100:.1f} %"
                )

                if confidence < 0.60:
                    st.warning(
                        "Das Modell ist sich bei dieser Erkennung unsicher."
                    )

                # --------------------------------
                # 9. Position anzeigen
                # --------------------------------

                result_image = after_array.copy()

                cv2.rectangle(
                    result_image,
                    (x1, y1),
                    (x2, y2),
                    (0, 255, 0),
                    4
                )

                st.subheader("Position deiner Zeichnung")
                st.image(
                    result_image,
                    use_container_width=True
                )

else:
    st.info("Bitte lade zuerst ein Stadtfoto hoch.")
