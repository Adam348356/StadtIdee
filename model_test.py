
import streamlit as st
import tensorflow as tf
import numpy as np
import cv2

from PIL import Image, ImageOps, ImageDraw


st.set_page_config(
    page_title="Meine Stadt. Meine Idee.",
    page_icon="🌳",
    layout="wide"
)

st.title("Meine Stadt. Meine Idee. 🌳")
st.write("Vergleiche deine Stadtbilder und erkenne neue Objekte.")


# --------------------------------
# Modell-Kompatibilität
# --------------------------------

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
    with open("model/labels.txt", "r", encoding="utf-8") as file:
        return [
            line.strip().split(" ", 1)[-1]
            for line in file.readlines()
        ]


try:
    model = load_model()
    labels = load_labels()
    st.success("Modell erfolgreich geladen!")

except Exception as error:
    st.error("Fehler beim Laden des Modells:")
    st.exception(error)
    st.stop()


# --------------------------------
# Objekt klassifizieren
# --------------------------------

def classify_crop(crop):

    image = Image.fromarray(
        cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
    )

    image = ImageOps.fit(
        image,
        (224, 224),
        Image.Resampling.LANCZOS
    )

    image_array = np.asarray(image)

    normalized = (
        image_array.astype(np.float32) / 127.5
    ) - 1

    data = np.expand_dims(normalized, axis=0)

    prediction = model.predict(data, verbose=0)[0]

    index = np.argmax(prediction)

    return labels[index], float(prediction[index])


# --------------------------------
# Bilder hochladen
# --------------------------------

st.subheader("1. Stadtbilder hochladen")

col1, col2 = st.columns(2)

with col1:
    before_file = st.file_uploader(
        "Vorher-Bild",
        type=["jpg", "jpeg", "png"],
        key="before"
    )

with col2:
    after_file = st.file_uploader(
        "Nachher-Bild",
        type=["jpg", "jpeg", "png"],
        key="after"
    )


# --------------------------------
# Bilder vergleichen
# --------------------------------

if before_file is not None and after_file is not None:

    before = Image.open(before_file).convert("RGB")
    after = Image.open(after_file).convert("RGB")

    # Nachher-Bild auf gleiche Größe bringen
    after = after.resize(
        before.size,
        Image.Resampling.LANCZOS
    )

    before_array = np.array(before)
    after_array = np.array(after)

    st.subheader("2. Vorher und Nachher")

    col1, col2 = st.columns(2)

    with col1:
        st.image(before, caption="Vorher")

    with col2:
        st.image(after, caption="Nachher")

    # --------------------------------
    # Bilddifferenz berechnen
    # --------------------------------

    st.subheader("3. Veränderungen erkennen")

    threshold = st.slider(
        "Empfindlichkeit des Vergleichs",
        min_value=10,
        max_value=100,
        value=30
    )

    min_area = st.slider(
        "Mindestgröße einer Veränderung",
        min_value=100,
        max_value=5000,
        value=500,
        step=100
    )

    before_bgr = cv2.cvtColor(
        before_array,
        cv2.COLOR_RGB2BGR
    )

    after_bgr = cv2.cvtColor(
        after_array,
        cv2.COLOR_RGB2BGR
    )

    difference = cv2.absdiff(
        before_bgr,
        after_bgr
    )

    gray = cv2.cvtColor(
        difference,
        cv2.COLOR_BGR2GRAY
    )

    _, mask = cv2.threshold(
        gray,
        threshold,
        255,
        cv2.THRESH_BINARY
    )

    kernel = np.ones((5, 5), np.uint8)

    mask = cv2.morphologyEx(
        mask,
        cv2.MORPH_OPEN,
        kernel
    )

    mask = cv2.dilate(
        mask,
        kernel,
        iterations=2
    )

    contours, _ = cv2.findContours(
        mask.copy(),
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    # --------------------------------
    # Veränderungen klassifizieren
    # --------------------------------

    result_image = after.copy()
    draw = ImageDraw.Draw(result_image)

    results = []

    for contour in contours:

        area = cv2.contourArea(contour)

        if area < min_area:
            continue

        x, y, w, h = cv2.boundingRect(contour)

        padding = 15

        x1 = max(0, x - padding)
        y1 = max(0, y - padding)

        x2 = min(after.width, x + w + padding)
        y2 = min(after.height, y + h + padding)

        crop = after_bgr[y1:y2, x1:x2]

        if crop.size == 0:
            continue

        category, confidence = classify_crop(crop)

        results.append({
            "category": category,
            "confidence": confidence,
            "position": (x, y, w, h),
            "crop": crop
        })

        draw.rectangle(
            [x1, y1, x2, y2],
            outline="red",
            width=4
        )

    st.subheader("4. Erkannte Veränderungen")

    st.image(
        result_image,
        caption="Nachher-Bild mit markierten Veränderungen"
    )

    if len(results) == 0:

        st.info(
            "Keine ausreichend großen Veränderungen gefunden. "
            "Versuche die Empfindlichkeit oder Mindestgröße anzupassen."
        )

    else:

        st.success(
            f"{len(results)} veränderte Bildbereiche gefunden."
        )

        for i, result in enumerate(results):

            st.markdown(f"### Veränderung {i + 1}")

            crop_rgb = cv2.cvtColor(
                result["crop"],
                cv2.COLOR_BGR2RGB
            )

            st.image(
                crop_rgb,
                caption="Erkannter Bildausschnitt",
                width=250
            )

            st.write(
                f"**Kategorie:** {result['category']}"
            )

            st.write(
                f"**Sicherheit:** "
                f"{result['confidence'] * 100:.2f} %"
            )

            x, y, w, h = result["position"]

            st.write(
                f"**Position:** X={x}, Y={y}"
            )

            st.divider()
