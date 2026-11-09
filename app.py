
import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image, ImageOps

st.set_page_config(
    page_title="Meine Stadt. Meine Idee.",
    page_icon="🌳",
    layout="wide"
)

st.title("Meine Stadt. Meine Idee. 🌳")
st.write("Teste dein Teachable-Machine-Modell!")

# --------------------------------
# Modell kompatibel laden
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


# --------------------------------
# Kategorien laden
# --------------------------------

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


# --------------------------------
# Modell starten
# --------------------------------

try:

    model = load_model()
    labels = load_labels()

    st.success("Dein Modell wurde erfolgreich geladen!")

except Exception as error:

    st.error("Das Modell konnte nicht geladen werden.")
    st.code(str(error))
    st.stop()


# --------------------------------
# Testbild hochladen
# --------------------------------

st.subheader("Modell testen")

uploaded_image = st.file_uploader(
    "Lade ein Bild hoch",
    type=["jpg", "jpeg", "png"]
)


if uploaded_image is not None:

    image = Image.open(
        uploaded_image
    ).convert("RGB")

    st.image(
        image,
        caption="Dein Testbild",
        use_container_width=True
    )

    # Bild vorbereiten
    prepared_image = ImageOps.fit(
        image,
        (224, 224),
        Image.Resampling.LANCZOS
    )

    image_array = np.asarray(
        prepared_image
    )

    normalized = (
        image_array.astype(np.float32) / 127.5
    ) - 1

    data = np.expand_dims(
        normalized,
        axis=0
    )

    # Vorhersage
    prediction = model.predict(
        data,
        verbose=0
    )[0]

    index = np.argmax(prediction)

    # Ergebnis anzeigen
    st.subheader("Ergebnis")

    st.success(
        f"Erkannt: {labels[index]}"
    )

    st.metric(
        "Sicherheit",
        f"{prediction[index] * 100:.2f} %"
    )

    # Alle Kategorien anzeigen
    st.subheader("Alle Erkennungen")

    for label, probability in zip(
        labels,
        prediction
    ):

        st.write(
            f"**{label}:** {probability * 100:.2f} %"
        )

        st.progress(
            float(probability)
        )
