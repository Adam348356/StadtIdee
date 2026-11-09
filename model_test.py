
import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image, ImageOps



st.write("Python-Version:", __import__("sys").version)
st.write("TensorFlow-Version:", tf.__version__)
st.write("Keras-Version:", tf.keras.__version__)
# Modell laden und zwischenspeichern
@st.cache_resource
def load_model():
    return tf.keras.models.load_model(
        "model/keras_model.h5",
        compile=False
    )

model = load_model()

# Kategorien laden
with open("model/labels.txt", "r", encoding="utf-8") as f:
    labels = [
        line.strip().split(" ", 1)[-1]
        for line in f.readlines()
    ]

# Bild hochladen und testen
st.subheader("Mein Modell testen")

uploaded_image = st.file_uploader(
    "Lade ein Bild hoch",
    type=["jpg", "jpeg", "png"]
)

if uploaded_image is not None:
    image = Image.open(uploaded_image).convert("RGB")
    st.image(image, caption="Dein Testbild")

    # Bild für Teachable Machine vorbereiten
    prepared_image = ImageOps.fit(
        image, (224, 224), Image.Resampling.LANCZOS
    )

    image_array = np.asarray(prepared_image)
    normalized = (image_array.astype(np.float32) / 127.5) - 1
    data = np.expand_dims(normalized, axis=0)

    # Vorhersage
    prediction = model.predict(data, verbose=0)[0]
    index = np.argmax(prediction)

    st.success(f"Erkannt: {labels[index]}")
    st.write(
        f"Sicherheit: {prediction[index] * 100:.2f} %"
    )

    st.write("Alle Ergebnisse:")
    for label, probability in zip(labels, prediction):
        st.write(f"{label}: {probability * 100:.2f} %")
