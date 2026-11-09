
import tensorflow as tf
import numpy as np
from PIL import Image, ImageOps

# Modell laden
model = tf.keras.models.load_model(
    "model/keras_model.h5",
    compile=False
)

# Kategorien laden
with open("model/labels.txt", "r", encoding="utf-8") as file:
    labels = [
        line.strip().split(" ", 1)[-1]
        for line in file.readlines()
    ]

def classify_image(image_path):

    # Bild öffnen
    image = Image.open(image_path).convert("RGB")

    # Auf die Modellgröße vorbereiten
    image = ImageOps.fit(
        image,
        (224, 224),
        Image.Resampling.LANCZOS
    )

    # In NumPy umwandeln
    image_array = np.asarray(image)

    # Pixelwerte normalisieren
    normalized = (
        image_array.astype(np.float32) / 127.5
    ) - 1

    # Batch-Dimension ergänzen
    data = np.expand_dims(normalized, axis=0)

    # Vorhersage durchführen
    prediction = model.predict(data, verbose=0)[0]

    # Beste Kategorie finden
    index = np.argmax(prediction)

    category = labels[index]
    confidence = float(prediction[index])

    return category, confidence


# TEST
category, confidence = classify_image("testbild.jpg")

print("Erkannt:", category)
print("Sicherheit:", round(confidence * 100, 2), "%")
