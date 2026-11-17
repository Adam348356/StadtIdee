"""Meine Stadt. Meine Idee. – zeichne Ergänzungen direkt auf ein Ortsfoto."""

from __future__ import annotations

from collections import Counter
from pathlib import Path

import cv2
import numpy as np
import streamlit as st
from PIL import Image
from streamlit_drawable_canvas import st_canvas


ROOT = Path(__file__).resolve().parent
MODEL_PATH = ROOT / "model" / "keras_model.h5"
LABELS_PATH = ROOT / "model" / "labels.txt"
EXPECTED_CLASSES = ["Bäume", "Blumenbeete", "Fahrradständer", "Bänke"]
CANVAS_MAX_WIDTH = 1000

st.set_page_config(page_title="Meine Stadt. Meine Idee.", page_icon="🏙️", layout="wide")
st.title("Meine Stadt. Meine Idee.")
st.write("Lade ein Foto eines öffentlichen Ortes hoch und zeichne deine Ideen direkt darauf ein.")


def read_labels(path: Path) -> list[str]:
    """Read Teachable Machine labels, accepting either plain or index-prefixed lines."""
    labels = []
    for line in path.read_text(encoding="utf-8").splitlines():
        value = line.strip()
        if not value:
            continue
        parts = value.split(maxsplit=1)
        if len(parts) == 2 and parts[0].isdigit():
            value = parts[1].strip()
        labels.append(value)
    return labels


@st.cache_resource(show_spinner="KI-Modell wird geladen …")
def load_model(model_path: str):
    from tensorflow.keras.models import load_model as keras_load_model

    return keras_load_model(model_path, compile=False)


def prepare_for_canvas(image: Image.Image) -> tuple[Image.Image, int, int]:
    image = image.convert("RGB")
    scale = min(1.0, CANVAS_MAX_WIDTH / image.width)
    width = max(1, round(image.width * scale))
    height = max(1, round(image.height * scale))
    return image.resize((width, height), Image.Resampling.LANCZOS), width, height


def find_drawings(original: np.ndarray, composite: np.ndarray) -> list[tuple[int, int, int, int]]:
    """Find changed connected regions after suppressing antialiasing and tiny noise."""
    diff = cv2.absdiff(composite, original)
    # Require a visible RGB change; the alpha channel is deliberately ignored because
    # drawable-canvas may return the already composited background in that channel.
    changed = np.max(diff[:, :, :3], axis=2) >= 28
    mask = (changed.astype(np.uint8) * 255)
    kernel = np.ones((3, 3), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    count, _, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    boxes = []
    for i in range(1, count):
        x, y, w, h, area = map(int, stats[i])
        if area >= 30 and w >= 3 and h >= 3:
            boxes.append((x, y, w, h))
    return boxes


def classify_crop(model, crop: np.ndarray, labels: list[str]) -> tuple[str, float] | None:
    if crop.size == 0 or crop.shape[0] < 8 or crop.shape[1] < 8:
        return None
    # Teachable Machine image models expect a 224x224 RGB image normalized to [-1, 1].
    rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
    input_shape = model.input_shape
    target_height, target_width = (224, 224)
    if isinstance(input_shape, list):
        input_shape = input_shape[0]
    if len(input_shape) == 4 and input_shape[1] and input_shape[2]:
        target_height, target_width = int(input_shape[1]), int(input_shape[2])
    resized = cv2.resize(rgb, (target_width, target_height), interpolation=cv2.INTER_AREA)
    batch = np.expand_dims(resized.astype(np.float32) / 127.5 - 1.0, axis=0)
    predictions = np.asarray(model.predict(batch, verbose=0)).reshape(-1)
    if predictions.size != len(labels):
        raise ValueError(f"Das Modell liefert {predictions.size} Ausgaben; erwartet werden 4.")
    idx = int(np.argmax(predictions))
    return labels[idx], float(predictions[idx])


uploaded = st.file_uploader("Foto auswählen", type=["png", "jpg", "jpeg", "webp"])
if uploaded is None:
    st.info("Bitte lade zuerst ein Foto hoch.")
    st.stop()

try:
    original_image = Image.open(uploaded).convert("RGB")
except Exception as exc:
    st.error(f"Das Bild konnte nicht geöffnet werden: {exc}")
    st.stop()

canvas_image, canvas_width, canvas_height = prepare_for_canvas(original_image)
original_array = np.asarray(canvas_image)
st.subheader("Zeichne deine Ergänzungen ein")
st.caption("Wähle Farbe und Strichbreite. Nutze den Radierer oder die Schaltfläche zum Zurücksetzen.")
tool = st.selectbox("Werkzeug", ["freihand", "Rechteck", "Kreis"], index=0)
tool_map = {"freihand": "freedraw", "Rechteck": "rect", "Kreis": "circle"}
color = st.color_picker("Zeichenfarbe", "#e53935")
stroke = st.slider("Strichbreite", 2, 40, 8)
if st.button("Zeichnung zurücksetzen"):
    st.session_state["canvas_reset"] = st.session_state.get("canvas_reset", 0) + 1

canvas_result = st_canvas(
    fill_color="rgba(0, 0, 0, 0)",
    stroke_width=stroke,
    stroke_color=color,
    background_image=canvas_image,
    update_streamlit=True,
    height=canvas_height,
    width=canvas_width,
    drawing_mode=tool_map[tool],
    key=f"city_canvas_{st.session_state.get('canvas_reset', 0)}",
    display_toolbar=True,
    return_image_data=True,
)

if canvas_result.image_data is None:
    st.info("Zeichne oben eine oder mehrere Ideen ein.")
    st.stop()

canvas_rgba = np.asarray(canvas_result.image_data, dtype=np.uint8)
if canvas_rgba.shape[:2] != original_array.shape[:2]:
    canvas_rgba = cv2.resize(canvas_rgba, (canvas_width, canvas_height), interpolation=cv2.INTER_NEAREST)
composite = canvas_rgba[:, :, :3]
boxes = find_drawings(original_array, composite)

if not boxes:
    st.info("Es wurden noch keine sichtbaren Zeichnungen erkannt. Zeichne mit einer kontrastierenden Farbe.")
    st.stop()

st.subheader("Ergebnisse")
left, middle, right = st.columns(3)
left.image(canvas_image, caption="Originalfoto", use_container_width=True)
middle.image(composite, caption="Bearbeitetes Foto", use_container_width=True)

annotated = composite.copy()
try:
    if not MODEL_PATH.is_file():
        raise FileNotFoundError(f"Modelldatei fehlt: {MODEL_PATH.relative_to(ROOT)}")
    if not LABELS_PATH.is_file():
        raise FileNotFoundError(f"Label-Datei fehlt: {LABELS_PATH.relative_to(ROOT)}")
    labels = read_labels(LABELS_PATH)
    if len(labels) != 4 or set(labels) != set(EXPECTED_CLASSES):
        raise ValueError("labels.txt muss genau diese Klassen enthalten: " + ", ".join(EXPECTED_CLASSES))
    model = load_model(str(MODEL_PATH))
    if model.output_shape[-1] != 4:
        raise ValueError("Das Modell muss genau vier Ausgabeklassen besitzen.")
except Exception as exc:
    right.image(composite, caption="Erkannte Bereiche", use_container_width=True)
    st.error(f"Die Klassifikation ist derzeit nicht verfügbar: {exc}")
    st.caption("Lege keras_model.h5 und labels.txt im Unterordner model ab.")
    st.stop()

results = []
failures = []
for number, (x, y, w, h) in enumerate(boxes, start=1):
    # Add modest context around the drawn mark so the model can see its visual form.
    pad = max(4, round(max(w, h) * 0.12))
    x0, y0 = max(0, x - pad), max(0, y - pad)
    x1, y1 = min(canvas_width, x + w + pad), min(canvas_height, y + h + pad)
    crop_rgb = composite[y0:y1, x0:x1]
    try:
        prediction = classify_crop(model, cv2.cvtColor(crop_rgb, cv2.COLOR_RGB2BGR), labels)
        if prediction is None:
            failures.append(f"Bereich {number}: Ausschnitt zu klein.")
            continue
        label, confidence = prediction
        results.append({"Nr.": number, "Klasse": label, "Konfidenz": confidence, "Position (x, y)": f"{x}, {y}", "Größe": f"{w} × {h} px"})
        cv2.rectangle(annotated, (x, y), (x + w, y + h), (0, 200, 0), 2)
        cv2.putText(annotated, f"{label} {confidence:.0%}", (x, max(18, y - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 120, 0), 2)
    except Exception as exc:
        failures.append(f"Bereich {number}: {exc}")

right.image(annotated, caption="Bearbeitetes Foto mit Erkennungsrahmen", use_container_width=True)
if results:
    st.subheader("Erkannte Elemente")
    st.dataframe(results, use_container_width=True, hide_index=True)
    counts = Counter(item["Klasse"] for item in results)
    st.write("**Anzahl je Klasse:** " + " · ".join(f"{name}: {counts.get(name, 0)}" for name in EXPECTED_CLASSES))
else:
    st.warning("Es konnten keine Bereiche klassifiziert werden.")
if failures:
    st.warning("Einige Bereiche wurden übersprungen: " + "; ".join(failures))
