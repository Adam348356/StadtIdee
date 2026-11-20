"""Loading and inference helpers for the existing Teachable Machine model."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

EXPECTED_CLASSES = ("Bäume", "Beete", "Fahrradständer", "Bänke")
LABEL_ALIASES = {"Blumenbeete": "Beete"}


def read_labels(path: Path) -> list[str]:
    """Read Teachable Machine labels and map its Blumenbeete label to Beete."""
    labels: list[str] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        value = line.strip()
        if not value:
            continue
        parts = value.split(maxsplit=1)
        if len(parts) == 2 and parts[0].isdigit():
            value = parts[1].strip()
        labels.append(LABEL_ALIASES.get(value, value))
    if len(labels) != 4 or set(labels) != set(EXPECTED_CLASSES):
        raise ValueError("labels.txt muss genau diese vier Klassen enthalten: " + ", ".join(EXPECTED_CLASSES))
    return labels


def load_model(model_path: str):
    """Load the supplied H5 model without compiling its training configuration."""
    from keras.layers import DepthwiseConv2D
    from keras.models import load_model as keras_load_model

    class CompatibleDepthwiseConv2D(DepthwiseConv2D):
        """Ignore the redundant groups=1 field written by newer Keras versions."""

        def __init__(self, *args, groups: int = 1, **kwargs):
            if groups != 1:
                raise ValueError("Das vorhandene Modell verwendet einen nicht unterstützten Gruppenwert.")
            super().__init__(*args, **kwargs)

    return keras_load_model(
        model_path,
        compile=False,
        custom_objects={"DepthwiseConv2D": CompatibleDepthwiseConv2D},
    )


def classify_crop(model, crop_rgb: np.ndarray, labels: list[str]) -> tuple[str, float]:
    """Classify an RGB crop using the dimensions and normalization expected by TM."""
    if crop_rgb.size == 0 or crop_rgb.shape[0] < 8 or crop_rgb.shape[1] < 8:
        raise ValueError("Der erkannte Ausschnitt ist zu klein.")
    shape = model.input_shape
    if isinstance(shape, list):
        shape = shape[0]
    height = int(shape[1]) if len(shape) == 4 and shape[1] else 224
    width = int(shape[2]) if len(shape) == 4 and shape[2] else 224
    rgb = cv2.resize(crop_rgb[:, :, :3], (width, height), interpolation=cv2.INTER_AREA)
    batch = np.expand_dims(rgb.astype(np.float32) / 127.5 - 1.0, axis=0)
    predictions = np.asarray(model.predict(batch, verbose=0)).reshape(-1)
    if predictions.size != len(labels):
        raise ValueError(f"Das Modell liefert {predictions.size} Ausgaben; erwartet werden {len(labels)}.")
    if not np.all(np.isfinite(predictions)):
        raise ValueError("Das Modell hat ungültige Konfidenzwerte geliefert.")
    index = int(np.argmax(predictions))
    return labels[index], float(predictions[index])
