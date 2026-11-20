"""Image preparation and annotation-region detection for the city idea app."""

from __future__ import annotations

import cv2
import numpy as np
from PIL import Image

CANVAS_MAX_WIDTH = 1000


def prepare_for_canvas(image: Image.Image) -> tuple[Image.Image, int, int]:
    """Convert uploads to RGB and scale large photos to a manageable canvas."""
    image = image.convert("RGB")
    scale = min(1.0, CANVAS_MAX_WIDTH / image.width)
    width = max(1, round(image.width * scale))
    height = max(1, round(image.height * scale))
    return image.resize((width, height), Image.Resampling.LANCZOS), width, height


def find_drawings(original: np.ndarray, composite: np.ndarray) -> list[tuple[int, int, int, int]]:
    """Return bounding boxes around visible changes made on the canvas."""
    if original.shape[:2] != composite.shape[:2]:
        raise ValueError("Original und bearbeitetes Bild haben unterschiedliche Abmessungen.")
    if original.ndim != 3 or composite.ndim != 3 or original.shape[2] < 3 or composite.shape[2] < 3:
        raise ValueError("Für die Bereichserkennung werden RGB-Bilder benötigt.")
    diff = cv2.absdiff(composite[:, :, :3], original[:, :, :3])
    changed = np.max(diff, axis=2) >= 28
    mask = changed.astype(np.uint8) * 255
    kernel = np.ones((3, 3), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    count, _, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    boxes = []
    for i in range(1, count):
        x, y, w, h, area = map(int, stats[i])
        if area >= 30 and w >= 3 and h >= 3:
            boxes.append((x, y, w, h))
    return sorted(boxes, key=lambda box: (box[1], box[0]))


def composite_canvas(original: np.ndarray, canvas_rgba: np.ndarray) -> np.ndarray:
    """Place the canvas drawing layer over the original RGB photo."""
    if original.ndim != 3 or original.shape[2] < 3:
        raise ValueError("Das Originalfoto muss ein RGB-Bild sein.")
    if canvas_rgba.ndim != 3 or canvas_rgba.shape[2] < 4:
        raise ValueError("Die Canvas-Zeichnung muss RGBA-Daten enthalten.")
    if canvas_rgba.shape[:2] != original.shape[:2]:
        canvas_rgba = cv2.resize(
            canvas_rgba,
            (original.shape[1], original.shape[0]),
            interpolation=cv2.INTER_NEAREST,
        )
    alpha = canvas_rgba[:, :, 3:4].astype(np.float32) / 255.0
    foreground = canvas_rgba[:, :, :3].astype(np.float32)
    background = original[:, :, :3].astype(np.float32)
    return np.clip(foreground * alpha + background * (1.0 - alpha), 0, 255).astype(np.uint8)


def white_background_crop(
    canvas_rgba: np.ndarray,
    box: tuple[int, int, int, int],
    padding_fraction: float = 0.18,
) -> np.ndarray:
    """Isolate one drawn region on a square white background for model inference.

    The original photo is deliberately not included. Transparent pixels from the
    drawable canvas become white; only the user's strokes remain in the crop.
    The square canvas preserves the drawing's proportions when resized by the
    Teachable Machine classifier.
    """
    if canvas_rgba.ndim != 3 or canvas_rgba.shape[2] < 4:
        raise ValueError("Die Zeichenebene muss RGBA-Daten enthalten.")
    x, y, width, height = box
    if width <= 0 or height <= 0:
        raise ValueError("Der erkannte Zeichenbereich ist leer.")
    if padding_fraction < 0:
        raise ValueError("Der Rand darf nicht negativ sein.")

    margin = max(4, round(max(width, height) * padding_fraction))
    side = max(width, height) + 2 * margin
    center_x, center_y = x + width / 2, y + height / 2
    source_x0 = round(center_x - side / 2)
    source_y0 = round(center_y - side / 2)
    source_x1, source_y1 = source_x0 + side, source_y0 + side

    layer = np.zeros((side, side, 4), dtype=np.uint8)
    image_height, image_width = canvas_rgba.shape[:2]
    copy_x0, copy_y0 = max(0, source_x0), max(0, source_y0)
    copy_x1, copy_y1 = min(image_width, source_x1), min(image_height, source_y1)
    if copy_x1 <= copy_x0 or copy_y1 <= copy_y0:
        raise ValueError("Der Zeichenbereich liegt außerhalb der Zeichenfläche.")

    target_x0, target_y0 = copy_x0 - source_x0, copy_y0 - source_y0
    layer[
        target_y0 : target_y0 + (copy_y1 - copy_y0),
        target_x0 : target_x0 + (copy_x1 - copy_x0),
    ] = canvas_rgba[copy_y0:copy_y1, copy_x0:copy_x1, :4]

    alpha = layer[:, :, 3:4].astype(np.float32) / 255.0
    foreground = layer[:, :, :3].astype(np.float32)
    white = np.full_like(foreground, 255, dtype=np.float32)
    return np.clip(foreground * alpha + white * (1.0 - alpha), 0, 255).astype(np.uint8)


def annotate_regions(image: np.ndarray, regions: list[dict]) -> np.ndarray:
    """Copy an RGB image and mark classified regions with labels and confidence."""
    annotated = image.copy()
    for item in regions:
        x, y, w, h = item["box"]
        color = (0, 180, 70)
        cv2.rectangle(annotated, (x, y), (x + w, y + h), color, 2)
        text = f'{item["label"]} {item["confidence"]:.0%}'
        cv2.putText(annotated, text, (x, max(18, y - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
    return annotated
