
import streamlit as st
import tensorflow as tf
import numpy as np
import cv2

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
    "Gestalte deine Stadt selbst: Lade ein Foto hoch, "
    "zeichne deine Ideen darauf und lasse die KI "
    "deine Veränderungen erkennen."
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

    st.error(
        f"Das KI-Modell konnte nicht geladen werden: {error}"
    )

    st.stop()


# ==========================================
# 3. FOTO HOCHLADEN
# ==========================================

st.header("1. Lade dein Stadtfoto hoch")

uploaded_file = st.file_uploader(
    "Wähle ein Foto aus",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:

    before_image = Image.open(
        uploaded_file
    ).convert("RGB")

    st.success("Dein Foto wurde hochgeladen.")

    # ==========================================
    # 4. ZEICHENFENSTER
    # ==========================================

    st.header("2. Gestalte deine Stadt")

    st.write(
        "Zeichne direkt auf das Foto. "
        "Du kannst beispielsweise einen Baum, "
        "eine Bank oder ein Blumenbeet hinzufügen."
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

    # Zeichenfläche passend zum Foto berechnen

    scale = min(
        1.0,
        900 / before_image.width,
        700 / before_image.height
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
        (canvas_width, canvas_height)
    )

    # Zeichenfläche

    canvas_result = st_canvas(

        fill_color="rgba(0, 0, 0, 0)",

        stroke_width=stroke_width,

        stroke_color=stroke_color,

        background_image=background,

        update_streamlit=True,

        height=canvas_height,

        width=canvas_width,

        drawing_mode="freedraw",

        key="stadt_zeichenfenster"
    )

    # ==========================================
    # 5. BEARBEITETES FOTO ERSTELLEN
    # ==========================================

    if canvas_result.image_data is not None:

        after_image = Image.fromarray(
            canvas_result.image_data.astype("uint8")
        ).convert("RGB")

        # Auf ursprüngliche Bildgröße zurückskalieren

        after_image = after_image.resize(
            before_image.size,
            Image.Resampling.LANCZOS
        )

        st.header("3. Dein fertiges Projekt")

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
                caption="Deine Stadtidee",
                use_container_width=True
            )

        # ==========================================
        # 6. DIFFERENZBILD UND KI-ERKENNUNG
        # ==========================================

        st.header("4. Veränderungen analysieren")

        if st.button(
            "🔍 Unterschiede erkennen und klassifizieren",
            type="primary",
            use_container_width=True
        ):

            before_array = np.array(
                before_image
            )

            after_array = np.array(
                after_image
            )

            # Farbunterschiede berechnen

            difference = cv2.absdiff(
                before_array,
                after_array
            )

            # Differenzbild in eine Graustufenmaske umwandeln

            difference_gray = cv2.cvtColor(
                difference,
                cv2.COLOR_RGB2GRAY
            )

            # Kleine Unterschiede herausfiltern

            threshold_value = 25

            _, mask = cv2.threshold(
                difference_gray,
                threshold_value,
                255,
                cv2.THRESH_BINARY
            )

            # Benachbarte Zeichnungsbereiche verbinden

            kernel_close = np.ones(
                (15, 15),
                np.uint8
            )

            mask = cv2.morphologyEx(
                mask,
                cv2.MORPH_CLOSE,
                kernel_close
            )

            # Bereiche etwas vergrößern

            kernel_dilate = np.ones(
                (9, 9),
                np.uint8
            )

            mask = cv2.dilate(
                mask,
                kernel_dilate,
                iterations=2
            )

            # ==========================================
            # 7. DIFFERENZBILD ANZEIGEN
            # ==========================================

            st.subheader("Dein Differenzbild")

            col1, col2 = st.columns(2)

            with col1:

                st.image(
                    difference,
                    caption="Bild mit den Farbunterschieden",
                    use_container_width=True
                )

            with col2:

                st.image(
                    mask,
                    caption="Erkannte veränderte Bereiche",
                    use_container_width=True
                )

            # ==========================================
            # 8. VERÄNDERTE BEREICHE FINDEN
            # ==========================================

            contours, _ = cv2.findContours(
                mask,
                cv2.RETR_EXTERNAL,
                cv2.CHAIN_APPROX_SIMPLE
            )

            # Sehr kleine Bereiche ignorieren

            valid_contours = [
                contour
                for contour in contours
                if cv2.contourArea(contour) > 100
            ]

            if not valid_contours:

                st.warning(
                    "Es wurden keine deutlichen Veränderungen "
                    "gefunden. Zeichne bitte etwas auf das Foto."
                )

            else:

                st.success(
                    f"{len(valid_contours)} veränderte Bereiche gefunden."
                )

                results = []

                result_image = after_array.copy()

                # ==========================================
                # 9. JEDEN BEREICH EINZELN KLASSIFIZIEREN
                # ==========================================

                for index, contour in enumerate(
                    valid_contours,
                    start=1
                ):

                    x, y, w, h = cv2.boundingRect(
                        contour
                    )

                    # Rand um den Ausschnitt hinzufügen

                    padding = 30

                    x1 = max(
                        0,
                        x - padding
                    )

                    y1 = max(
                        0,
                        y - padding
                    )

                    x2 = min(
                        after_image.width,
                        x + w + padding
                    )

                    y2 = min(
                        after_image.height,
                        y + h + padding
                    )

                    # Ausschnitt erstellen

                    crop = after_image.crop(
                        (x1, y1, x2, y2)
                    )

                    # Modell benötigt 224 x 224 Pixel

                    crop = crop.resize(
                        (224, 224)
                    )

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

                    # Vorhersage des trainierten Modells

                    prediction = model.predict(
                        image_array,
                        verbose=0
                    )[0]

                    class_index = int(
                        np.argmax(prediction)
                    )

                    confidence = float(
                        prediction[class_index]
                    )

                    detected_class = labels[
                        class_index
                    ]

                    # Ergebnis speichern

                    results.append({

                        "Bereich": index,

                        "Kategorie": detected_class,

                        "Sicherheit": confidence,

                        "Ausschnitt": crop,

                        "Position": (x1, y1, x2, y2)

                    })

                    # Rahmen ins Ergebnisbild zeichnen

                    cv2.rectangle(
                        result_image,
                        (x1, y1),
                        (x2, y2),
                        (0, 255, 0),
                        4
                    )

                    # Nummer des Bereichs anzeigen

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
                # 10. ERGEBNISSE ANZEIGEN
                # ==========================================

                st.header("5. Ergebnisse der KI")

                st.subheader(
                    "Übersicht der erkannten Veränderungen"
                )

                categories = Counter(
                    result["Kategorie"]
                    for result in results
                )

                for category, count in categories.items():

                    st.write(
                        f"**{category}:** {count}"
                    )

                st.divider()

                # Jeden Bereich anzeigen

                for result in results:

                    st.subheader(
                        f"Veränderter Bereich {result['Bereich']}"
                    )

                    col1, col2 = st.columns(2)

                    with col1:

                        st.image(
                            result["Ausschnitt"],
                            caption="Analysierter Bildausschnitt",
                            use_container_width=True
                        )

                    with col2:

                        st.write(
                            f"**Erkannte Kategorie:** "
                            f"{result['Kategorie']}"
                        )

                        st.write(
                            f"**Sicherheit:** "
                            f"{result['Sicherheit'] * 100:.1f} %"
                        )

                        if result["Sicherheit"] < 0.60:

                            st.warning(
                                "Das Modell ist sich bei dieser "
                                "Erkennung noch unsicher."
                            )

                        else:

                            st.success(
                                "Das Modell hat eine Kategorie erkannt."
                            )

                        st.write(
                            f"**Position:** "
                            f"{result['Position']}"
                        )

                    st.divider()

                # ==========================================
                # 11. GESAMTBILD MIT MARKIERUNGEN
                # ==========================================

                st.header("6. Alle Veränderungen im Stadtbild")

                st.image(
                    result_image,
                    caption="Markierte veränderte Bereiche",
                    use_container_width=True
                )

else:

    st.info(
        "Lade zuerst ein Stadtfoto hoch. "
        "Danach kannst du direkt darauf zeichnen."
    )
