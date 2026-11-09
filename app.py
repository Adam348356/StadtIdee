
import streamlit as st
import numpy as np
from PIL import Image
from gradio_client import Client, handle_file
from streamlit_drawable_canvas import st_canvas
import tempfile
import os
import io

# ==========================================
# 1. SEITENEINSTELLUNGEN
# ==========================================

st.set_page_config(
    page_title="Meine Stadt. Meine Idee.",
    page_icon="🌳",
    layout="wide"
)

# Öffentlicher Hugging-Face-Space mit Qwen-Image-Edit-2511
SPACE_ID = "techfreakworm/qwen-image-editor"

# ==========================================
# 2. DESIGN
# ==========================================

st.markdown("""
<style>
    .stApp {
        background-color: #F5F8F5;
    }

    h1, h2, h3 {
        color: #245943;
    }

    .intro {
        background-color: #E4EFE6;
        padding: 20px;
        border-radius: 12px;
        margin-bottom: 20px;
    }

    .stButton button {
        background-color: #347653;
        color: white;
        border-radius: 8px;
        border: none;
        padding: 10px 20px;
    }
</style>
""", unsafe_allow_html=True)


# ==========================================
# 3. VERBINDUNG ZUM QWEN-MODELL
# ==========================================

def get_client():

    # Optionaler Hugging-Face-Token aus Streamlit Secrets
    token = st.secrets.get("HF_TOKEN", None)

    if token:
        return Client(
            SPACE_ID,
            hf_token=token
        )

    # Öffentlicher Zugriff ohne Token
    return Client(SPACE_ID)


# ==========================================
# 4. BILD VORBEREITEN
# ==========================================

def prepare_image(image):

    image = image.convert("RGB")
    image.thumbnail((1024, 1024))

    return image


# ==========================================
# 5. KI-BILDGENERIERUNG MIT QWEN
# ==========================================

def generate_image(original_image, description):

    prompt = f"""
    Edit the provided photograph of a real urban location.

    Create a realistic architectural visualization of
    the proposed urban improvement.

    Requested changes:
    {description}

    IMPORTANT INSTRUCTIONS:

    - Preserve the original camera perspective.
    - Preserve existing buildings and surroundings.
    - Keep the original street layout.
    - Integrate all requested improvements naturally.
    - Use realistic materials, proportions and lighting.
    - Do not unnecessarily change existing structures.
    - Produce a photorealistic architectural visualization.

    The result should look like a photograph of the
    location after the proposed urban improvements
    have been implemented.
    """

    # Temporäre Bilddatei für die Gradio-Schnittstelle erstellen
    with tempfile.TemporaryDirectory() as temp_dir:

        image_path = os.path.join(temp_dir, "stadtfoto.png")

        original_image.convert("RGB").save(image_path)

        client = get_client()

        # Qwen-Image-Edit-2511 auf Hugging Face aufrufen
        result = client.predict(
            handle_file(image_path),
            prompt,
            "Fast",
            4,
            1.0,
            "blurry, low resolution, distorted",
            -1,
            "",
            None,
            0.9,
            api_name="/on_edit_generate"
        )

        if not result or result[0] is None:
            raise RuntimeError(
                "Das Modell hat kein Bild zurückgegeben."
            )

        generated_file = result[0]

        # Gradio kann das Ergebnis als Dateipfad oder Dictionary liefern
        if isinstance(generated_file, dict):
            generated_file = (
                generated_file.get("path")
                or generated_file.get("url")
            )

        if isinstance(generated_file, Image.Image):
            return generated_file.convert("RGB").copy()

        if not generated_file:
            raise RuntimeError(
                "Der Bildpfad des Ergebnisses fehlt."
            )

        # Bild öffnen und unabhängig von der temporären Datei speichern
        with Image.open(generated_file) as image:
            return image.convert("RGB").copy()


# ==========================================
# 6. ÜBERSCHRIFT
# ==========================================

st.title("🌳 Meine Stadt. Meine Idee.")

st.markdown("""
<div class="intro">
<h3>Gestalte deine Stadt der Zukunft!</h3>

Lade ein Foto eines öffentlichen Ortes hoch und
beschreibe, wie du diesen Ort verändern möchtest.

Unsere KI erstellt auf Basis deines Fotos und deiner
Beschreibung eine realistische Visualisierung deiner Idee.
</div>
""", unsafe_allow_html=True)


# ==========================================
# 7. FOTO HOCHLADEN
# ==========================================

st.header("1. Ursprüngliches Foto")

uploaded_file = st.file_uploader(
    "Wähle ein Foto deines gewünschten Ortes aus.",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file:

    original_image = Image.open(uploaded_file)
    original_image = prepare_image(original_image)

    st.image(
        original_image,
        caption="Ursprünglicher Zustand",
        use_container_width=True
    )

    st.divider()

    # ======================================
    # 8. BESCHREIBUNG
    # ======================================

    st.header("2. Beschreibe deine Veränderung")

    description = st.text_area(
        "Was möchtest du an diesem Ort verändern?",
        placeholder="""
Beispielsweise:

Ich möchte den Platz grüner und einladender gestalten.

Es sollen mehrere Bäume und Blumenbeete entstehen.
Außerdem wünsche ich mir neue Sitzbänke,
Fahrradständer und eine moderne Beleuchtung.
        """,
        height=180
    )

    st.divider()

    # ======================================
    # 9. OPTIONALE BILDBEARBEITUNG
    # ======================================

    st.header("3. Gestalte dein Foto (optional)")

    use_drawing = st.checkbox(
        "Ich möchte meine Veränderungen direkt in das Foto einzeichnen."
    )

    edited_image = original_image

    if use_drawing:

        st.write("""
        Verwende die Zeichenwerkzeuge, um deine Ideen
        direkt auf dem Foto zu markieren.
        """)

        col1, col2 = st.columns(2)

        with col1:

            drawing_mode = st.selectbox(
                "Zeichenwerkzeug",
                [
                    "freedraw",
                    "line",
                    "rect",
                    "circle",
                    "transform"
                ]
            )

        with col2:

            stroke_width = st.slider(
                "Strichstärke",
                1,
                25,
                5
            )

        stroke_color = st.color_picker(
            "Farbe",
            "#00AA55"
        )

        canvas_result = st_canvas(
            fill_color="rgba(0, 170, 85, 0.25)",
            stroke_width=stroke_width,
            stroke_color=stroke_color,
            background_image=original_image,
            update_streamlit=True,
            height=original_image.height,
            width=original_image.width,
            drawing_mode=drawing_mode,
            return_image_data=True,
            key="city_canvas"
        )

        if canvas_result.image_data is not None:

            edited_image = Image.fromarray(
                canvas_result.image_data.astype(np.uint8)
            ).convert("RGB")

            st.image(
                edited_image,
                caption="Deine bearbeitete Vorlage",
                use_container_width=True
            )

    st.divider()

    # ======================================
    # 10. KI-BILD GENERIEREN
    # ======================================

    st.header("4. Erstelle deine Stadtvision")

    st.write("""
    Die KI verwendet dein Foto und deine Beschreibung,
    um eine neue Visualisierung deines Projekts zu erstellen.
    """)

    if st.button(
        "✨ Mein Projektbild generieren",
        use_container_width=True
    ):

        if not description.strip():

            st.warning(
                "Bitte beschreibe zuerst deine gewünschte Veränderung."
            )

        else:

            with st.spinner(
                "Die KI gestaltet deine Stadtvision. "
                "Das kann einen Moment dauern..."
            ):

                try:

                    generated_image = generate_image(
                        edited_image,
                        description
                    )

                    st.session_state["generated_image"] = generated_image
                    st.session_state["source_image"] = original_image

                    st.success(
                        "Dein Projektbild wurde erfolgreich erstellt!"
                    )

                except Exception as error:

                    st.error(
                        "Bei der Bildgenerierung ist ein Fehler aufgetreten."
                    )

                    st.code(str(error))

                    st.info(
                        "Mögliche Ursachen sind eine volle ZeroGPU-Warteschlange, "
                        "ein vorübergehend nicht erreichbarer Space oder ein "
                        "aufgebrauchtes Tageslimit."
                    )

    # ======================================
    # 11. ERGEBNIS ANZEIGEN
    # ======================================

    if "generated_image" in st.session_state:

        st.divider()

        st.header("5. Deine fertige Stadtvision")

        col1, col2 = st.columns(2)

        with col1:

            st.image(
                st.session_state["source_image"],
                caption="Vorher – ursprünglicher Zustand",
                use_container_width=True
            )

        with col2:

            st.image(
                st.session_state["generated_image"],
                caption="Nachher – KI-generierte Vision",
                use_container_width=True
            )

        st.success(
            "Aus deiner Idee ist eine visuelle Stadtvision geworden!"
        )

        # Bild herunterladen

        buffer = io.BytesIO()

        st.session_state["generated_image"].save(
            buffer,
            format="PNG"
        )

        st.download_button(
            label="📥 Projektbild herunterladen",
            data=buffer.getvalue(),
            file_name="meine_stadtvision.png",
            mime="image/png",
            use_container_width=True
        )

else:

    st.info(
        "Bitte lade zunächst ein Foto hoch, um zu beginnen."
    )


# ==========================================
# 12. FOOTER
# ==========================================

st.divider()

st.markdown("""
<center>
<b>Meine Stadt. Meine Idee.</b>

<br>

Gemeinsam gestalten wir lebenswerte Städte.
</center>
""", unsafe_allow_html=True)
