
import streamlit as st
from PIL import Image, ImageOps
from streamlit_drawable_canvas import st_canvas
import io

st.set_page_config(
    page_title="Meine Stadt. Meine Idee.",
    layout="wide"
)

st.title("🏙️ Meine Stadt. Meine Idee.")
st.write("Gestalte deinen eigenen Stadtplatz!")


# Foto hochladen
uploaded = st.file_uploader(
    "Lade ein Foto deiner Stadt hoch",
    type=["jpg", "jpeg", "png"]
)

if uploaded is not None:

    image = Image.open(
        io.BytesIO(uploaded.getvalue())
    )

    image = ImageOps.exif_transpose(image).convert("RGB")
    image.thumbnail((800, 600))

    width, height = image.size

    st.success("Foto erfolgreich geladen!")

    # Farbpalette
    st.subheader("🎨 Wähle deine Zeichenfarbe")

    colors = {
        "🟢 Grün": "#00AA00",
        "🌳 Dunkelgrün": "#006400",
        "🟤 Braun": "#8B4513",
        "🔴 Rot": "#FF0000",
        "🔵 Blau": "#0000FF",
        "🟡 Gelb": "#FFD700",
        "🟠 Orange": "#FF8C00",
        "🌸 Pink": "#FF69B4",
        "🟣 Lila": "#800080",
        "⚫ Schwarz": "#000000",
        "⚪ Weiß": "#FFFFFF",
        "🩵 Türkis": "#40E0D0"
    }

    selected_color = st.selectbox(
        "Farbe auswählen",
        list(colors.keys())
    )

    drawing_color = colors[selected_color]

    # Eigene Farbe
    st.subheader("Eigene Farbe")

    custom_color = st.color_picker(
        "Wähle eine beliebige Farbe",
        "#00AA00"
    )

    use_custom = st.checkbox(
        "Eigene Farbe verwenden"
    )

    if use_custom:
        drawing_color = custom_color

    # Strichstärke
    st.subheader("🖌️ Stiftgröße")

    stroke_width = st.slider(
        "Wie dick soll dein Stift sein?",
        min_value=2,
        max_value=40,
        value=8
    )

    # Zeichenmodus
    st.subheader("✏️ Zeichenwerkzeuge")

    drawing_mode = st.selectbox(
        "Werkzeug auswählen",
        [
            "freedraw",
            "line",
            "rect",
            "circle",
            "transform"
        ],
        format_func=lambda x: {
            "freedraw": "Freihand zeichnen",
            "line": "Linie zeichnen",
            "rect": "Rechteck zeichnen",
            "circle": "Kreis zeichnen",
            "transform": "Objekte verschieben"
        }[x]
    )

    # Zeichenfläche
    st.subheader("🏙️ Dein Stadtplatz")

    canvas_result = st_canvas(
        fill_color="rgba(0, 170, 0, 0.25)",
        stroke_width=stroke_width,
        stroke_color=drawing_color,
        background_color="#FFFFFF",
        background_image=image,
        background_image_fit="contain",
        width=width,
        height=height,
        drawing_mode=drawing_mode,
        update_streamlit=True,
        return_image_data=True,
        key="city_canvas"
    )

    # Ergebnis anzeigen
    if canvas_result.image_data is not None:

        st.subheader("Dein gestalteter Stadtplatz")

        st.image(
            canvas_result.image_data.astype("uint8"),
            caption="Deine Zeichnung",
            width=width
        )

else:

    st.info(
        "Bitte lade zuerst ein Foto hoch."
    )
