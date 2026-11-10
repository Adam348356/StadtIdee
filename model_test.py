
import streamlit as st
from PIL import Image, ImageOps
from streamlit_drawable_canvas import st_canvas

import io

st.set_page_config(
    page_title="Zeichenflächen-Test",
    layout="wide"
)

st.title("Test: Foto als Zeichenhintergrund")

st.write("Streamlit-Zeichenflächen-Test")

uploaded = st.file_uploader(
    "Foto hochladen",
    type=["jpg", "jpeg", "png"]
)

if uploaded is not None:

    image = Image.open(
        io.BytesIO(uploaded.getvalue())
    )

    image = ImageOps.exif_transpose(image).convert("RGB")

    image.thumbnail((800, 600))

    width, height = image.size

    st.success(
        f"Foto geladen: {width} × {height} Pixel"
    )

    st.subheader("Zeichenfläche")

    result = st_canvas(
        fill_color="rgba(0, 255, 0, 0.3)",
        stroke_width=8,
        stroke_color="#00AA00",
        background_color="#FFFFFF",
        background_image=image,
        background_image_fit="contain",
        width=width,
        height=height,
        drawing_mode="freedraw",
        update_streamlit=True,
        return_image_data=True,
        key="canvas_test"
    )

    st.write("Zeichenfläche wurde aufgerufen.")

    if result.image_data is not None:
        st.write("Bilddaten werden zurückgegeben.")
