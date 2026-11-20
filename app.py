"""Meine Stadt. Meine Idee. – draw ideas on a photo of a public place."""

from __future__ import annotations

from collections import Counter
from datetime import datetime
from base64 import b64encode
from hashlib import sha256
from html import escape
import json
from pathlib import Path
from uuid import uuid4

import cv2
import numpy as np
import streamlit as st
from PIL import Image, ImageOps, UnidentifiedImageError
from streamlit_drawable_canvas import st_canvas

from classifier import EXPECTED_CLASSES, classify_crop, load_model, read_labels
from image_processing import (
    annotate_regions,
    composite_canvas,
    find_drawings,
    prepare_for_canvas,
    white_background_crop,
)
from project_store import (
    PROJECTS_DB,
    add_comment,
    describe_project,
    load_comments,
    load_projects,
    save_project,
    toggle_like,
)
from ui_theme import apply_app_theme, render_editor_step, render_page_header


ROOT = Path(__file__).resolve().parent
MODEL_PATH = ROOT / "model" / "keras_model.h5"
LABELS_PATH = ROOT / "model" / "labels.txt"

st.set_page_config(page_title="Meine Stadt. Meine Idee.", page_icon="🏙️", layout="wide")
apply_app_theme()


def _submit_comment(project_id: str, voter_id: str, author_key: str, body_key: str) -> None:
    try:
        add_comment(
            project_id,
            st.session_state.get(body_key, ""),
            st.session_state.get(author_key, ""),
            PROJECTS_DB,
        )
    except ValueError as exc:
        st.session_state["project_comment_error"] = str(exc)
    else:
        st.session_state[body_key] = ""
        st.session_state["project_comment_success"] = project_id


def _toggle_project_analysis(project_id: str) -> None:
    key = "project_analysis_open_id"
    if st.session_state.get(key) == project_id:
        st.session_state.pop(key, None)
    else:
        st.session_state[key] = project_id


def _project_difference_text(project: dict) -> str:
    singular = {
        "Bäume": "Baum",
        "Beete": "Beet",
        "Fahrradständer": "Fahrradständer",
        "Bänke": "Bank",
    }
    additions = []
    for label in EXPECTED_CLASSES:
        count = int(project["counts"].get(label, 0))
        if count:
            noun = label if count != 1 else singular.get(label, label)
            additions.append(f"{count} {noun}")
    if not additions:
        return "Im Entwurf wurden keine zusätzlichen Objekte erkannt."
    return "Hinzugefügt: " + " · ".join(additions)


def render_comparison(project: dict, container_key: str) -> None:
    with st.container(border=True, key=container_key):
        st.markdown('<div class="comparison-heading">Vorher-Nachher-Vergleich</div>', unsafe_allow_html=True)
        st.caption("So verändert der Vorschlag den ausgewählten Ort.")
        before_column, arrow_column, after_column = st.columns([1, 0.12, 1], vertical_alignment="center")
        with before_column:
            st.html('<span class="comparison-label">VORHER</span>')
            st.image(project["original_png"], use_container_width=True)
        with arrow_column:
            st.html('<div class="comparison-arrow" aria-hidden="true">→</div>')
        with after_column:
            st.html('<span class="comparison-label">NACHHER</span>')
            st.image(project["annotated_png"], use_container_width=True)
        st.markdown(f'<div class="comparison-card-note">{escape(_project_difference_text(project))}</div>', unsafe_allow_html=True)


def render_project_cards(projects: list[dict]) -> None:
    for row_start in range(0, len(projects), 2):
        card_columns = st.columns(2)
        for column, project in zip(card_columns, projects[row_start : row_start + 2]):
            with column:
                description_preview = project["description"].split(".", 1)[0].strip()
                # Start previews show the unboxed draft; classification outlines stay in analysis.
                image_data = b64encode(project["edited_png"]).decode("ascii")
                project_url = f"?projekt={project['id']}"
                location_label = project["location"] or "Ort nicht angegeben"
                card_html = f"""
                    <a class="city-project-card" href="{project_url}" aria-label="Projekt {escape(project['title'])} öffnen">
                      <div class="project-card-visual">
                        <img src="data:image/png;base64,{image_data}" alt="Entwurf: {escape(project['title'])}">
                        <div class="project-card-overlay">
                          <span class="project-location">📍 {escape(location_label)}</span>
                          <h3>{escape(project['title'])}</h3>
                        </div>
                        <span class="project-card-open" aria-hidden="true">↗</span>
                      </div>
                      <div class="project-card-copy">
                        <p>{escape(description_preview)}</p>
                        <span class="project-card-link">Projekt ansehen <b>→</b></span>
                      </div>
                    </a>
                """
                column.html(card_html)


def show_project_details(project: dict, voter_id: str) -> None:
    if st.button("← Zurück zum Start", key=f"back_to_projects_{project['id']}"):
        st.query_params.pop("projekt", None)
        st.session_state.pop("project_analysis_open_id", None)
        st.rerun()

    submitted_at = datetime.fromisoformat(project["created_at"]).astimezone()
    render_page_header(
        "PROJEKTDETAILS",
        project["title"],
        f'{project["location"] or "Ort nicht angegeben"} · Eingereicht am {submitted_at:%d.%m.%Y um %H:%M}',
        "🌿",
    )

    st.image(project["annotated_png"], caption="Projektbild · Entwurf", width=560)

    like_column, analysis_column = st.columns([1, 2])
    like_label = "♥ Gefällt dir nicht mehr" if project["liked_by_viewer"] else "♡ Gefällt mir"
    if like_column.button(f'{like_label} · {project["like_count"]}', key=f'detail_like_{project["id"]}'):
        toggle_like(project["id"], voter_id, PROJECTS_DB)
        st.rerun()

    analysis_is_open = st.session_state.get("project_analysis_open_id") == project["id"]
    analysis_column.button(
        "Analyse schließen" if analysis_is_open else "⌕  Analyse starten",
        key=f'open_analysis_{project["id"]}',
        use_container_width=True,
        on_click=_toggle_project_analysis,
        args=(project["id"],),
    )

    if analysis_is_open:
        st.subheader("Das wurde dem Ort hinzugefügt")
        render_comparison(project, f"detail_comparison_{project['id']}")
        st.html(
            f'<div class="project-description-strip"><span class="leaf">🌿</span>'
            f'<span><strong>{escape(_project_difference_text(project))}</strong><br>{escape(project["description"])}</span></div>'
        )
        metric_columns = st.columns(len(EXPECTED_CLASSES))
        for column, label in zip(metric_columns, EXPECTED_CLASSES):
            column.metric(label, project["counts"].get(label, 0))

    if analysis_is_open and project["items"]:
        with st.expander("Erkannte Objekte und Positionen", expanded=analysis_is_open):
            for item in project["items"]:
                st.write(
                    f'{item["number"]}. {item["label"]} · Konfidenz '
                    f'{item["confidence"]:.0%} · {item["position"]} · {item["size"]}'
                )

    st.subheader(f'Kommentare ({project["comment_count"]})')
    comments = load_comments(project["id"], PROJECTS_DB)
    if comments:
        for comment in comments:
            comment_date = datetime.fromisoformat(comment["created_at"]).astimezone()
            st.markdown(f'**{comment["author"]}** · {comment_date:%d.%m.%Y %H:%M}')
            st.write(comment["body"])
    else:
        st.caption("Noch keine Kommentare. Teile einen freundlichen und konstruktiven Gedanken.")

    error = st.session_state.pop("project_comment_error", None)
    if error:
        st.error(error)
    success_for = st.session_state.pop("project_comment_success", None)
    if success_for == project["id"]:
        st.success("Dein Kommentar wurde hinzugefügt.")

    author_key = f"comment_author_{project['id']}"
    body_key = f"comment_body_{project['id']}"
    st.text_input("Name (optional)", key=author_key, max_chars=80)
    st.text_area("Kommentar", key=body_key, max_chars=2000, height=90)
    st.button(
        "Kommentar hinzufügen",
        key=f"add_comment_{project['id']}",
        on_click=_submit_comment,
        args=(project["id"], voter_id, author_key, body_key),
    )


def show_start_page(voter_id: str) -> None:
    render_page_header(
        "GEMEINSAM GESTALTEN",
        "Projektübersicht",
        "Klicke ein Projektbild an, um seine Reaktionen und den Entwurf zu entdecken.",
        "🏙️",
    )
    projects = load_projects(PROJECTS_DB, voter_id)
    selected_id = st.query_params.get("projekt")
    if selected_id:
        selected = next((item for item in projects if item["id"] == selected_id), None)
        if selected is not None:
            show_project_details(selected, voter_id)
            return
        st.query_params.pop("projekt", None)

    if not projects:
        st.info("Es gibt noch keine Projekte. Wechsle zu „Eigene Projekte“, um einen Vorschlag einzureichen.")
        return

    location_query = st.text_input(
        "Projekte nach Ortsnamen suchen",
        placeholder="Zum Beispiel: Marktplatz oder Stadtpark",
        key="project_location_search",
    ).strip()
    if location_query:
        normalized_query = location_query.casefold()
        visible_projects = [
            project for project in projects
            if normalized_query in project["location"].casefold()
        ]
        project_word = "Projekt" if len(visible_projects) == 1 else "Projekte"
        st.caption(f'{len(visible_projects)} {project_word} für „{location_query}“')
    else:
        visible_projects = projects
        st.caption("Beliebteste Projekte" if visible_projects else "")

    if visible_projects:
        render_project_cards(visible_projects)
    elif location_query:
        st.info(f'Für „{location_query}“ wurden keine Projekte gefunden.')


if st.session_state.pop("open_projects_after_submit", False):
    st.session_state["main_navigation"] = "Start"
legacy_navigation = st.session_state.get("main_navigation")
if legacy_navigation == "Projektübersicht":
    st.session_state["main_navigation"] = "Start"
elif legacy_navigation == "Idee einreichen":
    st.session_state["main_navigation"] = "Eigene Projekte"
elif legacy_navigation == "Analyse":
    st.session_state["main_navigation"] = "Eigene Projekte"
if st.query_params.get("projekt") and "main_navigation" not in st.session_state:
    st.session_state["main_navigation"] = "Start"

with st.sidebar:
    st.html(
        "<div class='sidebar-brand'><div class='sidebar-brand-icon'>🏙️</div>"
        "<div>Meine Stadt.<br>Meine Idee.<small>GEMEINSAM GESTALTEN</small></div></div>"
    )
    selected_page = st.radio(
        "Navigation",
        ["Start", "Eigene Projekte"],
        key="main_navigation",
        format_func=lambda page: {
            "Start": "⌂   Start",
            "Eigene Projekte": "▥   Eigene Projekte",
        }[page],
    )

if selected_page == "Start":
    voter_id = st.session_state.setdefault("project_voter_id", uuid4().hex)
    show_start_page(voter_id)
    st.stop()

if st.query_params.get("projekt"):
    st.query_params.pop("projekt", None)

render_page_header(
    "EIGENE PROJEKTE",
    "Neuen Vorschlag gestalten.",
    "Lade einen Ort hoch, zeichne deine Ergänzungen ein und prüfe den Vorher-Nachher-Effekt.",
    "🔎",
)


@st.cache_resource(show_spinner="KI-Modell wird geladen …")
def cached_model(model_path: str, model_mtime_ns: int):
    # Include the file timestamp so replacing the H5 invalidates Streamlit's model cache.
    return load_model(model_path)


with st.container(border=True, key="upload_panel"):
    render_editor_step("01", "Foto und Ort", "Lade ein Bild hoch und gib an, wo sich der Platz befindet.")
    image_source = st.radio(
        "Bildquelle",
        ["Datei hochladen", "Foto aufnehmen"],
        horizontal=True,
        key="image_source",
    )
    if image_source == "Datei hochladen":
        uploaded = st.file_uploader("Foto auswählen", type=["png", "jpg", "jpeg", "webp"])
    else:
        uploaded = st.camera_input(
            "Foto aufnehmen",
            help="Erlaube den Kamerazugriff im Browser und nimm direkt ein Foto auf.",
        )

    if uploaded is not None:
        upload_signature = sha256(uploaded.getvalue()).hexdigest()[:12]
        location_key = f"project_location_{upload_signature}"
        location_name = st.text_input(
            "Ortsname (erforderlich)",
            placeholder="Zum Beispiel: Marktplatz Mainz",
            help="Der Ortsname wird in der Projektübersicht zur Suche verwendet.",
            max_chars=120,
            key=location_key,
        )
    elif image_source == "Datei hochladen":
        st.caption("Unterstützte Formate: PNG, JPG und WEBP.")
    else:
        st.caption("Sobald du ein Foto aufgenommen hast, kannst du es mit dem Ortsnamen versehen und bearbeiten.")

if uploaded is None:
    st.stop()

try:
    original_image = ImageOps.exif_transpose(Image.open(uploaded)).convert("RGB")
except (UnidentifiedImageError, OSError, ValueError) as exc:
    st.error(f"Das Bild konnte nicht geöffnet werden. Bitte wähle eine gültige PNG-, JPG- oder WEBP-Datei. ({exc})")
    st.stop()

canvas_image, canvas_width, canvas_height = prepare_for_canvas(original_image)
original_array = np.asarray(canvas_image)
with st.container(border=True, key="drawing_panel"):
    render_editor_step("02", "Deine Idee einzeichnen", "Zeichne direkt auf das Foto. Werkzeugleiste und Rückgängig-Funktion findest du auf der Zeichenfläche.")
    tool = st.selectbox("Werkzeug", ["Freihand", "Rechteck", "Kreis", "Auswählen und anpassen"], index=0)
    tool_map = {
        "Freihand": "freedraw",
        "Rechteck": "rect",
        "Kreis": "circle",
        "Auswählen und anpassen": "transform",
    }
    control_col, color_col = st.columns([1, 2])
    with control_col:
        stroke = st.slider("Strichbreite", 2, 40, 8)
    with color_col:
        color = st.color_picker("Zeichenfarbe", "#e53935")

    if st.button("Zeichnung zurücksetzen"):
        st.session_state["canvas_reset"] = st.session_state.get("canvas_reset", 0) + 1

    canvas_key = sha256(uploaded.getvalue()).hexdigest()[:16]
    canvas_result = st_canvas(
        fill_color="rgba(0, 0, 0, 0)",
        stroke_width=stroke,
        stroke_color=color,
        background_image=canvas_image,
        update_streamlit=True,
        height=canvas_height,
        width=canvas_width,
        drawing_mode=tool_map[tool],
        key=f"city_canvas_{canvas_key}_{st.session_state.get('canvas_reset', 0)}",
        display_toolbar=True,
    )

if canvas_result.image_data is None:
    st.info("Zeichne oben eine oder mehrere Ideen ein.")
    st.stop()

canvas_rgba = np.asarray(canvas_result.image_data, dtype=np.uint8)
try:
    composite = composite_canvas(original_array, canvas_rgba)
    boxes = find_drawings(original_array, composite)
except (ValueError, cv2.error) as exc:
    st.error(f"Die Zeichnung konnte nicht ausgewertet werden: {exc}")
    st.stop()

if not boxes:
    st.info("Es wurden noch keine sichtbaren Zeichnungen erkannt. Zeichne mit einer kontrastierenden Farbe.")
    st.stop()

with st.container(border=True, key="comparison_panel"):
    render_editor_step("03", "Vorher und nachher", "Vergleiche den Ort mit deiner eingezeichneten Idee.")
    original_col, edited_col = st.columns(2)
    original_col.image(canvas_image, caption="Vorher · Originalfoto", use_container_width=True)
    edited_col.image(composite, caption="Nachher · Deine Idee", use_container_width=True)

try:
    if not MODEL_PATH.is_file():
        raise FileNotFoundError("Die Modelldatei model/keras_model.h5 fehlt.")
    if not LABELS_PATH.is_file():
        raise FileNotFoundError("Die Label-Datei model/labels.txt fehlt.")
    labels = read_labels(LABELS_PATH)
    model = cached_model(str(MODEL_PATH), MODEL_PATH.stat().st_mtime_ns)
    output_shape = model.output_shape
    if isinstance(output_shape, list):
        output_shape = output_shape[0]
    if output_shape[-1] != len(EXPECTED_CLASSES):
        raise ValueError("Das vorhandene Modell muss genau vier Ausgabeklassen besitzen.")
except Exception as exc:
    st.error(f"Die Klassifikation ist derzeit nicht verfügbar: {exc}")
    st.caption("Das vorhandene Modell wird aus dem Ordner model geladen. Lege dort keras_model.h5 und labels.txt ab.")
    st.stop()

results: list[dict] = []
failures: list[str] = []
for number, (x, y, width, height) in enumerate(boxes, start=1):
    try:
        classification_image = white_background_crop(canvas_rgba, (x, y, width, height))
        label, confidence = classify_crop(model, classification_image, labels)
        scale_x = original_image.width / canvas_width
        scale_y = original_image.height / canvas_height
        results.append({
            "number": number,
            "label": label,
            "confidence": confidence,
            "position": f"x {round(x * scale_x)}, y {round(y * scale_y)} px",
            "size": f"{round(width * scale_x)} × {round(height * scale_y)} px",
            "box": (x, y, width, height),
            "classification_image": classification_image,
        })
    except Exception as exc:
        failures.append(f"Bereich {number}: {exc}")

annotated = annotate_regions(composite, results)
counts = Counter(item["label"] for item in results)
with st.container(border=True, key="results_panel"):
    render_editor_step("04", "Erkannte Ergänzungen", "Das Modell ordnet jede Zeichnung einzeln einer Kategorie zu.")
    annotated_col, count_col = st.columns([1.5, 1])
    annotated_col.image(annotated, caption="Dein Entwurf mit klassifizierten Bereichen", use_container_width=True)

    with count_col:
        st.markdown("#### Anzahl je Kategorie")
        for name in EXPECTED_CLASSES:
            st.metric(name, counts.get(name, 0))
        if results:
            st.markdown("#### Einzelne Bereiche")
            for item in results:
                st.image(
                    item["classification_image"],
                    caption=f'Weißer KI-Ausschnitt · {item["label"]}',
                    width=140,
                )
                st.write(f'**{item["number"]}. {item["label"]}** · Konfidenz {item["confidence"]:.0%} · {item["position"]} · {item["size"]}')
        else:
            st.warning("Es konnten keine Bereiche klassifiziert werden.")

    if failures:
        st.warning("Einige Bereiche wurden übersprungen: " + "; ".join(failures))
    st.caption("Die KI klassifiziert jeden Bereich einzeln auf weißem Hintergrund. Das Ortsfoto bleibt im Entwurf sichtbar, wird aber nicht als Klassifikationshintergrund verwendet.")

with st.container(border=True, key="proposal_panel"):
    render_editor_step("05", "Projektvorschlag", "Prüfe den automatisch erstellten Text und teile deine Idee.")
    try:
        suggested_title, suggested_description = describe_project(dict(counts))
    except ValueError:
        st.info("Es gibt noch keine klassifizierten Objekte, aus denen ein Projektvorschlag erstellt werden kann.")
    else:
        count_signature = sha256(
            json.dumps(dict(counts), ensure_ascii=False, sort_keys=True).encode("utf-8")
        ).hexdigest()[:12]
        st.caption("Titel und Beschreibung werden aus den erkannten Objekten vorgeschlagen. Den Titel kannst du vor dem Einreichen anpassen.")
        title_key = f"project_title_{count_signature}"
        st.markdown("**Vorgeschlagene Beschreibung**")
        st.write(suggested_description)
        with st.form(f"submit_project_{count_signature}"):
            project_title = st.text_input(
                "Projekttitel",
                value=suggested_title,
                key=title_key,
            )
            submitted = st.form_submit_button("Projekt einreichen")

        if submitted:
            try:
                if not location_name.strip():
                    raise ValueError("Bitte gib den Ortsnamen ein, damit das Projekt in der Übersicht gefunden werden kann.")
                save_project(
                    title=st.session_state.get(title_key, project_title),
                    description=suggested_description,
                    counts=dict(counts),
                    items=results,
                    original=original_image,
                    edited=composite,
                    annotated=annotated,
                    database=PROJECTS_DB,
                    location=location_name,
                )
            except (OSError, ValueError) as exc:
                st.error(f"Das Projekt konnte nicht gespeichert werden: {exc}")
            except Exception as exc:
                st.error(f"Beim Speichern ist ein Datenbankfehler aufgetreten: {exc}")
            else:
                st.session_state["open_projects_after_submit"] = True
                st.rerun()
